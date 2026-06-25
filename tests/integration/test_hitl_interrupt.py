"""
tests/integration/test_hitl_interrupt.py
-----------------------------------------
Integration tests for the LangGraph HITL interrupt/resume mechanism in
agent/orchestrator.py.

These are INTEGRATION tests, not unit tests, because they exercise real
LangGraph machinery (StateGraph, MemorySaver checkpointer, interrupt_before,
invoke/update_state/resume). Some scenarios use a real small PDF with real
OpenAI API calls; others mock specific pipeline stages to force deterministic
HITL/no-HITL outcomes without depending on the LLM's non-determinism.

LangGraph version confirmed: 0.1.19
Interrupt mechanism: compile(interrupt_before=["check_human_review"])
Checkpointer: MemorySaver (module-level singleton in orchestrator.py)

What this file tests:
    1. CLEAN RUN: a document that produces NO HIGH risk findings completes
       without pausing. Demonstrates the non-interrupt path.
       (Achieved by mocking flag_risks to return all LOW findings.)

    2. INTERRUPT: a document with a missing critical clause triggers HITL.
       The graph pauses, a ReviewItem lands in the queue with the correct
       thread_id, and the graph does NOT proceed past the interrupt.
       (Achieved by mocking flag_risks to return a HIGH missing-clause finding.)

    3. RESUME with approve: resume_after_review() with action="approve"
       causes the graph to complete. The final_state is populated and
       human_decisions_applied > 0.

    4. OVERRIDE: resume_after_review() with action="correct" and a
       corrected_value changes what ends up in the final state compared
       to what the AI originally found. This proves the override actually
       takes effect, not just that the function runs.

How to run:
    pytest tests/integration/test_hitl_interrupt.py -v -s

Note: Tests 1-4 mock the LLM-calling nodes to avoid real API costs and
non-determinism in CI. The real pipeline (extraction, chunking, embeddings)
still runs on a real PDF for test 1, but flag_risks is mocked to return
predictable findings.
"""

import sys
import os
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# We need to patch agent.human_review's queue paths before the orchestrator
# module imports, so we use importlib to control load order.
import agent.human_review as hr_module

# ---------------------------------------------------------------------------
# Real PDF for extraction/chunking (no API needed for these stages)
# ---------------------------------------------------------------------------

BASE = Path(__file__).parent.parent.parent / "data" / "pdf"
REAL_PDF = BASE / "ndas" / "NDA_2_Bakhu_Holdings,_Corp._(BKUH).pdf"


# ---------------------------------------------------------------------------
# Fixture: isolate review queue storage per test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_queue(tmp_path):
    """Redirect JSON queue files to a temp dir for each test."""
    queue_dir = tmp_path / "review_queue"
    pending_file = queue_dir / "pending_reviews.json"
    resolved_file = queue_dir / "resolved_reviews.json"
    with (
        patch.object(hr_module, "_QUEUE_DIR", queue_dir),
        patch.object(hr_module, "_PENDING_FILE", pending_file),
        patch.object(hr_module, "_RESOLVED_FILE", resolved_file),
    ):
        yield


# ---------------------------------------------------------------------------
# Helpers: build mock findings and clauses
# ---------------------------------------------------------------------------

def _make_low_risk_findings():
    """10 findings all LOW risk, none missing -- should not trigger HITL."""
    from schemas.risk import RiskFinding
    from config import CLAUSE_CATEGORIES
    return [
        RiskFinding(
            clause_type=cat,
            risk_level="LOW",
            is_missing=False,
            reason="Clause present and standard.",
            precedent_applied=False,
            precedent_note=None,
        )
        for cat in CLAUSE_CATEGORIES
    ]


def _make_high_risk_missing_findings():
    """One HIGH risk missing-clause finding plus 9 LOW -- triggers HITL."""
    from schemas.risk import RiskFinding
    from config import CLAUSE_CATEGORIES
    findings = []
    for i, cat in enumerate(CLAUSE_CATEGORIES):
        if i == 0:
            findings.append(RiskFinding(
                clause_type=cat,
                risk_level="HIGH",
                is_missing=True,
                reason="Critical clause absent from document.",
                precedent_applied=False,
                precedent_note=None,
            ))
        else:
            findings.append(RiskFinding(
                clause_type=cat,
                risk_level="LOW",
                is_missing=False,
                reason="Clause present and standard.",
                precedent_applied=False,
                precedent_note=None,
            ))
    return findings


def _make_mock_clauses():
    """10 ExtractedClause objects (no requires_human_review)."""
    from schemas.clause import ExtractedClause
    from config import CLAUSE_CATEGORIES
    return [
        ExtractedClause(
            clause_type=cat,
            is_present=True,
            extracted_text=f"Sample text for {cat}.",
            page_reference=1,
            confidence=0.85,
            source_chunk_id=f"chunk-001",
            retry_count=0,
            requires_human_review=False,
        )
        for cat in CLAUSE_CATEGORIES
    ]


# ---------------------------------------------------------------------------
# Shared mock for classify + extract_clauses (to avoid real API calls)
# ---------------------------------------------------------------------------

def _mock_classification():
    from schemas.clause import DocumentClassification
    return DocumentClassification(
        document_type="NDA",
        confidence=0.95,
        reasoning="Mock classification for test.",
        retry_count=0,
    )


# ---------------------------------------------------------------------------
# TEST 1: Clean run — no HITL triggered, graph completes without pausing
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not REAL_PDF.exists(), reason="Test PDF not found")
def test_clean_run_no_interrupt():
    """
    A run with all LOW risk findings should complete without interrupting.
    Extraction and chunking use the real PDF; classify/extract_clauses/
    flag_risks are mocked so we don't make real API calls and the outcome
    is deterministic.
    """
    from agent.orchestrator import run_pipeline, get_graph_state

    thread_id = f"test-clean-{uuid.uuid4().hex[:8]}"
    mock_clauses = _make_mock_clauses()
    low_findings = _make_low_risk_findings()
    mock_comparisons = []

    with (
        patch("agent.orchestrator.classify_document", return_value=_mock_classification()),
        patch("agent.orchestrator.extract_clauses", return_value=mock_clauses),
        patch("agent.orchestrator.flag_risks", return_value=low_findings),
        patch("agent.orchestrator.compare_to_templates", return_value=mock_comparisons),
    ):
        result = run_pipeline(str(REAL_PDF), thread_id=thread_id)

    print(f"\n[TEST 1 - CLEAN RUN]")
    print(f"  status             : {result['status']}")
    print(f"  needs_human_review : {result['needs_human_review']}")
    print(f"  final_state        : {result['final_state']}")

    assert result["status"] == "completed", f"Expected completed, got {result['status']}"
    assert result["needs_human_review"] is False
    assert result["final_state"] is not None
    assert result["final_state"]["completed"] is True
    assert result["final_state"]["risk_counts"]["HIGH"] == 0

    # Verify graph is NOT paused
    snap = get_graph_state(thread_id)
    assert not snap["is_paused"], f"Graph should not be paused: next={snap['next']}"

    # Verify NO review items in the queue
    pending = hr_module.get_pending_reviews()
    assert len(pending) == 0, f"Expected 0 pending reviews, got {len(pending)}"

    print(f"  PASS: graph completed without pausing, 0 review items queued")


# ---------------------------------------------------------------------------
# TEST 2: Interrupt — missing critical clause triggers HITL pause
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not REAL_PDF.exists(), reason="Test PDF not found")
def test_interrupt_fires_for_missing_critical_clause():
    """
    A HIGH risk missing-clause finding must cause the graph to pause
    (interrupt fires), a ReviewItem must land in the pending queue with
    the correct thread_id, and the graph must NOT proceed to finalize.
    """
    from agent.orchestrator import run_pipeline, get_graph_state

    thread_id = f"test-interrupt-{uuid.uuid4().hex[:8]}"
    mock_clauses = _make_mock_clauses()
    high_findings = _make_high_risk_missing_findings()
    mock_comparisons = []

    with (
        patch("agent.orchestrator.classify_document", return_value=_mock_classification()),
        patch("agent.orchestrator.extract_clauses", return_value=mock_clauses),
        patch("agent.orchestrator.flag_risks", return_value=high_findings),
        patch("agent.orchestrator.compare_to_templates", return_value=mock_comparisons),
    ):
        result = run_pipeline(str(REAL_PDF), thread_id=thread_id)

    print(f"\n[TEST 2 - INTERRUPT]")
    print(f"  status             : {result['status']}")
    print(f"  needs_human_review : {result['needs_human_review']}")
    print(f"  review_items count : {len(result['review_items'])}")

    # Graph must be interrupted
    assert result["status"] == "interrupted", (
        f"Expected 'interrupted', got {result['status']!r}"
    )
    assert result["needs_human_review"] is True
    assert result["final_state"] is None

    # Review items must be in the result
    assert len(result["review_items"]) >= 1

    # Graph must be paused
    snap = get_graph_state(thread_id)
    assert snap["is_paused"], f"Graph should be paused but next={snap['next']}"
    print(f"  next node          : {snap['next']}")

    # ReviewItem must be in the persistent queue with correct thread_id
    pending = hr_module.get_pending_reviews()
    assert len(pending) >= 1
    assert pending[0].thread_id == thread_id
    assert pending[0].trigger_reason in ("missing_critical_clause", "high_risk_finding")

    print(f"  pending review_id  : {pending[0].review_id}")
    print(f"  trigger_reason     : {pending[0].trigger_reason}")
    print(f"  PASS: graph paused, ReviewItem queued with correct thread_id")


# ---------------------------------------------------------------------------
# TEST 3: Resume with approve — graph completes after human approves
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not REAL_PDF.exists(), reason="Test PDF not found")
def test_resume_approve_completes_graph():
    """
    After an interrupt, calling resume_after_review with action='approve'
    must cause the graph to complete and return a populated final_state.
    """
    from agent.orchestrator import run_pipeline, resume_after_review, get_graph_state
    from schemas.review import ReviewDecision

    thread_id = f"test-resume-{uuid.uuid4().hex[:8]}"
    mock_clauses = _make_mock_clauses()
    high_findings = _make_high_risk_missing_findings()
    mock_comparisons = []

    # First: run to interrupt
    with (
        patch("agent.orchestrator.classify_document", return_value=_mock_classification()),
        patch("agent.orchestrator.extract_clauses", return_value=mock_clauses),
        patch("agent.orchestrator.flag_risks", return_value=high_findings),
        patch("agent.orchestrator.compare_to_templates", return_value=mock_comparisons),
    ):
        run_result = run_pipeline(str(REAL_PDF), thread_id=thread_id)

    assert run_result["status"] == "interrupted"
    pending = hr_module.get_pending_reviews()
    assert len(pending) >= 1
    first_review_id = pending[0].review_id

    # Human approves the first review item
    decision = ReviewDecision(review_id=first_review_id, action="approve")
    resume_result = resume_after_review(thread_id, decision)

    print(f"\n[TEST 3 - RESUME APPROVE]")
    print(f"  resume status            : {resume_result['status']}")
    print(f"  final_state              : {resume_result['final_state']}")

    assert resume_result["status"] == "completed", (
        f"Expected 'completed' after approve, got {resume_result['status']!r}"
    )
    assert resume_result["final_state"] is not None
    assert resume_result["final_state"]["completed"] is True

    # Graph must no longer be paused
    snap = get_graph_state(thread_id)
    assert not snap["is_paused"], f"Graph should be done: next={snap['next']}"

    # human_decisions must reflect the approve
    final = resume_result["final_state"]
    assert final["human_decisions_applied"] >= 1

    print(f"  human_decisions_applied  : {final['human_decisions_applied']}")
    print(f"  PASS: graph completed after approve, final_state populated")


# ---------------------------------------------------------------------------
# TEST 4: Override — action="correct" changes what ends up in final state
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not REAL_PDF.exists(), reason="Test PDF not found")
def test_correct_action_changes_final_state():
    """
    resume_after_review with action='correct' must store the corrected_value
    in the final state's human_decisions list, proving the override genuinely
    takes effect (not just that the function runs without error).
    """
    from agent.orchestrator import run_pipeline, resume_after_review
    from schemas.review import ReviewDecision

    thread_id = f"test-correct-{uuid.uuid4().hex[:8]}"
    mock_clauses = _make_mock_clauses()
    high_findings = _make_high_risk_missing_findings()
    mock_comparisons = []

    with (
        patch("agent.orchestrator.classify_document", return_value=_mock_classification()),
        patch("agent.orchestrator.extract_clauses", return_value=mock_clauses),
        patch("agent.orchestrator.flag_risks", return_value=high_findings),
        patch("agent.orchestrator.compare_to_templates", return_value=mock_comparisons),
    ):
        run_result = run_pipeline(str(REAL_PDF), thread_id=thread_id)

    assert run_result["status"] == "interrupted"
    pending = hr_module.get_pending_reviews()
    first_review_id = pending[0].review_id

    # AI said: missing clause / HIGH risk
    # Human corrects it: "actually this clause is present in exhibit A"
    human_correction = {
        "clause_type": pending[0].clause_category,
        "is_present": True,
        "extracted_text": "Human-verified: clause exists in Exhibit A, paragraph 3.",
        "confidence": 1.0,
        "human_overridden": True,
    }
    decision = ReviewDecision(
        review_id=first_review_id,
        action="correct",
        corrected_value=human_correction,
        reviewer_note="Clause was in exhibit A, AI missed it.",
    )
    resume_result = resume_after_review(thread_id, decision)

    print(f"\n[TEST 4 - CORRECT OVERRIDE]")
    print(f"  resume status      : {resume_result['status']}")
    print(f"  final_state        : {resume_result['final_state']}")

    assert resume_result["status"] == "completed"
    final = resume_result["final_state"]
    assert final["completed"] is True

    # The human_decisions_applied count must reflect the correction
    assert final["human_decisions_applied"] >= 1

    # Retrieve the injected state to confirm the corrected value is there
    from agent.orchestrator import get_graph_state
    snap = get_graph_state(thread_id)
    decisions_in_state = snap["values"].get("human_decisions", [])

    print(f"  decisions in state : {decisions_in_state}")

    assert len(decisions_in_state) >= 1
    applied = decisions_in_state[0]
    assert applied["action"] == "correct"
    assert applied["discarded"] is False
    assert applied["value"] is not None
    assert applied["value"].get("human_overridden") is True, (
        "Corrected value must carry through into graph state -- "
        "proves the override took effect, not just that the function ran"
    )

    print(f"  corrected value in state: {applied['value']}")
    print(f"  PASS: correction confirmed in graph state, override took effect")
