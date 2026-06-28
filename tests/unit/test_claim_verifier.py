"""
tests/unit/test_claim_verifier.py
------------------------------------
Unit tests for guardrails/claim_verifier.py.

Tests verify:
  1. A clause with verified evidence and valid page reference -> passed
  2. A LOW risk clause with unverifiable evidence -> removed
  3. A HIGH risk clause with unverifiable evidence -> escalated_to_human_review
     AND produces a real ReviewItem in the queue
  4. A clause citing an out-of-range page -> escalated (if HIGH) or removed (if LOW)
  5. An absent clause (is_present=False) -> passed (structural, no evidence check)
"""

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import agent.human_review as hr_module
from schemas.clause import ExtractedClause
from schemas.document import DocumentExtraction, ExtractionMethod, PageExtraction
from schemas.risk import RiskFinding
from guardrails.claim_verifier import verify_final_claims


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_queue(tmp_path):
    """Redirect JSON queue files so tests don't pollute the real review queue."""
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
# Helper builders
# ---------------------------------------------------------------------------

def _make_doc(pages: dict[int, str]) -> DocumentExtraction:
    page_list = [
        PageExtraction(
            page_number=n,
            text=text,
            method=ExtractionMethod.PYMUPDF,
        )
        for n, text in sorted(pages.items())
    ]
    return DocumentExtraction(
        file_path="/fake/test.pdf",
        file_name="test.pdf",
        file_hash="a" * 64,
        total_pages=max(pages.keys()),
        pages=page_list,
        extraction_method_summary="pymupdf",
    )


def _clause(
    clause_type: str,
    is_present: bool,
    extracted_text: str | None,
    page_reference: int | None,
    confidence: float = 0.85,
    risk_level_override: str = "LOW",  # passed for context only, not stored in clause
) -> ExtractedClause:
    return ExtractedClause(
        clause_type=clause_type,
        is_present=is_present,
        extracted_text=extracted_text,
        page_reference=page_reference,
        confidence=confidence,
        source_chunk_id="chunk-001",
        retry_count=0,
        requires_human_review=False,
    )


def _finding(clause_type: str, risk_level: str, is_missing: bool = False) -> RiskFinding:
    return RiskFinding(
        clause_type=clause_type,
        risk_level=risk_level,
        is_missing=is_missing,
        reason="Test finding.",
        precedent_applied=False,
        precedent_note=None,
    )


# ---------------------------------------------------------------------------
# TEST 1: Clean clause — evidence verified, page valid -> passed
# ---------------------------------------------------------------------------

def test_verified_clause_passes():
    """A clause whose extracted_text genuinely appears on its cited page -> passed."""
    real_text = "The parties agree to maintain confidentiality of all proprietary information."
    doc = _make_doc({1: real_text, 2: "Termination upon 30 days notice."})

    clauses = [
        _clause("Confidentiality / Non-Disclosure", True, real_text, page_reference=1),
    ]
    findings = [_finding("Confidentiality / Non-Disclosure", "LOW")]

    results = verify_final_claims(clauses, findings, doc)

    assert len(results) == 1
    r = results[0]
    assert r.action_taken == "passed"
    assert r.has_supporting_evidence is True
    assert r.evidence_source is not None


# ---------------------------------------------------------------------------
# TEST 2: LOW risk clause, evidence not found -> removed
# ---------------------------------------------------------------------------

def test_low_risk_unverifiable_evidence_removed():
    """
    A LOW risk clause whose extracted_text cannot be found in the source
    page should be marked action_taken='removed', not escalated.
    """
    doc = _make_doc({
        1: "The parties agree to maintain confidentiality.",
        2: "Governing law shall be Delaware.",
    })
    # Clause claims to be on page 1 but the text is fabricated / not there
    clauses = [
        _clause(
            "Governing Law / Jurisdiction", True,
            "This agreement shall be governed by the laws of California.",  # not in source
            page_reference=1,
        ),
    ]
    findings = [_finding("Governing Law / Jurisdiction", "LOW")]

    results = verify_final_claims(clauses, findings, doc)

    assert len(results) == 1
    r = results[0]
    assert r.action_taken == "removed", (
        f"Expected 'removed' for LOW risk with bad evidence, got {r.action_taken!r}"
    )
    assert r.has_supporting_evidence is False

    # No ReviewItem should be enqueued for a LOW risk removal
    pending = hr_module.get_pending_reviews()
    assert len(pending) == 0


# ---------------------------------------------------------------------------
# TEST 3: HIGH risk clause, evidence not found -> escalated + ReviewItem queued
# ---------------------------------------------------------------------------

def test_high_risk_unverifiable_evidence_escalated():
    """
    A HIGH risk clause whose extracted_text cannot be found in source must
    be escalated and produce a real ReviewItem in the HITL queue.
    This confirms the override took effect -- not just that the function ran.
    """
    doc = _make_doc({
        1: "Basic NDA terms.",
        2: "More NDA terms.",
    })
    clauses = [
        _clause(
            "Indemnification", True,
            "Each party shall indemnify and hold harmless the other.",  # not in source
            page_reference=1,
            confidence=0.72,
        ),
    ]
    findings = [_finding("Indemnification", "HIGH")]

    results = verify_final_claims(clauses, findings, doc)

    assert len(results) == 1
    r = results[0]
    assert r.action_taken == "escalated_to_human_review", (
        f"Expected escalation for HIGH risk with bad evidence, got {r.action_taken!r}"
    )
    assert r.has_supporting_evidence is False

    # A ReviewItem MUST be in the queue
    pending = hr_module.get_pending_reviews()
    assert len(pending) == 1, f"Expected 1 ReviewItem, got {len(pending)}"
    item = pending[0]
    assert item.trigger_reason == "evidence_verification_failure"
    assert item.clause_category == "Indemnification"
    assert item.risk_level == "HIGH"


# ---------------------------------------------------------------------------
# TEST 4: Clause citing out-of-range page
# ---------------------------------------------------------------------------

def test_high_risk_out_of_range_page_escalated():
    """
    A HIGH risk clause citing a non-existent page number (e.g. page 50 in a
    3-page document) should be escalated, never silently removed.
    """
    doc = _make_doc({1: "Page one.", 2: "Page two.", 3: "Page three."})
    clauses = [
        _clause(
            "Limitation of Liability", True,
            "Liability is capped at the fees paid in the prior 12 months.",
            page_reference=50,  # out of range
            confidence=0.80,
        ),
    ]
    findings = [_finding("Limitation of Liability", "HIGH")]

    results = verify_final_claims(clauses, findings, doc)

    assert len(results) == 1
    # HIGH risk + bad page -> escalated
    assert results[0].action_taken == "escalated_to_human_review"

    pending = hr_module.get_pending_reviews()
    assert len(pending) == 1


def test_low_risk_out_of_range_page_removed():
    """
    A LOW risk clause citing a non-existent page -> removed (not escalated).
    """
    doc = _make_doc({1: "Only page."})
    clauses = [
        _clause(
            "Renewal / Term", True,
            "The agreement shall renew automatically each year.",
            page_reference=99,
        ),
    ]
    findings = [_finding("Renewal / Term", "LOW")]

    results = verify_final_claims(clauses, findings, doc)

    assert results[0].action_taken == "removed"
    assert len(hr_module.get_pending_reviews()) == 0


# ---------------------------------------------------------------------------
# TEST 5: Absent clause (is_present=False) -> passed (structural, no text check)
# ---------------------------------------------------------------------------

def test_absent_clause_passes_without_evidence_check():
    """
    An absent clause has no extracted_text, so evidence verification is
    not applicable. It should always be passed -- the absence is the finding,
    not a fabrication.
    """
    doc = _make_doc({1: "Some contract text."})
    clauses = [
        _clause("Non-Compete / Non-Solicitation", False, None, page_reference=None),
    ]
    findings = [_finding("Non-Compete / Non-Solicitation", "HIGH", is_missing=True)]

    results = verify_final_claims(clauses, findings, doc)

    assert results[0].action_taken == "passed"
    assert results[0].has_supporting_evidence is False
    # No escalation for absent clauses (they go through HITL via orchestrator,
    # not through evidence verification)
    assert len(hr_module.get_pending_reviews()) == 0


# ---------------------------------------------------------------------------
# TEST 6: Mix of clauses — results are independent
# ---------------------------------------------------------------------------

def test_mixed_clauses():
    """
    Multiple clauses with different verification outcomes in one call.
    """
    conf_text = "The parties agree to maintain confidentiality of all information."
    doc = _make_doc({1: conf_text, 2: "Termination upon 30 days notice."})

    clauses = [
        _clause("Confidentiality / Non-Disclosure", True, conf_text, page_reference=1),
        _clause("Assignment", True, "Fabricated assignment text not in source.", page_reference=1),
    ]
    findings = [
        _finding("Confidentiality / Non-Disclosure", "LOW"),
        _finding("Assignment", "MEDIUM"),
    ]

    results = verify_final_claims(clauses, findings, doc)

    actions = {r.claim_text.split(":")[0]: r.action_taken for r in results}
    assert actions["Confidentiality / Non-Disclosure"] == "passed"
    assert actions["Assignment"] == "escalated_to_human_review"

    # Only the MEDIUM risk Assignment should be in the queue
    pending = hr_module.get_pending_reviews()
    assert len(pending) == 1
    assert pending[0].clause_category == "Assignment"


# ---------------------------------------------------------------------------
# TEST 7: Escalated ReviewItem has new HITL-deepening fields populated
# ---------------------------------------------------------------------------

class TestEscalatedReviewItemNewFields:
    """
    Confirm that the ReviewItem enqueued by claim_verifier populates all
    new HITL-deepening fields introduced in Stage 2 of the HITL upgrade.

    Call site: guardrails/claim_verifier.py -> _enqueue_escalation()
    """

    def test_escalated_item_fact_found_populated(self, isolated_queue):
        doc = _make_doc({1: "Basic contract text."})
        clause = _clause(
            "Indemnification", True,
            "Each party shall indemnify the other.",
            page_reference=1,
            confidence=0.72,
        )
        finding = _finding("Indemnification", "HIGH")

        verify_final_claims([clause], [finding], doc)

        pending = hr_module.get_pending_reviews()
        assert len(pending) == 1
        item = pending[0]
        assert item.fact_found != "", "fact_found must be non-empty for escalated items"
        assert "Indemnification" in item.fact_found

    def test_escalated_item_deviation_found_is_evidence_failure_message(self, isolated_queue):
        doc = _make_doc({1: "Basic contract text."})
        clause = _clause(
            "Indemnification", True,
            "Each party shall indemnify the other.",
            page_reference=1,
            confidence=0.72,
        )
        finding = _finding("Indemnification", "HIGH")

        verify_final_claims([clause], [finding], doc)

        pending = hr_module.get_pending_reviews()
        item = pending[0]
        assert item.deviation_found is not None
        assert "evidence" in item.deviation_found.lower() or "verification" in item.deviation_found.lower()

    def test_escalated_item_risk_rationale_populated(self, isolated_queue):
        doc = _make_doc({1: "Basic contract text."})
        clause = _clause(
            "Indemnification", True,
            "Each party shall indemnify the other.",
            page_reference=1,
            confidence=0.72,
        )
        finding = _finding("Indemnification", "HIGH")

        verify_final_claims([clause], [finding], doc)

        pending = hr_module.get_pending_reviews()
        item = pending[0]
        assert item.risk_rationale != "", "risk_rationale must be non-empty"
        assert "HIGH" in item.risk_rationale or "escalated" in item.risk_rationale.lower()

    def test_escalated_item_evidence_match_type_is_not_found(self, isolated_queue):
        doc = _make_doc({1: "Basic contract text."})
        clause = _clause(
            "Indemnification", True,
            "Each party shall indemnify the other.",
            page_reference=1,
            confidence=0.72,
        )
        finding = _finding("Indemnification", "HIGH")

        verify_final_claims([clause], [finding], doc)

        pending = hr_module.get_pending_reviews()
        item = pending[0]
        assert item.evidence_match_type == "not_found"

    def test_escalated_item_confidence_fields_populated(self, isolated_queue):
        doc = _make_doc({1: "Basic contract text."})
        clause = _clause(
            "Indemnification", True,
            "Each party shall indemnify the other.",
            page_reference=1,
            confidence=0.72,
        )
        finding = _finding("Indemnification", "HIGH")

        verify_final_claims([clause], [finding], doc)

        pending = hr_module.get_pending_reviews()
        item = pending[0]
        # Both confidence fields are populated from clause.confidence (same value, documented)
        assert item.extraction_confidence == pytest.approx(0.72)
        assert item.llm_confidence == pytest.approx(0.72)

    def test_escalated_item_coexists_with_expansion_fields(self, isolated_queue):
        """Expansion fields default correctly alongside new HITL-deepening fields."""
        doc = _make_doc({1: "Basic contract text."})
        clause = _clause(
            "Indemnification", True,
            "Each party shall indemnify the other.",
            page_reference=1,
            confidence=0.72,
        )
        finding = _finding("Indemnification", "HIGH")

        verify_final_claims([clause], [finding], doc)

        pending = hr_module.get_pending_reviews()
        item = pending[0]
        # Expansion fields (from prior task) default to their zero-values — no conflict
        assert item.expansion_triggered is False
        assert item.expanded_clause_text is None
        assert item.source_chunks_used == []
        assert item.expansion_boundary_reason is None
        # New fields are also present
        assert item.evidence_match_type == "not_found"
        assert item.fact_found != ""
