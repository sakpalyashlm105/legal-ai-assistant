"""
tests/integration/test_report_generation.py
---------------------------------------------
Integration tests verifying that the generate_report node produces a real
LegalDocumentReport in graph state after a completed pipeline run.

What is tested:
    1. Running the NDA_2_Bakhu_Holdings demo scenario end-to-end produces a
       generated_report in the final result (not None).
    2. The rendered markdown (re-generated from the report dict) contains all
       required Section-30 headers.
    3. The executive summary text does not contain any banned verdict phrases
       (real check against actual generated text, not asserted by assumption).
    4. Confidence language throughout the rendered report uses HIGH/MODERATE/LOW
       labels, never raw percentage probabilities.
    5. The HITL resume path also produces a generated_report after the human
       decision completes.

LLM calls (classify, extract_clauses, flag_risks, compare_to_templates) are
mocked. The executive summary LLM call is also mocked to a clean, valid
response so the test is deterministic and costs nothing.

Tests use unique thread_ids to avoid state leakage between runs.
"""

import sys
import uuid
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import agent.human_review as hr_module

BASE = Path(__file__).parent.parent.parent
NDA_PDF = BASE / "data" / "pdf" / "ndas" / "NDA_2_Bakhu_Holdings,_Corp._(BKUH).pdf"

pytestmark = pytest.mark.skipif(
    not NDA_PDF.exists(),
    reason="NDA_2_Bakhu_Holdings PDF not found",
)

# Banned phrases (same list as executive_summary.py)
BANNED_PHRASES = [
    "is enforceable",
    "is unenforceable",
    "is valid",
    "is invalid",
    "legally binding",
    "guarantees",
    "certifies that",
]

MOCK_EXECUTIVE_SUMMARY = (
    "This NDA contains 9 of 10 standard clause categories. "
    "The system identified one high-risk finding for the Indemnification clause, "
    "which was not found in the document. "
    "The system identified no medium-risk findings. "
    "Qualified legal review is recommended before execution."
)

REQUIRED_HEADERS = [
    "## Executive Summary",
    "## Document Metadata",
    "## Clause Analysis",
    "## High-Risk Findings",
    "## Medium-Risk Findings",
    "## Missing Clauses",
    "## Guardrail Summary",
    "## Human Review Decisions",
    "## Recommendations",
    "## Confidence and Limitations",
    "## Disclaimer",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_queue(tmp_path):
    queue_dir = tmp_path / "review_queue"
    pending_file = queue_dir / "pending_reviews.json"
    resolved_file = queue_dir / "resolved_reviews.json"
    with (
        patch.object(hr_module, "_QUEUE_DIR", queue_dir),
        patch.object(hr_module, "_PENDING_FILE", pending_file),
        patch.object(hr_module, "_RESOLVED_FILE", resolved_file),
    ):
        yield


def fresh_thread():
    return f"report-test-{uuid.uuid4().hex[:8]}"


def _mock_classification():
    from schemas.clause import DocumentClassification
    return DocumentClassification(
        document_type="NDA", confidence=0.95,
        reasoning="Mock NDA classification.", retry_count=0,
    )


def _mock_clauses_with_missing():
    """9 present + 1 absent (Indemnification) to force a HITL trigger."""
    from schemas.clause import ExtractedClause
    from config import CLAUSE_CATEGORIES
    clauses = []
    for cat in CLAUSE_CATEGORIES:
        if cat == "Indemnification":
            clauses.append(ExtractedClause(
                clause_type=cat, is_present=False, extracted_text=None,
                page_reference=None, confidence=0.91, source_chunk_id=None,
                retry_count=0, requires_human_review=False,
            ))
        else:
            clauses.append(ExtractedClause(
                clause_type=cat, is_present=True,
                extracted_text=f"[Demo] Standard {cat} language found on page 2.",
                page_reference=2, confidence=0.92,
                source_chunk_id="chunk-001", retry_count=0,
                requires_human_review=False,
            ))
    return clauses


def _mock_findings_with_high():
    from schemas.risk import RiskFinding
    from config import CLAUSE_CATEGORIES
    findings = []
    for cat in CLAUSE_CATEGORIES:
        if cat == "Indemnification":
            findings.append(RiskFinding(
                clause_type=cat, risk_level="HIGH", is_missing=True,
                reason="Critical clause absent from document.",
                precedent_applied=False, precedent_note=None,
            ))
        else:
            findings.append(RiskFinding(
                clause_type=cat, risk_level="LOW", is_missing=False,
                reason="Clause present and standard.",
                precedent_applied=False, precedent_note=None,
            ))
    return findings


def _mock_clauses_all_present():
    from schemas.clause import ExtractedClause
    from config import CLAUSE_CATEGORIES
    return [
        ExtractedClause(
            clause_type=cat, is_present=True,
            extracted_text=f"[Demo] {cat} on page 2.",
            page_reference=2, confidence=0.92,
            source_chunk_id="chunk-001", retry_count=0,
            requires_human_review=False,
        )
        for cat in CLAUSE_CATEGORIES
    ]


def _mock_findings_all_low():
    from schemas.risk import RiskFinding
    from config import CLAUSE_CATEGORIES
    return [
        RiskFinding(
            clause_type=cat, risk_level="LOW", is_missing=False,
            reason="Clause present and standard.",
            precedent_applied=False, precedent_note=None,
        )
        for cat in CLAUSE_CATEGORIES
    ]


# ---------------------------------------------------------------------------
# T1: Clean run (no HITL) produces a generated_report
# ---------------------------------------------------------------------------

def test_clean_run_produces_generated_report():
    from agent.orchestrator import run_pipeline

    thread_id = fresh_thread()
    clauses = _mock_clauses_all_present()
    findings = _mock_findings_all_low()

    with (
        patch("agent.orchestrator.classify_document", return_value=_mock_classification()),
        patch("agent.orchestrator.extract_clauses", return_value=clauses),
        patch("agent.orchestrator.flag_risks", return_value=findings),
        patch("agent.orchestrator.compare_to_templates", return_value=[]),
        patch("reporting.executive_summary._call_llm", return_value=MOCK_EXECUTIVE_SUMMARY),
    ):
        result = run_pipeline(str(NDA_PDF), thread_id=thread_id,
                              request_text="Analyze this NDA for risk.")

    assert result["status"] == "completed", f"Expected completed, got {result['status']}"
    assert result["generated_report"] is not None, (
        "generated_report must be present in result after a completed run"
    )

    report = result["generated_report"]
    assert report["document_type"] == "NDA"
    assert report["total_clauses_found"] == 10
    assert report["total_clauses_missing"] == 0
    assert len(report["clause_entries"]) == 10


# ---------------------------------------------------------------------------
# T2: Markdown contains all required Section-30 headers
# ---------------------------------------------------------------------------

def test_rendered_markdown_contains_all_required_headers():
    from agent.orchestrator import run_pipeline
    from reporting.report_generator import render_markdown
    from schemas.report import LegalDocumentReport

    thread_id = fresh_thread()
    clauses = _mock_clauses_all_present()
    findings = _mock_findings_all_low()

    with (
        patch("agent.orchestrator.classify_document", return_value=_mock_classification()),
        patch("agent.orchestrator.extract_clauses", return_value=clauses),
        patch("agent.orchestrator.flag_risks", return_value=findings),
        patch("agent.orchestrator.compare_to_templates", return_value=[]),
        patch("reporting.executive_summary._call_llm", return_value=MOCK_EXECUTIVE_SUMMARY),
    ):
        result = run_pipeline(str(NDA_PDF), thread_id=thread_id)

    assert result["generated_report"] is not None
    report_obj = LegalDocumentReport(**result["generated_report"])
    md = render_markdown(report_obj)

    for header in REQUIRED_HEADERS:
        assert header in md, f"Required header missing from markdown: '{header}'"


# ---------------------------------------------------------------------------
# T3: Executive summary contains no banned verdict phrases
# ---------------------------------------------------------------------------

def test_executive_summary_contains_no_banned_phrases():
    from agent.orchestrator import run_pipeline
    from schemas.report import LegalDocumentReport

    thread_id = fresh_thread()
    clauses = _mock_clauses_with_missing()
    findings = _mock_findings_with_high()

    with (
        patch("agent.orchestrator.classify_document", return_value=_mock_classification()),
        patch("agent.orchestrator.extract_clauses", return_value=clauses),
        patch("agent.orchestrator.flag_risks", return_value=findings),
        patch("agent.orchestrator.compare_to_templates", return_value=[]),
        # Use the deterministic fallback (don't call LLM at all)
        patch("reporting.executive_summary._call_llm", side_effect=Exception("No LLM in test")),
    ):
        result = run_pipeline(str(NDA_PDF), thread_id=thread_id)

    # With HITL triggered, graph pauses -- check report is populated after resume
    if result["status"] == "interrupted":
        from agent.orchestrator import resume_after_review
        from schemas.review import ReviewDecision
        pending = hr_module.get_pending_reviews()
        assert len(pending) >= 1
        decision = ReviewDecision(
            review_id=pending[0].review_id, action="approve",
        )
        with patch("reporting.executive_summary._call_llm",
                   side_effect=Exception("No LLM in test")):
            result = resume_after_review(thread_id, decision)

    assert result["generated_report"] is not None
    report_obj = LegalDocumentReport(**result["generated_report"])
    summary_lower = report_obj.executive_summary.lower()

    for phrase in BANNED_PHRASES:
        assert phrase.lower() not in summary_lower, (
            f"Banned verdict phrase '{phrase}' found in executive_summary: "
            f"{report_obj.executive_summary!r}"
        )


# ---------------------------------------------------------------------------
# T4: Confidence language uses labels, never raw percentage probabilities
# ---------------------------------------------------------------------------

def test_confidence_labels_not_raw_percentages():
    from agent.orchestrator import run_pipeline
    from reporting.report_generator import render_markdown
    from schemas.report import LegalDocumentReport

    thread_id = fresh_thread()
    clauses = _mock_clauses_all_present()
    findings = _mock_findings_all_low()

    with (
        patch("agent.orchestrator.classify_document", return_value=_mock_classification()),
        patch("agent.orchestrator.extract_clauses", return_value=clauses),
        patch("agent.orchestrator.flag_risks", return_value=findings),
        patch("agent.orchestrator.compare_to_templates", return_value=[]),
        patch("reporting.executive_summary._call_llm", return_value=MOCK_EXECUTIVE_SUMMARY),
    ):
        result = run_pipeline(str(NDA_PDF), thread_id=thread_id)

    report_obj = LegalDocumentReport(**result["generated_report"])
    md = render_markdown(report_obj)

    # The word "probability" must not appear (would indicate a calibrated claim)
    assert "probability" not in md.lower()

    # All clause entries must use label, not raw float
    for entry in report_obj.clause_entries:
        assert entry.confidence_label in ("HIGH", "MODERATE", "LOW")

    # The classification confidence must also be a label
    assert report_obj.classification_confidence_label in ("HIGH", "MODERATE", "LOW")


# ---------------------------------------------------------------------------
# T5: HITL resume path produces generated_report
# ---------------------------------------------------------------------------

def test_hitl_resume_produces_generated_report():
    from agent.orchestrator import run_pipeline, resume_after_review
    from schemas.review import ReviewDecision
    from schemas.report import LegalDocumentReport

    thread_id = fresh_thread()
    clauses = _mock_clauses_with_missing()
    findings = _mock_findings_with_high()

    with (
        patch("agent.orchestrator.classify_document", return_value=_mock_classification()),
        patch("agent.orchestrator.extract_clauses", return_value=clauses),
        patch("agent.orchestrator.flag_risks", return_value=findings),
        patch("agent.orchestrator.compare_to_templates", return_value=[]),
        patch("reporting.executive_summary._call_llm", return_value=MOCK_EXECUTIVE_SUMMARY),
    ):
        run_result = run_pipeline(str(NDA_PDF), thread_id=thread_id)

    assert run_result["status"] == "interrupted"
    pending = hr_module.get_pending_reviews()
    assert len(pending) >= 1

    decision = ReviewDecision(
        review_id=pending[0].review_id,
        action="approve",
    )
    with patch("reporting.executive_summary._call_llm", return_value=MOCK_EXECUTIVE_SUMMARY):
        resume_result = resume_after_review(thread_id, decision)

    assert resume_result["status"] == "completed"
    assert resume_result["generated_report"] is not None, (
        "generated_report must be present after HITL resume completes"
    )

    report_obj = LegalDocumentReport(**resume_result["generated_report"])
    assert report_obj.document_type == "NDA"
    assert report_obj.total_clauses_missing == 1
    assert "Indemnification" in report_obj.missing_clauses
