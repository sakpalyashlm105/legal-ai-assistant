"""
tests/unit/test_output_validator.py
-------------------------------------
Unit tests for guardrails/output_validator.py (Step 11).

All tests use synthetic LegalDocumentReport objects. No LLM calls, no PDF I/O.
The HITL queue (data/review_queue/) IS written to in escalation tests because
the validator calls add_to_review_queue() -- that's the real integration point
being tested. Tests clean up by checking the queue state after each escalation.

Test classes:
  TestCleanReport             -- a fully valid report passes with no fixes/escalations
  TestDisclaimerCheck         -- auto-fix cases for missing disclaimer
  TestClauseCountConsistency  -- count mismatch detection and escalation
  TestSchemaCompleteness      -- empty required fields escalation
  TestGuardrailDisconnect     -- evidence_verified=False + not_required detection
"""

from datetime import datetime
from typing import List, Optional

import pytest

from agent.human_review import get_pending_reviews
from guardrails.output_validator import validate_final_output
from reporting.report_generator import DISCLAIMER
from schemas.report import (
    ClauseReportEntry,
    FinalOutputValidationResult,
    GuardrailSummaryEntry,
    LegalDocumentReport,
    RiskFindingEntry,
)

# ---------------------------------------------------------------------------
# Test fixture helpers
# ---------------------------------------------------------------------------

ALL_CLAUSES = [
    "Confidentiality / Non-Disclosure",
    "Termination for Convenience",
    "Termination for Cause",
    "Governing Law / Jurisdiction",
    "Indemnification",
    "Limitation of Liability",
    "Non-Compete / Non-Solicitation",
    "Assignment",
    "Renewal / Term",
    "Dispute Resolution",
]


def _make_clause_entry(
    category: str,
    is_present: bool,
    evidence_verified: Optional[bool] = None,
    risk_level: Optional[str] = None,
) -> ClauseReportEntry:
    return ClauseReportEntry(
        clause_category=category,
        is_present=is_present,
        confidence_label="HIGH" if is_present else "LOW",
        confidence_explanation="Test explanation.",
        source_page=1 if is_present else None,
        evidence_verified=evidence_verified,
        risk_level=risk_level,
    )


def _make_risk_finding(
    category: str,
    risk_level: str = "HIGH",
    human_review_status: str = "not_required",
    finding_summary: str = "The system identified a risk. Legal review recommended.",
) -> RiskFindingEntry:
    return RiskFindingEntry(
        clause_category=category,
        risk_level=risk_level,
        finding_summary=finding_summary,
        human_review_status=human_review_status,
    )


def _make_clean_report(
    n_present: int = 4,
    n_missing: int = 6,
    disclaimer: str = DISCLAIMER,
    doc_name: str = "test_nda.pdf",
    doc_hash: str = "abc123" * 10 + "abcd",
) -> LegalDocumentReport:
    """
    Build a fully internally-consistent LegalDocumentReport with n_present
    clauses present and n_missing absent. All 10 must be accounted for.
    """
    assert n_present + n_missing == 10

    present_cats = ALL_CLAUSES[:n_present]
    missing_cats = ALL_CLAUSES[n_present:]

    clause_entries = []
    for cat in present_cats:
        clause_entries.append(_make_clause_entry(cat, is_present=True, evidence_verified=True, risk_level="LOW"))
    for cat in missing_cats:
        clause_entries.append(_make_clause_entry(cat, is_present=False, evidence_verified=None, risk_level="HIGH"))

    risk_findings = [
        _make_risk_finding(cat, risk_level="HIGH") for cat in missing_cats
    ]

    return LegalDocumentReport(
        document_name=doc_name,
        document_hash=doc_hash,
        document_type="NDA",
        classification_confidence_label="HIGH",
        total_pages=3,
        total_clauses_found=n_present,
        total_clauses_missing=n_missing,
        executive_summary=(
            f"This NDA was analysed. {n_present} clauses were present and passed "
            f"evidence verification; {n_missing} clauses were absent and not applicable."
        ),
        clause_entries=clause_entries,
        risk_findings=risk_findings,
        missing_clauses=list(missing_cats),
        guardrail_summary=[
            GuardrailSummaryEntry(
                guardrail_name="input_validator.all_checks",
                passed=True,
                severity="info",
                reason="All input checks passed.",
            )
        ],
        limitations_note="Template comparison was available for all categories.",
        disclaimer=disclaimer,
        generated_at=datetime.utcnow(),
    )


# ---------------------------------------------------------------------------
# TestCleanReport
# ---------------------------------------------------------------------------

class TestCleanReport:
    def test_clean_report_passes_all_checks(self):
        report = _make_clean_report()
        returned_report, result = validate_final_output(report, thread_id="test-clean")
        assert result.passed is True
        assert result.checks_failed == []
        assert result.auto_fixes_applied == []
        assert result.escalations_created == []
        assert set(result.checks_run) == {
            "disclaimer", "clause_count_consistency",
            "schema_completeness", "guardrail_disconnect",
        }

    def test_clean_report_returned_unchanged(self):
        report = _make_clean_report()
        returned_report, result = validate_final_output(report, thread_id="test-unchanged")
        # When no auto-fix needed, returned report is semantically identical
        assert returned_report.disclaimer == DISCLAIMER
        assert returned_report.document_name == report.document_name

    def test_all_present_clean(self):
        """All 10 clauses present, all verified -- should still pass."""
        clause_entries = [
            _make_clause_entry(cat, is_present=True, evidence_verified=True, risk_level="LOW")
            for cat in ALL_CLAUSES
        ]
        report = _make_clean_report(n_present=0, n_missing=10)  # start from base, then override
        # Build directly
        report2 = LegalDocumentReport(
            document_name="full_nda.pdf",
            document_hash="ff" * 32,
            document_type="NDA",
            classification_confidence_label="HIGH",
            total_pages=5,
            total_clauses_found=10,
            total_clauses_missing=0,
            executive_summary="All 10 clauses present and verified.",
            clause_entries=clause_entries,
            risk_findings=[],
            missing_clauses=[],
            guardrail_summary=[],
            limitations_note="Full analysis performed.",
            disclaimer=DISCLAIMER,
            generated_at=datetime.utcnow(),
        )
        _, result = validate_final_output(report2, thread_id="test-all-present")
        assert result.passed is True
        assert result.checks_failed == []


# ---------------------------------------------------------------------------
# TestDisclaimerCheck
# ---------------------------------------------------------------------------

class TestDisclaimerCheck:
    def test_empty_disclaimer_is_reinjected(self):
        # schema requires min_length=1, so use a single space (missing sentinel)
        report = _make_clean_report(disclaimer=" ")
        returned_report, result = validate_final_output(report, thread_id="test-disc-empty")
        # Auto-fix applied
        assert "disclaimer_reinjected" in result.auto_fixes_applied
        assert "disclaimer" in result.checks_failed
        # Returned report has the correct disclaimer
        assert "does not constitute legal advice" in returned_report.disclaimer
        # No escalation -- auto-fix is sufficient
        assert result.escalations_created == []

    def test_missing_sentinel_reinjected(self):
        report = _make_clean_report(disclaimer="This is a custom disclaimer without the key phrase.")
        returned_report, result = validate_final_output(report, thread_id="test-disc-partial")
        assert "disclaimer_reinjected" in result.auto_fixes_applied
        assert returned_report.disclaimer == DISCLAIMER

    def test_correct_disclaimer_passes(self):
        report = _make_clean_report(disclaimer=DISCLAIMER)
        _, result = validate_final_output(report, thread_id="test-disc-ok")
        assert "disclaimer" not in result.checks_failed
        assert "disclaimer_reinjected" not in result.auto_fixes_applied

    def test_auto_fix_does_not_trigger_escalation(self):
        """Disclaimer auto-fix should never create a ReviewItem."""
        report = _make_clean_report(disclaimer="No sentinel text here.")
        _, result = validate_final_output(report, thread_id="test-disc-no-esc")
        # passed=True because only auto-fix, no escalation
        assert result.passed is True
        assert len(result.escalations_created) == 0


# ---------------------------------------------------------------------------
# TestClauseCountConsistency
# ---------------------------------------------------------------------------

class TestClauseCountConsistency:
    def test_total_clauses_found_mismatch_escalated(self):
        """total_clauses_found=5 but only 4 entries are present -> escalation."""
        report = _make_clean_report(n_present=4, n_missing=6)
        # Tamper: claim 5 found when only 4 present
        tampered = report.model_copy(update={"total_clauses_found": 5})
        _, result = validate_final_output(tampered, thread_id="test-count-found")

        assert result.passed is False
        assert "clause_count_consistency" in result.checks_failed
        assert len(result.escalations_created) == 1

        # Verify the ReviewItem was actually enqueued
        pending = get_pending_reviews(document_hash=tampered.document_hash)
        matching = [p for p in pending if p.trigger_reason == "report_count_inconsistency"]
        assert len(matching) >= 1

    def test_total_clauses_missing_mismatch_escalated(self):
        """total_clauses_missing=5 but 6 entries are absent -> escalation."""
        report = _make_clean_report(n_present=4, n_missing=6)
        tampered = report.model_copy(update={"total_clauses_missing": 5})
        _, result = validate_final_output(tampered, thread_id="test-count-missing")

        assert result.passed is False
        assert "clause_count_consistency" in result.checks_failed
        assert len(result.escalations_created) >= 1

    def test_orphaned_missing_clause_name_escalated(self):
        """missing_clauses contains a name not present in clause_entries as absent."""
        report = _make_clean_report(n_present=4, n_missing=6)
        # Add a bogus name to missing_clauses
        bogus_missing = list(report.missing_clauses) + ["Fake Clause Category"]
        tampered = report.model_copy(update={"missing_clauses": bogus_missing})
        _, result = validate_final_output(tampered, thread_id="test-orphan")

        assert result.passed is False
        assert "clause_count_consistency" in result.checks_failed

    def test_wrong_entry_count_escalated(self):
        """clause_entries with 9 entries (not 10) is detected."""
        report = _make_clean_report(n_present=4, n_missing=6)
        # Drop one entry
        short_entries = list(report.clause_entries)[:-1]
        tampered = report.model_copy(update={"clause_entries": short_entries})
        _, result = validate_final_output(tampered, thread_id="test-9-entries")

        assert result.passed is False
        assert "clause_count_consistency" in result.checks_failed

    def test_11_entries_escalated(self):
        """clause_entries with 11 entries (one duplicate) is detected."""
        report = _make_clean_report(n_present=4, n_missing=6)
        extra_entries = list(report.clause_entries) + [report.clause_entries[0]]
        tampered = report.model_copy(update={"clause_entries": extra_entries})
        _, result = validate_final_output(tampered, thread_id="test-11-entries")

        assert result.passed is False
        assert "clause_count_consistency" in result.checks_failed

    def test_consistent_report_passes_count_check(self):
        """A report with correct counts passes the count consistency check."""
        report = _make_clean_report(n_present=4, n_missing=6)
        _, result = validate_final_output(report, thread_id="test-count-ok")
        assert "clause_count_consistency" not in result.checks_failed


# ---------------------------------------------------------------------------
# TestSchemaCompleteness
# ---------------------------------------------------------------------------

class TestSchemaCompleteness:
    def test_empty_document_name_escalated(self):
        report = _make_clean_report()
        tampered = report.model_copy(update={"document_name": "  "})
        _, result = validate_final_output(tampered, thread_id="test-empty-name")
        assert result.passed is False
        assert "schema_completeness" in result.checks_failed
        assert len(result.escalations_created) >= 1

    def test_empty_executive_summary_escalated(self):
        report = _make_clean_report()
        tampered = report.model_copy(update={"executive_summary": ""})
        _, result = validate_final_output(tampered, thread_id="test-empty-exec")
        assert result.passed is False
        assert "schema_completeness" in result.checks_failed

    def test_high_risk_finding_with_empty_summary_escalated(self):
        """A HIGH-risk finding with empty finding_summary is a schema defect."""
        report = _make_clean_report(n_present=4, n_missing=6)
        # Replace the first risk finding with one that has an empty summary
        bad_finding = _make_risk_finding(
            ALL_CLAUSES[4],  # first missing clause
            risk_level="HIGH",
            finding_summary="",  # empty!
        )
        new_findings = [bad_finding] + list(report.risk_findings)[1:]
        tampered = report.model_copy(update={"risk_findings": new_findings})
        _, result = validate_final_output(tampered, thread_id="test-empty-summary")
        assert result.passed is False
        assert "schema_completeness" in result.checks_failed

    def test_medium_risk_finding_with_empty_summary_escalated(self):
        """A MEDIUM-risk finding with empty finding_summary is also a defect."""
        report = _make_clean_report(n_present=4, n_missing=6)
        bad_finding = _make_risk_finding(
            ALL_CLAUSES[4], risk_level="MEDIUM", finding_summary=""
        )
        new_findings = [bad_finding]
        tampered = report.model_copy(update={"risk_findings": new_findings})
        _, result = validate_final_output(tampered, thread_id="test-medium-empty")
        assert result.passed is False
        assert "schema_completeness" in result.checks_failed

    def test_low_risk_empty_summary_passes(self):
        """LOW-risk finding with empty summary is not checked (LOW is lower-stakes)."""
        report = _make_clean_report(n_present=10, n_missing=0)
        low_finding = _make_risk_finding(
            ALL_CLAUSES[0], risk_level="LOW", finding_summary=""
        )
        tampered = report.model_copy(update={"risk_findings": [low_finding]})
        _, result = validate_final_output(tampered, thread_id="test-low-empty")
        assert "schema_completeness" not in result.checks_failed


# ---------------------------------------------------------------------------
# TestGuardrailDisconnect
# ---------------------------------------------------------------------------

class TestGuardrailDisconnect:
    def test_evidence_failed_not_reviewed_high_risk_escalated(self):
        """
        The most important test in this file.

        Scenario: The pipeline ran evidence verification on the
        Confidentiality/Non-Disclosure clause and found no verbatim match
        in the source PDF (evidence_verified=False). The risk engine rated
        this clause HIGH. BUT the assembled report's RiskFindingEntry shows
        human_review_status='not_required' -- as if the guardrail chain
        never flagged it.

        This is the guardrail-to-report disconnect: the report implies the
        clause was a normal high-risk finding that didn't need review, when
        in reality the evidence guardrail flagged it as unverifiable.

        The validate_final_output check must detect this and escalate.
        """
        # Build a report where:
        #   clause_entries[0] = Confidentiality, is_present=True, evidence_verified=False
        #   risk_findings[0] = Confidentiality, HIGH, human_review_status='not_required'
        cat = "Confidentiality / Non-Disclosure"

        # Other 9 clauses are absent (clean)
        clause_entries = [
            _make_clause_entry(cat, is_present=True, evidence_verified=False, risk_level="HIGH"),
        ] + [
            _make_clause_entry(c, is_present=False, evidence_verified=None, risk_level="HIGH")
            for c in ALL_CLAUSES[1:]
        ]

        risk_findings = [
            # The disconnect: evidence failed but human_review_status says 'not_required'
            _make_risk_finding(cat, risk_level="HIGH", human_review_status="not_required"),
        ] + [
            _make_risk_finding(c, risk_level="HIGH") for c in ALL_CLAUSES[1:]
        ]

        report = LegalDocumentReport(
            document_name="disconnect_test.pdf",
            document_hash="dd" * 32,
            document_type="NDA",
            classification_confidence_label="HIGH",
            total_pages=3,
            total_clauses_found=1,
            total_clauses_missing=9,
            executive_summary=(
                "1 present clause passed evidence verification; "
                "9 clauses were absent and not applicable."
            ),
            clause_entries=clause_entries,
            risk_findings=risk_findings,
            missing_clauses=ALL_CLAUSES[1:],
            guardrail_summary=[],
            limitations_note="Test report.",
            disclaimer=DISCLAIMER,
            generated_at=datetime.utcnow(),
        )

        _, result = validate_final_output(report, thread_id="test-disconnect-high")

        assert result.passed is False
        assert "guardrail_disconnect" in result.checks_failed
        assert len(result.escalations_created) >= 1

        # Confirm a ReviewItem was actually created and queued
        pending = get_pending_reviews(document_hash=report.document_hash)
        disconnect_items = [p for p in pending if p.trigger_reason == "guardrail_disconnect"]
        assert len(disconnect_items) >= 1
        assert disconnect_items[0].ai_finding_summary is not None

    def test_evidence_failed_medium_risk_not_reviewed_escalated(self):
        """Same disconnect scenario but MEDIUM risk -- also must escalate."""
        cat = "Governing Law / Jurisdiction"
        clause_entries = [
            _make_clause_entry(cat, is_present=True, evidence_verified=False, risk_level="MEDIUM"),
        ] + [
            _make_clause_entry(c, is_present=False) for c in ALL_CLAUSES if c != cat
        ]
        risk_findings = [
            _make_risk_finding(cat, risk_level="MEDIUM", human_review_status="not_required"),
        ]
        report = LegalDocumentReport(
            document_name="medium_disconnect.pdf",
            document_hash="ee" * 32,
            document_type="Contract",
            classification_confidence_label="MODERATE",
            total_pages=2,
            total_clauses_found=1,
            total_clauses_missing=9,
            executive_summary="1 clause present; 9 absent and N/A.",
            clause_entries=clause_entries,
            risk_findings=risk_findings,
            missing_clauses=[c for c in ALL_CLAUSES if c != cat],
            guardrail_summary=[],
            limitations_note="Test.",
            disclaimer=DISCLAIMER,
            generated_at=datetime.utcnow(),
        )
        _, result = validate_final_output(report, thread_id="test-disconnect-medium")
        assert result.passed is False
        assert "guardrail_disconnect" in result.checks_failed

    def test_evidence_failed_but_reviewed_passes_disconnect(self):
        """
        If evidence_verified=False but human_review_status='resolved',
        the human already reviewed it -- no disconnect.
        """
        cat = "Confidentiality / Non-Disclosure"
        clause_entries = [
            _make_clause_entry(cat, is_present=True, evidence_verified=False, risk_level="HIGH"),
        ] + [
            _make_clause_entry(c, is_present=False) for c in ALL_CLAUSES[1:]
        ]
        risk_findings = [
            # Already resolved -- human reviewed it
            _make_risk_finding(cat, risk_level="HIGH", human_review_status="resolved"),
        ]
        report = LegalDocumentReport(
            document_name="reviewed_ok.pdf",
            document_hash="f1" * 32,
            document_type="NDA",
            classification_confidence_label="HIGH",
            total_pages=3,
            total_clauses_found=1,
            total_clauses_missing=9,
            executive_summary="1 clause present, reviewed; 9 absent and N/A.",
            clause_entries=clause_entries,
            risk_findings=risk_findings,
            missing_clauses=ALL_CLAUSES[1:],
            guardrail_summary=[],
            limitations_note="Test.",
            disclaimer=DISCLAIMER,
            generated_at=datetime.utcnow(),
        )
        _, result = validate_final_output(report, thread_id="test-resolved-ok")
        assert "guardrail_disconnect" not in result.checks_failed

    def test_evidence_failed_low_risk_passes_disconnect(self):
        """
        If evidence_verified=False but risk is LOW, no escalation needed.
        The disconnect check only applies to HIGH and MEDIUM risk.
        """
        cat = "Renewal / Term"
        clause_entries = [
            _make_clause_entry(cat, is_present=True, evidence_verified=False, risk_level="LOW"),
        ] + [
            _make_clause_entry(c, is_present=False) for c in ALL_CLAUSES if c != cat
        ]
        risk_findings = [
            _make_risk_finding(cat, risk_level="LOW", human_review_status="not_required"),
        ]
        report = LegalDocumentReport(
            document_name="low_risk_ok.pdf",
            document_hash="a2" * 32,
            document_type="NDA",
            classification_confidence_label="HIGH",
            total_pages=2,
            total_clauses_found=1,
            total_clauses_missing=9,
            executive_summary="1 clause present; 9 absent and N/A.",
            clause_entries=clause_entries,
            risk_findings=risk_findings,
            missing_clauses=[c for c in ALL_CLAUSES if c != cat],
            guardrail_summary=[],
            limitations_note="Test.",
            disclaimer=DISCLAIMER,
            generated_at=datetime.utcnow(),
        )
        _, result = validate_final_output(report, thread_id="test-low-risk-ok")
        assert "guardrail_disconnect" not in result.checks_failed

    def test_absent_clause_evidence_none_passes_disconnect(self):
        """Absent clauses have evidence_verified=None. That should NOT trigger disconnect."""
        report = _make_clean_report(n_present=4, n_missing=6)
        _, result = validate_final_output(report, thread_id="test-absent-ok")
        assert "guardrail_disconnect" not in result.checks_failed


# ---------------------------------------------------------------------------
# TestEscalatedReviewItemNewFields
# Stage 2: confirm output_validator's enqueued ReviewItems populate the
# HITL-deepening fields introduced in the schema upgrade.
# ---------------------------------------------------------------------------

class TestEscalatedReviewItemNewFields:
    """
    Call site: guardrails/output_validator.py -> _enqueue_report_escalation()

    These tests verify that when output_validator escalates a report-level issue
    to the HITL queue, the resulting ReviewItem has the new Stage-2 fields populated
    (fact_found, deviation_found, risk_rationale, document_type_context) and that
    the clause-expansion fields (from the prior task) default correctly -- confirming
    the two sets of fields coexist without conflict.
    """

    def test_escalated_item_fact_found_non_empty(self, tmp_path, monkeypatch):
        import agent.human_review as hr_mod
        queue_dir = tmp_path / "review_queue"
        pending_file = queue_dir / "pending_reviews.json"
        resolved_file = queue_dir / "resolved_reviews.json"
        monkeypatch.setattr(hr_mod, "_QUEUE_DIR", queue_dir)
        monkeypatch.setattr(hr_mod, "_PENDING_FILE", pending_file)
        monkeypatch.setattr(hr_mod, "_RESOLVED_FILE", resolved_file)

        # Trigger a count-mismatch escalation (total_clauses_found wrong)
        report = _make_clean_report(n_present=4, n_missing=6)
        bad_report = report.model_copy(update={"total_clauses_found": 99})

        validate_final_output(bad_report, thread_id="test-fields")

        pending = get_pending_reviews()
        assert len(pending) >= 1
        item = pending[0]
        assert item.fact_found != "", "fact_found must be non-empty on escalated report item"
        assert "report" in item.fact_found.lower() or "structural" in item.fact_found.lower() or "issue" in item.fact_found.lower()

    def test_escalated_item_deviation_found_non_empty(self, tmp_path, monkeypatch):
        import agent.human_review as hr_mod
        queue_dir = tmp_path / "review_queue"
        pending_file = queue_dir / "pending_reviews.json"
        resolved_file = queue_dir / "resolved_reviews.json"
        monkeypatch.setattr(hr_mod, "_QUEUE_DIR", queue_dir)
        monkeypatch.setattr(hr_mod, "_PENDING_FILE", pending_file)
        monkeypatch.setattr(hr_mod, "_RESOLVED_FILE", resolved_file)

        report = _make_clean_report(n_present=4, n_missing=6)
        bad_report = report.model_copy(update={"total_clauses_found": 99})

        validate_final_output(bad_report, thread_id="test-deviation")

        pending = get_pending_reviews()
        item = pending[0]
        assert item.deviation_found is not None and item.deviation_found != ""

    def test_escalated_item_risk_rationale_non_empty(self, tmp_path, monkeypatch):
        import agent.human_review as hr_mod
        queue_dir = tmp_path / "review_queue"
        pending_file = queue_dir / "pending_reviews.json"
        resolved_file = queue_dir / "resolved_reviews.json"
        monkeypatch.setattr(hr_mod, "_QUEUE_DIR", queue_dir)
        monkeypatch.setattr(hr_mod, "_PENDING_FILE", pending_file)
        monkeypatch.setattr(hr_mod, "_RESOLVED_FILE", resolved_file)

        report = _make_clean_report(n_present=4, n_missing=6)
        bad_report = report.model_copy(update={"total_clauses_found": 99})

        validate_final_output(bad_report, thread_id="test-rationale")

        pending = get_pending_reviews()
        item = pending[0]
        assert item.risk_rationale != ""
        assert "Step 11" in item.risk_rationale or "output_validator" in item.risk_rationale

    def test_escalated_item_document_type_context_populated(self, tmp_path, monkeypatch):
        import agent.human_review as hr_mod
        queue_dir = tmp_path / "review_queue"
        pending_file = queue_dir / "pending_reviews.json"
        resolved_file = queue_dir / "resolved_reviews.json"
        monkeypatch.setattr(hr_mod, "_QUEUE_DIR", queue_dir)
        monkeypatch.setattr(hr_mod, "_PENDING_FILE", pending_file)
        monkeypatch.setattr(hr_mod, "_RESOLVED_FILE", resolved_file)

        report = _make_clean_report(n_present=4, n_missing=6)
        bad_report = report.model_copy(update={"total_clauses_found": 99})

        validate_final_output(bad_report, thread_id="test-doc-type")

        pending = get_pending_reviews()
        item = pending[0]
        assert item.document_type_context == "NDA"

    def test_escalated_item_expansion_fields_default_correctly(self, tmp_path, monkeypatch):
        """
        Expansion fields (prior task) default to zero-values alongside
        new Stage-2 fields -- confirms the two field sets coexist without conflict.
        """
        import agent.human_review as hr_mod
        queue_dir = tmp_path / "review_queue"
        pending_file = queue_dir / "pending_reviews.json"
        resolved_file = queue_dir / "resolved_reviews.json"
        monkeypatch.setattr(hr_mod, "_QUEUE_DIR", queue_dir)
        monkeypatch.setattr(hr_mod, "_PENDING_FILE", pending_file)
        monkeypatch.setattr(hr_mod, "_RESOLVED_FILE", resolved_file)

        report = _make_clean_report(n_present=4, n_missing=6)
        bad_report = report.model_copy(update={"total_clauses_found": 99})

        validate_final_output(bad_report, thread_id="test-coexist")

        pending = get_pending_reviews()
        item = pending[0]
        # Expansion fields: report-level escalations have no clause boundary data
        assert item.expansion_triggered is False
        assert item.expanded_clause_text is None
        assert item.source_chunks_used == []
        assert item.expansion_boundary_reason is None
        # New Stage-2 fields: must be populated
        assert item.fact_found != ""
        assert item.risk_rationale != ""


# ---------------------------------------------------------------------------
# TestRejectedFindingsCountInteraction
# Stage 4: Step 11 interaction analysis for rejected findings.
#
# Design decision (documented here):
#   Rejected findings DO still count toward total_clauses_found.
#   Rationale: reject categories (hallucinated_deviation, wrong_clause_category,
#   etc.) describe why the AI's RISK ANALYSIS was wrong, not whether the clause
#   was detected. "total_clauses_found" reflects extraction results (what the
#   system detected), not risk assessment results. A reviewer who rejects a
#   "hallucinated_deviation" finding is saying "the deviation is wrong" -- not
#   "the clause doesn't exist." The clause_entries list (the source of truth
#   for count checks) is unchanged by HITL decisions.
#   Auto-fix vs escalate: unchanged -- the interaction with rejected findings
#   does not create a new inconsistency that requires a fix to output_validator.
# ---------------------------------------------------------------------------

class TestRejectedFindingsCountInteraction:
    """
    Verify that Step 11's count-consistency check handles reports containing
    rejected findings correctly.
    """

    def test_rejected_finding_in_report_passes_count_check(self):
        """
        A report with a rejected decision still has the clause in clause_entries
        (is_present=True). The count check uses clause_entries, so it should
        still pass when total_clauses_found matches clause_entries.
        """
        report = _make_clean_report(n_present=4, n_missing=6)
        # Inject a rejected human decision -- clause_entries unchanged
        rejected_decision = {
            "action": "reject",
            "discarded": True,
            "clause_category": "Confidentiality / Non-Disclosure",
            "reviewer_note": "AI hallucinated a deviation that does not exist.",
        }
        report_with_rejection = report.model_copy(
            update={"human_review_decisions": [rejected_decision]}
        )

        _, result = validate_final_output(report_with_rejection, thread_id="test-rejected-count")
        # Count check must pass: total_clauses_found=4 still matches clause_entries
        assert "clause_count_consistency" not in result.checks_failed

    def test_rejected_finding_excluded_from_high_risk_render(self):
        """
        The report renderer excludes rejected findings from High-Risk section.
        This is a render-level check, not a validator check -- but we verify
        the report object is structured so the renderer can do this correctly.
        """
        from reporting.report_generator import render_markdown

        report = _make_clean_report(n_present=4, n_missing=6)
        # Add a HIGH-risk finding that is also rejected
        high_finding = _make_risk_finding(
            "Confidentiality / Non-Disclosure",
            risk_level="HIGH",
            finding_summary="The system found a HIGH risk deviation. Legal review recommended.",
        )
        rejected_decision = {
            "action": "reject",
            "discarded": True,
            "clause_category": "Confidentiality / Non-Disclosure",
            "reviewer_note": "Hallucinated; no real deviation exists.",
        }
        report_with_rejected_high = report.model_copy(
            update={
                "risk_findings": [high_finding],
                "human_review_decisions": [rejected_decision],
            }
        )

        md = render_markdown(report_with_rejected_high)

        # The High-Risk section must NOT contain the rejected clause's findings
        # (look for finding_summary content in the HIGH section specifically)
        high_section_start = md.find("## High-Risk Findings")
        medium_section_start = md.find("## Medium-Risk Findings")
        high_section = md[high_section_start:medium_section_start]

        assert "No high-risk findings were identified." in high_section, (
            "Rejected HIGH finding must be excluded from the High-Risk section.\n"
            f"High section was:\n{high_section}"
        )
