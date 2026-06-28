"""
tests/unit/test_report_schema.py
---------------------------------
Construction tests for all four Pydantic models in schemas/report.py.

These tests verify:
    - Each model can be constructed with valid data.
    - confidence_label only accepts the three allowed values.
    - Optional fields default correctly (None / empty list).
    - Frozen config means mutation raises an error.
"""

import sys
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from schemas.report import (
    ClauseReportEntry,
    GuardrailSummaryEntry,
    LegalDocumentReport,
    RiskFindingEntry,
)


# ---------------------------------------------------------------------------
# ClauseReportEntry
# ---------------------------------------------------------------------------

class TestClauseReportEntry:
    def _minimal(self, **overrides):
        data = dict(
            clause_category="Indemnification",
            is_present=True,
            confidence_label="HIGH",
            confidence_explanation="High confidence: clause text verified in source document.",
        )
        data.update(overrides)
        return ClauseReportEntry(**data)

    def test_construction_minimal(self):
        entry = self._minimal()
        assert entry.clause_category == "Indemnification"
        assert entry.is_present is True
        assert entry.confidence_label == "HIGH"
        assert entry.source_page is None
        assert entry.evidence_verified is None
        assert entry.risk_level is None
        assert entry.deviation_summary is None
        assert entry.recommendation is None

    def test_construction_full(self):
        entry = self._minimal(
            source_page=3,
            evidence_verified=True,
            risk_level="MEDIUM",
            deviation_summary="The system identified a non-standard jurisdiction clause on page 3.",
            recommendation="Qualified legal review is recommended for this clause.",
        )
        assert entry.source_page == 3
        assert entry.evidence_verified is True
        assert entry.risk_level == "MEDIUM"

    def test_confidence_label_accepts_all_three_values(self):
        for label in ("HIGH", "MODERATE", "LOW"):
            entry = self._minimal(confidence_label=label)
            assert entry.confidence_label == label

    def test_confidence_label_rejects_invalid(self):
        with pytest.raises(ValidationError):
            self._minimal(confidence_label="VERY_HIGH")

    def test_confidence_label_rejects_raw_float_string(self):
        # Ensures nobody accidentally passes a stringified float
        with pytest.raises(ValidationError):
            self._minimal(confidence_label="0.87")

    def test_source_page_must_be_positive(self):
        with pytest.raises(ValidationError):
            self._minimal(source_page=0)

    def test_frozen_model_rejects_mutation(self):
        entry = self._minimal()
        with pytest.raises(Exception):
            entry.is_present = False  # type: ignore[misc]

    def test_absent_clause(self):
        entry = self._minimal(
            clause_category="Non-Compete / Non-Solicitation",
            is_present=False,
            confidence_label="LOW",
            confidence_explanation="Low confidence: clause was not located in retrieved context.",
            evidence_verified=None,
        )
        assert entry.is_present is False
        assert entry.evidence_verified is None


# ---------------------------------------------------------------------------
# RiskFindingEntry
# ---------------------------------------------------------------------------

class TestRiskFindingEntry:
    def _minimal(self, **overrides):
        data = dict(
            clause_category="Governing Law / Jurisdiction",
            risk_level="HIGH",
            finding_summary=(
                "The system identified an absent Governing Law clause. "
                "Qualified legal review is recommended."
            ),
        )
        data.update(overrides)
        return RiskFindingEntry(**data)

    def test_construction_minimal(self):
        entry = self._minimal()
        assert entry.risk_level == "HIGH"
        assert entry.source_page is None
        assert entry.precedent_applied is False
        assert entry.human_review_status == "not_required"

    def test_construction_full(self):
        entry = self._minimal(
            source_page=7,
            precedent_applied=True,
            human_review_status="resolved",
        )
        assert entry.source_page == 7
        assert entry.precedent_applied is True
        assert entry.human_review_status == "resolved"

    def test_all_risk_levels_accepted(self):
        for level in ("HIGH", "MEDIUM", "LOW"):
            entry = self._minimal(risk_level=level)
            assert entry.risk_level == level

    def test_invalid_risk_level_rejected(self):
        with pytest.raises(ValidationError):
            self._minimal(risk_level="CRITICAL")

    def test_all_human_review_statuses_accepted(self):
        for status in ("not_required", "pending", "resolved"):
            entry = self._minimal(human_review_status=status)
            assert entry.human_review_status == status

    def test_invalid_human_review_status_rejected(self):
        with pytest.raises(ValidationError):
            self._minimal(human_review_status="unknown")


# ---------------------------------------------------------------------------
# GuardrailSummaryEntry
# ---------------------------------------------------------------------------

class TestGuardrailSummaryEntry:
    def _make(self, **overrides):
        data = dict(
            guardrail_name="input_validator.all_checks",
            passed=True,
            severity="info",
            reason="File size and format are within accepted limits.",
        )
        data.update(overrides)
        return GuardrailSummaryEntry(**data)

    def test_construction_pass(self):
        entry = self._make()
        assert entry.passed is True
        assert entry.severity == "info"

    def test_construction_fail_blocking(self):
        entry = self._make(
            guardrail_name="scope_validator",
            passed=False,
            severity="blocking",
            reason="Request was identified as requesting autonomous legal approval.",
        )
        assert entry.passed is False
        assert entry.severity == "blocking"

    def test_construction_warning(self):
        entry = self._make(
            passed=False,
            severity="warning",
            reason="File size is close to the maximum accepted limit.",
        )
        assert entry.severity == "warning"

    def test_frozen_model_rejects_mutation(self):
        entry = self._make()
        with pytest.raises(Exception):
            entry.passed = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# LegalDocumentReport
# ---------------------------------------------------------------------------

def _make_clause_entry(clause_type: str, is_present: bool = True) -> ClauseReportEntry:
    return ClauseReportEntry(
        clause_category=clause_type,
        is_present=is_present,
        confidence_label="HIGH",
        confidence_explanation="High confidence: clause verified in source.",
        risk_level="LOW" if is_present else "HIGH",
    )


def _make_risk_entry(clause_type: str, risk_level: str = "LOW") -> RiskFindingEntry:
    return RiskFindingEntry(
        clause_category=clause_type,
        risk_level=risk_level,
        finding_summary=(
            f"The system identified the {clause_type} clause. "
            "No material deviation from standard template was found."
        ),
    )


def _make_guardrail_entry() -> GuardrailSummaryEntry:
    return GuardrailSummaryEntry(
        guardrail_name="input_validator.all_checks",
        passed=True,
        severity="info",
        reason="All input checks passed.",
    )


def _make_full_report(**overrides) -> LegalDocumentReport:
    from config import CLAUSE_CATEGORIES
    clause_entries = [_make_clause_entry(c) for c in CLAUSE_CATEGORIES]
    risk_findings = [_make_risk_entry(c) for c in CLAUSE_CATEGORIES]

    data = dict(
        document_name="NDA_2_Bakhu_Holdings.pdf",
        document_hash="abc123def456",
        document_type="NDA",
        classification_confidence_label="HIGH",
        total_pages=5,
        total_clauses_found=10,
        total_clauses_missing=0,
        executive_summary=(
            "This NDA contains all 10 standard clause categories. "
            "No high-risk findings were identified. "
            "Qualified legal review is recommended before execution."
        ),
        clause_entries=clause_entries,
        risk_findings=risk_findings,
        missing_clauses=[],
        guardrail_summary=[_make_guardrail_entry()],
        limitations_note="Template comparison was available for all 10 clause categories.",
        disclaimer=(
            "This report was generated by an AI-assisted legal analysis system. "
            "It does not constitute legal advice. Qualified legal counsel should "
            "review this document before any binding decision is made."
        ),
        generated_at=datetime(2026, 6, 26, 12, 0, 0),
    )
    data.update(overrides)
    return LegalDocumentReport(**data)


class TestLegalDocumentReport:
    def test_construction_full(self):
        report = _make_full_report()
        assert report.document_type == "NDA"
        assert report.total_clauses_found == 10
        assert report.total_clauses_missing == 0
        assert len(report.clause_entries) == 10
        assert len(report.risk_findings) == 10
        assert report.human_review_decisions == []
        assert report.processing_notes is None

    def test_confidence_label_valid_values(self):
        for label in ("HIGH", "MODERATE", "LOW"):
            report = _make_full_report(classification_confidence_label=label)
            assert report.classification_confidence_label == label

    def test_confidence_label_invalid_rejected(self):
        with pytest.raises(ValidationError):
            _make_full_report(classification_confidence_label="VERY_HIGH")

    def test_missing_clauses_list(self):
        from config import CLAUSE_CATEGORIES
        # One clause absent
        clause_entries = [
            _make_clause_entry(c, is_present=(c != "Indemnification"))
            for c in CLAUSE_CATEGORIES
        ]
        report = _make_full_report(
            clause_entries=clause_entries,
            total_clauses_found=9,
            total_clauses_missing=1,
            missing_clauses=["Indemnification"],
        )
        assert "Indemnification" in report.missing_clauses
        assert report.total_clauses_missing == 1

    def test_generated_at_is_datetime(self):
        report = _make_full_report()
        assert isinstance(report.generated_at, datetime)

    def test_disclaimer_never_empty(self):
        with pytest.raises(ValidationError):
            _make_full_report(disclaimer="")

    def test_processing_notes_optional(self):
        report = _make_full_report(processing_notes="OCR was used on 2 pages.")
        assert "OCR" in report.processing_notes

    def test_human_review_decisions_default_empty(self):
        report = _make_full_report()
        assert report.human_review_decisions == []

    def test_total_pages_non_negative(self):
        with pytest.raises(ValidationError):
            _make_full_report(total_pages=-1)
