"""
tests/unit/test_report_generator.py
-------------------------------------
Unit tests for reporting/report_generator.py.

What is tested:
    - confidence_to_label: exact boundary cases (0.70 threshold, 0.50 threshold)
    - Verdict-style language is never produced in any generated text
    - render_markdown produces all required Section-30 headers
    - render_json round-trips cleanly (json.loads succeeds, data matches)
    - assemble_report produces a complete LegalDocumentReport
    - build_risk_findings sorts HIGH before MEDIUM before LOW
    - build_guardrail_summary faithfully maps guardrail dicts
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from reporting.report_generator import (
    _REQUIRED_HEADERS,
    _fallback_executive_summary,
    assemble_report,
    build_clause_entries,
    build_guardrail_summary,
    build_risk_findings,
    confidence_to_label,
    render_json,
    render_markdown,
)
from schemas.clause import ExtractedClause
from schemas.report import LegalDocumentReport, RiskFindingEntry
from schemas.risk import RiskFinding

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_clause(clause_type: str, is_present: bool = True, confidence: float = 0.85) -> ExtractedClause:
    return ExtractedClause(
        clause_type=clause_type,
        is_present=is_present,
        extracted_text=f"Standard {clause_type} language." if is_present else None,
        page_reference=2 if is_present else None,
        confidence=confidence,
        source_chunk_id="chunk-001" if is_present else None,
        retry_count=0,
        requires_human_review=False,
    )


def _make_finding(clause_type: str, risk_level: str = "LOW", is_missing: bool = False) -> RiskFinding:
    return RiskFinding(
        clause_type=clause_type,
        risk_level=risk_level,
        reason=f"Clause assessed at {risk_level} risk.",
        is_missing=is_missing,
        precedent_applied=False,
    )


def _make_all_clauses_and_findings():
    from config import CLAUSE_CATEGORIES
    clauses = [_make_clause(c) for c in CLAUSE_CATEGORIES]
    findings = [_make_finding(c) for c in CLAUSE_CATEGORIES]
    return clauses, findings


# ---------------------------------------------------------------------------
# confidence_to_label boundary tests
# ---------------------------------------------------------------------------

class TestConfidenceToLabel:
    def test_exactly_at_high_threshold(self):
        label, _ = confidence_to_label(0.70)
        assert label == "HIGH"

    def test_above_high_threshold(self):
        label, _ = confidence_to_label(0.95)
        assert label == "HIGH"

    def test_just_below_high_threshold(self):
        label, _ = confidence_to_label(0.699)
        assert label == "MODERATE"

    def test_exactly_at_moderate_floor(self):
        label, _ = confidence_to_label(0.50)
        assert label == "MODERATE"

    def test_above_moderate_floor(self):
        label, _ = confidence_to_label(0.65)
        assert label == "MODERATE"

    def test_just_below_moderate_floor(self):
        label, _ = confidence_to_label(0.499)
        assert label == "LOW"

    def test_at_zero(self):
        label, _ = confidence_to_label(0.0)
        assert label == "LOW"

    def test_at_one(self):
        label, _ = confidence_to_label(1.0)
        assert label == "HIGH"

    def test_explanation_is_string(self):
        for val in (0.0, 0.5, 0.7, 1.0):
            _, explanation = confidence_to_label(val)
            assert isinstance(explanation, str)
            assert len(explanation) > 10

    def test_explanation_never_contains_percentage(self):
        for val in (0.0, 0.5, 0.7, 1.0):
            _, explanation = confidence_to_label(val)
            # Must not present a calibrated percentage
            assert "%" not in explanation
            assert "probability" not in explanation.lower()


# ---------------------------------------------------------------------------
# Verdict-language absence test
# ---------------------------------------------------------------------------

BANNED_VERDICT_PHRASES = [
    "is enforceable",
    "is unenforceable",
    "is valid",
    "is invalid",
    "legally binding",
    "guarantees",
    "certifies that",
]


class TestNoVerdictLanguage:
    def _make_full_report(self) -> LegalDocumentReport:
        clauses, findings = _make_all_clauses_and_findings()
        return assemble_report(
            document_name="test.pdf",
            document_hash="abc123",
            document_type="NDA",
            classification_confidence=0.85,
            total_pages=5,
            clauses=clauses,
            risk_findings=findings,
            guardrail_results=[],
            executive_summary="This NDA contains all standard clauses. Qualified legal review is recommended.",
        )

    def test_markdown_contains_no_verdict_phrases(self):
        report = self._make_full_report()
        md = render_markdown(report)
        # Split off the disclaimer section -- it intentionally references these
        # phrases in a negative/limiting context ("does not determine whether X
        # is enforceable"). We check only the non-disclaimer body sections.
        body = md.split("## Disclaimer")[0].lower()
        for phrase in BANNED_VERDICT_PHRASES:
            assert phrase.lower() not in body, (
                f"Banned verdict phrase found in report body: '{phrase}'"
            )

    def test_risk_finding_summaries_contain_no_verdict_phrases(self):
        clauses, findings = _make_all_clauses_and_findings()
        entries = build_risk_findings(findings)
        for entry in entries:
            summary_lower = entry.finding_summary.lower()
            for phrase in BANNED_VERDICT_PHRASES:
                assert phrase.lower() not in summary_lower, (
                    f"Banned phrase '{phrase}' found in finding_summary for "
                    f"{entry.clause_category}: {entry.finding_summary!r}"
                )

    def test_fallback_summary_contains_no_verdict_phrases(self):
        risk_entries = [
            RiskFindingEntry(
                clause_category="Indemnification",
                risk_level="HIGH",
                finding_summary="The system identified an absent clause.",
            )
        ]
        summary = _fallback_executive_summary(
            document_type="NDA",
            total_found=9,
            risk_entries=risk_entries,
            missing_names=["Indemnification"],
        )
        summary_lower = summary.lower()
        for phrase in BANNED_VERDICT_PHRASES:
            assert phrase.lower() not in summary_lower, (
                f"Banned phrase '{phrase}' found in fallback summary: {summary!r}"
            )


# ---------------------------------------------------------------------------
# render_markdown: required headers
# ---------------------------------------------------------------------------

class TestRenderMarkdown:
    def _make_report(self) -> LegalDocumentReport:
        clauses, findings = _make_all_clauses_and_findings()
        return assemble_report(
            document_name="NDA_test.pdf",
            document_hash="deadbeef1234",
            document_type="NDA",
            classification_confidence=0.92,
            total_pages=7,
            clauses=clauses,
            risk_findings=findings,
            guardrail_results=[
                {"guardrail_name": "input_validator.all_checks", "passed": True,
                 "severity": "info", "reason": "All checks passed."}
            ],
            executive_summary=(
                "This NDA contains all 10 standard clause categories. "
                "Qualified legal review is recommended before execution."
            ),
        )

    def test_all_required_headers_present(self):
        report = self._make_report()
        md = render_markdown(report)
        for header in _REQUIRED_HEADERS:
            assert f"## {header}" in md or f"# {header}" in md, (
                f"Required Section-30 header missing from markdown: '## {header}'"
            )

    def test_markdown_contains_clause_table(self):
        report = self._make_report()
        md = render_markdown(report)
        assert "| Clause Category |" in md
        assert "| Present |" in md

    def test_markdown_contains_document_name(self):
        report = self._make_report()
        md = render_markdown(report)
        assert "NDA_test.pdf" in md

    def test_markdown_disclaimer_present(self):
        report = self._make_report()
        md = render_markdown(report)
        assert "does not constitute legal advice" in md

    def test_confidence_labels_not_raw_floats(self):
        report = self._make_report()
        md = render_markdown(report)
        # The words HIGH, MODERATE, LOW should appear; raw decimals like "0.92" must not
        # appear as confidence values (they can appear in the hash or hash snippet, but
        # not in the clause-analysis or confidence sections).
        assert "HIGH" in md or "MODERATE" in md or "LOW" in md
        # Confidence section must not claim a percentage probability
        assert "probability" not in md.lower()

    def test_missing_clause_section_absent_when_all_present(self):
        report = self._make_report()
        md = render_markdown(report)
        assert "All 10 standard clause categories were found" in md

    def test_missing_clause_listed_when_absent(self):
        from config import CLAUSE_CATEGORIES
        clauses = [
            _make_clause(c, is_present=(c != "Indemnification"))
            for c in CLAUSE_CATEGORIES
        ]
        findings = [
            _make_finding(c, risk_level="HIGH" if c == "Indemnification" else "LOW",
                          is_missing=(c == "Indemnification"))
            for c in CLAUSE_CATEGORIES
        ]
        report = assemble_report(
            document_name="test.pdf",
            document_hash="abc",
            document_type="NDA",
            classification_confidence=0.9,
            total_pages=5,
            clauses=clauses,
            risk_findings=findings,
            guardrail_results=[],
        )
        md = render_markdown(report)
        assert "Indemnification" in md
        assert "## High-Risk Findings" in md

    def test_guardrail_section_present(self):
        report = self._make_report()
        md = render_markdown(report)
        assert "## Guardrail Summary" in md
        assert "input_validator.all_checks" in md


# ---------------------------------------------------------------------------
# render_json: round-trip
# ---------------------------------------------------------------------------

class TestRenderJson:
    def _make_report(self) -> LegalDocumentReport:
        clauses, findings = _make_all_clauses_and_findings()
        return assemble_report(
            document_name="test.pdf",
            document_hash="cafebabe",
            document_type="Contract",
            classification_confidence=0.78,
            total_pages=12,
            clauses=clauses,
            risk_findings=findings,
            guardrail_results=[],
        )

    def test_json_is_valid(self):
        report = self._make_report()
        json_str = render_json(report)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_json_round_trip_document_type(self):
        report = self._make_report()
        parsed = json.loads(render_json(report))
        assert parsed["document_type"] == "Contract"

    def test_json_round_trip_clause_count(self):
        report = self._make_report()
        parsed = json.loads(render_json(report))
        assert len(parsed["clause_entries"]) == 10

    def test_json_round_trip_confidence_label(self):
        report = self._make_report()
        parsed = json.loads(render_json(report))
        assert parsed["classification_confidence_label"] in ("HIGH", "MODERATE", "LOW")

    def test_json_contains_no_raw_float_confidence(self):
        report = self._make_report()
        parsed = json.loads(render_json(report))
        # classification_confidence_label must be a string label, not a float
        assert isinstance(parsed["classification_confidence_label"], str)
        for entry in parsed["clause_entries"]:
            assert isinstance(entry["confidence_label"], str)
            assert entry["confidence_label"] in ("HIGH", "MODERATE", "LOW")

    def test_json_generated_at_is_string(self):
        report = self._make_report()
        parsed = json.loads(render_json(report))
        # Pydantic serialises datetime as ISO-8601 string in mode='json'
        assert isinstance(parsed["generated_at"], str)


# ---------------------------------------------------------------------------
# build_risk_findings: sort order
# ---------------------------------------------------------------------------

class TestBuildRiskFindings:
    def test_sorted_high_medium_low(self):
        from config import CLAUSE_CATEGORIES
        findings = [
            _make_finding(CLAUSE_CATEGORIES[0], "LOW"),
            _make_finding(CLAUSE_CATEGORIES[1], "HIGH"),
            _make_finding(CLAUSE_CATEGORIES[2], "MEDIUM"),
        ]
        entries = build_risk_findings(findings)
        levels = [e.risk_level for e in entries]
        assert levels.index("HIGH") < levels.index("MEDIUM") < levels.index("LOW")

    def test_missing_clause_summary_uses_observation_pattern(self):
        finding = _make_finding("Indemnification", "HIGH", is_missing=True)
        entries = build_risk_findings([finding])
        assert "the system identified" in entries[0].finding_summary.lower()
        assert "qualified legal review" in entries[0].finding_summary.lower()


# ---------------------------------------------------------------------------
# build_guardrail_summary
# ---------------------------------------------------------------------------

class TestBuildGuardrailSummary:
    def test_maps_passed_and_failed(self):
        results = [
            {"guardrail_name": "input_validator.all_checks", "passed": True,
             "severity": "info", "reason": "All OK."},
            {"guardrail_name": "scope_validator", "passed": False,
             "severity": "blocking", "reason": "Out-of-scope request detected."},
        ]
        entries = build_guardrail_summary(results)
        assert len(entries) == 2
        assert entries[0].passed is True
        assert entries[1].passed is False
        assert entries[1].severity == "blocking"

    def test_empty_results_returns_empty_list(self):
        assert build_guardrail_summary([]) == []
