"""
tests/unit/test_guardrails_schema.py
--------------------------------------
Construction and validation tests for all six models in schemas/guardrails.py.

Tests verify:
  - Required fields are enforced
  - Optional fields default correctly
  - Literal/enum constraints reject bad values
  - Numeric bounds (ge/le) on match_score, cited_page, etc.
  - Frozen config prevents mutation after construction
"""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from schemas.guardrails import (
    GuardrailResult,
    ScopeValidationResult,
    InjectionScanResult,
    EvidenceVerificationResult,
    PageReferenceVerificationResult,
    ClaimVerificationResult,
)


# ---------------------------------------------------------------------------
# GuardrailResult
# ---------------------------------------------------------------------------

class TestGuardrailResult:
    def test_basic_pass(self):
        r = GuardrailResult(
            guardrail_name="input_validator.file_exists",
            passed=True,
            reason="File exists and is readable.",
            severity="info",
        )
        assert r.passed is True
        assert r.severity == "info"

    def test_basic_fail_blocking(self):
        r = GuardrailResult(
            guardrail_name="input_validator.size_check",
            passed=False,
            reason="File exceeds 50 MB size limit.",
            severity="blocking",
        )
        assert r.passed is False
        assert r.severity == "blocking"

    def test_warning_severity(self):
        r = GuardrailResult(
            guardrail_name="duplicate_check",
            passed=False,
            reason="Document hash matches a previously processed file.",
            severity="warning",
        )
        assert r.severity == "warning"

    def test_invalid_severity_rejected(self):
        with pytest.raises(ValidationError):
            GuardrailResult(
                guardrail_name="test",
                passed=True,
                reason="ok",
                severity="critical",  # not in Literal
            )

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            GuardrailResult(passed=True, reason="ok", severity="info")  # no guardrail_name

    def test_frozen(self):
        r = GuardrailResult(
            guardrail_name="test", passed=True, reason="ok", severity="info"
        )
        with pytest.raises(Exception):  # frozen model raises on attribute set
            r.passed = False


# ---------------------------------------------------------------------------
# ScopeValidationResult
# ---------------------------------------------------------------------------

class TestScopeValidationResult:
    def test_pass_no_intent(self):
        r = ScopeValidationResult(
            guardrail_name="scope_validator",
            passed=True,
            reason="Request is within analysis scope.",
            severity="info",
            detected_intent=None,
        )
        assert r.detected_intent is None

    def test_fail_with_intent(self):
        r = ScopeValidationResult(
            guardrail_name="scope_validator",
            passed=False,
            reason="Request asks for autonomous contract approval.",
            severity="blocking",
            detected_intent="binding_legal_approval",
        )
        assert r.detected_intent == "binding_legal_approval"

    def test_detected_intent_defaults_none(self):
        r = ScopeValidationResult(
            guardrail_name="scope_validator",
            passed=True,
            reason="ok",
            severity="info",
        )
        assert r.detected_intent is None

    def test_inherits_guardrail_fields(self):
        r = ScopeValidationResult(
            guardrail_name="scope_validator",
            passed=False,
            reason="Out of scope.",
            severity="blocking",
            detected_intent="contract_execution",
        )
        assert r.guardrail_name == "scope_validator"
        assert r.severity == "blocking"

    def test_frozen(self):
        r = ScopeValidationResult(
            guardrail_name="scope_validator",
            passed=True,
            reason="ok",
            severity="info",
        )
        with pytest.raises(Exception):
            r.passed = False


# ---------------------------------------------------------------------------
# InjectionScanResult
# ---------------------------------------------------------------------------

class TestInjectionScanResult:
    def test_clean_scan(self):
        r = InjectionScanResult(
            guardrail_name="prompt_injection.scan",
            passed=True,
            reason="No injection patterns detected.",
            severity="info",
            technique_categories=[],
            flagged_segments=[],
        )
        assert r.technique_categories == []
        assert r.flagged_segments == []

    def test_injection_detected(self):
        r = InjectionScanResult(
            guardrail_name="prompt_injection.scan",
            passed=False,
            reason="Instruction override attempt detected.",
            severity="blocking",
            technique_categories=["instruction_override"],
            flagged_segments=["ignore previous instructions"],
        )
        assert "instruction_override" in r.technique_categories
        assert len(r.flagged_segments) == 1

    def test_multiple_categories(self):
        r = InjectionScanResult(
            guardrail_name="prompt_injection.scan",
            passed=False,
            reason="Multiple injection techniques detected.",
            severity="blocking",
            technique_categories=["instruction_override", "role_manipulation"],
            flagged_segments=["segment1", "segment2"],
        )
        assert len(r.technique_categories) == 2

    def test_defaults_empty_lists(self):
        r = InjectionScanResult(
            guardrail_name="prompt_injection.scan",
            passed=True,
            reason="ok",
            severity="info",
        )
        assert r.technique_categories == []
        assert r.flagged_segments == []

    def test_frozen(self):
        r = InjectionScanResult(
            guardrail_name="prompt_injection.scan",
            passed=True,
            reason="ok",
            severity="info",
        )
        with pytest.raises(Exception):
            r.passed = False


# ---------------------------------------------------------------------------
# EvidenceVerificationResult
# ---------------------------------------------------------------------------

class TestEvidenceVerificationResult:
    def test_exact_match(self):
        r = EvidenceVerificationResult(
            extracted_text="The parties agree to maintain confidentiality.",
            found_in_source=True,
            match_type="exact",
            match_score=None,
            source_page_checked=3,
        )
        assert r.found_in_source is True
        assert r.match_type == "exact"
        assert r.match_score is None

    def test_fuzzy_match(self):
        r = EvidenceVerificationResult(
            extracted_text="Both parties shall maintain confidentiality.",
            found_in_source=True,
            match_type="fuzzy",
            match_score=0.85,
            source_page_checked=3,
        )
        assert r.match_type == "fuzzy"
        assert r.match_score == pytest.approx(0.85)

    def test_not_found(self):
        r = EvidenceVerificationResult(
            extracted_text="Clause that doesn't exist.",
            found_in_source=False,
            match_type="not_found",
        )
        assert r.found_in_source is False
        assert r.match_score is None
        assert r.source_page_checked is None

    def test_invalid_match_type(self):
        with pytest.raises(ValidationError):
            EvidenceVerificationResult(
                extracted_text="text",
                found_in_source=True,
                match_type="partial",  # not in Literal
            )

    def test_match_score_bounds(self):
        with pytest.raises(ValidationError):
            EvidenceVerificationResult(
                extracted_text="text",
                found_in_source=True,
                match_type="fuzzy",
                match_score=1.5,  # > 1.0
            )
        with pytest.raises(ValidationError):
            EvidenceVerificationResult(
                extracted_text="text",
                found_in_source=True,
                match_type="fuzzy",
                match_score=-0.1,  # < 0.0
            )

    def test_page_checked_must_be_positive(self):
        with pytest.raises(ValidationError):
            EvidenceVerificationResult(
                extracted_text="text",
                found_in_source=True,
                match_type="exact",
                source_page_checked=0,  # ge=1
            )


# ---------------------------------------------------------------------------
# PageReferenceVerificationResult
# ---------------------------------------------------------------------------

class TestPageReferenceVerificationResult:
    def test_valid_page_found(self):
        r = PageReferenceVerificationResult(
            cited_page=3,
            page_exists_in_document=True,
            text_found_near_cited_page=True,
        )
        assert r.page_exists_in_document is True
        assert r.text_found_near_cited_page is True
        assert r.notes is None

    def test_invalid_page(self):
        r = PageReferenceVerificationResult(
            cited_page=50,
            page_exists_in_document=False,
            text_found_near_cited_page=False,
            notes="Cited page 50 exceeds document length of 10 pages.",
        )
        assert r.page_exists_in_document is False
        assert r.notes is not None

    def test_found_on_adjacent_page(self):
        r = PageReferenceVerificationResult(
            cited_page=5,
            page_exists_in_document=True,
            text_found_near_cited_page=True,
            notes="Text found on page 6 (adjacent to cited page 5).",
        )
        assert r.text_found_near_cited_page is True

    def test_cited_page_accepts_any_integer(self):
        """
        cited_page records what the LLM claimed -- may be 0 or negative.
        The verifier sets page_exists_in_document=False for such values;
        the schema itself does not enforce ge=1.
        """
        r = PageReferenceVerificationResult(
            cited_page=0,  # invalid page, but schema accepts it
            page_exists_in_document=False,
            text_found_near_cited_page=False,
        )
        assert r.cited_page == 0
        assert r.page_exists_in_document is False

    def test_frozen(self):
        r = PageReferenceVerificationResult(
            cited_page=1,
            page_exists_in_document=True,
            text_found_near_cited_page=True,
        )
        with pytest.raises(Exception):
            r.cited_page = 2


# ---------------------------------------------------------------------------
# ClaimVerificationResult
# ---------------------------------------------------------------------------

class TestClaimVerificationResult:
    def test_passed(self):
        r = ClaimVerificationResult(
            claim_text="Confidentiality clause present on page 3.",
            has_supporting_evidence=True,
            evidence_source="chunk-042",
            action_taken="passed",
        )
        assert r.action_taken == "passed"
        assert r.has_supporting_evidence is True

    def test_removed(self):
        r = ClaimVerificationResult(
            claim_text="LOW risk clause with unverifiable text.",
            has_supporting_evidence=False,
            evidence_source=None,
            action_taken="removed",
        )
        assert r.action_taken == "removed"
        assert r.evidence_source is None

    def test_escalated(self):
        r = ClaimVerificationResult(
            claim_text="HIGH risk: Indemnification clause absent.",
            has_supporting_evidence=False,
            evidence_source=None,
            action_taken="escalated_to_human_review",
        )
        assert r.action_taken == "escalated_to_human_review"

    def test_invalid_action(self):
        with pytest.raises(ValidationError):
            ClaimVerificationResult(
                claim_text="some claim",
                has_supporting_evidence=True,
                action_taken="ignored",  # not in Literal
            )

    def test_defaults(self):
        r = ClaimVerificationResult(
            claim_text="claim",
            has_supporting_evidence=True,
            action_taken="passed",
        )
        assert r.evidence_source is None

    def test_frozen(self):
        r = ClaimVerificationResult(
            claim_text="claim",
            has_supporting_evidence=True,
            action_taken="passed",
        )
        with pytest.raises(Exception):
            r.action_taken = "removed"
