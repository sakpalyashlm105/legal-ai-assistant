"""
tests/unit/test_review_schema_v2.py
--------------------------------------
Unit tests for the HITL-deepening upgrades to schemas/review.py.

Covers:
  1.  ReviewItem: new reviewer-context fields construct correctly.
  2.  ReviewItem: new fields default correctly when not supplied.
  3.  ReviewDecision: HIGH-risk item requires non-empty reason (validator).
  4.  ReviewDecision: MEDIUM-risk item allows empty reason.
  5.  ReviewDecision: LOW-risk item allows empty reason.
  6.  ReviewDecision: reason NOT required when risk_level_on_item is None.
  7.  ReviewDecision: reject requires reject_category (validator).
  8.  ReviewDecision: reject without reject_category raises ValidationError.
  9.  ReviewDecision: correct action populates corrected_risk_level/summary/rationale.
  10. ReviewDecision: flag_for_regression_dataset defaults False, can be set True.
  11. ReviewDecision: all 7 reject_category values are valid.
  12. Original ReviewItem construction still works (no regression).
  13. Original ReviewDecision approve/correct/select_alternative still work.
  14. ReviewItem serialization round-trip includes new fields.

How to run:
    pytest tests/unit/test_review_schema_v2.py tests/unit/test_review_schema.py -v
"""

import pytest
from pydantic import ValidationError

from schemas.review import ReviewItem, ReviewDecision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_item(**overrides) -> ReviewItem:
    defaults = dict(
        review_id="rev-v2-001",
        document_hash="b" * 64,
        source_text="The indemnifying party shall hold harmless...",
        trigger_reason="high_risk_finding",
        ai_finding_summary="PRESENT | Indemnification | HIGH | major deviation from template",
        thread_id="thread-v2-001",
        risk_level="HIGH",
    )
    defaults.update(overrides)
    return ReviewItem(**defaults)


def _make_approve(review_id="rev-v2-001", risk_level=None, reason="", **overrides) -> ReviewDecision:
    return ReviewDecision(
        review_id=review_id,
        action="approve",
        risk_level_on_item=risk_level,
        reason=reason,
        **overrides,
    )


# ---------------------------------------------------------------------------
# 1. ReviewItem: new reviewer-context fields construct correctly
# ---------------------------------------------------------------------------

class TestReviewItemNewFields:
    def test_full_context_fields(self):
        item = _make_item(
            template_clause_text="Each party shall indemnify and hold harmless...",
            missing_elements=["mutuality", "survival period"],
            extra_risky_language=["one-sided obligation"],
            fact_found="Indemnification clause is present in the document.",
            deviation_found="Clause is one-sided; template expected mutual obligations.",
            risk_rationale="Marked HIGH because template expected mutual indemnification.",
            evidence_match_type="fuzzy",
            evidence_match_score=0.78,
            page_reference_valid=True,
            extraction_confidence=0.91,
            llm_confidence=0.91,
            previous_chunk_text="...the terms of this Agreement shall...",
            next_chunk_text="...subject to applicable law and...",
            section_heading=None,
            document_type_context="NDA",
        )
        assert item.template_clause_text == "Each party shall indemnify and hold harmless..."
        assert item.missing_elements == ["mutuality", "survival period"]
        assert item.extra_risky_language == ["one-sided obligation"]
        assert item.fact_found == "Indemnification clause is present in the document."
        assert item.deviation_found == "Clause is one-sided; template expected mutual obligations."
        assert item.risk_rationale == "Marked HIGH because template expected mutual indemnification."
        assert item.evidence_match_type == "fuzzy"
        assert item.evidence_match_score == 0.78
        assert item.page_reference_valid is True
        assert item.extraction_confidence == 0.91
        assert item.llm_confidence == 0.91
        assert item.previous_chunk_text == "...the terms of this Agreement shall..."
        assert item.next_chunk_text == "...subject to applicable law and..."
        assert item.section_heading is None
        assert item.document_type_context == "NDA"

    def test_evidence_match_type_values(self):
        for match_type in ("exact", "fuzzy", "not_found"):
            item = _make_item(evidence_match_type=match_type)
            assert item.evidence_match_type == match_type

    def test_invalid_evidence_match_type_raises(self):
        with pytest.raises(ValidationError):
            _make_item(evidence_match_type="partial")


# ---------------------------------------------------------------------------
# 2. ReviewItem: new fields default correctly when not supplied
# ---------------------------------------------------------------------------

class TestReviewItemNewFieldDefaults:
    def test_all_new_fields_default_to_none_or_empty(self):
        item = _make_item()
        assert item.template_clause_text is None
        assert item.missing_elements == []
        assert item.extra_risky_language == []
        assert item.fact_found == ""
        assert item.deviation_found is None
        assert item.risk_rationale == ""
        assert item.evidence_match_type is None
        assert item.evidence_match_score is None
        assert item.page_reference_valid is None
        assert item.extraction_confidence is None
        assert item.llm_confidence is None
        assert item.previous_chunk_text is None
        assert item.next_chunk_text is None
        assert item.section_heading is None
        assert item.document_type_context is None


# ---------------------------------------------------------------------------
# 3-6. ReviewDecision: HIGH-risk reason validator
# ---------------------------------------------------------------------------

class TestHighRiskRequiresReason:
    def test_high_risk_with_reason_passes(self):
        d = ReviewDecision(
            review_id="rev-001",
            action="approve",
            risk_level_on_item="HIGH",
            reason="Reviewed the clause text; AI finding is correct.",
        )
        assert d.reason == "Reviewed the clause text; AI finding is correct."

    def test_high_risk_empty_reason_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            ReviewDecision(
                review_id="rev-001",
                action="approve",
                risk_level_on_item="HIGH",
                reason="",
            )
        assert "reason is required" in str(exc_info.value)

    def test_high_risk_whitespace_reason_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            ReviewDecision(
                review_id="rev-001",
                action="approve",
                risk_level_on_item="HIGH",
                reason="   ",
            )
        assert "reason is required" in str(exc_info.value)

    def test_medium_risk_empty_reason_allowed(self):
        d = ReviewDecision(
            review_id="rev-001",
            action="approve",
            risk_level_on_item="MEDIUM",
            reason="",
        )
        assert d.reason == ""

    def test_low_risk_empty_reason_allowed(self):
        d = ReviewDecision(
            review_id="rev-001",
            action="approve",
            risk_level_on_item="LOW",
            reason="",
        )
        assert d.reason == ""

    def test_none_risk_level_empty_reason_allowed(self):
        d = ReviewDecision(
            review_id="rev-001",
            action="approve",
            risk_level_on_item=None,
        )
        assert d.reason == ""


# ---------------------------------------------------------------------------
# 7-8. ReviewDecision: reject requires reject_category
# ---------------------------------------------------------------------------

class TestRejectCategoryRequired:
    def test_reject_with_category_passes(self):
        d = ReviewDecision(
            review_id="rev-001",
            action="reject",
            reject_category="hallucinated_deviation",
            reason="AI cited evidence that is not in the document.",
        )
        assert d.reject_category == "hallucinated_deviation"

    def test_reject_without_category_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            ReviewDecision(review_id="rev-001", action="reject")
        assert "reject_category is required" in str(exc_info.value)

    def test_all_seven_reject_categories_valid(self):
        categories = [
            "duplicate_finding",
            "wrong_clause_category",
            "hallucinated_deviation",
            "evidence_mismatch",
            "low_materiality",
            "template_mismatch",
            "not_legally_relevant",
        ]
        for cat in categories:
            d = ReviewDecision(
                review_id="rev-001",
                action="reject",
                reject_category=cat,
            )
            assert d.reject_category == cat

    def test_invalid_reject_category_raises(self):
        with pytest.raises(ValidationError):
            ReviewDecision(
                review_id="rev-001",
                action="reject",
                reject_category="made_up_category",
            )

    def test_approve_does_not_require_category(self):
        d = ReviewDecision(review_id="rev-001", action="approve", risk_level_on_item="LOW")
        assert d.reject_category is None


# ---------------------------------------------------------------------------
# 9. ReviewDecision: correct action with new corrected_* fields
# ---------------------------------------------------------------------------

class TestCorrectActionFields:
    def test_correct_with_structured_fields(self):
        d = ReviewDecision(
            review_id="rev-001",
            action="correct",
            corrected_value={"clause_type": "Indemnification", "is_present": True},
            corrected_risk_level="MEDIUM",
            corrected_summary="Clause is present but one-sided; not absent as AI claimed.",
            corrected_rationale="Human reviewer located clause in Exhibit A paragraph 3.",
            reason="AI missed clause that appears in an exhibit.",
            risk_level_on_item="HIGH",
        )
        assert d.corrected_risk_level == "MEDIUM"
        assert d.corrected_summary == "Clause is present but one-sided; not absent as AI claimed."
        assert d.corrected_rationale == "Human reviewer located clause in Exhibit A paragraph 3."
        # corrected_value backward-compat field still present
        assert d.corrected_value["clause_type"] == "Indemnification"

    def test_corrected_risk_level_values(self):
        for level in ("LOW", "MEDIUM", "HIGH"):
            d = ReviewDecision(
                review_id="rev-001",
                action="correct",
                corrected_value={"x": 1},
                corrected_risk_level=level,
            )
            assert d.corrected_risk_level == level

    def test_corrected_fields_default_none_for_approve(self):
        d = ReviewDecision(review_id="rev-001", action="approve")
        assert d.corrected_risk_level is None
        assert d.corrected_summary is None
        assert d.corrected_rationale is None


# ---------------------------------------------------------------------------
# 10. flag_for_regression_dataset field
# ---------------------------------------------------------------------------

class TestFlagForRegressionDataset:
    def test_defaults_false(self):
        d = ReviewDecision(review_id="rev-001", action="approve")
        assert d.flag_for_regression_dataset is False

    def test_can_be_set_true(self):
        d = ReviewDecision(
            review_id="rev-001",
            action="approve",
            flag_for_regression_dataset=True,
        )
        assert d.flag_for_regression_dataset is True


# ---------------------------------------------------------------------------
# 12-13. Regression: original construction patterns still work
# ---------------------------------------------------------------------------

class TestOriginalConstructionRegression:
    def test_minimal_review_item_still_works(self):
        item = ReviewItem(
            review_id="rev-orig-001",
            document_hash="a" * 64,
            source_text="Some clause text here.",
            trigger_reason="low_confidence_after_retry",
            ai_finding_summary="PRESENT | Confidentiality | MEDIUM | confidence=0.43",
            thread_id="thread-orig-001",
        )
        assert item.review_id == "rev-orig-001"
        assert item.status == "pending"

    def test_approve_decision_still_works(self):
        d = ReviewDecision(review_id="rev-orig-001", action="approve")
        assert d.action == "approve"

    def test_correct_decision_still_works(self):
        d = ReviewDecision(
            review_id="rev-orig-001",
            action="correct",
            corrected_value={"clause_type": "Indemnification", "is_present": True},
        )
        assert d.corrected_value["is_present"] is True

    def test_select_alternative_still_works(self):
        d = ReviewDecision(
            review_id="rev-orig-001",
            action="select_alternative",
            selected_alternative_id="alt-1",
        )
        assert d.selected_alternative_id == "alt-1"


# ---------------------------------------------------------------------------
# 14. Serialization round-trip
# ---------------------------------------------------------------------------

class TestSerializationRoundTrip:
    def test_review_item_round_trip(self):
        item = _make_item(
            template_clause_text="Standard template text.",
            missing_elements=["survival period"],
            fact_found="Clause is present.",
            risk_rationale="Marked HIGH due to missing mutuality.",
            evidence_match_type="exact",
            evidence_match_score=1.0,
            document_type_context="NDA",
        )
        d = item.model_dump(mode="json")
        item2 = ReviewItem(**d)
        assert item2.template_clause_text == item.template_clause_text
        assert item2.missing_elements == item.missing_elements
        assert item2.fact_found == item.fact_found
        assert item2.evidence_match_type == item.evidence_match_type
        assert item2.document_type_context == item.document_type_context

    def test_review_decision_round_trip(self):
        d = ReviewDecision(
            review_id="rev-001",
            action="reject",
            reject_category="evidence_mismatch",
            reason="Evidence cited by AI does not appear in the document.",
            flag_for_regression_dataset=True,
        )
        dumped = d.model_dump(mode="json")
        d2 = ReviewDecision(**dumped)
        assert d2.reject_category == "evidence_mismatch"
        assert d2.flag_for_regression_dataset is True
