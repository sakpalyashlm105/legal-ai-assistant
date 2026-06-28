"""
tests/unit/test_review_schema.py
---------------------------------
Unit tests for schemas/review.py (ReviewItem and ReviewDecision).

No OpenAI API calls, no file I/O, no LangGraph machinery -- purely
testing the Pydantic schemas and their validators in isolation.

What this file tests:
    1.  Valid ReviewItem construction with all required fields.
    2.  Valid ReviewDecision for each of the 4 allowed actions.
    3.  action="correct" without corrected_value raises ValidationError.
    4.  action="correct" with selected_alternative_id set also raises.
    5.  action="select_alternative" without selected_alternative_id raises.
    6.  action="select_alternative" with corrected_value set also raises.
    7.  action="approve" with corrected_value set raises.
    8.  action="approve" with selected_alternative_id set raises.
    9.  action="reject" with corrected_value set raises.
    10. action="reject" with selected_alternative_id set raises.
    11. ReviewItem status only accepts "pending" / "resolved".
    12. ReviewItem with alternatives list (for ToT select_alternative path).
    13. ReviewItem with optional fields left as None (minimal construction).

How to run:
    pytest tests/unit/test_review_schema.py -v
"""

from datetime import datetime
import pytest
from pydantic import ValidationError

from schemas.review import ReviewItem, ReviewDecision


# ---------------------------------------------------------------------------
# Helpers: minimal valid ReviewItem and ReviewDecision builders
# ---------------------------------------------------------------------------

def _make_review_item(**overrides) -> ReviewItem:
    defaults = dict(
        review_id="rev-001",
        document_hash="a" * 64,
        source_text="The disclosing party shall keep all information confidential.",
        trigger_reason="low_confidence_after_retry",
        ai_finding_summary="PRESENT | Confidentiality/Non-Disclosure | MEDIUM risk | confidence=0.43",
        thread_id="thread-abc-123",
    )
    defaults.update(overrides)
    return ReviewItem(**defaults)


def _make_approve_decision(**overrides) -> ReviewDecision:
    defaults = dict(review_id="rev-001", action="approve")
    defaults.update(overrides)
    return ReviewDecision(**defaults)


# ---------------------------------------------------------------------------
# Test 1: Valid ReviewItem construction
# ---------------------------------------------------------------------------

def test_review_item_valid_minimal():
    """Minimal required fields should construct without error."""
    item = _make_review_item()
    assert item.review_id == "rev-001"
    assert item.status == "pending"
    assert item.clause_category is None
    assert item.alternatives == []
    assert isinstance(item.created_at, datetime)


def test_review_item_valid_full():
    """ReviewItem with all optional fields populated."""
    item = _make_review_item(
        source_chunk_id="chunk-007",
        clause_category="Indemnification",
        source_page=4,
        confidence_signals={"final_confidence": 0.43, "retry_count": 1},
        alternatives=[{"id": "alt-1", "summary": "interpretation A"}],
        risk_level="HIGH",
        template_comparison_summary="Clause is broader than template standard.",
        status="pending",
    )
    assert item.source_chunk_id == "chunk-007"
    assert item.clause_category == "Indemnification"
    assert item.source_page == 4
    assert item.risk_level == "HIGH"
    assert len(item.alternatives) == 1


# ---------------------------------------------------------------------------
# Test 2: Valid ReviewDecision for all 4 actions
# ---------------------------------------------------------------------------

def test_decision_approve_valid():
    """approve requires no corrected_value or selected_alternative_id."""
    d = ReviewDecision(review_id="rev-001", action="approve")
    assert d.action == "approve"
    assert d.corrected_value is None
    assert d.selected_alternative_id is None


def test_decision_reject_valid():
    """reject is valid with reject_category specified."""
    d = ReviewDecision(
        review_id="rev-001",
        action="reject",
        reviewer_note="Hallucinated text",
        reject_category="hallucinated_deviation",
    )
    assert d.action == "reject"
    assert d.reviewer_note == "Hallucinated text"


def test_decision_correct_valid():
    """correct requires corrected_value; selected_alternative_id must be absent."""
    d = ReviewDecision(
        review_id="rev-001",
        action="correct",
        corrected_value={"clause_type": "Indemnification", "is_present": True},
        reviewer_note="AI missed this clause in paragraph 3.",
    )
    assert d.action == "correct"
    assert d.corrected_value["is_present"] is True
    assert d.selected_alternative_id is None


def test_decision_select_alternative_valid():
    """select_alternative requires selected_alternative_id; corrected_value must be absent."""
    d = ReviewDecision(
        review_id="rev-001",
        action="select_alternative",
        selected_alternative_id="alt-2",
    )
    assert d.action == "select_alternative"
    assert d.selected_alternative_id == "alt-2"
    assert d.corrected_value is None


# ---------------------------------------------------------------------------
# Test 3: correct without corrected_value raises
# ---------------------------------------------------------------------------

def test_correct_without_corrected_value_raises():
    with pytest.raises(ValidationError) as exc_info:
        ReviewDecision(review_id="rev-001", action="correct")
    assert "corrected_value is required" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 4: correct with selected_alternative_id set also raises
# ---------------------------------------------------------------------------

def test_correct_with_selected_alternative_id_raises():
    with pytest.raises(ValidationError) as exc_info:
        ReviewDecision(
            review_id="rev-001",
            action="correct",
            corrected_value={"x": 1},
            selected_alternative_id="alt-1",
        )
    assert "selected_alternative_id must be None" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 5: select_alternative without selected_alternative_id raises
# ---------------------------------------------------------------------------

def test_select_alternative_without_id_raises():
    with pytest.raises(ValidationError) as exc_info:
        ReviewDecision(review_id="rev-001", action="select_alternative")
    assert "selected_alternative_id is required" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 6: select_alternative with corrected_value set also raises
# ---------------------------------------------------------------------------

def test_select_alternative_with_corrected_value_raises():
    with pytest.raises(ValidationError) as exc_info:
        ReviewDecision(
            review_id="rev-001",
            action="select_alternative",
            selected_alternative_id="alt-1",
            corrected_value={"x": 1},
        )
    assert "corrected_value must be None" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 7: approve with corrected_value set raises
# ---------------------------------------------------------------------------

def test_approve_with_corrected_value_raises():
    with pytest.raises(ValidationError) as exc_info:
        ReviewDecision(
            review_id="rev-001",
            action="approve",
            corrected_value={"something": "wrong"},
        )
    assert "corrected_value must be None" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 8: approve with selected_alternative_id set raises
# ---------------------------------------------------------------------------

def test_approve_with_selected_alternative_raises():
    with pytest.raises(ValidationError) as exc_info:
        ReviewDecision(
            review_id="rev-001",
            action="approve",
            selected_alternative_id="alt-1",
        )
    assert "selected_alternative_id must be None" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 9: reject with corrected_value set raises
# ---------------------------------------------------------------------------

def test_reject_with_corrected_value_raises():
    with pytest.raises(ValidationError) as exc_info:
        ReviewDecision(
            review_id="rev-001",
            action="reject",
            corrected_value="something",
            reject_category="hallucinated_deviation",
        )
    assert "corrected_value must be None" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 10: reject with selected_alternative_id set raises
# ---------------------------------------------------------------------------

def test_reject_with_selected_alternative_raises():
    with pytest.raises(ValidationError) as exc_info:
        ReviewDecision(
            review_id="rev-001",
            action="reject",
            selected_alternative_id="alt-1",
            reject_category="hallucinated_deviation",
        )
    assert "selected_alternative_id must be None" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 11: ReviewItem status only accepts "pending" / "resolved"
# ---------------------------------------------------------------------------

def test_review_item_invalid_status_raises():
    with pytest.raises(ValidationError):
        _make_review_item(status="in_progress")  # not a valid Literal


def test_review_item_resolved_status_valid():
    item = _make_review_item(status="resolved")
    assert item.status == "resolved"


# ---------------------------------------------------------------------------
# Test 12: ReviewItem with alternatives list for ToT path
# ---------------------------------------------------------------------------

def test_review_item_with_tot_alternatives():
    alternatives = [
        {"id": "alt-1", "summary": "Clause is present, confidence 0.62", "score": 0.62},
        {"id": "alt-2", "summary": "Clause is absent, confidence 0.38", "score": 0.38},
    ]
    item = _make_review_item(
        trigger_reason="tot_unresolved_tie",
        alternatives=alternatives,
    )
    assert len(item.alternatives) == 2
    assert item.alternatives[0]["id"] == "alt-1"


# ---------------------------------------------------------------------------
# Test 13: ReviewItem with all optional fields as None (minimal)
# ---------------------------------------------------------------------------

def test_review_item_minimal_all_optionals_none():
    item = ReviewItem(
        review_id="rev-min",
        document_hash="b" * 64,
        source_text="Some clause text.",
        trigger_reason="missing_critical_clause",
        ai_finding_summary="ABSENT | Indemnification | HIGH",
        thread_id="thread-xyz",
    )
    assert item.source_chunk_id is None
    assert item.clause_category is None
    assert item.source_page is None
    assert item.risk_level is None
    assert item.template_comparison_summary is None
    assert item.confidence_signals == {}
    assert item.alternatives == []
