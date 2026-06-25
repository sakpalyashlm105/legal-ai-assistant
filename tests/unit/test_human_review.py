"""
tests/unit/test_human_review.py
---------------------------------
Unit tests for agent/human_review.py.

Uses a temporary directory for the JSON queue files so tests never touch
the real data/review_queue/ and always start with a clean slate.

What this file tests:
    1.  add_to_review_queue + get_pending_reviews round-trips correctly.
    2.  get_pending_reviews filters correctly by document_hash.
    3.  Resolved items no longer appear in get_pending_reviews.
    4.  record_review_decision for action="approve" marks item resolved.
    5.  record_review_decision for action="correct" marks item resolved.
    6.  record_review_decision for action="select_alternative" validates ID.
    7.  record_review_decision for action="reject" marks item resolved.
    8.  recording a decision for a non-existent review_id raises KeyError.
    9.  apply_human_decision("approve") returns discarded=False, value=None.
    10. apply_human_decision("correct") returns the corrected_value.
    11. apply_human_decision("select_alternative") returns the chosen alt dict.
    12. apply_human_decision("reject") returns discarded=True.
    13. select_alternative with an ID not in item.alternatives raises ValueError.
    14. add_to_review_queue is idempotent (re-adding same review_id replaces, not duplicates).

How to run:
    pytest tests/unit/test_human_review.py -v
"""

import os
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

import agent.human_review as hr_module
from schemas.review import ReviewDecision, ReviewItem


# ---------------------------------------------------------------------------
# Fixture: redirect queue storage to a temp dir for each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_queue(tmp_path):
    """
    Patch the three path constants in agent.human_review to point at a
    fresh temp directory for every test. This ensures tests are fully
    isolated: no test sees queue state from any other test.
    """
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
# Helpers
# ---------------------------------------------------------------------------

def _make_item(
    review_id: str = None,
    document_hash: str = None,
    trigger: str = "low_confidence_after_retry",
    alternatives: list = None,
    **kwargs,
) -> ReviewItem:
    return ReviewItem(
        review_id=review_id or f"rev-{uuid.uuid4().hex[:8]}",
        document_hash=document_hash or "a" * 64,
        source_text="Each party shall keep information confidential.",
        trigger_reason=trigger,
        ai_finding_summary="PRESENT | Confidentiality/Non-Disclosure | MEDIUM | conf=0.43",
        thread_id="thread-test-001",
        alternatives=alternatives or [],
        **kwargs,
    )


def _approve(review_id: str) -> ReviewDecision:
    return ReviewDecision(review_id=review_id, action="approve")


# ---------------------------------------------------------------------------
# Test 1: add and retrieve from pending queue
# ---------------------------------------------------------------------------

def test_add_and_retrieve_pending():
    item = _make_item()
    hr_module.add_to_review_queue(item)
    pending = hr_module.get_pending_reviews()
    assert len(pending) == 1
    assert pending[0].review_id == item.review_id
    assert pending[0].status == "pending"


# ---------------------------------------------------------------------------
# Test 2: get_pending_reviews filters by document_hash
# ---------------------------------------------------------------------------

def test_filter_pending_by_document_hash():
    hash_a = "a" * 64
    hash_b = "b" * 64
    item_a = _make_item(document_hash=hash_a)
    item_b = _make_item(document_hash=hash_b)
    hr_module.add_to_review_queue(item_a)
    hr_module.add_to_review_queue(item_b)

    filtered = hr_module.get_pending_reviews(document_hash=hash_a)
    assert len(filtered) == 1
    assert filtered[0].document_hash == hash_a


# ---------------------------------------------------------------------------
# Test 3: resolved items no longer appear in get_pending_reviews
# ---------------------------------------------------------------------------

def test_resolved_item_not_in_pending():
    item = _make_item()
    hr_module.add_to_review_queue(item)
    hr_module.record_review_decision(_approve(item.review_id))
    pending = hr_module.get_pending_reviews()
    assert len(pending) == 0


# ---------------------------------------------------------------------------
# Test 4: record_review_decision approve -> item marked resolved
# ---------------------------------------------------------------------------

def test_record_approve_marks_resolved():
    item = _make_item()
    hr_module.add_to_review_queue(item)
    resolved_item = hr_module.record_review_decision(_approve(item.review_id))
    assert resolved_item.status == "resolved"
    assert resolved_item.review_id == item.review_id


# ---------------------------------------------------------------------------
# Test 5: record_review_decision correct -> item marked resolved
# ---------------------------------------------------------------------------

def test_record_correct_marks_resolved():
    item = _make_item()
    hr_module.add_to_review_queue(item)
    decision = ReviewDecision(
        review_id=item.review_id,
        action="correct",
        corrected_value={"clause_type": "Indemnification", "is_present": True},
    )
    resolved_item = hr_module.record_review_decision(decision)
    assert resolved_item.status == "resolved"


# ---------------------------------------------------------------------------
# Test 6: record select_alternative validates the ID
# ---------------------------------------------------------------------------

def test_record_select_alternative_valid_id():
    alternatives = [
        {"id": "alt-1", "summary": "Clause present, high confidence"},
        {"id": "alt-2", "summary": "Clause absent, low confidence"},
    ]
    item = _make_item(alternatives=alternatives)
    hr_module.add_to_review_queue(item)
    decision = ReviewDecision(
        review_id=item.review_id,
        action="select_alternative",
        selected_alternative_id="alt-1",
    )
    resolved_item = hr_module.record_review_decision(decision)
    assert resolved_item.status == "resolved"


def test_record_select_alternative_invalid_id_raises():
    alternatives = [{"id": "alt-1", "summary": "Only option"}]
    item = _make_item(alternatives=alternatives)
    hr_module.add_to_review_queue(item)
    decision = ReviewDecision(
        review_id=item.review_id,
        action="select_alternative",
        selected_alternative_id="alt-NONEXISTENT",
    )
    with pytest.raises(ValueError, match="does not match any alternative"):
        hr_module.record_review_decision(decision)


# ---------------------------------------------------------------------------
# Test 7: record reject -> item marked resolved
# ---------------------------------------------------------------------------

def test_record_reject_marks_resolved():
    item = _make_item()
    hr_module.add_to_review_queue(item)
    decision = ReviewDecision(
        review_id=item.review_id,
        action="reject",
        reviewer_note="AI hallucinated this clause.",
    )
    resolved_item = hr_module.record_review_decision(decision)
    assert resolved_item.status == "resolved"


# ---------------------------------------------------------------------------
# Test 8: recording a decision for non-existent review_id raises KeyError
# ---------------------------------------------------------------------------

def test_record_decision_nonexistent_id_raises():
    with pytest.raises(KeyError, match="No pending review item found"):
        hr_module.record_review_decision(_approve("nonexistent-id"))


# ---------------------------------------------------------------------------
# Test 9: apply_human_decision approve
# ---------------------------------------------------------------------------

def test_apply_approve_returns_no_change_signal():
    item = _make_item()
    decision = _approve(item.review_id)
    result = hr_module.apply_human_decision(decision, item)
    assert result["action"] == "approve"
    assert result["value"] is None
    assert result["discarded"] is False


# ---------------------------------------------------------------------------
# Test 10: apply_human_decision correct returns corrected_value
# ---------------------------------------------------------------------------

def test_apply_correct_returns_corrected_value():
    item = _make_item()
    correction = {"clause_type": "Indemnification", "is_present": True, "confidence": 0.9}
    decision = ReviewDecision(
        review_id=item.review_id,
        action="correct",
        corrected_value=correction,
    )
    result = hr_module.apply_human_decision(decision, item)
    assert result["action"] == "correct"
    assert result["value"] == correction
    assert result["discarded"] is False


# ---------------------------------------------------------------------------
# Test 11: apply_human_decision select_alternative returns chosen alt dict
# ---------------------------------------------------------------------------

def test_apply_select_alternative_returns_chosen_alt():
    alternatives = [
        {"id": "alt-1", "summary": "Interpretation A", "score": 0.7},
        {"id": "alt-2", "summary": "Interpretation B", "score": 0.3},
    ]
    item = _make_item(alternatives=alternatives)
    decision = ReviewDecision(
        review_id=item.review_id,
        action="select_alternative",
        selected_alternative_id="alt-2",
    )
    result = hr_module.apply_human_decision(decision, item)
    assert result["action"] == "select_alternative"
    assert result["value"]["id"] == "alt-2"
    assert result["value"]["summary"] == "Interpretation B"
    assert result["discarded"] is False


# ---------------------------------------------------------------------------
# Test 12: apply_human_decision reject returns discarded=True
# ---------------------------------------------------------------------------

def test_apply_reject_returns_discarded():
    item = _make_item()
    decision = ReviewDecision(review_id=item.review_id, action="reject")
    result = hr_module.apply_human_decision(decision, item)
    assert result["action"] == "reject"
    assert result["discarded"] is True
    assert result["value"] is None


# ---------------------------------------------------------------------------
# Test 13: select_alternative with unknown ID raises ValueError in apply
# ---------------------------------------------------------------------------

def test_apply_select_alternative_bad_id_raises():
    alternatives = [{"id": "alt-1", "summary": "Only option"}]
    item = _make_item(alternatives=alternatives)
    decision = ReviewDecision(
        review_id=item.review_id,
        action="select_alternative",
        selected_alternative_id="alt-GHOST",
    )
    with pytest.raises(ValueError, match="not found in original_item.alternatives"):
        hr_module.apply_human_decision(decision, item)


# ---------------------------------------------------------------------------
# Test 14: add_to_review_queue is idempotent (re-add replaces, not duplicates)
# ---------------------------------------------------------------------------

def test_add_is_idempotent():
    item = _make_item(review_id="rev-unique")
    hr_module.add_to_review_queue(item)
    hr_module.add_to_review_queue(item)  # same ID, second add
    pending = hr_module.get_pending_reviews()
    assert len(pending) == 1, "Duplicate add should replace, not create two entries"
