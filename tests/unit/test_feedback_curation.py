"""
tests/unit/test_feedback_curation.py
-------------------------------------
Unit tests for agent/feedback_curation.py — approve_feedback_as_precedent().

Test inventory (9 tests):
  1. Bakhu HIGH/not_eligible record cannot be promoted (FeedbackNotEligibleError)
  2. Non-existent feedback_id raises FeedbackRecordNotFoundError
  3. clause_category=None record cannot be promoted (FeedbackPromotionValidationError)
  4. Synthetic MEDIUM-risk happy-path promotion end-to-end
  5. Re-promotion of an already-approved record raises FeedbackAlreadyPromotedError
  6. document_type auto-fills from the record when curator omits it from scope
  7. _clear_feedback_cache() is called after successful promotion
  8. Atomic-write safety: os.replace failure leaves original file untouched
  9. Empty approval_note is rejected (FeedbackPromotionValidationError)
"""

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from agent.feedback_writer import save_feedback
from agent.feedback_curation import (
    approve_feedback_as_precedent,
    FeedbackRecordNotFoundError,
    FeedbackNotEligibleError,
    FeedbackAlreadyPromotedError,
    FeedbackPromotionValidationError,
    _FEEDBACK_LOG,
    _FEEDBACK_LOG_TMP,
)
from schemas.feedback import PrecedentScope, FeedbackRecord
from schemas.review import ReviewItem, ReviewDecision


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated_feedback_log(tmp_path, monkeypatch):
    """
    Redirect both feedback_writer and feedback_curation to a fresh temp log.
    Ensures each test starts with an empty log and never touches the real one.
    """
    log_file = tmp_path / "feedback_log.jsonl"
    tmp_file = tmp_path / "feedback_log.jsonl.tmp"

    monkeypatch.setattr("agent.feedback_writer._FEEDBACK_DIR", tmp_path)
    monkeypatch.setattr("agent.feedback_writer._FEEDBACK_LOG", log_file)
    monkeypatch.setattr("agent.feedback_curation._FEEDBACK_DIR", tmp_path)
    monkeypatch.setattr("agent.feedback_curation._FEEDBACK_LOG", log_file)
    monkeypatch.setattr("agent.feedback_curation._FEEDBACK_LOG_TMP", tmp_file)

    return log_file


def _make_medium_item(
    review_id: str = "rev-medium-001",
    document_hash: str = "docsup4567ab",
    clause_category: str = "Governing Law / Jurisdiction",
) -> ReviewItem:
    """
    Build a realistic MEDIUM-risk ReviewItem: governing-law clause with a
    minor deviation (wrong jurisdiction), evidence found, page reference valid.
    """
    return ReviewItem(
        review_id=review_id,
        document_hash=document_hash,
        document_name="Supplier_Agreement_Acme_Corp_2024.pdf",
        source_text=(
            "This Agreement shall be governed by and construed in accordance with "
            "the laws of the State of Illinois, without regard to its conflict of "
            "law provisions."
        ),
        trigger_reason="deviation_detected",
        ai_finding_summary=(
            "Governing law clause specifies Illinois. Standard template requires "
            "Delaware. Minor jurisdiction deviation — not a protection gap."
        ),
        thread_id="thread-medium-001",
        clause_category=clause_category,
        risk_level="MEDIUM",
        risk_rationale=(
            "Jurisdiction deviation from template (Illinois vs Delaware) is a minor "
            "deviation. Legal counsel has historically accepted Illinois jurisdiction "
            "for Midwest suppliers."
        ),
        fact_found=(
            "Clause is present. Governing law is Illinois. Template specifies Delaware."
        ),
        extraction_confidence=0.88,
        evidence_match_type="exact",
        page_reference_valid=True,
        document_type_context="Contract",
        source_page=4,
        source_chunk_id="docsup4567_chunk_0004",
    )


def _make_medium_decision(
    review_id: str = "rev-medium-001",
    mark_as_precedent: bool = True,
) -> ReviewDecision:
    return ReviewDecision(
        review_id=review_id,
        action="approve",
        reviewer_note=(
            "Illinois governing law is acceptable for Midwest suppliers. "
            "Clause language should be approved as a business precedent."
        ),
        mark_clause_language_as_precedent_candidate=mark_as_precedent,
    )


def _make_scope(
    clause_category: str = "Governing Law / Jurisdiction",
    document_type: str | None = None,
) -> PrecedentScope:
    return PrecedentScope(
        clause_category=clause_category,
        document_type=document_type,
        jurisdiction="Illinois",
    )


def _read_raw_lines(log_path: Path) -> list[dict]:
    lines = []
    with log_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
    return lines


# ---------------------------------------------------------------------------
# Test 1: Bakhu HIGH / not_eligible cannot be promoted
# ---------------------------------------------------------------------------

class TestBakhuNotEligibleRejected:
    """
    The Bakhu Confidentiality record is not_eligible (HIGH risk, reviewer did
    not flag clause language). Attempting to promote it must raise
    FeedbackNotEligibleError, not silently pass.
    """

    def test_bakhu_not_eligible_cannot_be_promoted(self, isolated_feedback_log):
        item = ReviewItem(
            review_id="bakhu-live-demo-001",
            document_hash="b8fc19abcd1234ef5678",
            document_name="NDA_2_Bakhu_Holdings,_Corp._(BKUH).pdf",
            source_text=(
                "Director hereto agrees as follows: Protected Information Defined as "
                "any information disclosed by Company."
            ),
            trigger_reason="deviation_detected",
            ai_finding_summary=(
                "Confidentiality clause has a major deviation: clause introduces a "
                "specific definition of Protected Information that differs from standard "
                "mutual NDA template."
            ),
            thread_id="thread-bakhu-001",
            clause_category="Confidentiality / Non-Disclosure",
            risk_level="HIGH",
            risk_rationale=(
                "One-sided definition of Protected Information is a major deviation — "
                "the company loses mutual protection."
            ),
            fact_found="Clause present. Definition is one-sided.",
            extraction_confidence=0.91,
            evidence_match_type="exact",
            page_reference_valid=True,
            document_type_context="NDA",
            source_page=1,
            source_chunk_id="b8fc19_chunk_0001",
        )
        decision = ReviewDecision(
            review_id="bakhu-live-demo-001",
            action="approve",
            reviewer_note="HIGH risk assessment is correct — the definition is one-sided.",
            # Reviewer agreed with the finding but did NOT flag clause language as reusable
            mark_clause_language_as_precedent_candidate=False,
        )

        record = save_feedback(item, decision)
        assert record.feedback_status == "not_eligible"

        scope = _make_scope(clause_category="Confidentiality / Non-Disclosure")
        with pytest.raises(FeedbackNotEligibleError) as exc_info:
            approve_feedback_as_precedent(
                feedback_id="fb_bakhu-live-demo-001",
                approved_by="curator@legalteam.com",
                final_accepted_risk="MEDIUM",
                precedent_scope=scope,
                approval_note="Should fail — this is a HIGH-risk finding.",
            )
        assert "not_eligible" in str(exc_info.value)
        assert "fb_bakhu-live-demo-001" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 2: Non-existent feedback_id raises FeedbackRecordNotFoundError
# ---------------------------------------------------------------------------

class TestRecordNotFound:
    def test_unknown_feedback_id_raises_not_found(self, isolated_feedback_log):
        scope = _make_scope()
        with pytest.raises(FeedbackRecordNotFoundError) as exc_info:
            approve_feedback_as_precedent(
                feedback_id="fb_does-not-exist-999",
                approved_by="curator@legalteam.com",
                final_accepted_risk="MEDIUM",
                precedent_scope=scope,
                approval_note="This record does not exist.",
            )
        assert "fb_does-not-exist-999" in str(exc_info.value)
        assert "not found" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Test 3: clause_category=None cannot be promoted
# ---------------------------------------------------------------------------

class TestNullClauseCategory:
    """
    Defense-in-depth: even if a record somehow reached pending_precedent_review
    with clause_category=None, promotion must be rejected.
    We simulate this by writing a crafted record directly to the JSONL.
    """

    def test_none_clause_category_cannot_be_promoted(self, isolated_feedback_log):
        # Write a synthetic record with clause_category=None but
        # feedback_status=pending_precedent_review directly to the log
        # (bypassing save_feedback) to simulate the defense-in-depth scenario.
        raw_record = {
            "feedback_id": "fb_null-cat-defense-001",
            "document_id": "docnullcat001",
            "document_name": "TestDoc.pdf",
            "document_type": "Contract",
            "clause_category": None,
            "source_page": 2,
            "source_chunk_ids": ["chunk-001"],
            "evidence_excerpt": "This clause governs the payment terms of the agreement.",
            "original_model_category": None,
            "original_model_risk": "MEDIUM",
            "original_model_reason": "Minor deviation detected.",
            "original_confidence": 0.75,
            "review_action": "approve",
            "final_category": None,
            "final_risk": "MEDIUM",
            "reviewer_comment": None,
            "model_finding_accepted": True,
            "clause_language_accepted_as_business_precedent": True,
            "is_clause_present": True,
            "feedback_status": "pending_precedent_review",
            "approved_for_precedent": False,
            "precedent_scope": None,
            "precedent_approved_by": None,
            "precedent_approved_at": None,
            "prompt_version": None,
            "model_name": "gpt-4o-mini",
            "template_version": None,
            "created_at": "2026-06-28T00:00:00",
        }
        isolated_feedback_log.write_text(
            json.dumps(raw_record) + "\n", encoding="utf-8"
        )

        scope = PrecedentScope(clause_category="Governing Law / Jurisdiction")
        with pytest.raises(FeedbackPromotionValidationError) as exc_info:
            approve_feedback_as_precedent(
                feedback_id="fb_null-cat-defense-001",
                approved_by="curator@legalteam.com",
                final_accepted_risk="MEDIUM",
                precedent_scope=scope,
                approval_note="Defense-in-depth test.",
            )
        assert "clause_category=None" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 4: Happy-path MEDIUM-risk promotion end-to-end
# ---------------------------------------------------------------------------

class TestHappyPathPromotion:
    def test_medium_risk_promotion_end_to_end(self, isolated_feedback_log):
        item = _make_medium_item()
        decision = _make_medium_decision()

        # Stage 1: save_feedback → pending_precedent_review
        saved = save_feedback(item, decision)
        assert saved.feedback_status == "pending_precedent_review"
        assert saved.approved_for_precedent is False

        # Stage 4: promote
        scope = _make_scope(document_type="Contract")
        promoted = approve_feedback_as_precedent(
            feedback_id="fb_rev-medium-001",
            approved_by="legal.curator@company.com",
            final_accepted_risk="MEDIUM",
            precedent_scope=scope,
            approval_note=(
                "Illinois governing law is an accepted jurisdiction for Midwest "
                "suppliers. This clause language is approved as a business precedent."
            ),
        )

        # Verify returned record
        assert promoted.feedback_status == "approved_precedent"
        assert promoted.approved_for_precedent is True
        assert promoted.precedent_approved_by == "legal.curator@company.com"
        assert promoted.precedent_approved_at is not None
        assert promoted.precedent_scope is not None
        assert promoted.precedent_scope.clause_category == "Governing Law / Jurisdiction"
        assert promoted.precedent_scope.document_type == "Contract"
        assert promoted.precedent_scope.jurisdiction == "Illinois"

        # Verify the log on disk was rewritten correctly
        lines = _read_raw_lines(isolated_feedback_log)
        assert len(lines) == 1  # only one record was ever written
        assert lines[0]["feedback_status"] == "approved_precedent"
        assert lines[0]["approved_for_precedent"] is True
        assert lines[0]["precedent_approved_by"] == "legal.curator@company.com"
        assert lines[0]["precedent_approved_at"] is not None
        # Original fields preserved
        assert lines[0]["final_risk"] == "MEDIUM"
        assert lines[0]["clause_category"] == "Governing Law / Jurisdiction"
        assert lines[0]["document_name"] == "Supplier_Agreement_Acme_Corp_2024.pdf"


# ---------------------------------------------------------------------------
# Test 5: Re-promotion of an already-approved record is rejected
# ---------------------------------------------------------------------------

class TestRePromotionRejected:
    def test_already_approved_cannot_be_re_promoted(self, isolated_feedback_log):
        item = _make_medium_item(review_id="rev-medium-repromote-001")
        decision = _make_medium_decision(review_id="rev-medium-repromote-001")
        save_feedback(item, decision)

        scope = _make_scope(document_type="Contract")
        # First promotion — should succeed
        approve_feedback_as_precedent(
            feedback_id="fb_rev-medium-repromote-001",
            approved_by="curator@company.com",
            final_accepted_risk="MEDIUM",
            precedent_scope=scope,
            approval_note="First promotion — valid.",
        )

        # Second promotion — must be rejected
        with pytest.raises(FeedbackAlreadyPromotedError) as exc_info:
            approve_feedback_as_precedent(
                feedback_id="fb_rev-medium-repromote-001",
                approved_by="curator@company.com",
                final_accepted_risk="MEDIUM",
                precedent_scope=scope,
                approval_note="Second promotion attempt — must fail.",
            )
        assert "already at" in str(exc_info.value).lower() or "already" in str(exc_info.value).lower()
        assert "approved_precedent" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 6: document_type auto-fills from record when curator omits it
# ---------------------------------------------------------------------------

class TestDocumentTypeAutoFill:
    def test_document_type_autofills_from_record(self, isolated_feedback_log):
        item = _make_medium_item(review_id="rev-medium-autofill-001")
        decision = _make_medium_decision(review_id="rev-medium-autofill-001")
        saved = save_feedback(item, decision)
        # The item had document_type_context="Contract" → maps to "Contract"
        assert saved.document_type == "Contract"

        # Curator provides a scope with document_type=None
        scope_no_dtype = PrecedentScope(
            clause_category="Governing Law / Jurisdiction",
            jurisdiction="Illinois",
            document_type=None,
        )
        promoted = approve_feedback_as_precedent(
            feedback_id="fb_rev-medium-autofill-001",
            approved_by="curator@company.com",
            final_accepted_risk="MEDIUM",
            precedent_scope=scope_no_dtype,
            approval_note="Scope has no document_type — should auto-fill from record.",
        )

        # Auto-fill must have picked up "Contract" from the record
        assert promoted.precedent_scope is not None
        assert promoted.precedent_scope.document_type == "Contract"

        # Verify on disk too
        lines = _read_raw_lines(isolated_feedback_log)
        assert lines[0]["precedent_scope"]["document_type"] == "Contract"


# ---------------------------------------------------------------------------
# Test 7: _clear_feedback_cache() is called after successful promotion
# ---------------------------------------------------------------------------

class TestCacheClearCalledOnPromotion:
    def test_clear_cache_called_after_promotion(self, isolated_feedback_log, monkeypatch):
        item = _make_medium_item(review_id="rev-medium-cache-001")
        decision = _make_medium_decision(review_id="rev-medium-cache-001")
        save_feedback(item, decision)

        call_count = {"n": 0}

        def spy_clear():
            call_count["n"] += 1

        monkeypatch.setattr("agent.feedback_curation._clear_feedback_cache", spy_clear)

        scope = _make_scope(document_type="Contract")
        approve_feedback_as_precedent(
            feedback_id="fb_rev-medium-cache-001",
            approved_by="curator@company.com",
            final_accepted_risk="MEDIUM",
            precedent_scope=scope,
            approval_note="Checking that cache is cleared after promotion.",
        )

        assert call_count["n"] == 1, (
            f"Expected _clear_feedback_cache() to be called exactly once after a "
            f"successful promotion; got {call_count['n']} call(s)."
        )


# ---------------------------------------------------------------------------
# Test 8: Atomic-write safety — original file untouched on os.replace failure
# ---------------------------------------------------------------------------

class TestAtomicWriteSafety:
    def test_original_log_untouched_if_replace_fails(self, isolated_feedback_log, monkeypatch):
        item = _make_medium_item(review_id="rev-medium-atomic-001")
        decision = _make_medium_decision(review_id="rev-medium-atomic-001")
        save_feedback(item, decision)

        # Capture the original file contents before the simulated failure
        original_content = isolated_feedback_log.read_bytes()
        original_line_count = len(_read_raw_lines(isolated_feedback_log))
        assert original_line_count == 1

        # Monkeypatch os.replace to raise after the .tmp file is written
        def failing_replace(src, dst):
            raise OSError("Simulated disk-full / rename failure mid-write")

        monkeypatch.setattr("agent.feedback_curation.os.replace", failing_replace)

        scope = _make_scope(document_type="Contract")
        with pytest.raises(OSError, match="Simulated disk-full"):
            approve_feedback_as_precedent(
                feedback_id="fb_rev-medium-atomic-001",
                approved_by="curator@company.com",
                final_accepted_risk="MEDIUM",
                precedent_scope=scope,
                approval_note="Testing atomic-write safety.",
            )

        # Original log must be byte-for-byte identical after the failure
        post_failure_content = isolated_feedback_log.read_bytes()
        assert post_failure_content == original_content, (
            "Original feedback_log.jsonl was modified despite os.replace() failure. "
            "The atomic-write guarantee was violated."
        )

        # Verify the record is still in its pre-promotion state
        lines = _read_raw_lines(isolated_feedback_log)
        assert len(lines) == original_line_count
        assert lines[0]["feedback_status"] == "pending_precedent_review"
        assert lines[0]["approved_for_precedent"] is False


# ---------------------------------------------------------------------------
# Test 9: Empty approval_note is rejected
# ---------------------------------------------------------------------------

class TestEmptyApprovalNoteRejected:
    @pytest.mark.parametrize("note", ["", "   ", "\t\n"])
    def test_empty_approval_note_raises(self, isolated_feedback_log, note):
        item = _make_medium_item(review_id="rev-medium-note-001")
        decision = _make_medium_decision(review_id="rev-medium-note-001")
        save_feedback(item, decision)

        scope = _make_scope(document_type="Contract")
        with pytest.raises(FeedbackPromotionValidationError) as exc_info:
            approve_feedback_as_precedent(
                feedback_id="fb_rev-medium-note-001",
                approved_by="curator@company.com",
                final_accepted_risk="MEDIUM",
                precedent_scope=scope,
                approval_note=note,
            )
        assert "approval_note" in str(exc_info.value)
        assert "non-empty" in str(exc_info.value) or "required" in str(exc_info.value)
