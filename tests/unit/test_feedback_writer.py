"""
tests/unit/test_feedback_writer.py
-----------------------------------
Unit tests for agent/feedback_writer.save_feedback().

Coverage (8 required tests per Step 12 Stage 3 spec):
    a. Every HITL action (approve/correct/reject/select_alternative) can be saved
    b. New records always default to approved_for_precedent=False
    c. Bakhu case: approve + HIGH risk -> feedback_status="not_eligible"
    d. correct + select_alternative preserve both original and final values
    e. Idempotency: second call with same review_id writes exactly one line
    f. Full contract text never written (byte-length check)
    g. Override attempt: mark_candidate=True + HIGH -> not_eligible; field stored faithfully
    h. Positive path: mark_candidate=True + MEDIUM + present + verified -> pending_precedent_review
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import agent.feedback_writer as fw
from agent.feedback_writer import save_feedback
from schemas.feedback import FeedbackRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_feedback_log(tmp_path, monkeypatch):
    """
    Redirect _FEEDBACK_DIR and _FEEDBACK_LOG to a tmp_path so tests never
    touch the real data/feedback/ directory and never share state with each other.
    """
    feedback_dir = tmp_path / "feedback"
    feedback_log = feedback_dir / "feedback_log.jsonl"
    monkeypatch.setattr(fw, "_FEEDBACK_DIR", feedback_dir)
    monkeypatch.setattr(fw, "_FEEDBACK_LOG", feedback_log)
    return feedback_log


def _make_item(
    review_id: str = "rev-001",
    risk_level: str = "MEDIUM",
    trigger_reason: str = "high_risk_finding",
    source_text: str = "The parties agree to maintain confidentiality.",
    evidence_match_type: str = "exact",
    page_reference_valid: bool = True,
    clause_category: str = "Confidentiality / Non-Disclosure",
    document_type_context: str = "NDA",
    fact_found: str = "A Confidentiality clause is present.",
    risk_rationale: str = "Minor deviation from standard template.",
    extraction_confidence: float = 0.88,
    document_hash: str = "b8fc19abcd1234",
    source_page: int = 1,
) -> "ReviewItem":
    from schemas.review import ReviewItem

    return ReviewItem(
        review_id=review_id,
        document_hash=document_hash,
        source_text=source_text,
        trigger_reason=trigger_reason,
        ai_finding_summary=f"PRESENT | {clause_category} | {risk_level} risk",
        thread_id="thread-test-001",
        clause_category=clause_category,
        risk_level=risk_level,
        risk_rationale=risk_rationale,
        fact_found=fact_found,
        extraction_confidence=extraction_confidence,
        evidence_match_type=evidence_match_type,
        page_reference_valid=page_reference_valid,
        source_page=source_page,
        document_type_context=document_type_context,
    )


def _make_decision(
    review_id: str = "rev-001",
    action: str = "approve",
    corrected_value=None,
    corrected_risk_level=None,
    selected_alternative_id: str = None,
    reviewer_note: str = None,
    reject_category: str = None,
    mark_clause_language_as_precedent_candidate: bool = False,
    risk_level_on_item: str = None,
    reason: str = "",
) -> "ReviewDecision":
    from schemas.review import ReviewDecision

    kwargs = dict(
        review_id=review_id,
        action=action,
        reviewer_note=reviewer_note,
        mark_clause_language_as_precedent_candidate=mark_clause_language_as_precedent_candidate,
        reason=reason,
    )
    if corrected_value is not None:
        kwargs["corrected_value"] = corrected_value
    if corrected_risk_level is not None:
        kwargs["corrected_risk_level"] = corrected_risk_level
    if selected_alternative_id is not None:
        kwargs["selected_alternative_id"] = selected_alternative_id
    if reject_category is not None:
        kwargs["reject_category"] = reject_category
    if risk_level_on_item is not None:
        kwargs["risk_level_on_item"] = risk_level_on_item
    return ReviewDecision(**kwargs)


def _read_all_records(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    records = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# a. All four HITL actions can be saved
# ---------------------------------------------------------------------------


class TestAllActionsCanBeSaved:
    def test_approve_action(self, isolated_feedback_log):
        item = _make_item(review_id="rev-approve-001")
        decision = _make_decision(review_id="rev-approve-001", action="approve")
        record = save_feedback(item, decision)
        assert record.review_action == "approve"
        assert isolated_feedback_log.exists()
        lines = _read_all_records(isolated_feedback_log)
        assert len(lines) == 1
        assert lines[0]["review_action"] == "approve"

    def test_correct_action(self, isolated_feedback_log):
        item = _make_item(review_id="rev-correct-001", risk_level="HIGH")
        decision = _make_decision(
            review_id="rev-correct-001",
            action="correct",
            corrected_value={"summary": "Clause present with minor deviation"},
            corrected_risk_level="MEDIUM",
            risk_level_on_item="HIGH",
            reason="AI over-assessed risk; clause is standard.",
        )
        record = save_feedback(item, decision)
        assert record.review_action == "correct"
        lines = _read_all_records(isolated_feedback_log)
        assert len(lines) == 1

    def test_reject_action(self, isolated_feedback_log):
        item = _make_item(review_id="rev-reject-001")
        decision = _make_decision(
            review_id="rev-reject-001",
            action="reject",
            reject_category="hallucinated_deviation",
        )
        record = save_feedback(item, decision)
        assert record.review_action == "reject"
        lines = _read_all_records(isolated_feedback_log)
        assert len(lines) == 1

    def test_select_alternative_action(self, isolated_feedback_log):
        from schemas.review import ReviewItem

        item = ReviewItem(
            review_id="rev-alt-001",
            document_hash="abc123",
            source_text="The parties agree to keep information confidential.",
            trigger_reason="tot_unresolved_tie",
            ai_finding_summary="PRESENT | Confidentiality | MEDIUM risk",
            thread_id="thread-alt-001",
            clause_category="Confidentiality / Non-Disclosure",
            risk_level="MEDIUM",
            risk_rationale="ToT tie between two interpretations.",
            fact_found="Confidentiality clause found.",
            extraction_confidence=0.75,
            evidence_match_type="fuzzy",
            page_reference_valid=True,
            document_type_context="NDA",
            alternatives=[{"id": "alt-1", "summary": "Standard NDA language"}],
        )
        decision = _make_decision(
            review_id="rev-alt-001",
            action="select_alternative",
            selected_alternative_id="alt-1",
        )
        record = save_feedback(item, decision)
        assert record.review_action == "select_alternative"
        lines = _read_all_records(isolated_feedback_log)
        assert len(lines) == 1


# ---------------------------------------------------------------------------
# b. New records always default to approved_for_precedent=False
# ---------------------------------------------------------------------------


class TestApprovedForPrecedentAlwaysFalse:
    def test_approve_never_auto_promotes(self, isolated_feedback_log):
        item = _make_item(review_id="rev-promo-001", risk_level="MEDIUM")
        decision = _make_decision(review_id="rev-promo-001", action="approve")
        record = save_feedback(item, decision)
        assert record.approved_for_precedent is False

    def test_mark_candidate_true_does_not_set_approved(self, isolated_feedback_log):
        """mark_clause_language_as_precedent_candidate=True does not set approved_for_precedent."""
        item = _make_item(review_id="rev-promo-002", risk_level="MEDIUM")
        decision = _make_decision(
            review_id="rev-promo-002",
            action="approve",
            mark_clause_language_as_precedent_candidate=True,
        )
        record = save_feedback(item, decision)
        assert record.approved_for_precedent is False  # still False — Stage 4 promotes


# ---------------------------------------------------------------------------
# c. Bakhu case: approve + HIGH risk -> not_eligible
# ---------------------------------------------------------------------------


class TestBakhuCase:
    def test_bakhu_high_risk_approve_is_not_eligible(self, isolated_feedback_log):
        """
        Canonical Bakhu Confidentiality case:
        - Reviewer approves the AI's HIGH-risk finding (agreed with AI)
        - model_finding_accepted = True (reviewer agreed)
        - But feedback_status must be "not_eligible" — HIGH risk can never become precedent
        - approved_for_precedent must be False
        """
        item = _make_item(
            review_id="rev-bakhu-001",
            document_hash="b8fc19abcd1234",
            risk_level="HIGH",
            trigger_reason="high_risk_finding",
            source_text=(
                "Director hereto agrees as follows: Protected Information Defined "
                "as any information disclosed by Company."
            ),
            clause_category="Confidentiality / Non-Disclosure",
            document_type_context="NDA",
            fact_found="Confidentiality clause is present with a non-standard definition.",
            risk_rationale=(
                "Confidentiality clause has a major deviation: clause introduces "
                "a specific definition of Protected Information that differs from "
                "standard mutual NDA template."
            ),
            extraction_confidence=0.91,
            evidence_match_type="exact",
            page_reference_valid=True,
            source_page=1,
        )
        decision = _make_decision(
            review_id="rev-bakhu-001",
            action="approve",
            reviewer_note=(
                "Boundary expansion correctly identified the full clause. "
                "HIGH risk assessment is correct — the definition is one-sided."
            ),
            risk_level_on_item="HIGH",
            reason="HIGH risk assessment is correct.",
        )
        record = save_feedback(item, decision)

        assert record.review_action == "approve"
        assert record.model_finding_accepted is True
        assert record.final_risk == "HIGH"
        assert record.feedback_status == "not_eligible"
        assert record.approved_for_precedent is False
        assert record.clause_language_accepted_as_business_precedent is False

        lines = _read_all_records(isolated_feedback_log)
        assert len(lines) == 1
        raw = lines[0]
        assert raw["feedback_status"] == "not_eligible"
        assert raw["model_finding_accepted"] is True
        assert raw["approved_for_precedent"] is False


# ---------------------------------------------------------------------------
# d. correct and select_alternative preserve original and final values
# ---------------------------------------------------------------------------


class TestOriginalAndFinalPreserved:
    def test_correct_preserves_both(self, isolated_feedback_log):
        item = _make_item(
            review_id="rev-correct-pres-001",
            risk_level="HIGH",
            risk_rationale="AI's original rationale for HIGH.",
            extraction_confidence=0.72,
        )
        decision = _make_decision(
            review_id="rev-correct-pres-001",
            action="correct",
            corrected_value={"summary": "Minor deviation only"},
            corrected_risk_level="MEDIUM",
            risk_level_on_item="HIGH",
            reason="Risk was overstated — deviation is minor.",
        )
        record = save_feedback(item, decision)

        # Original values preserved
        assert record.original_model_risk == "HIGH"
        assert record.original_model_reason == "AI's original rationale for HIGH."
        assert record.original_confidence == pytest.approx(0.72)

        # Final values reflect correction
        assert record.final_risk == "MEDIUM"
        assert record.review_action == "correct"

    def test_select_alternative_preserves_original(self, isolated_feedback_log):
        from schemas.review import ReviewItem

        item = ReviewItem(
            review_id="rev-alt-pres-001",
            document_hash="docalt123",
            source_text="The parties shall maintain secrecy of all information.",
            trigger_reason="tot_unresolved_tie",
            ai_finding_summary="PRESENT | Confidentiality | LOW risk",
            thread_id="thread-alt-pres-001",
            clause_category="Confidentiality / Non-Disclosure",
            risk_level="LOW",
            risk_rationale="ToT original rationale LOW.",
            fact_found="Clause found.",
            extraction_confidence=0.65,
            evidence_match_type="fuzzy",
            page_reference_valid=True,
            document_type_context="NDA",
            alternatives=[{"id": "alt-x", "summary": "Weaker confidentiality"}],
        )
        decision = _make_decision(
            review_id="rev-alt-pres-001",
            action="select_alternative",
            selected_alternative_id="alt-x",
        )
        record = save_feedback(item, decision)

        assert record.original_model_risk == "LOW"
        assert record.original_model_reason == "ToT original rationale LOW."
        assert record.review_action == "select_alternative"


# ---------------------------------------------------------------------------
# e. Idempotency: two calls with same review_id -> exactly one line in file
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_duplicate_save_yields_one_line(self, isolated_feedback_log):
        item = _make_item(review_id="rev-idem-001", risk_level="MEDIUM")
        decision = _make_decision(review_id="rev-idem-001", action="approve")

        record1 = save_feedback(item, decision)
        record2 = save_feedback(item, decision)

        # In-Python: same feedback_id
        assert record1.feedback_id == record2.feedback_id

        # File-level: exactly one line exists
        content = isolated_feedback_log.read_text(encoding="utf-8")
        lines = [l for l in content.splitlines() if l.strip()]
        assert len(lines) == 1, (
            f"Expected exactly 1 line in feedback_log.jsonl after two save calls, "
            f"got {len(lines)}"
        )


# ---------------------------------------------------------------------------
# f. Full contract text is never written — byte-length check
# ---------------------------------------------------------------------------


class TestNoFullContractText:
    def test_long_source_text_truncated_in_log(self, isolated_feedback_log):
        """
        Source text of 50,000 chars must be truncated before write.
        Each JSON line must remain well under a sane ceiling.
        """
        long_text = "Legal clause text. " * 3000  # ~57,000 chars
        item = _make_item(
            review_id="rev-trunc-001",
            source_text=long_text,
            risk_level="MEDIUM",
        )
        decision = _make_decision(review_id="rev-trunc-001", action="approve")
        record = save_feedback(item, decision)

        # Schema enforced: evidence_excerpt must be <= 500 chars
        from schemas.feedback import MAX_EVIDENCE_EXCERPT_CHARS
        assert len(record.evidence_excerpt) <= MAX_EVIDENCE_EXCERPT_CHARS

        # File-level: the stored JSON line must not be enormous
        content = isolated_feedback_log.read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            byte_len = len(line.encode("utf-8"))
            assert byte_len < 10_000, (
                f"JSON line is {byte_len} bytes — suspiciously large; "
                f"check that full contract text is not being written. "
                f"MAX_EVIDENCE_EXCERPT_CHARS={MAX_EVIDENCE_EXCERPT_CHARS}"
            )

    def test_evidence_excerpt_ceiling_matches_schema(self, isolated_feedback_log):
        """evidence_excerpt in the stored JSON is exactly the schema ceiling or less."""
        from schemas.feedback import MAX_EVIDENCE_EXCERPT_CHARS

        at_limit = "x" * MAX_EVIDENCE_EXCERPT_CHARS
        item = _make_item(review_id="rev-trunc-002", source_text=at_limit)
        decision = _make_decision(review_id="rev-trunc-002", action="approve")
        record = save_feedback(item, decision)

        assert len(record.evidence_excerpt) == MAX_EVIDENCE_EXCERPT_CHARS

        stored = json.loads(
            isolated_feedback_log.read_text(encoding="utf-8").strip()
        )
        assert len(stored["evidence_excerpt"]) == MAX_EVIDENCE_EXCERPT_CHARS


# ---------------------------------------------------------------------------
# g. Override-attempt: mark_candidate=True + HIGH -> not_eligible; field stored faithfully
# ---------------------------------------------------------------------------


class TestOverrideAttempt:
    def test_mark_candidate_true_with_high_risk_overridden(self, isolated_feedback_log):
        """
        Reviewer sets mark_clause_language_as_precedent_candidate=True but
        final_risk is HIGH.

        Expected outcome:
        - feedback_status = "not_eligible"  (safety rule wins)
        - approved_for_precedent = False
        - clause_language_accepted_as_business_precedent = True
          (stored faithfully — audit trail of reviewer's intent preserved)
        """
        item = _make_item(
            review_id="rev-override-001",
            risk_level="HIGH",
            evidence_match_type="exact",
            page_reference_valid=True,
            trigger_reason="high_risk_finding",
        )
        decision = _make_decision(
            review_id="rev-override-001",
            action="approve",
            mark_clause_language_as_precedent_candidate=True,  # reviewer tries to flag it
            risk_level_on_item="HIGH",
            reason="HIGH risk is correct.",
        )
        record = save_feedback(item, decision)

        # Eligibility rules override the flag
        assert record.feedback_status == "not_eligible"
        assert record.approved_for_precedent is False

        # Reviewer's intent is preserved in the stored field (audit trail)
        assert record.clause_language_accepted_as_business_precedent is True

        # Verify the same in the raw stored JSON
        stored = json.loads(isolated_feedback_log.read_text(encoding="utf-8").strip())
        assert stored["feedback_status"] == "not_eligible"
        assert stored["approved_for_precedent"] is False
        assert stored["clause_language_accepted_as_business_precedent"] is True


# ---------------------------------------------------------------------------
# h. Positive path: mark_candidate=True + MEDIUM + present + verified -> pending_precedent_review
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# i. None clause_category: does not crash, resolves to not_eligible
# ---------------------------------------------------------------------------


class TestNoneClauseCategory:
    def test_none_clause_category_does_not_crash(self, isolated_feedback_log):
        """
        When item.clause_category is None (document-level trigger, no specific
        clause category), save_feedback() must NOT raise. It must write a valid
        record with feedback_status="not_eligible" — a record without a clause
        category cannot responsibly become a precedent.

        This guards against the silent failure mode where "Other" fallback caused
        a Pydantic ValidationError swallowed by apply_human_decision()'s try/except.
        """
        from schemas.review import ReviewItem

        item = ReviewItem(
            review_id="rev-none-cat-001",
            document_hash="docnonecat123",
            source_text="The document does not contain a recognizable clause.",
            trigger_reason="low_confidence_after_retry",
            ai_finding_summary="Document-level flag; no clause category",
            thread_id="thread-none-cat-001",
            clause_category=None,  # explicitly None — document-level trigger
            risk_level="MEDIUM",
            risk_rationale="Confidence too low after retry.",
            fact_found="No specific clause identified.",
            extraction_confidence=0.45,
            evidence_match_type="not_found",
            page_reference_valid=None,
            document_type_context="NDA",
        )
        decision = _make_decision(
            review_id="rev-none-cat-001",
            action="approve",
            mark_clause_language_as_precedent_candidate=True,  # flagged but still not_eligible
        )

        # Must not raise
        record = save_feedback(item, decision)

        # No category → not_eligible regardless of other flags
        assert record.feedback_status == "not_eligible"
        assert record.clause_category is None
        assert record.approved_for_precedent is False

        # File must contain exactly one line
        lines = _read_all_records(isolated_feedback_log)
        assert len(lines) == 1
        assert lines[0]["feedback_status"] == "not_eligible"
        assert lines[0]["clause_category"] is None


class TestPositivePrecedentPath:
    def test_medium_risk_present_verified_pending(self, isolated_feedback_log):
        """
        When all eligibility conditions are met, the record lands at
        feedback_status="pending_precedent_review".

        Conditions:
        - final_risk = "MEDIUM"
        - is_clause_present = True
        - evidence_match_type = "exact" (verified)
        - page_reference_valid = True
        - review_action = "approve" (not "reject")
        - mark_clause_language_as_precedent_candidate = True
        """
        item = _make_item(
            review_id="rev-pending-001",
            risk_level="MEDIUM",
            trigger_reason="high_risk_finding",
            evidence_match_type="exact",
            page_reference_valid=True,
            fact_found="Confidentiality clause is present.",
        )
        decision = _make_decision(
            review_id="rev-pending-001",
            action="approve",
            mark_clause_language_as_precedent_candidate=True,
        )
        record = save_feedback(item, decision)

        assert record.feedback_status == "pending_precedent_review"
        assert record.clause_language_accepted_as_business_precedent is True
        assert record.approved_for_precedent is False
        assert record.final_risk == "MEDIUM"
        assert record.is_clause_present is True

        stored = json.loads(isolated_feedback_log.read_text(encoding="utf-8").strip())
        assert stored["feedback_status"] == "pending_precedent_review"


# ---------------------------------------------------------------------------
# i. possible_clause_under_different_heading trigger handling
# ---------------------------------------------------------------------------


class TestPossibleClauseUnderDifferentHeading:
    """
    Verify that feedback decisions on 'possible_clause_under_different_heading'
    items are captured correctly and never silently coerced into is_clause_present=True.

    This trigger is set by node_verify_missing_clauses (Step 15) when a clause
    was not extracted but keyword evidence was found. The human must confirm whether
    the clause is genuinely absent or present under a non-standard heading.

    Until the human confirms presence, the system treats the clause as NOT present
    for eligibility purposes — preventing promotion to a language precedent on the
    basis of unverified keyword hits alone.
    """

    def test_approve_on_keyword_escalation_is_not_eligible(self, isolated_feedback_log):
        """
        A human approving a 'possible_clause_under_different_heading' item should NOT
        be treated as confirming clause presence for precedent purposes.
        is_clause_present must be False, feedback_status must be not_eligible.
        """
        item = _make_item(
            review_id="rev-keyword-001",
            trigger_reason="possible_clause_under_different_heading",
            risk_level="HIGH",
            fact_found="Governing Law clause was NOT extracted by the extractor.",
            clause_category="Governing Law / Jurisdiction",
            evidence_match_type=None,
            page_reference_valid=None,
        )
        decision = _make_decision(
            review_id="rev-keyword-001",
            action="approve",
            mark_clause_language_as_precedent_candidate=True,
        )
        record = save_feedback(item, decision)

        # Human said "approve" but is_clause_present must still be False
        # because the trigger is possible_clause_under_different_heading —
        # the clause was never actually extracted and verified.
        assert record.is_clause_present is False
        # HIGH risk with is_clause_present=False → always not_eligible
        assert record.feedback_status == "not_eligible"
        # The reviewer's intent IS preserved (audit trail)
        assert record.clause_language_accepted_as_business_precedent is True
        # But approved_for_precedent must remain False (Stage 4 controls this)
        assert record.approved_for_precedent is False

    def test_correct_on_keyword_escalation_captures_decision(self, isolated_feedback_log):
        """
        A human correcting a 'possible_clause_under_different_heading' item
        (e.g. adjusting risk level after confirming absence) is persisted faithfully.
        """
        item = _make_item(
            review_id="rev-keyword-002",
            trigger_reason="possible_clause_under_different_heading",
            risk_level="HIGH",
            fact_found="Assignment clause was NOT extracted by the extractor.",
            clause_category="Assignment",
            evidence_match_type=None,
            page_reference_valid=None,
        )
        decision = _make_decision(
            review_id="rev-keyword-002",
            action="correct",
            corrected_value={"clause_type": "Assignment", "is_present": True},
            corrected_risk_level="MEDIUM",
            reviewer_note="Found under 'Transfer of Rights' heading — confirmed present.",
        )
        record = save_feedback(item, decision)

        assert record.review_action == "correct"
        assert record.final_risk == "MEDIUM"
        assert record.reviewer_comment == "Found under 'Transfer of Rights' heading — confirmed present."
        # Still not_eligible: even with corrected risk, is_clause_present=False
        # because _derive_is_clause_present uses trigger_reason, not corrected_value
        assert record.is_clause_present is False
        assert record.feedback_status == "not_eligible"
