"""
tests/unit/test_feedback_schema.py
------------------------------------
Unit tests for schemas/feedback.py.

Coverage:
    - Valid PrecedentScope and FeedbackRecord construction
    - evidence_excerpt validators (empty, over ceiling, placeholder)
    - Safety invariant 1: approved_for_precedent=True + HIGH risk -> rejected
    - Safety invariant 1: approved_for_precedent=True + missing clause -> rejected
    - Safety invariant 2: approved_for_precedent=True but feedback_status not "approved_precedent" is fine
    - Safety invariant 2: feedback_status="approved_precedent" requires MEDIUM + approved_for_precedent=True
    - Default values
    - mark_clause_language_as_precedent_candidate on ReviewDecision (confirm additive, default False)
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from schemas.feedback import FeedbackRecord, PrecedentScope, MAX_EVIDENCE_EXCERPT_CHARS
from schemas.review import ReviewDecision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_record(**overrides) -> dict:
    """Return a minimal valid FeedbackRecord payload, with optional overrides."""
    base = {
        "feedback_id": "fb_test-uuid-001",
        "document_id": "b8fc19abcd1234",
        "document_name": "NDA_test.pdf",
        "document_type": "NDA",
        "clause_category": "Confidentiality / Non-Disclosure",
        "source_page": 2,
        "source_chunk_ids": ["chunk_001"],
        "evidence_excerpt": "The parties agree to maintain in strict confidence all proprietary information.",
        "original_model_category": "Confidentiality / Non-Disclosure",
        "original_model_risk": "MEDIUM",
        "original_model_reason": "Minor deviation: missing survival period.",
        "original_confidence": 0.88,
        "review_action": "approve",
        "final_category": "Confidentiality / Non-Disclosure",
        "final_risk": "MEDIUM",
        "reviewer_comment": "Clause is acceptable for NDA context.",
        "model_finding_accepted": True,
        "clause_language_accepted_as_business_precedent": True,
        "is_clause_present": True,
        "feedback_status": "pending_precedent_review",
        "approved_for_precedent": False,
        "model_name": "gpt-4o-mini",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# PrecedentScope tests
# ---------------------------------------------------------------------------

class TestPrecedentScope:
    def test_valid_full_scope(self):
        scope = PrecedentScope(
            document_type="NDA",
            clause_category="Confidentiality / Non-Disclosure",
            jurisdiction="Delaware",
            template_version="v2.1",
        )
        assert scope.clause_category == "Confidentiality / Non-Disclosure"
        assert scope.jurisdiction == "Delaware"

    def test_minimal_scope_only_required(self):
        scope = PrecedentScope(clause_category="Indemnification")
        assert scope.document_type is None
        assert scope.jurisdiction is None
        assert scope.template_version is None

    def test_invalid_clause_category_rejected(self):
        with pytest.raises(Exception):
            PrecedentScope(clause_category="Fake Clause Type")

    def test_invalid_document_type_rejected(self):
        with pytest.raises(Exception):
            PrecedentScope(clause_category="Assignment", document_type="Lease")


# ---------------------------------------------------------------------------
# FeedbackRecord — valid construction
# ---------------------------------------------------------------------------

class TestFeedbackRecordValid:
    def test_minimal_valid_record(self):
        record = FeedbackRecord(**_base_record())
        assert record.feedback_id == "fb_test-uuid-001"
        assert record.clause_category == "Confidentiality / Non-Disclosure"
        assert record.feedback_status == "pending_precedent_review"
        assert record.approved_for_precedent is False

    def test_default_values(self):
        record = FeedbackRecord(**_base_record())
        assert record.model_name == "gpt-4o-mini"
        assert record.approved_for_precedent is False
        assert record.precedent_scope is None
        assert record.precedent_approved_by is None
        assert record.precedent_approved_at is None
        assert record.prompt_version is None
        assert record.template_version is None
        assert record.source_chunk_ids == ["chunk_001"]

    def test_not_eligible_record_valid(self):
        record = FeedbackRecord(**_base_record(
            final_risk="HIGH",
            feedback_status="not_eligible",
            model_finding_accepted=True,
            clause_language_accepted_as_business_precedent=False,
            approved_for_precedent=False,
        ))
        assert record.feedback_status == "not_eligible"
        assert record.approved_for_precedent is False

    def test_approved_precedent_record_valid(self):
        """A fully promoted record with all required fields passes."""
        from datetime import datetime, timezone
        scope = PrecedentScope(
            document_type="NDA",
            clause_category="Governing Law / Jurisdiction",
        )
        record = FeedbackRecord(**_base_record(
            clause_category="Governing Law / Jurisdiction",
            original_model_risk="MEDIUM",
            final_risk="MEDIUM",
            feedback_status="approved_precedent",
            approved_for_precedent=True,
            precedent_scope=scope,
            precedent_approved_by="legal_reviewer_1",
            precedent_approved_at=datetime.now(timezone.utc),
        ))
        assert record.feedback_status == "approved_precedent"
        assert record.approved_for_precedent is True
        assert record.precedent_scope.clause_category == "Governing Law / Jurisdiction"

    def test_reject_action_saves_correctly(self):
        record = FeedbackRecord(**_base_record(
            review_action="reject",
            final_risk="MEDIUM",
            model_finding_accepted=False,
            clause_language_accepted_as_business_precedent=False,
            feedback_status="not_eligible",
        ))
        assert record.review_action == "reject"
        assert record.model_finding_accepted is False

    def test_correct_action_saves_both_original_and_final(self):
        record = FeedbackRecord(**_base_record(
            review_action="correct",
            original_model_risk="HIGH",
            final_risk="MEDIUM",
            model_finding_accepted=False,
            clause_language_accepted_as_business_precedent=True,
            feedback_status="pending_precedent_review",
        ))
        assert record.original_model_risk == "HIGH"
        assert record.final_risk == "MEDIUM"
        assert record.review_action == "correct"


# ---------------------------------------------------------------------------
# evidence_excerpt validator
# ---------------------------------------------------------------------------

class TestEvidenceExcerpt:
    def test_empty_excerpt_rejected(self):
        with pytest.raises(Exception, match="non-empty"):
            FeedbackRecord(**_base_record(evidence_excerpt=""))

    def test_whitespace_only_excerpt_rejected(self):
        with pytest.raises(Exception, match="non-empty"):
            FeedbackRecord(**_base_record(evidence_excerpt="   "))

    def test_exactly_at_ceiling_accepted(self):
        at_ceiling = "x" * MAX_EVIDENCE_EXCERPT_CHARS
        record = FeedbackRecord(**_base_record(evidence_excerpt=at_ceiling))
        assert len(record.evidence_excerpt) == MAX_EVIDENCE_EXCERPT_CHARS

    def test_one_over_ceiling_rejected(self):
        over_ceiling = "x" * (MAX_EVIDENCE_EXCERPT_CHARS + 1)
        with pytest.raises(Exception, match=str(MAX_EVIDENCE_EXCERPT_CHARS)):
            FeedbackRecord(**_base_record(evidence_excerpt=over_ceiling))

    def test_full_document_length_rejected(self):
        """A full contract (e.g. 50,000 chars) must be rejected."""
        fake_contract = "Legal text " * 5000  # ~55,000 chars
        with pytest.raises(Exception):
            FeedbackRecord(**_base_record(evidence_excerpt=fake_contract))


# ---------------------------------------------------------------------------
# Safety invariant 1: approved_for_precedent=True + HIGH/missing rejected
# ---------------------------------------------------------------------------

class TestSafetyInvariantHighRiskNeverApproved:
    def test_high_risk_approved_for_precedent_rejected(self):
        """
        A HIGH-risk finding must NEVER have approved_for_precedent=True.
        This is the Bakhu Confidentiality case: reviewer approved the AI's
        HIGH-risk finding (agreed with the finding) but that does NOT make
        the clause language acceptable as a future benchmark.
        """
        with pytest.raises(Exception, match="approved_for_precedent cannot be True when final_risk='HIGH'"):
            FeedbackRecord(**_base_record(
                final_risk="HIGH",
                feedback_status="not_eligible",
                approved_for_precedent=True,  # <<< must be rejected
                clause_language_accepted_as_business_precedent=False,
            ))

    def test_missing_clause_approved_for_precedent_rejected(self):
        """
        A missing-clause finding must NEVER have approved_for_precedent=True.
        Precedent matching requires actual clause language — a missing clause
        has no language to match against.

        Use final_risk="MEDIUM" so the HIGH-risk check does NOT fire first,
        isolating the is_clause_present=False check specifically.
        (A MEDIUM-risk missing clause is unusual but possible if the model
        incorrectly scored it — the invariant must hold regardless of risk level.)
        """
        with pytest.raises(Exception, match="is_clause_present=False"):
            FeedbackRecord(**_base_record(
                is_clause_present=False,
                final_risk="MEDIUM",   # not HIGH, so HIGH-check doesn't shadow this
                feedback_status="not_eligible",
                approved_for_precedent=True,  # <<< must be rejected
                clause_language_accepted_as_business_precedent=False,
            ))

    def test_medium_risk_present_clause_can_be_approved(self):
        """The invariant does NOT block MEDIUM-risk present-clause approval."""
        record = FeedbackRecord(**_base_record(
            final_risk="MEDIUM",
            is_clause_present=True,
            feedback_status="approved_precedent",
            approved_for_precedent=True,
        ))
        assert record.approved_for_precedent is True

    def test_bakhu_case_not_eligible(self):
        """
        Canonical Bakhu example:
            review_action="approve", original/final risk=HIGH.
            model_finding_accepted=True (reviewer agreed with the AI).
            clause_language_accepted_as_business_precedent=False (HIGH finding
            means the clause language is problematic — not reusable).
            feedback_status="not_eligible", approved_for_precedent=False.
        This must construct without error.
        """
        record = FeedbackRecord(
            feedback_id="fb_bakhu-conf-001",
            document_id="b8fc19abcd1234",
            document_name="NDA_2_Bakhu_Holdings,_Corp._(BKUH).pdf",
            document_type="NDA",
            clause_category="Confidentiality / Non-Disclosure",
            source_page=1,
            source_chunk_ids=["b8fc19_chunk_0001", "b8fc19_chunk_0002"],
            evidence_excerpt=(
                "Director hereto agrees as follows: Protected Information Defined "
                "as any information disclosed by Company."
            ),
            original_model_category="Confidentiality / Non-Disclosure",
            original_model_risk="HIGH",
            original_model_reason=(
                "Confidentiality clause has a major deviation: clause introduces "
                "a specific definition of Protected Information that differs from "
                "standard mutual NDA template."
            ),
            original_confidence=0.91,
            review_action="approve",
            final_category="Confidentiality / Non-Disclosure",
            final_risk="HIGH",
            reviewer_comment=(
                "Boundary expansion correctly identified the full clause. "
                "HIGH risk assessment is correct — the definition is one-sided."
            ),
            model_finding_accepted=True,
            clause_language_accepted_as_business_precedent=False,
            is_clause_present=True,
            feedback_status="not_eligible",
            approved_for_precedent=False,
            model_name="gpt-4o-mini",
        )
        assert record.feedback_status == "not_eligible"
        assert record.model_finding_accepted is True
        assert record.clause_language_accepted_as_business_precedent is False
        assert record.approved_for_precedent is False


# ---------------------------------------------------------------------------
# Safety invariant 2: approved_precedent status requirements
# ---------------------------------------------------------------------------

class TestSafetyInvariantApprovedPrecedentStatus:
    def test_approved_precedent_without_medium_risk_rejected(self):
        """feedback_status='approved_precedent' requires final_risk='MEDIUM'."""
        with pytest.raises(Exception, match="final_risk='MEDIUM'"):
            FeedbackRecord(**_base_record(
                final_risk="LOW",
                feedback_status="approved_precedent",
                approved_for_precedent=True,
            ))

    def test_approved_precedent_without_approved_flag_rejected(self):
        """feedback_status='approved_precedent' requires approved_for_precedent=True."""
        with pytest.raises(Exception, match="approved_for_precedent=True"):
            FeedbackRecord(**_base_record(
                final_risk="MEDIUM",
                feedback_status="approved_precedent",
                approved_for_precedent=False,  # <<< must be rejected
            ))

    def test_high_risk_approved_precedent_rejected_by_both_validators(self):
        """HIGH + approved_precedent + approved_for_precedent=True violates both invariants."""
        with pytest.raises(Exception):
            FeedbackRecord(**_base_record(
                final_risk="HIGH",
                feedback_status="approved_precedent",
                approved_for_precedent=True,
            ))


# ---------------------------------------------------------------------------
# ReviewDecision new field: mark_clause_language_as_precedent_candidate
# ---------------------------------------------------------------------------

class TestReviewDecisionNewField:
    def test_new_field_defaults_false(self):
        """mark_clause_language_as_precedent_candidate defaults to False."""
        decision = ReviewDecision(
            review_id="rev-001",
            action="approve",
        )
        assert decision.mark_clause_language_as_precedent_candidate is False

    def test_new_field_can_be_set_true(self):
        decision = ReviewDecision(
            review_id="rev-001",
            action="approve",
            mark_clause_language_as_precedent_candidate=True,
        )
        assert decision.mark_clause_language_as_precedent_candidate is True

    def test_new_field_does_not_break_existing_validators(self):
        """Existing ReviewDecision validators still fire correctly."""
        # approve + no corrected_value -> valid
        d = ReviewDecision(
            review_id="rev-001",
            action="approve",
            mark_clause_language_as_precedent_candidate=True,
        )
        assert d.action == "approve"

    def test_existing_reject_validator_still_requires_category(self):
        """Confirm the new field doesn't suppress the existing reject_category check."""
        with pytest.raises(Exception, match="reject_category"):
            ReviewDecision(
                review_id="rev-002",
                action="reject",
                mark_clause_language_as_precedent_candidate=True,
                # missing reject_category — must still raise
            )

    def test_high_risk_reason_still_required(self):
        """HIGH-risk reason requirement survives the new field addition."""
        with pytest.raises(Exception, match="reason is required"):
            ReviewDecision(
                review_id="rev-003",
                action="approve",
                risk_level_on_item="HIGH",
                reason="",  # empty — must still raise
                mark_clause_language_as_precedent_candidate=False,
            )
