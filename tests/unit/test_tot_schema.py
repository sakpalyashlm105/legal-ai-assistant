"""
tests/unit/test_tot_schema.py
------------------------------
Unit tests for schemas/tot.py (ToTCandidate and ToTResult).

Tests cover:
  - Valid construction of both models
  - composite_score is computed from sub-scores when omitted
  - composite_score is bounded 0-1
  - pruned=True requires pruned_at_depth to be set (Pydantic validator)
  - pruned=False must not have pruned_at_depth set
  - winning_candidates length must be 1 or 2 (Pydantic validator)
"""

import pytest
from pydantic import ValidationError

from schemas.tot import ToTCandidate, ToTResult


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_candidate(
    candidate_id: str = "cand_1",
    clause_category: str = "Indemnification",
    textual_evidence_score: float = 0.8,
    category_fit_score: float = 0.7,
    template_alignment_score: float = 0.75,
    exclusivity_score: float = 0.6,
    composite_score: float | None = None,
    depth_reached: int = 1,
    pruned: bool = False,
    pruned_at_depth: int | None = None,
    rationale: str = "Strong indemnification language present.",
) -> dict:
    d = dict(
        candidate_id=candidate_id,
        clause_category=clause_category,
        textual_evidence_score=textual_evidence_score,
        category_fit_score=category_fit_score,
        template_alignment_score=template_alignment_score,
        exclusivity_score=exclusivity_score,
        depth_reached=depth_reached,
        pruned=pruned,
        pruned_at_depth=pruned_at_depth,
        rationale=rationale,
    )
    if composite_score is not None:
        d["composite_score"] = composite_score
    return d


def _make_tot_candidate(**kwargs) -> ToTCandidate:
    return ToTCandidate(**_make_candidate(**kwargs))


# ---------------------------------------------------------------------------
# ToTCandidate tests
# ---------------------------------------------------------------------------

class TestToTCandidateValidConstruction:
    def test_basic_valid_candidate(self):
        c = _make_tot_candidate()
        assert c.candidate_id == "cand_1"
        assert c.clause_category == "Indemnification"
        assert c.pruned is False
        assert c.pruned_at_depth is None

    def test_composite_score_computed_from_sub_scores_when_omitted(self):
        # 0.8 + 0.7 + 0.75 + 0.6 = 2.85 / 4 = 0.7125
        c = _make_tot_candidate(
            textual_evidence_score=0.8,
            category_fit_score=0.7,
            template_alignment_score=0.75,
            exclusivity_score=0.6,
        )
        assert abs(c.composite_score - 0.7125) < 0.001

    def test_composite_score_explicitly_provided_is_accepted(self):
        c = _make_tot_candidate(
            textual_evidence_score=0.5,
            category_fit_score=0.5,
            template_alignment_score=0.5,
            exclusivity_score=0.5,
            composite_score=0.5,
        )
        assert c.composite_score == 0.5

    def test_composite_score_bounded_to_one(self):
        c = _make_tot_candidate(
            textual_evidence_score=1.0,
            category_fit_score=1.0,
            template_alignment_score=1.0,
            exclusivity_score=1.0,
        )
        assert c.composite_score <= 1.0

    def test_composite_score_bounded_above_zero(self):
        c = _make_tot_candidate(
            textual_evidence_score=0.0,
            category_fit_score=0.0,
            template_alignment_score=0.0,
            exclusivity_score=0.0,
        )
        assert c.composite_score >= 0.0

    def test_pruned_candidate_with_depth(self):
        c = _make_tot_candidate(pruned=True, pruned_at_depth=1)
        assert c.pruned is True
        assert c.pruned_at_depth == 1

    def test_all_valid_clause_categories_accepted(self):
        from schemas.clause import ClauseType
        import typing
        categories = typing.get_args(ClauseType)
        for cat in categories:
            c = _make_tot_candidate(clause_category=cat)
            assert c.clause_category == cat

    def test_depth_reached_valid_range(self):
        for d in (1, 2, 3):
            c = _make_tot_candidate(depth_reached=d)
            assert c.depth_reached == d


class TestToTCandidateValidators:
    def test_pruned_true_without_pruned_at_depth_raises(self):
        with pytest.raises(ValidationError, match="pruned_at_depth is None"):
            _make_tot_candidate(pruned=True, pruned_at_depth=None)

    def test_pruned_false_with_pruned_at_depth_raises(self):
        with pytest.raises(ValidationError, match="pruned_at_depth=1"):
            _make_tot_candidate(pruned=False, pruned_at_depth=1)

    def test_invalid_clause_category_raises(self):
        with pytest.raises(ValidationError):
            _make_tot_candidate(clause_category="Made Up Category")

    def test_confidence_above_1_raises(self):
        with pytest.raises(ValidationError):
            _make_tot_candidate(textual_evidence_score=1.5)

    def test_confidence_below_0_raises(self):
        with pytest.raises(ValidationError):
            _make_tot_candidate(category_fit_score=-0.1)

    def test_depth_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            _make_tot_candidate(depth_reached=0)
        with pytest.raises(ValidationError):
            _make_tot_candidate(depth_reached=4)


# ---------------------------------------------------------------------------
# ToTResult tests
# ---------------------------------------------------------------------------

def _make_winner(candidate_id: str = "cand_1") -> ToTCandidate:
    return _make_tot_candidate(candidate_id=candidate_id, depth_reached=3)


def _make_pruned(candidate_id: str = "cand_2") -> ToTCandidate:
    return _make_tot_candidate(
        candidate_id=candidate_id,
        clause_category="Limitation of Liability",
        pruned=True,
        pruned_at_depth=1,
        depth_reached=1,
    )


class TestToTResultValidConstruction:
    def test_single_winner_valid(self):
        r = ToTResult(
            source_chunk_id="abc_chunk_0001",
            all_candidates=[_make_winner(), _make_pruned()],
            winning_candidates=[_make_winner()],
        )
        assert len(r.winning_candidates) == 1
        assert r.requires_human_review is False
        assert r.critic_step_triggered is False

    def test_dual_label_two_winners_valid(self):
        w1 = _make_winner("cand_1")
        w2 = _make_tot_candidate(
            candidate_id="cand_2",
            clause_category="Limitation of Liability",
            depth_reached=3,
        )
        r = ToTResult(
            source_chunk_id="abc_chunk_0001",
            all_candidates=[w1, w2],
            winning_candidates=[w1, w2],
            critic_step_triggered=True,
            critic_rationale="Both clause types genuinely present in this paragraph.",
        )
        assert len(r.winning_candidates) == 2
        assert r.critic_step_triggered is True

    def test_human_review_result_valid(self):
        r = ToTResult(
            source_chunk_id="abc_chunk_0001",
            all_candidates=[_make_winner()],
            winning_candidates=[_make_winner()],
            requires_human_review=True,
            human_review_reason="Critic step could not resolve tie.",
        )
        assert r.requires_human_review is True
        assert r.human_review_reason is not None

    def test_clause_category_input_optional(self):
        r = ToTResult(
            clause_category_input=None,
            source_chunk_id="abc_chunk_0001",
            all_candidates=[_make_winner()],
            winning_candidates=[_make_winner()],
        )
        assert r.clause_category_input is None

    def test_clause_category_input_set(self):
        r = ToTResult(
            clause_category_input="Indemnification",
            source_chunk_id="abc_chunk_0001",
            all_candidates=[_make_winner()],
            winning_candidates=[_make_winner()],
        )
        assert r.clause_category_input == "Indemnification"


class TestToTResultValidators:
    def test_zero_winners_raises(self):
        with pytest.raises(ValidationError, match="must not be empty"):
            ToTResult(
                source_chunk_id="abc_chunk_0001",
                all_candidates=[_make_winner()],
                winning_candidates=[],
            )

    def test_three_winners_raises(self):
        w = _make_winner()
        with pytest.raises(ValidationError, match="maximum is 2"):
            ToTResult(
                source_chunk_id="abc_chunk_0001",
                all_candidates=[w],
                winning_candidates=[w, w, w],
            )
