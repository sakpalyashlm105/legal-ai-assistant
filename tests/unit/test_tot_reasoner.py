"""
tests/unit/test_tot_reasoner.py
--------------------------------
Unit tests for agent/tot_reasoner.py.

All OpenAI API calls are mocked -- no real API key or network access needed.
Mocking pattern follows test_classifier.py and test_extractor.py.

Tests cover:
  - get_local_context: boundary cases, same-document-only constraint,
    metadata preservation
  - run_tree_of_thought: clean single-winner run, tie + critic picks one,
    tie + critic declares dual-label, tie + critic declares human review,
    all-candidates-pruned-at-depth-1 edge case, API failure at depth 1,
    pruned candidates retained in all_candidates
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from schemas.chunk import DocumentChunk
from schemas.tot import ToTCandidate, ToTResult
from agent.tot_reasoner import (
    get_local_context,
    run_tree_of_thought,
    _prune,
)


# ---------------------------------------------------------------------------
# Helpers: chunk factories
# ---------------------------------------------------------------------------

DOC_HASH_A = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"  # 64 hex chars
DOC_HASH_B = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


def _make_chunk(
    index: int,
    total: int = 3,
    doc_hash: str = DOC_HASH_A,
    doc_name: str = "test.pdf",
    text: str | None = None,
) -> DocumentChunk:
    if text is None:
        text = f"Chunk text number {index}."
    return DocumentChunk(
        chunk_id=f"{doc_hash[:12]}_chunk_{index:04d}",
        document_hash=doc_hash,
        document_name=doc_name,
        text=text,
        char_count=len(text),
        token_count=len(text) // 4,
        start_page=index + 1,
        end_page=index + 1,
        chunk_index=index,
        total_chunks=total,
        overlap_tokens=0,
    )


def _make_candidate_dict(
    candidate_id: str = "cand_1",
    clause_category: str = "Indemnification",
    textual_evidence_score: float = 0.8,
    category_fit_score: float = 0.8,
    template_alignment_score: float = 0.8,
    exclusivity_score: float = 0.8,
    rationale: str = "Strong match.",
) -> dict:
    composite = (textual_evidence_score + category_fit_score +
                 template_alignment_score + exclusivity_score) / 4
    return {
        "candidate_id": candidate_id,
        "clause_category": clause_category,
        "textual_evidence_score": textual_evidence_score,
        "category_fit_score": category_fit_score,
        "template_alignment_score": template_alignment_score,
        "exclusivity_score": exclusivity_score,
        "composite_score": round(composite, 4),
        "rationale": rationale,
    }


def _mock_llm_response(data: dict) -> MagicMock:
    mock_msg = MagicMock()
    mock_msg.content = json.dumps(data)
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    return mock_resp


# ---------------------------------------------------------------------------
# Tests: get_local_context
# ---------------------------------------------------------------------------

class TestGetLocalContext:
    def test_middle_chunk_returns_prev_target_next(self):
        chunks = [_make_chunk(i, total=5) for i in range(5)]
        result = get_local_context(chunks, target_chunk_index=2)
        assert len(result) == 3
        assert result[0].chunk_index == 1  # previous
        assert result[1].chunk_index == 2  # target
        assert result[2].chunk_index == 3  # next

    def test_first_chunk_returns_target_and_next_only(self):
        chunks = [_make_chunk(i, total=3) for i in range(3)]
        result = get_local_context(chunks, target_chunk_index=0)
        assert len(result) == 2
        assert result[0].chunk_index == 0  # target (no previous)
        assert result[1].chunk_index == 1  # next

    def test_last_chunk_returns_previous_and_target_only(self):
        chunks = [_make_chunk(i, total=3) for i in range(3)]
        result = get_local_context(chunks, target_chunk_index=2)
        assert len(result) == 2
        assert result[0].chunk_index == 1  # previous
        assert result[1].chunk_index == 2  # target (no next)

    def test_single_chunk_document_returns_target_only(self):
        chunks = [_make_chunk(0, total=1)]
        result = get_local_context(chunks, target_chunk_index=0)
        assert len(result) == 1
        assert result[0].chunk_index == 0

    def test_different_document_chunks_excluded(self):
        # Mix chunks from two documents; target is from doc A (index=1).
        chunks_a = [_make_chunk(i, total=3, doc_hash=DOC_HASH_A) for i in range(3)]
        chunks_b = [_make_chunk(i, total=2, doc_hash=DOC_HASH_B) for i in range(2)]
        all_chunks = chunks_a + chunks_b

        result = get_local_context(all_chunks, target_chunk_index=1)
        assert all(c.document_hash == DOC_HASH_A for c in result)
        # Must not include any chunk from doc B.
        assert all(c.document_hash != DOC_HASH_B for c in result)

    def test_metadata_preserved_on_returned_chunks(self):
        chunks = [_make_chunk(i, total=5) for i in range(5)]
        result = get_local_context(chunks, target_chunk_index=2)
        for chunk in result:
            # chunk_index, start_page, end_page must be unchanged.
            original = next(c for c in chunks if c.chunk_index == chunk.chunk_index)
            assert chunk.chunk_index == original.chunk_index
            assert chunk.start_page == original.start_page
            assert chunk.end_page == original.end_page
            assert chunk.document_hash == original.document_hash

    def test_empty_chunks_returns_empty(self):
        result = get_local_context([], target_chunk_index=0)
        assert result == []

    def test_target_not_found_returns_empty(self):
        chunks = [_make_chunk(0, total=1)]
        result = get_local_context(chunks, target_chunk_index=99)
        assert result == []


# ---------------------------------------------------------------------------
# Tests: _prune edge case (all-pruned protection)
# ---------------------------------------------------------------------------

class TestPrune:
    def _make_tot_candidate(self, candidate_id: str, score: float) -> ToTCandidate:
        return ToTCandidate(
            candidate_id=candidate_id,
            clause_category="Indemnification",
            textual_evidence_score=score,
            category_fit_score=score,
            template_alignment_score=score,
            exclusivity_score=score,
            composite_score=score,
            depth_reached=1,
            pruned=False,
            pruned_at_depth=None,
            rationale="test",
        )

    def test_all_pruned_keeps_best(self):
        candidates = [
            self._make_tot_candidate("cand_1", 0.20),
            self._make_tot_candidate("cand_2", 0.25),  # best of the bad ones
        ]
        survivors, pruned = _prune(candidates, threshold=0.30, depth=1)
        assert len(survivors) == 1
        assert survivors[0].candidate_id == "cand_2"  # highest score kept
        assert len(pruned) == 1

    def test_normal_pruning(self):
        candidates = [
            self._make_tot_candidate("cand_1", 0.10),
            self._make_tot_candidate("cand_2", 0.80),
        ]
        survivors, pruned = _prune(candidates, threshold=0.30, depth=1)
        assert len(survivors) == 1
        assert survivors[0].candidate_id == "cand_2"
        assert len(pruned) == 1
        assert pruned[0].pruned is True
        assert pruned[0].pruned_at_depth == 1


# ---------------------------------------------------------------------------
# Helpers: build mock LLM responses for run_tree_of_thought
# ---------------------------------------------------------------------------

def _good_generator_response(
    cand1_category: str = "Indemnification",
    cand1_score: float = 0.85,
    cand2_category: str = "Limitation of Liability",
    cand2_score: float = 0.35,
) -> MagicMock:
    return _mock_llm_response({
        "candidates": [
            _make_candidate_dict("cand_1", cand1_category,
                                 *[cand1_score] * 4),
            _make_candidate_dict("cand_2", cand2_category,
                                 *[cand2_score] * 4),
        ]
    })


def _tied_generator_response() -> MagicMock:
    """Both candidates at same score -- will trigger critic."""
    return _mock_llm_response({
        "candidates": [
            _make_candidate_dict("cand_1", "Indemnification", 0.75, 0.75, 0.75, 0.75),
            _make_candidate_dict("cand_2", "Limitation of Liability", 0.75, 0.75, 0.75, 0.75),
        ]
    })


def _critic_pick_one_response(winner_id: str = "cand_1") -> MagicMock:
    return _mock_llm_response({
        "decision": "pick_one",
        "winner_candidate_id": winner_id,
        "critic_rationale": "Candidate 1 clearly fits indemnification.",
    })


def _critic_dual_label_response() -> MagicMock:
    return _mock_llm_response({
        "decision": "dual_label",
        "winner_candidate_id": None,
        "critic_rationale": "Both clause types genuinely present.",
    })


def _critic_human_review_response() -> MagicMock:
    return _mock_llm_response({
        "decision": "human_review",
        "winner_candidate_id": None,
        "critic_rationale": "Cannot distinguish between the two.",
    })


# ---------------------------------------------------------------------------
# Tests: run_tree_of_thought
# ---------------------------------------------------------------------------

CHUNKS = [_make_chunk(i, total=3) for i in range(3)]
CHUNK_ID = CHUNKS[1].chunk_id  # target = middle chunk


@patch("agent.tot_reasoner._get_client")
class TestRunTreeOfThought:

    def test_clean_single_winner_no_critic(self, mock_get_client):
        """
        When one candidate clearly outscores the other (gap > 0.05),
        no critic step fires and winning_candidates has length 1.
        """
        # Depth 1 and depth 2 both return a clear winner.
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = (
            _good_generator_response(cand1_score=0.85, cand2_score=0.35)
        )

        result = run_tree_of_thought(
            clause_category="Indemnification",
            clause_text="Party A shall indemnify Party B against all losses.",
            source_chunk_id=CHUNK_ID,
            document_chunks=CHUNKS,
        )

        assert isinstance(result, ToTResult)
        assert len(result.winning_candidates) == 1
        assert result.critic_step_triggered is False
        assert result.requires_human_review is False
        assert result.winning_candidates[0].clause_category == "Indemnification"

    def test_tie_critic_picks_one(self, mock_get_client):
        """
        When top-2 candidates are within 0.05, critic fires.
        Critic picks one -> winning_candidates length 1.
        """
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        # First two calls (depth 1 + depth 2) return tied candidates.
        # Third call is the critic.
        mock_client.chat.completions.create.side_effect = [
            _tied_generator_response(),   # depth 1
            _tied_generator_response(),   # depth 2
            _critic_pick_one_response("cand_1"),  # critic
        ]

        result = run_tree_of_thought(
            clause_category="Indemnification",
            clause_text="Some ambiguous text.",
            source_chunk_id=CHUNK_ID,
            document_chunks=CHUNKS,
        )

        assert result.critic_step_triggered is True
        assert len(result.winning_candidates) == 1
        assert result.requires_human_review is False

    def test_tie_critic_dual_label(self, mock_get_client):
        """
        Critic declares dual-label -> winning_candidates has 2 entries.
        """
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.side_effect = [
            _tied_generator_response(),
            _tied_generator_response(),
            _critic_dual_label_response(),
        ]

        result = run_tree_of_thought(
            clause_category="Indemnification",
            clause_text="Ambiguous text combining two clause types.",
            source_chunk_id=CHUNK_ID,
            document_chunks=CHUNKS,
        )

        assert result.critic_step_triggered is True
        assert len(result.winning_candidates) == 2
        assert result.requires_human_review is False

    def test_tie_critic_human_review(self, mock_get_client):
        """
        Critic declares unresolved ambiguity -> requires_human_review=True.
        """
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.side_effect = [
            _tied_generator_response(),
            _tied_generator_response(),
            _critic_human_review_response(),
        ]

        result = run_tree_of_thought(
            clause_category="Indemnification",
            clause_text="Very ambiguous text.",
            source_chunk_id=CHUNK_ID,
            document_chunks=CHUNKS,
        )

        assert result.critic_step_triggered is True
        assert result.requires_human_review is True
        assert result.human_review_reason is not None
        assert len(result.winning_candidates) == 1  # best candidate kept as placeholder

    def test_all_candidates_pruned_at_depth_1_keeps_best(self, mock_get_client):
        """
        If all depth-1 candidates score below 0.30, the best one is kept alive
        instead of leaving the beam empty. Pipeline must not crash.
        """
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        # Both candidates below 0.30.
        low_response = _mock_llm_response({
            "candidates": [
                _make_candidate_dict("cand_1", "Indemnification", 0.10, 0.10, 0.10, 0.10),
                _make_candidate_dict("cand_2", "Limitation of Liability", 0.20, 0.20, 0.20, 0.20),
            ]
        })
        mock_client.chat.completions.create.return_value = low_response

        result = run_tree_of_thought(
            clause_category="Indemnification",
            clause_text="Very unclear text.",
            source_chunk_id=CHUNK_ID,
            document_chunks=CHUNKS,
        )

        assert isinstance(result, ToTResult)
        assert len(result.winning_candidates) >= 1  # at least one survived

    def test_api_failure_at_depth_1_returns_human_review(self, mock_get_client):
        """
        An API exception at depth 1 must return a ToTResult with
        requires_human_review=True. The exception must not propagate.
        """
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")

        result = run_tree_of_thought(
            clause_category="Indemnification",
            clause_text="Some text.",
            source_chunk_id=CHUNK_ID,
            document_chunks=CHUNKS,
        )

        assert isinstance(result, ToTResult)
        assert result.requires_human_review is True
        assert result.human_review_reason is not None
        assert "Depth-1" in result.human_review_reason

    def test_pruned_candidates_remain_in_all_candidates(self, mock_get_client):
        """
        Candidates pruned at depth 1 must appear in all_candidates with
        pruned=True and the correct pruned_at_depth, even though they are
        absent from winning_candidates.
        """
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        # cand_1 scores 0.85 (survives), cand_2 scores 0.10 (pruned at depth 1).
        mock_client.chat.completions.create.return_value = (
            _good_generator_response(cand1_score=0.85, cand2_score=0.10)
        )

        result = run_tree_of_thought(
            clause_category="Indemnification",
            clause_text="Clear indemnification language.",
            source_chunk_id=CHUNK_ID,
            document_chunks=CHUNKS,
        )

        # The pruned candidate must be in all_candidates.
        pruned = [c for c in result.all_candidates if c.pruned]
        assert len(pruned) >= 1
        for p in pruned:
            assert p.pruned_at_depth is not None
