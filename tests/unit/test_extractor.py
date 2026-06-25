"""
tests/unit/test_extractor.py
-----------------------------
Automated test suite for agent/extractor.py.

All OpenAI API calls are mocked -- no real API key or network access needed.

What this file tests:
    1.  A successful extraction returns exactly 10 ExtractedClause objects.
    2.  A present clause has is_present=True and non-None extracted_text.
    3.  An absent clause has is_present=False and extracted_text=None.
    4.  extract_clauses called with an empty chunk list returns 10 absent clauses
        without calling the API.
    5.  An API exception returns 10 absent clauses gracefully (no crash).
    6.  Malformed JSON from the LLM returns 10 absent clauses gracefully.
    7.  A response with the wrong number of clause entries (not 10) returns
        10 absent clauses gracefully.
    8.  The context_text built from chunks includes chunk_id and page range
        headers for each chunk.
    9.  All 10 CLAUSE_CATEGORIES are represented in the result (even if absent).
    10. clause_type values in the result exactly match the CLAUSE_CATEGORIES list.

How to run:
    From legal-agent/ with the venv active:
        pytest tests/unit/test_extractor.py -v
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from schemas.chunk import DocumentChunk
from schemas.clause import ExtractedClause
from agent.extractor import (
    extract_clauses,
    _build_context_text,
    _all_absent_clauses,
    _apply_confidence_routing,
)
from config import CLAUSE_CATEGORIES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(index: int, text: str, start_page: int = 1, end_page: int = 2) -> DocumentChunk:
    doc_hash = "ab" * 32
    return DocumentChunk(
        chunk_id=f"{doc_hash[:12]}_chunk_{index:04d}",
        document_hash=doc_hash,
        document_name="test_contract.pdf",
        text=text,
        char_count=len(text),
        token_count=len(text) // 4,
        start_page=start_page,
        end_page=end_page,
        chunk_index=index,
        total_chunks=3,
        overlap_tokens=0,
    )


def _make_llm_response(clauses: list[dict]) -> MagicMock:
    """Wrap a list of clause dicts in a mock OpenAI response."""
    content = json.dumps({"clauses": clauses})
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content=content))]
    return mock_resp


def _full_clause_list(
    present_type: str = "Indemnification",
    present_text: str = "The indemnifying party shall defend and hold harmless...",
    present_page: int = 3,
) -> list[dict]:
    """
    Build a 10-element clause list where one clause is present and the rest absent.
    """
    result = []
    for category in CLAUSE_CATEGORIES:
        if category == present_type:
            result.append({
                "clause_type": category,
                "is_present": True,
                "extracted_text": present_text,
                "page_reference": present_page,
                "confidence": 0.92,
                "source_chunk_id": "abababababab_chunk_0001",
            })
        else:
            result.append({
                "clause_type": category,
                "is_present": False,
                "extracted_text": None,
                "page_reference": None,
                "confidence": 0.85,
                "source_chunk_id": None,
            })
    return result


# ---------------------------------------------------------------------------
# Test 1: Successful extraction returns exactly 10 clauses
# ---------------------------------------------------------------------------

def test_successful_extraction_returns_ten_clauses():
    chunks = [_make_chunk(i, f"Legal text about clause {i}.") for i in range(3)]

    with patch("agent.extractor._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_llm_response(
            _full_clause_list()
        )

        result = extract_clauses(chunks, document_name="test.pdf")

    assert len(result) == 10, f"Expected 10 clauses, got {len(result)}"


# ---------------------------------------------------------------------------
# Test 2: Present clause has is_present=True and non-None text
# ---------------------------------------------------------------------------

def test_present_clause_has_text():
    chunks = [_make_chunk(0, "The indemnifying party shall defend and hold harmless...")]
    expected_text = "The indemnifying party shall defend and hold harmless..."

    with patch("agent.extractor._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_llm_response(
            _full_clause_list(present_type="Indemnification", present_text=expected_text)
        )

        result = extract_clauses(chunks)

    indemnification = next(c for c in result if c.clause_type == "Indemnification")
    assert indemnification.is_present is True
    assert indemnification.extracted_text == expected_text
    assert indemnification.page_reference is not None


# ---------------------------------------------------------------------------
# Test 3: Absent clause has is_present=False and None text
# ---------------------------------------------------------------------------

def test_absent_clause_has_no_text():
    chunks = [_make_chunk(0, "General contract terms about payment and delivery.")]

    with patch("agent.extractor._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_llm_response(
            _full_clause_list(present_type="Indemnification")  # only indemnification present
        )

        result = extract_clauses(chunks)

    # Termination for Convenience should be absent
    term_conv = next(c for c in result if c.clause_type == "Termination for Convenience")
    assert term_conv.is_present is False
    assert term_conv.extracted_text is None


# ---------------------------------------------------------------------------
# Test 4: Empty chunk list -> 10 absent clauses, no API call
# ---------------------------------------------------------------------------

def test_empty_chunks_returns_absent_clauses_without_api_call():
    with patch("agent.extractor._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        result = extract_clauses([], document_name="empty.pdf")

    assert len(result) == 10
    assert all(not c.is_present for c in result)
    mock_client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# Test 5: API exception -> 10 absent clauses, no crash
# ---------------------------------------------------------------------------

def test_api_exception_returns_absent_clauses_gracefully():
    chunks = [_make_chunk(0, "Some contract text.")]

    with patch("agent.extractor._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("Network error")

        result = extract_clauses(chunks, document_name="test.pdf")

    assert len(result) == 10
    assert all(not c.is_present for c in result)
    assert all(c.confidence == 0.0 for c in result)


# ---------------------------------------------------------------------------
# Test 6: Malformed JSON -> 10 absent clauses, no crash
# ---------------------------------------------------------------------------

def test_malformed_json_returns_absent_clauses_gracefully():
    chunks = [_make_chunk(0, "Some contract text.")]

    bad_response = MagicMock()
    bad_response.choices = [MagicMock(message=MagicMock(content="not valid json {{}"))]

    with patch("agent.extractor._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = bad_response

        result = extract_clauses(chunks, document_name="test.pdf")

    assert len(result) == 10
    assert all(not c.is_present for c in result)


# ---------------------------------------------------------------------------
# Test 7: Wrong number of clause entries in response -> 10 absent clauses
# ---------------------------------------------------------------------------

def test_wrong_clause_count_in_response_returns_absent_clauses():
    """
    If the LLM returns only 5 entries instead of 10, extract_clauses must
    catch the mismatch and return the safe fallback (10 absent clauses).
    """
    chunks = [_make_chunk(0, "Some contract text.")]
    only_five = _full_clause_list()[:5]  # truncate to 5

    with patch("agent.extractor._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_llm_response(only_five)

        result = extract_clauses(chunks, document_name="test.pdf")

    assert len(result) == 10
    assert all(not c.is_present for c in result)


# ---------------------------------------------------------------------------
# Test 8: Context text includes chunk_id and page range headers
# ---------------------------------------------------------------------------

def test_context_text_includes_source_headers():
    """
    _build_context_text should prefix each chunk with its chunk_id and page range.
    This is how the LLM knows which page to cite in page_reference.
    """
    chunk = _make_chunk(2, "Governing law shall be New York.", start_page=7, end_page=7)
    context = _build_context_text([chunk])

    assert 'chunk_id: "' in context
    assert chunk.chunk_id in context
    assert "pages: 7-7" in context
    assert "Governing law shall be New York." in context


# ---------------------------------------------------------------------------
# Test 9: All 10 CLAUSE_CATEGORIES are represented in the result
# ---------------------------------------------------------------------------

def test_all_clause_categories_represented():
    chunks = [_make_chunk(0, "Contract text.")]

    with patch("agent.extractor._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_llm_response(
            _full_clause_list()
        )

        result = extract_clauses(chunks)

    result_types = {c.clause_type for c in result}
    for category in CLAUSE_CATEGORIES:
        assert category in result_types, f"Missing clause category: {category}"


# ---------------------------------------------------------------------------
# Test 10: clause_type values exactly match CLAUSE_CATEGORIES
# ---------------------------------------------------------------------------

def test_clause_types_match_config_categories():
    """
    The clause_type on every returned ExtractedClause must be one of the
    10 locked categories from config.CLAUSE_CATEGORIES. No extras, no typos.
    """
    chunks = [_make_chunk(0, "Contract text.")]

    with patch("agent.extractor._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_llm_response(
            _full_clause_list()
        )

        result = extract_clauses(chunks)

    for clause in result:
        assert clause.clause_type in CLAUSE_CATEGORIES, (
            f"Unexpected clause_type: {clause.clause_type!r}"
        )


# ---------------------------------------------------------------------------
# Stage 4 tests: three-tier confidence routing
# ---------------------------------------------------------------------------

def _make_clause(
    clause_type: str = "Indemnification",
    is_present: bool = True,
    confidence: float = 0.90,
    source_chunk_id: str = "abababababab_chunk_0001",
    retry_count: int = 0,
) -> ExtractedClause:
    return ExtractedClause(
        clause_type=clause_type,
        is_present=is_present,
        extracted_text="Some clause text." if is_present else None,
        page_reference=1 if is_present else None,
        confidence=confidence,
        source_chunk_id=source_chunk_id if is_present else None,
        retry_count=retry_count,
    )


def _make_chunks_for_routing(n: int = 3) -> list[DocumentChunk]:
    doc_hash = "ab" * 32
    chunks = []
    for i in range(n):
        text = f"Clause text {i}."
        chunks.append(DocumentChunk(
            chunk_id=f"{doc_hash[:12]}_chunk_{i:04d}",
            document_hash=doc_hash,
            document_name="test.pdf",
            text=text,
            char_count=len(text),
            token_count=len(text) // 4,
            start_page=i + 1,
            end_page=i + 1,
            chunk_index=i,
            total_chunks=n,
            overlap_tokens=0,
        ))
    return chunks


# Test 11: confidence > 0.7 does NOT trigger ToT or retry
def test_high_confidence_clause_no_tot_no_retry():
    """
    Clauses with confidence > 0.7 must pass through _apply_confidence_routing
    unchanged. Neither ToT nor retry should be called.
    """
    clause = _make_clause(confidence=0.92)
    chunks = _make_chunks_for_routing()

    with patch("agent.extractor.run_tree_of_thought") as mock_tot, \
         patch("agent.extractor._get_client") as mock_client:

        result = _apply_confidence_routing(clause, chunks, "test.pdf")

    mock_tot.assert_not_called()
    assert result.confidence == 0.92
    assert result.requires_human_review is False
    assert result.retry_count == 0


# Test 12: confidence in 0.5-0.7 band calls run_tree_of_thought
def test_mid_confidence_clause_calls_tot():
    """
    Clauses with 0.5 <= confidence < 0.7 must trigger ToT.
    Mock ToT to return a clean single-winner result.
    """
    from schemas.tot import ToTCandidate, ToTResult

    clause = _make_clause(confidence=0.60)
    chunks = _make_chunks_for_routing()

    winner = ToTCandidate(
        candidate_id="cand_1",
        clause_category="Indemnification",
        textual_evidence_score=0.8,
        category_fit_score=0.8,
        template_alignment_score=0.8,
        exclusivity_score=0.8,
        composite_score=0.8,
        depth_reached=3,
        pruned=False,
        pruned_at_depth=None,
        rationale="Clear indemnification match.",
    )
    mock_tot_result = ToTResult(
        clause_category_input="Indemnification",
        source_chunk_id="abababababab_chunk_0001",
        all_candidates=[winner],
        winning_candidates=[winner],
        requires_human_review=False,
    )

    with patch("agent.extractor.run_tree_of_thought", return_value=mock_tot_result) as mock_tot:
        result = _apply_confidence_routing(clause, chunks, "test.pdf")

    mock_tot.assert_called_once_with(
        clause_category="Indemnification",
        clause_text="Some clause text.",
        source_chunk_id="abababababab_chunk_0001",
        document_chunks=chunks,
    )
    assert result.confidence == 0.8  # updated from ToT winner composite_score
    assert result.requires_human_review is False


# Test 13: confidence < 0.5 triggers exactly ONE retry; if still low -> human review
def test_low_confidence_retries_once_then_escalates():
    """
    A clause with confidence < 0.5 must be retried exactly once.
    If the retry also returns confidence < 0.5, requires_human_review=True.
    The retry must not fire a second time (retry_count capped at 1).
    """
    clause = _make_clause(confidence=0.30, retry_count=0)
    chunks = _make_chunks_for_routing()

    # First retry also returns low confidence.
    low_retry_list = []
    for category in CLAUSE_CATEGORIES:
        conf = 0.30 if category == "Indemnification" else 0.85
        low_retry_list.append({
            "clause_type": category,
            "is_present": category == "Indemnification",
            "extracted_text": "Low confidence text." if category == "Indemnification" else None,
            "page_reference": 1 if category == "Indemnification" else None,
            "confidence": conf,
            "source_chunk_id": "abababababab_chunk_0001" if category == "Indemnification" else None,
        })
    low_retry_response = _make_llm_response(low_retry_list)

    with patch("agent.extractor._get_client") as mock_get_client, \
         patch("agent.extractor.run_tree_of_thought") as mock_tot:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = low_retry_response

        result = _apply_confidence_routing(clause, chunks, "test.pdf")

    # ToT must NOT have been called (this is the <0.5 path, not 0.5-0.7).
    mock_tot.assert_not_called()
    # Retry must have fired once.
    assert result.retry_count == 1
    # Still low after retry -> human review.
    assert result.requires_human_review is True
    assert result.human_review_reason is not None


# Test 14: clause already retried (retry_count=1) is escalated without another retry
def test_already_retried_clause_escalates_without_another_retry():
    """
    A clause with retry_count=1 and confidence < 0.5 must be escalated
    immediately without making another LLM call (no infinite retry loop).
    """
    clause = _make_clause(confidence=0.30, retry_count=1)
    chunks = _make_chunks_for_routing()

    with patch("agent.extractor._get_client") as mock_get_client, \
         patch("agent.extractor.run_tree_of_thought") as mock_tot:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        result = _apply_confidence_routing(clause, chunks, "test.pdf")

    # Neither ToT nor another LLM retry should fire.
    mock_tot.assert_not_called()
    mock_client.chat.completions.create.assert_not_called()
    assert result.requires_human_review is True


# Test 15: absent clause bypasses all routing
def test_absent_clause_bypasses_confidence_routing():
    """
    Absent clauses (is_present=False) are structural gaps, not ambiguous
    classifications. The routing function must return them unchanged.
    """
    clause = _make_clause(is_present=False, confidence=0.0)
    chunks = _make_chunks_for_routing()

    with patch("agent.extractor.run_tree_of_thought") as mock_tot, \
         patch("agent.extractor._get_client") as mock_client:

        result = _apply_confidence_routing(clause, chunks, "test.pdf")

    mock_tot.assert_not_called()
    assert result.is_present is False
    assert result.requires_human_review is False
