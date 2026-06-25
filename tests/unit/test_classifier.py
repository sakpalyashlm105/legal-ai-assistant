"""
tests/unit/test_classifier.py
------------------------------
Automated test suite for agent/classifier.py.

All OpenAI API calls are mocked -- no real API key or network access needed.

What this file tests:
    1.  A confident NDA classification (confidence > 0.7) returns immediately
        with the correct document_type and retry_count=0.
    2.  A confident Contract classification works correctly.
    3.  A confident Amendment classification works correctly.
    4.  Low confidence (< 0.5) on the first call triggers exactly one retry.
    5.  Low confidence on both the first call AND the retry returns with
        retry_count=1 (signals HITL escalation without crashing).
    6.  Medium confidence (0.5-0.7) returns immediately for ToT routing
        without triggering a retry.
    7.  A document with no text returns "Other" with confidence=0.0 without
        calling the API.
    8.  An LLM error (API exception) returns "Other" with confidence=0.0
        gracefully without raising.
    9.  An LLM that returns malformed JSON returns "Other" with confidence=0.0
        gracefully without raising.
    10. The preamble sent to the LLM is at most PREAMBLE_CHAR_LIMIT chars,
        even for a very long document.

How to run:
    From legal-agent/ with the venv active:
        pytest tests/unit/test_classifier.py -v
"""

import json
import pytest
from unittest.mock import MagicMock, patch, call

from schemas.document import DocumentExtraction, ExtractionMethod, PageExtraction
from schemas.clause import DocumentClassification
from agent.classifier import classify_document, PREAMBLE_CHAR_LIMIT, CONFIDENCE_TOT_FLOOR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(text: str, file_name: str = "test.pdf") -> DocumentExtraction:
    page = PageExtraction(
        page_number=1,
        text=text,
        char_count=len(text),
        method=ExtractionMethod.PYMUPDF,
    )
    return DocumentExtraction(
        file_path=f"/tmp/{file_name}",
        file_name=file_name,
        file_hash="a" * 64,
        total_pages=1,
        pages=[page],
        full_text=text,
        pages_failed=0,
        pages_ocr=0,
        extraction_successful=True,
        error_message=None,
    )


def _mock_llm_response(document_type: str, confidence: float, reasoning: str = "test") -> MagicMock:
    """Return a mock OpenAI chat completion response with the given JSON body."""
    content = json.dumps({
        "document_type": document_type,
        "confidence": confidence,
        "reasoning": reasoning,
    })
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content=content))]
    return mock_resp


NDA_TEXT = "THIS NON-DISCLOSURE AGREEMENT is entered into between Party A and Party B."
CONTRACT_TEXT = "MASTER SERVICES AGREEMENT between Vendor Corp and Client Inc."
AMENDMENT_TEXT = "Amendment No. 2 to the Master Services Agreement dated January 1, 2023."


# ---------------------------------------------------------------------------
# Test 1: Confident NDA
# ---------------------------------------------------------------------------

def test_confident_nda_classification():
    """confidence > 0.7 for NDA -> auto-proceed, retry_count=0."""
    doc = _make_doc(NDA_TEXT)

    with patch("agent.classifier._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_llm_response(
            "NDA", 0.95
        )

        result = classify_document(doc)

    assert result.document_type == "NDA"
    assert result.confidence == 0.95
    assert result.retry_count == 0
    assert mock_client.chat.completions.create.call_count == 1


# ---------------------------------------------------------------------------
# Test 2: Confident Contract
# ---------------------------------------------------------------------------

def test_confident_contract_classification():
    doc = _make_doc(CONTRACT_TEXT)

    with patch("agent.classifier._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_llm_response(
            "Contract", 0.88
        )

        result = classify_document(doc)

    assert result.document_type == "Contract"
    assert result.retry_count == 0


# ---------------------------------------------------------------------------
# Test 3: Confident Amendment
# ---------------------------------------------------------------------------

def test_confident_amendment_classification():
    doc = _make_doc(AMENDMENT_TEXT)

    with patch("agent.classifier._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_llm_response(
            "Amendment", 0.82
        )

        result = classify_document(doc)

    assert result.document_type == "Amendment"
    assert result.retry_count == 0


# ---------------------------------------------------------------------------
# Test 4: Low confidence triggers exactly one retry
# ---------------------------------------------------------------------------

def test_low_confidence_triggers_one_retry():
    """
    First call returns confidence=0.3 (< 0.5) -> retry.
    Second call returns confidence=0.85 -> auto-proceed with retry_count=1.
    """
    doc = _make_doc(NDA_TEXT)

    responses = [
        _mock_llm_response("NDA", 0.3),   # first call: low confidence
        _mock_llm_response("NDA", 0.85),  # retry: high confidence
    ]

    with patch("agent.classifier._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.side_effect = responses

        result = classify_document(doc)

    assert result.document_type == "NDA"
    assert result.retry_count == 1
    assert mock_client.chat.completions.create.call_count == 2


# ---------------------------------------------------------------------------
# Test 5: Low confidence on both calls -> returns for HITL (no crash)
# ---------------------------------------------------------------------------

def test_low_confidence_after_retry_returns_for_hitl():
    """
    Both calls return confidence < 0.5. The function must return with
    retry_count=1 so the orchestrator can escalate to human review.
    It must NOT loop infinitely or raise an exception.
    """
    doc = _make_doc(NDA_TEXT)

    with patch("agent.classifier._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_llm_response(
            "Other", 0.2
        )

        result = classify_document(doc)

    assert result.retry_count == 1
    assert result.confidence < 0.5
    assert mock_client.chat.completions.create.call_count == 2


# ---------------------------------------------------------------------------
# Test 6: Medium confidence -> ToT routing (no retry triggered)
# ---------------------------------------------------------------------------

def test_medium_confidence_routes_to_tot_without_retry():
    """
    0.5 <= confidence < 0.7 must return immediately for Tree-of-Thought routing.
    It must NOT retry (retry is only for < 0.5).
    """
    doc = _make_doc(CONTRACT_TEXT)

    with patch("agent.classifier._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_llm_response(
            "Contract", 0.6
        )

        result = classify_document(doc)

    assert result.confidence == 0.6
    assert result.retry_count == 0
    # Only one LLM call -- no retry
    assert mock_client.chat.completions.create.call_count == 1


# ---------------------------------------------------------------------------
# Test 7: Empty document -> "Other", confidence=0.0, no API call
# ---------------------------------------------------------------------------

def test_empty_document_returns_other_without_api_call():
    """
    A document with no extractable text must return "Other" with confidence=0.0
    without ever calling the OpenAI API.
    """
    doc = _make_doc("")

    with patch("agent.classifier._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        result = classify_document(doc)

    assert result.document_type == "Other"
    assert result.confidence == 0.0
    mock_client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# Test 8: API exception -> "Other", confidence=0.0, no crash
# ---------------------------------------------------------------------------

def test_api_exception_returns_other_gracefully():
    """
    If the OpenAI API raises an exception, classify_document must return
    "Other" with confidence=0.0 rather than propagating the exception.
    """
    doc = _make_doc(NDA_TEXT)

    with patch("agent.classifier._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("API timeout")

        result = classify_document(doc)

    assert result.document_type == "Other"
    assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# Test 9: Malformed JSON response -> "Other", confidence=0.0, no crash
# ---------------------------------------------------------------------------

def test_malformed_json_response_returns_other_gracefully():
    """
    If the LLM returns a non-JSON string, the function must catch the parse
    error and return "Other" with confidence=0.0 without crashing.
    """
    doc = _make_doc(NDA_TEXT)

    bad_response = MagicMock()
    bad_response.choices = [MagicMock(message=MagicMock(content="This is not JSON."))]

    with patch("agent.classifier._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = bad_response

        result = classify_document(doc)

    assert result.document_type == "Other"
    assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# Test 10: Preamble is capped at PREAMBLE_CHAR_LIMIT
# ---------------------------------------------------------------------------

def test_preamble_is_capped_at_limit():
    """
    Even if the document is very long, only the first PREAMBLE_CHAR_LIMIT
    characters should be sent to the LLM.
    """
    long_text = "word " * 5000   # ~25,000 characters
    doc = _make_doc(long_text)

    captured_messages = []

    def capture_call(**kwargs):
        captured_messages.append(kwargs["messages"])
        return _mock_llm_response("Contract", 0.9)

    with patch("agent.classifier._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.side_effect = capture_call

        classify_document(doc)

    assert captured_messages, "No LLM call was made"
    user_message = captured_messages[0][1]["content"]   # index 1 = user message

    # The full message = prompt boilerplate + preamble (capped at PREAMBLE_CHAR_LIMIT).
    # We verify the preamble portion itself is capped, not the whole message length.
    # Extract the text between the --- delimiters that wrap the preamble.
    import re as _re
    preamble_match = _re.search(r'---\n(.*?)\n---', user_message, _re.DOTALL)
    assert preamble_match, "Could not find preamble delimiters in user message"
    preamble_in_message = preamble_match.group(1)
    assert len(preamble_in_message) <= PREAMBLE_CHAR_LIMIT, (
        f"Preamble in message is {len(preamble_in_message)} chars, "
        f"expected <= {PREAMBLE_CHAR_LIMIT}"
    )
