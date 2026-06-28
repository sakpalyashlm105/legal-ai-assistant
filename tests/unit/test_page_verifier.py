"""
tests/unit/test_page_verifier.py
----------------------------------
Unit tests for guardrails/page_verifier.py.

Uses minimal DocumentExtraction / PageExtraction stubs constructed inline
rather than real PDFs to keep tests fast and deterministic.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from schemas.document import DocumentExtraction, PageExtraction, ExtractionMethod
from guardrails.page_verifier import verify_page_reference


# ---------------------------------------------------------------------------
# Helper: build a minimal DocumentExtraction with known page texts
# ---------------------------------------------------------------------------

def _make_doc(pages: dict[int, str]) -> DocumentExtraction:
    """
    Create a DocumentExtraction where pages[n] = text of page n.
    All other required fields are set to minimal valid values.
    """
    page_list = [
        PageExtraction(
            page_number=n,
            text=text,
            method=ExtractionMethod.PYMUPDF,
        )
        for n, text in sorted(pages.items())
    ]
    return DocumentExtraction(
        file_path="/fake/test.pdf",
        file_name="test.pdf",
        file_hash="a" * 64,
        total_pages=max(pages.keys()),
        pages=page_list,
        extraction_method_summary="pymupdf",
    )


# ---------------------------------------------------------------------------
# Structural checks (page existence)
# ---------------------------------------------------------------------------

class TestPageExistence:
    def test_valid_page_exists(self):
        doc = _make_doc({1: "Confidentiality clause text.", 2: "Termination clause."})
        result = verify_page_reference(1, doc)
        assert result.page_exists_in_document is True
        assert result.cited_page == 1

    def test_cited_page_beyond_total_fails(self):
        doc = _make_doc({1: "Page one.", 2: "Page two."})
        result = verify_page_reference(50, doc)
        assert result.page_exists_in_document is False
        assert result.text_found_near_cited_page is False
        assert "50" in (result.notes or "")

    def test_cited_page_zero_fails(self):
        """Page numbers are 1-based; 0 is always invalid."""
        doc = _make_doc({1: "Some text."})
        result = verify_page_reference(0, doc)
        assert result.page_exists_in_document is False

    def test_negative_page_fails(self):
        doc = _make_doc({1: "Some text."})
        result = verify_page_reference(-5, doc)
        assert result.page_exists_in_document is False

    def test_exactly_last_page_is_valid(self):
        doc = _make_doc({1: "Page 1.", 2: "Page 2.", 3: "Page 3."})
        result = verify_page_reference(3, doc)
        assert result.page_exists_in_document is True


# ---------------------------------------------------------------------------
# Text found on cited page
# ---------------------------------------------------------------------------

class TestTextOnCitedPage:
    def test_text_found_on_exact_cited_page(self):
        doc = _make_doc({
            1: "Introduction and parties.",
            2: "The Receiving Party agrees to maintain confidentiality of all information.",
            3: "Termination upon 30 days notice.",
        })
        result = verify_page_reference(
            2, doc,
            extracted_text="The Receiving Party agrees to maintain confidentiality",
        )
        assert result.page_exists_in_document is True
        assert result.text_found_near_cited_page is True
        assert result.notes is None  # exact page match, no note needed

    def test_text_found_on_adjacent_page(self):
        """Clause spanning pages: cited page 2 but text is actually on page 3."""
        doc = _make_doc({
            1: "Introduction.",
            2: "Section 4. Indemnification. The parties agree to",
            3: "indemnify and hold harmless each other from all claims.",
            4: "Governing law.",
        })
        # LLM cites page 2 but the key phrase is actually on page 3
        result = verify_page_reference(
            2, doc,
            extracted_text="indemnify and hold harmless each other from all claims",
        )
        assert result.page_exists_in_document is True
        assert result.text_found_near_cited_page is True
        # The note should mention which actual page was found
        assert "3" in (result.notes or "")

    def test_text_found_on_previous_adjacent_page(self):
        """Clause cited on page 3 but actually starts on page 2."""
        doc = _make_doc({
            1: "Introduction.",
            2: "Section 5. Non-Compete. For a period of two years after termination,",
            3: "the Employee shall not engage in competing business activities.",
        })
        result = verify_page_reference(
            3, doc,
            extracted_text="For a period of two years after termination",
        )
        assert result.page_exists_in_document is True
        assert result.text_found_near_cited_page is True
        assert "2" in (result.notes or "")


# ---------------------------------------------------------------------------
# Text not found
# ---------------------------------------------------------------------------

class TestTextNotFound:
    def test_text_not_present_anywhere_near_cited_page(self):
        doc = _make_doc({
            1: "Page one about confidentiality.",
            2: "Page two about termination.",
            3: "Page three about governing law.",
        })
        result = verify_page_reference(
            2, doc,
            extracted_text="This clause does not exist anywhere in the document at all.",
        )
        assert result.page_exists_in_document is True
        assert result.text_found_near_cited_page is False

    def test_out_of_range_page_with_text_returns_false(self):
        doc = _make_doc({1: "Only page."})
        result = verify_page_reference(
            99, doc,
            extracted_text="The parties agree to confidentiality.",
        )
        assert result.page_exists_in_document is False
        assert result.text_found_near_cited_page is False


# ---------------------------------------------------------------------------
# No extracted_text provided
# ---------------------------------------------------------------------------

class TestNoExtractedText:
    def test_no_text_valid_page_returns_exists_true(self):
        doc = _make_doc({1: "Some text.", 2: "More text."})
        result = verify_page_reference(1, doc, extracted_text=None)
        assert result.page_exists_in_document is True
        # Cannot confirm text found without extracted_text
        assert result.text_found_near_cited_page is False
        assert result.notes is not None  # should note why text check was skipped

    def test_no_text_invalid_page_returns_exists_false(self):
        doc = _make_doc({1: "Some text."})
        result = verify_page_reference(5, doc, extracted_text=None)
        assert result.page_exists_in_document is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_page_document_adjacent_check_safe(self):
        """Pages 0 and 2 don't exist; the verifier must clamp safely."""
        doc = _make_doc({1: "The only page. Confidentiality clause here."})
        result = verify_page_reference(
            1, doc,
            extracted_text="Confidentiality clause here",
        )
        assert result.page_exists_in_document is True
        assert result.text_found_near_cited_page is True

    def test_first_page_no_page_zero_check(self):
        """Cited page 1 of a 5-page doc -- adjacent check should not try page 0."""
        doc = _make_doc({
            1: "Target clause text.",
            2: "Other content.",
            3: "More content.",
        })
        # Should work without errors even though page 0 doesn't exist
        result = verify_page_reference(1, doc, extracted_text="Target clause text")
        assert result.page_exists_in_document is True
        assert result.text_found_near_cited_page is True
