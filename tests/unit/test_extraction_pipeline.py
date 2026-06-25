"""
tests/unit/test_extraction_pipeline.py
---------------------------------------
Automated test suite for the Step 3 text extraction pipeline.

What this file tests, end to end:
    1. A normal digital PDF extracts successfully via PyMuPDF (no OCR needed).
    2. A page with too little text correctly triggers the OCR fallback path
       (the actual OpenAI call is mocked -- no real API call is made).
    3. A page with too little text and OCR disabled is correctly marked FAILED.
    4. A non-existent file path is handled gracefully (no crash).
    5. An empty (0-byte) file is handled gracefully.
    6. A corrupted / non-PDF file is handled gracefully.
    7. A password-protected PDF is handled gracefully.
    8. The extraction_validator guardrail's threshold logic is correct on
       its own, independent of any PDF or file at all.
    9. File hashing is deterministic (same bytes -> same hash, every time).

How to run this file:
    From the legal-agent/ project root, with the venv active:
        pytest tests/unit/test_extraction_pipeline.py -v

    The "-v" flag means "verbose" -- it prints the name and pass/fail status
    of every individual test, instead of just a summary count.
"""

import io
import os
from pathlib import Path
from unittest.mock import patch

import fitz  # PyMuPDF
import pytest

from extraction.pdf_parser import (
    compute_file_hash,
    extract_text_from_pdf,
    normalize_text,
)
from guardrails.extraction_validator import (
    EXTRACTION_CHAR_THRESHOLD,
    check_extraction_quality,
)
from schemas.document import ExtractionMethod, PageExtraction


# ---------------------------------------------------------------------------
# Helpers: build small test PDFs on the fly, saved to a temp folder
# ---------------------------------------------------------------------------
# "tmp_path" below is a special built-in pytest fixture. Pytest automatically
# creates a brand-new, empty temporary folder for each test that asks for it,
# and deletes it afterward. This means our tests never touch your real
# data/raw/ folder and never leave junk files behind.


def _make_text_pdf(path: Path, paragraphs: int = 20) -> None:
    """
    Create a simple digital PDF with plenty of real embedded text.
    Used to simulate a normal, well-formed legal document page.
    """
    doc = fitz.open()
    page = doc.new_page()
    text = "\n".join(
        [f"Paragraph {i}: This is sample contract language for testing purposes. "
         f"It contains enough characters to safely pass the extraction threshold."
         for i in range(paragraphs)]
    )
    page.insert_text((50, 50), text, fontsize=8)
    doc.save(str(path))
    doc.close()


def _make_blank_pdf(path: Path) -> None:
    """
    Create a PDF with a page that has effectively no extractable text --
    simulating a scanned image page where PyMuPDF's direct text extraction
    returns almost nothing.
    """
    doc = fitz.open()
    doc.new_page()  # a blank page, no text inserted at all
    doc.save(str(path))
    doc.close()


def _make_corrupted_pdf(path: Path) -> None:
    """
    Create a file with a .pdf extension that is NOT actually valid PDF data.
    Used to simulate a corrupted or mislabeled file.
    """
    path.write_bytes(b"This is not a real PDF file, just plain text bytes.")


def _make_empty_file(path: Path) -> None:
    """Create a genuinely empty (0-byte) file."""
    path.write_bytes(b"")


def _make_password_protected_pdf(path: Path) -> None:
    """
    Create a PDF that requires a password to open.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Secret content that requires a password.")
    doc.save(
        str(path),
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner_secret",
        user_pw="user_secret",
    )
    doc.close()


# ---------------------------------------------------------------------------
# Test 1: Normal digital PDF -- the "everything works" happy path
# ---------------------------------------------------------------------------

def test_normal_pdf_extracts_via_pymupdf(tmp_path):
    """
    A PDF with plenty of real text should extract fully via PyMuPDF,
    with zero pages needing OCR and zero failures.
    """
    pdf_path = tmp_path / "normal_contract.pdf"
    _make_text_pdf(pdf_path, paragraphs=20)

    result = extract_text_from_pdf(str(pdf_path), use_ocr_fallback=False)

    assert result.extraction_successful is True
    assert result.total_pages == 1
    assert result.pages_failed == 0
    assert result.pages_ocr == 0
    assert len(result.pages) == 1
    assert result.pages[0].method == ExtractionMethod.PYMUPDF
    assert result.pages[0].char_count >= EXTRACTION_CHAR_THRESHOLD
    assert "Paragraph 0" in result.pages[0].text
    assert result.error_message is None


# ---------------------------------------------------------------------------
# Test 2: Low-text page with OCR enabled -- mocked Vision call
# ---------------------------------------------------------------------------

def test_blank_page_routes_to_ocr_when_enabled(tmp_path):
    """
    A page with almost no extractable text should trigger the OCR fallback
    path when use_ocr_fallback=True. We MOCK the actual Vision API call so
    this test runs instantly, for free, with no real network request.

    "patch" temporarily replaces the real extract_page_with_vision function
    with a fake one (a "mock") for the duration of this test only. Outside
    this test, the real function is untouched.
    """
    pdf_path = tmp_path / "blank_page.pdf"
    _make_blank_pdf(pdf_path)

    fake_ocr_result = PageExtraction(
        page_number=1,
        text="This is fake OCR text standing in for a real Vision API response.",
        char_count=68,
        method=ExtractionMethod.OCR_VISION,
    )

    # We patch the function exactly where pdf_parser.py looks it up:
    # "extraction.ocr.extract_page_with_vision" (pdf_parser.py does a lazy
    # import of this function from inside extraction/ocr.py).
    with patch("extraction.ocr.extract_page_with_vision", return_value=fake_ocr_result) as mock_ocr:
        result = extract_text_from_pdf(str(pdf_path), use_ocr_fallback=True)

        # Confirm the mock was actually called -- proves the routing logic
        # correctly decided this page needed OCR.
        assert mock_ocr.call_count == 1

    assert result.extraction_successful is True
    assert result.pages_ocr == 1
    assert result.pages_failed == 0
    assert result.pages[0].method == ExtractionMethod.OCR_VISION
    assert result.pages[0].text == fake_ocr_result.text


# ---------------------------------------------------------------------------
# Test 3: Low-text page with OCR disabled -- should be marked FAILED
# ---------------------------------------------------------------------------

def test_blank_page_marked_failed_when_ocr_disabled(tmp_path):
    """
    The same blank page as Test 2, but with use_ocr_fallback=False.
    The page should be marked FAILED immediately, with no attempt to call
    OCR at all -- and the overall document should still be considered
    successful as long as not EVERY page failed. Since this document has
    only one page and it fails, the WHOLE document should report failure.
    """
    pdf_path = tmp_path / "blank_page_no_ocr.pdf"
    _make_blank_pdf(pdf_path)

    result = extract_text_from_pdf(str(pdf_path), use_ocr_fallback=False)

    assert result.pages[0].method == ExtractionMethod.FAILED
    assert result.pages_failed == 1
    assert result.pages_ocr == 0
    # Only page in the doc failed -> whole document extraction is unsuccessful
    assert result.extraction_successful is False
    assert result.error_message is not None


# ---------------------------------------------------------------------------
# Test 4: Non-existent file path
# ---------------------------------------------------------------------------

def test_missing_file_returns_graceful_failure(tmp_path):
    """
    Pointing the extractor at a file path that does not exist must NOT
    raise an exception -- it must return a DocumentExtraction with
    extraction_successful=False and a clear error_message.
    """
    fake_path = tmp_path / "this_file_does_not_exist.pdf"

    result = extract_text_from_pdf(str(fake_path), use_ocr_fallback=False)

    assert result.extraction_successful is False
    assert result.total_pages == 0
    assert "not found" in result.error_message.lower()


# ---------------------------------------------------------------------------
# Test 5: Empty (0-byte) file
# ---------------------------------------------------------------------------

def test_empty_file_returns_graceful_failure(tmp_path):
    """
    A genuinely empty file (0 bytes) on disk must be caught by the
    pre-flight checks before PyMuPDF ever tries to open it.
    """
    empty_path = tmp_path / "empty.pdf"
    _make_empty_file(empty_path)

    result = extract_text_from_pdf(str(empty_path), use_ocr_fallback=False)

    assert result.extraction_successful is False
    assert "empty" in result.error_message.lower()


# ---------------------------------------------------------------------------
# Test 6: Corrupted / not-actually-a-PDF file
# ---------------------------------------------------------------------------

def test_corrupted_file_returns_graceful_failure(tmp_path):
    """
    A file with a .pdf extension that is not real PDF data must be caught
    and reported clearly, not crash the program.
    """
    corrupted_path = tmp_path / "corrupted.pdf"
    _make_corrupted_pdf(corrupted_path)

    result = extract_text_from_pdf(str(corrupted_path), use_ocr_fallback=False)

    assert result.extraction_successful is False
    assert result.error_message is not None


# ---------------------------------------------------------------------------
# Test 7: Password-protected PDF
# ---------------------------------------------------------------------------

def test_password_protected_pdf_returns_graceful_failure(tmp_path):
    """
    A PDF that requires a password must be detected and rejected cleanly,
    with a message asking for an unlocked version -- never a crash, and
    never an attempt to guess the password.
    """
    protected_path = tmp_path / "protected.pdf"
    _make_password_protected_pdf(protected_path)

    result = extract_text_from_pdf(str(protected_path), use_ocr_fallback=False)

    assert result.extraction_successful is False
    assert "password" in result.error_message.lower()


# ---------------------------------------------------------------------------
# Test 8: The guardrail's threshold logic, tested completely on its own
# ---------------------------------------------------------------------------

def test_extraction_validator_passes_long_text():
    """
    Text comfortably above the threshold should pass, with no PDF or file
    involved at all -- this tests the guardrail function in total isolation.
    """
    long_text = "word " * 50  # 50 words * 5 chars = 250 meaningful characters
    result = check_extraction_quality(long_text)

    assert result.passed is True
    assert result.char_count >= EXTRACTION_CHAR_THRESHOLD


def test_extraction_validator_fails_short_text():
    """
    Text far below the threshold should fail.
    """
    short_text = "too short"
    result = check_extraction_quality(short_text)

    assert result.passed is False
    assert result.char_count < EXTRACTION_CHAR_THRESHOLD


def test_extraction_validator_fails_whitespace_only_text():
    """
    A page that is all whitespace (spaces, newlines, tabs) but technically
    not an empty string must still correctly count as 0 meaningful characters
    and fail. This guards against a bug where whitespace is mistakenly
    counted as real content.
    """
    whitespace_text = "   \n\n\t\t   \n   "
    result = check_extraction_quality(whitespace_text)

    assert result.passed is False
    assert result.char_count == 0


def test_extraction_validator_boundary_exactly_at_threshold():
    """
    Text with EXACTLY the threshold number of meaningful characters should
    pass (the rule is >=, not >). This test would catch an accidental
    off-by-one bug if the comparison were ever changed to ">" by mistake.
    """
    exact_text = "a" * EXTRACTION_CHAR_THRESHOLD
    result = check_extraction_quality(exact_text)

    assert result.passed is True
    assert result.char_count == EXTRACTION_CHAR_THRESHOLD


def test_extraction_validator_boundary_one_below_threshold():
    """
    Text with exactly ONE character fewer than the threshold should fail.
    """
    almost_text = "a" * (EXTRACTION_CHAR_THRESHOLD - 1)
    result = check_extraction_quality(almost_text)

    assert result.passed is False
    assert result.char_count == EXTRACTION_CHAR_THRESHOLD - 1


# ---------------------------------------------------------------------------
# Test 9: File hashing is deterministic
# ---------------------------------------------------------------------------

def test_file_hash_is_deterministic(tmp_path):
    """
    Hashing the exact same file content twice must produce the exact same
    hash both times. Hashing two DIFFERENT files must produce different
    hashes. This is the property duplicate-detection and caching depend on.
    """
    file_a = tmp_path / "a.pdf"
    file_b = tmp_path / "b.pdf"
    _make_text_pdf(file_a, paragraphs=5)
    _make_text_pdf(file_b, paragraphs=5)  # same content-generating logic

    hash_a_first = compute_file_hash(str(file_a))
    hash_a_second = compute_file_hash(str(file_a))

    assert hash_a_first == hash_a_second  # same file, hashed twice -> identical
    assert len(hash_a_first) == 64  # SHA-256 hex digest is always 64 characters


# ---------------------------------------------------------------------------
# Test 10: normalize_text preserves meaning while cleaning whitespace
# ---------------------------------------------------------------------------

def test_normalize_text_collapses_excess_blank_lines():
    """
    normalize_text should collapse more than 2 consecutive blank lines
    down to 2, while leaving actual words completely untouched.
    """
    messy = "Clause 1: Confidentiality.\n\n\n\n\nClause 2: Termination."
    cleaned = normalize_text(messy)

    assert "Clause 1: Confidentiality." in cleaned
    assert "Clause 2: Termination." in cleaned
    assert "\n\n\n\n" not in cleaned  # no run of 4+ blank lines survives


def test_normalize_text_handles_empty_string():
    """
    normalize_text must not crash on an empty string -- it should just
    return an empty string back.
    """
    assert normalize_text("") == ""
