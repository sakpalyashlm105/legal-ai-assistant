"""
tests/unit/test_input_validator.py
------------------------------------
Unit tests for guardrails/input_validator.py.

Tests cover:
  - A valid PDF passes all checks
  - Missing file returns blocking failure
  - Empty file returns blocking failure
  - Oversized file returns blocking (size threshold overridden via parameter)
  - Password-protected PDF returns blocking
  - Unknown / corrupt file returns blocking
  - A new hash passes duplicate check
  - A seen hash fails with severity="warning"
  - record_processed_document followed by check_duplicate_document detects duplicate
"""

import json
import sys
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import fitz

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from guardrails.input_validator import (
    validate_input,
    check_duplicate_document,
    record_processed_document,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# A real PDF that definitely exists in the project
REAL_PDF = (
    Path(__file__).parent.parent.parent
    / "data" / "pdf" / "ndas"
    / "NDA_2_Bakhu_Holdings,_Corp._(BKUH).pdf"
)


@pytest.fixture
def tmp_hash_file(tmp_path):
    """Return a path to a temporary hash record file."""
    return str(tmp_path / "processed_hashes.json")


# ---------------------------------------------------------------------------
# validate_input — passing case
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not REAL_PDF.exists(), reason="Test PDF not found")
def test_valid_pdf_passes():
    result = validate_input(str(REAL_PDF))
    assert result.passed is True
    assert result.severity == "info"
    assert "passed" in result.reason.lower()


# ---------------------------------------------------------------------------
# validate_input — failing cases
# ---------------------------------------------------------------------------

def test_missing_file_fails_blocking(tmp_path):
    result = validate_input(str(tmp_path / "nonexistent.pdf"))
    assert result.passed is False
    assert result.severity == "blocking"
    assert "not found" in result.reason.lower()


def test_empty_file_fails_blocking(tmp_path):
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    result = validate_input(str(empty))
    assert result.passed is False
    assert result.severity == "blocking"
    assert "empty" in result.reason.lower()


def test_oversized_file_fails_blocking(tmp_path):
    """
    Use max_file_size_bytes=10 to simulate an oversized file without
    needing a real large file. The real limit (50 MB) is unchanged in
    production -- this override exists only for test efficiency.
    """
    small_but_oversize = tmp_path / "oversize.pdf"
    small_but_oversize.write_bytes(b"X" * 100)  # 100 bytes > limit of 10
    result = validate_input(str(small_but_oversize), max_file_size_bytes=10)
    assert result.passed is False
    assert result.severity == "blocking"
    assert "limit" in result.reason.lower()


def test_corrupt_file_fails_blocking(tmp_path):
    """A file that is not a valid PDF should fail at the open step."""
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"this is not a PDF file at all %$#@!")
    result = validate_input(str(corrupt))
    assert result.passed is False
    assert result.severity == "blocking"


def test_path_is_directory_fails_blocking(tmp_path):
    """Passing a directory path should fail the is_file check."""
    result = validate_input(str(tmp_path))
    assert result.passed is False
    assert result.severity == "blocking"


@pytest.mark.skipif(not REAL_PDF.exists(), reason="Test PDF not found")
def test_password_protected_pdf_fails_blocking(tmp_path):
    """
    Create a password-protected PDF programmatically using PyMuPDF.
    The actual NDA PDF is not password-protected, so we make one in tmp_path.
    """
    protected_path = str(tmp_path / "protected.pdf")
    doc = fitz.open(str(REAL_PDF))
    # Encrypt with user password -- this causes needs_pass=True when opened without it
    doc.save(
        protected_path,
        encryption=fitz.PDF_ENCRYPT_AES_256,
        user_pw="secret",
        owner_pw="secret",
    )
    doc.close()

    result = validate_input(protected_path)
    assert result.passed is False
    assert result.severity == "blocking"
    assert "password" in result.reason.lower()


@pytest.mark.skipif(not REAL_PDF.exists(), reason="Test PDF not found")
def test_page_count_limit_fails_blocking():
    """Force a page-count failure by setting max_page_count=0."""
    # Any real PDF has >= 1 page, so max_page_count=0 always triggers the limit check.
    result = validate_input(str(REAL_PDF), max_page_count=0)
    assert result.passed is False
    assert result.severity == "blocking"
    assert "page" in result.reason.lower()


# ---------------------------------------------------------------------------
# check_duplicate_document
# ---------------------------------------------------------------------------

def test_new_hash_passes_duplicate_check(tmp_hash_file):
    result = check_duplicate_document("abc123" * 10 + "abcd", tmp_hash_file)
    assert result.passed is True
    assert result.severity == "info"


def test_seen_hash_fails_with_warning(tmp_hash_file):
    """Write a hash directly into the JSON file, then check it."""
    fake_hash = "a" * 64
    Path(tmp_hash_file).parent.mkdir(parents=True, exist_ok=True)
    Path(tmp_hash_file).write_text(json.dumps({fake_hash: {"processed_at": "2026-01-01"}}), encoding="utf-8")

    result = check_duplicate_document(fake_hash, tmp_hash_file)
    assert result.passed is False
    assert result.severity == "warning"  # NOT blocking -- see module docstring
    assert fake_hash[:12] in result.reason


def test_duplicate_detected_after_record(tmp_hash_file):
    """
    Round-trip: record a hash, then verify check_duplicate_document finds it.
    """
    fake_hash = "b" * 64
    # Before recording: passes
    r1 = check_duplicate_document(fake_hash, tmp_hash_file)
    assert r1.passed is True

    # Record it
    record_processed_document(fake_hash, tmp_hash_file, metadata={"file_name": "test.pdf"})

    # After recording: detected as duplicate
    r2 = check_duplicate_document(fake_hash, tmp_hash_file)
    assert r2.passed is False
    assert r2.severity == "warning"


def test_record_persists_to_disk(tmp_hash_file):
    """Verify the JSON file actually exists and contains the hash after recording."""
    fake_hash = "c" * 64
    record_processed_document(fake_hash, tmp_hash_file)

    data = json.loads(Path(tmp_hash_file).read_text(encoding="utf-8"))
    assert fake_hash in data
    assert "processed_at" in data[fake_hash]


def test_check_creates_no_file_if_not_found(tmp_path):
    """check_duplicate_document should not create the hash file on a first check."""
    path = str(tmp_path / "hashes.json")
    result = check_duplicate_document("x" * 64, path)
    assert result.passed is True
    assert not Path(path).exists()  # file must NOT be created by check, only by record


def test_missing_hash_file_treated_as_empty(tmp_path):
    """If the hash file doesn't exist, check_duplicate_document should treat it as empty."""
    result = check_duplicate_document("d" * 64, str(tmp_path / "missing.json"))
    assert result.passed is True


def test_corrupt_hash_file_treated_as_empty(tmp_path):
    """A corrupt JSON hash file should fall back to an empty record without crashing."""
    path = tmp_path / "corrupt_hashes.json"
    path.write_text("{ NOT VALID JSON }", encoding="utf-8")
    result = check_duplicate_document("e" * 64, str(path))
    assert result.passed is True  # treated as empty, not a crash
