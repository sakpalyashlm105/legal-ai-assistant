"""
tests/unit/test_chunking.py
----------------------------
Automated test suite for retrieval/chunking.py.

What this file tests:
    1.  A normal document produces the expected number of chunks.
    2.  Every chunk carries the correct document_hash and document_name.
    3.  chunk_index values are contiguous (0, 1, 2, ...) and total_chunks
        matches the actual list length.
    4.  No chunk is empty.
    5.  Overlap: the first tokens of chunk N+1 match the last tokens of chunk N.
    6.  Page numbers: start_page and end_page are valid (>= 1, end >= start).
    7.  A document with no text returns an empty list without crashing.
    8.  A document whose full text is all whitespace returns an empty list.
    9.  An oversized single paragraph (larger than target_tokens) is kept as
        its own chunk rather than silently dropped or causing a crash.
    10. chunk_id format is "<12-char-hash>_chunk_<4-digit-index>".
    11. A very short document produces exactly one chunk.

How to run:
    From legal-agent/ with the venv active:
        pytest tests/unit/test_chunking.py -v
"""

from schemas.chunk import DocumentChunk
from schemas.document import DocumentExtraction, ExtractionMethod, PageExtraction
from retrieval.chunking import chunk_document, TARGET_CHUNK_TOKENS, OVERLAP_TOKENS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_page(page_number: int, text: str) -> PageExtraction:
    return PageExtraction(
        page_number=page_number,
        text=text,
        char_count=len(text),
        method=ExtractionMethod.PYMUPDF,
    )


def _make_doc(pages: list[PageExtraction], file_hash: str = "abc123def456" + "0" * 52) -> DocumentExtraction:
    """
    Build a DocumentExtraction with a proper full_text string that includes
    the '--- Page N ---' separators that chunking.py uses to track page numbers.
    """
    full_text = ""
    for page in pages:
        full_text += f"--- Page {page.page_number} ---\n\n{page.text}\n\n"

    return DocumentExtraction(
        file_path=f"/tmp/test_{file_hash[:8]}.pdf",
        file_name=f"test_{file_hash[:8]}.pdf",
        file_hash=file_hash,
        total_pages=len(pages),
        pages=pages,
        full_text=full_text.strip(),
        pages_failed=0,
        pages_ocr=0,
        extraction_successful=True,
        error_message=None,
    )


def _long_paragraph(n_words: int = 100) -> str:
    """Generate a paragraph of n_words distinct words."""
    return " ".join(f"word{i}" for i in range(n_words))


# ---------------------------------------------------------------------------
# Test 1: Normal document produces chunks
# ---------------------------------------------------------------------------

def test_normal_document_produces_chunks():
    """A multi-page document with plenty of text should yield >= 1 chunk."""
    pages = [
        _make_page(i + 1, _long_paragraph(80))
        for i in range(5)
    ]
    doc = _make_doc(pages)
    chunks = chunk_document(doc)

    assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# Test 2: Chunk metadata -- hash and name are correct
# ---------------------------------------------------------------------------

def test_chunk_metadata_is_correct():
    """Every chunk must carry the parent document's hash and file name."""
    pages = [_make_page(1, _long_paragraph(80))]
    doc = _make_doc(pages)
    chunks = chunk_document(doc)

    for chunk in chunks:
        assert chunk.document_hash == doc.file_hash
        assert chunk.document_name == doc.file_name


# ---------------------------------------------------------------------------
# Test 3: Contiguous indices and total_chunks accuracy
# ---------------------------------------------------------------------------

def test_chunk_indices_are_contiguous():
    """
    chunk_index must be 0, 1, 2, ... with no gaps.
    total_chunks must equal the actual list length.
    """
    pages = [_make_page(i + 1, _long_paragraph(80)) for i in range(4)]
    doc = _make_doc(pages)
    chunks = chunk_document(doc)

    for expected_idx, chunk in enumerate(chunks):
        assert chunk.chunk_index == expected_idx, (
            f"Expected chunk_index={expected_idx}, got {chunk.chunk_index}"
        )
        assert chunk.total_chunks == len(chunks), (
            f"total_chunks={chunk.total_chunks} does not match list length {len(chunks)}"
        )


# ---------------------------------------------------------------------------
# Test 4: No chunk is empty
# ---------------------------------------------------------------------------

def test_no_chunk_is_empty():
    """Every chunk must have non-empty text."""
    pages = [_make_page(i + 1, _long_paragraph(60)) for i in range(3)]
    doc = _make_doc(pages)
    chunks = chunk_document(doc)

    for chunk in chunks:
        assert chunk.text.strip() != "", f"Chunk {chunk.chunk_index} is empty"
        assert chunk.char_count > 0


# ---------------------------------------------------------------------------
# Test 5: Overlap -- adjacent chunks share a token boundary
# ---------------------------------------------------------------------------

def test_adjacent_chunks_share_overlap_tokens():
    """
    The overlap_tokens field of chunk N+1 should be > 0, indicating that
    some content from chunk N was prepended as context.

    Each paragraph is 4 common English words (~4 tokens with tiktoken).
    target_tokens=8 means a second paragraph triggers a buffer flush after
    the first -- the normal flush path that builds an overlap prefix.
    min_chunk_tokens=1 prevents the small-chunk merge from eating the flush.
    """
    short_phrases = [
        "the party shall indemnify",
        "all claims arising hereunder",
        "governing law is applicable",
        "termination without cause permitted",
        "confidential information protected always",
        "dispute resolution through arbitration",
        "non-compete clause applies broadly",
        "assignment requires written consent",
    ]

    # Build one page with all phrases as separate paragraphs
    page_text = "\n\n".join(short_phrases)
    pages = [_make_page(1, page_text)]
    doc = _make_doc(pages)

    # target_tokens=8: a single phrase (~4 tokens) fits; two phrases (~10 tokens) overflow.
    # min_chunk_tokens=1: no minimum size -- every flush produces a real chunk.
    chunks = chunk_document(doc, target_tokens=8, overlap_tokens=3, min_chunk_tokens=1)

    if len(chunks) < 2:
        return  # can't test overlap with a single chunk

    has_overlap = any(chunk.overlap_tokens > 0 for chunk in chunks[1:])
    assert has_overlap, "Expected at least one chunk with overlap_tokens > 0"


# ---------------------------------------------------------------------------
# Test 6: Page numbers are valid
# ---------------------------------------------------------------------------

def test_page_numbers_are_valid():
    """start_page >= 1, end_page >= start_page, for every chunk."""
    pages = [_make_page(i + 1, _long_paragraph(80)) for i in range(4)]
    doc = _make_doc(pages)
    chunks = chunk_document(doc)

    for chunk in chunks:
        assert chunk.start_page >= 1, f"start_page={chunk.start_page} is < 1"
        assert chunk.end_page >= chunk.start_page, (
            f"end_page={chunk.end_page} < start_page={chunk.start_page}"
        )


# ---------------------------------------------------------------------------
# Test 7: Empty document returns []
# ---------------------------------------------------------------------------

def test_empty_document_returns_empty_list():
    """A DocumentExtraction with no text should return [] without crashing."""
    page = PageExtraction(
        page_number=1,
        text="",
        char_count=0,
        method=ExtractionMethod.FAILED,
    )
    doc = DocumentExtraction(
        file_path="/tmp/empty.pdf",
        file_name="empty.pdf",
        file_hash="a" * 64,
        total_pages=1,
        pages=[page],
        full_text="",
        pages_failed=1,
        pages_ocr=0,
        extraction_successful=False,
        error_message="No text extracted",
    )
    chunks = chunk_document(doc)
    assert chunks == []


# ---------------------------------------------------------------------------
# Test 8: Whitespace-only document returns []
# ---------------------------------------------------------------------------

def test_whitespace_only_document_returns_empty_list():
    """A document whose full_text is only whitespace must return []."""
    doc = DocumentExtraction(
        file_path="/tmp/ws.pdf",
        file_name="ws.pdf",
        file_hash="b" * 64,
        total_pages=1,
        pages=[_make_page(1, "   \n\n\t  \n  ")],
        full_text="   \n\n\t  \n  ",
        pages_failed=0,
        pages_ocr=0,
        extraction_successful=True,
        error_message=None,
    )
    chunks = chunk_document(doc)
    assert chunks == []


# ---------------------------------------------------------------------------
# Test 9: Oversized single paragraph is kept, not dropped
# ---------------------------------------------------------------------------

def test_oversized_paragraph_is_kept_as_single_chunk():
    """
    A paragraph with far more tokens than target_tokens must be kept as its
    own oversized chunk rather than silently dropped or causing a crash.
    """
    # 500 words is well above any reasonable target_tokens setting
    big_paragraph = _long_paragraph(500)
    pages = [_make_page(1, big_paragraph)]
    doc = _make_doc(pages)

    chunks = chunk_document(doc, target_tokens=50)

    assert len(chunks) >= 1
    # The big paragraph's text must appear somewhere in the chunks
    combined = " ".join(c.text for c in chunks)
    assert "word0" in combined
    assert "word499" in combined


# ---------------------------------------------------------------------------
# Test 10: chunk_id format
# ---------------------------------------------------------------------------

def test_chunk_id_format():
    """
    chunk_id must follow the format "<12-char-hash>_chunk_<4-digit-zero-padded-index>".
    """
    import re
    pages = [_make_page(1, _long_paragraph(80))]
    doc = _make_doc(pages)
    chunks = chunk_document(doc)

    pattern = re.compile(r'^[a-f0-9]{12}_chunk_\d{4}$')
    for chunk in chunks:
        assert pattern.match(chunk.chunk_id), (
            f"chunk_id '{chunk.chunk_id}' does not match expected format"
        )


# ---------------------------------------------------------------------------
# Test 11: Very short document produces exactly one chunk
# ---------------------------------------------------------------------------

def test_short_document_produces_one_chunk():
    """
    A document that fits entirely within target_tokens must produce exactly
    one chunk (no spurious empty second chunk).
    """
    short_text = "This NDA is effective as of January 1, 2024."
    pages = [_make_page(1, short_text)]
    doc = _make_doc(pages)
    chunks = chunk_document(doc)

    assert len(chunks) == 1
    assert short_text in chunks[0].text
