"""
retrieval/chunking.py
---------------------
Paragraph-aware document chunker for the Legal AI Assistant.

What this file does:
    Takes a DocumentExtraction (the full text of a PDF, already extracted
    by pdf_parser.py) and splits it into smaller, overlapping text windows
    called "chunks". Each chunk is stored as a DocumentChunk (schemas/chunk.py)
    and is ready to be embedded and inserted into the FAISS vector index.

Why do we chunk at all?
    Embedding models have an input token limit (8192 tokens for
    text-embedding-3-small). A 50-page contract can easily be 40,000+ tokens.
    We can't embed the whole document at once. Instead, we embed each chunk
    separately, store all of them in the vector index, and at query time we
    find the specific chunk(s) most relevant to each question.

    Think of it like an index at the back of a textbook: instead of reading
    the whole book to find "indemnification clauses", you look up the index
    and go straight to pages 31-33. The vector index is that index, and chunks
    are the individual index entries.

Why paragraph-aware splitting?
    A naive fixed-size splitter (every 500 tokens, cut hard) would frequently
    split a clause right in the middle of a sentence, losing context. Legal
    text is organized into paragraphs and numbered clauses. Splitting at
    paragraph boundaries keeps each chunk semantically coherent -- a full
    clause stays together rather than being split across two chunks.

Why overlap?
    Even with paragraph-aware splitting, important context sometimes sits right
    at the boundary. Overlapping consecutive chunks by ~50 tokens ensures that
    a sentence at the end of chunk N also appears at the start of chunk N+1.
    This means retrieval for either half of that sentence will find a chunk
    where the sentence is in full context, not truncated.

Dependencies:
    tiktoken (for token counting), schemas/document.py, schemas/chunk.py
"""

import logging
import re
from typing import Optional

from schemas.chunk import DocumentChunk
from schemas.document import DocumentExtraction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Target number of tokens per chunk. Chosen to be well within
# text-embedding-3-small's 8192-token limit while being large enough to
# contain a full legal clause with context.
TARGET_CHUNK_TOKENS = 500

# How many tokens from the end of chunk N to prepend to the start of chunk N+1.
# This "overlap window" ensures clause boundaries never lose their surrounding
# context entirely.
OVERLAP_TOKENS = 50

# Minimum meaningful chunk size. Chunks smaller than this (e.g., a lone header
# line like "SECTION 4. GOVERNING LAW") are merged into the next chunk rather
# than stored as their own entry in the vector index.
MIN_CHUNK_TOKENS = 50

# tiktoken encoding to use for token counting.
# "cl100k_base" is the encoding used by GPT-4 and text-embedding-3-small.
TIKTOKEN_ENCODING = "cl100k_base"


# ---------------------------------------------------------------------------
# Helper: lazy tiktoken loader
# ---------------------------------------------------------------------------

_encoder = None


def _get_encoder():
    """
    Load the tiktoken encoder once and cache it.

    Why lazy? tiktoken adds ~100ms on first import. Loading it at module
    level would slow down every import of chunking.py, even in tests where
    we don't need it.
    """
    global _encoder
    if _encoder is None:
        try:
            import tiktoken
            _encoder = tiktoken.get_encoding(TIKTOKEN_ENCODING)
        except ImportError:
            logger.warning(
                "tiktoken not installed -- token counts will be estimated "
                "from character count (divide by 4). Install with: pip install tiktoken"
            )
            _encoder = None
    return _encoder


def _count_tokens(text: str) -> int:
    """
    Count tokens in a string using tiktoken, or estimate if unavailable.

    Why estimate at 4 chars/token?
        English text averages ~4 characters per token in the cl100k_base
        encoding. This is a rough approximation -- legal text with long
        Latin phrases may be slightly different -- but it's close enough
        for chunking decisions when tiktoken is unavailable.
    """
    enc = _get_encoder()
    if enc is not None:
        return len(enc.encode(text))
    return max(1, len(text) // 4)


def _encode_tokens(text: str) -> list:
    """Return raw token IDs for a string (needed for precise overlap slicing)."""
    enc = _get_encoder()
    if enc is not None:
        return enc.encode(text)
    # Fallback: approximate with character-level "tokens" (each 4 chars = 1 token)
    chars = list(text)
    return [chars[i:i+4] for i in range(0, len(chars), 4)]


def _decode_tokens(tokens: list) -> str:
    """Convert token IDs back to a string."""
    enc = _get_encoder()
    if enc is not None:
        return enc.decode(tokens)
    # Fallback: rejoin character groups
    return "".join("".join(t) if isinstance(t, list) else t for t in tokens)


# ---------------------------------------------------------------------------
# Helper: split text into paragraphs
# ---------------------------------------------------------------------------

def _split_into_paragraphs(text: str) -> list[str]:
    """
    Split full document text into a list of paragraph strings.

    What counts as a paragraph boundary?
        - Two or more consecutive newline characters (\n\n, \n\n\n, etc.)
        - The "--- Page N ---" separator lines inserted by pdf_parser.py

    Each returned string is one paragraph (may be multiple lines, but is
    separated from others by a blank line). Empty strings are discarded.

    Why keep page separators as their own "paragraph"?
        They act as structural anchors. When we build chunks, encountering
        a page separator tells us we've crossed a page boundary, so we can
        update the `current_page` tracker accurately.
    """
    # Split on blank lines (2+ newlines)
    raw_parts = re.split(r'\n{2,}', text)

    paragraphs = []
    for part in raw_parts:
        stripped = part.strip()
        if stripped:
            paragraphs.append(stripped)

    return paragraphs


def _parse_page_number_from_separator(text: str) -> Optional[int]:
    """
    If the text is a pdf_parser.py page separator ("--- Page N ---"),
    return N as an int. Otherwise return None.
    """
    match = re.match(r'^---\s*Page\s+(\d+)\s*---$', text.strip())
    if match:
        return int(match.group(1))
    return None


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def chunk_document(
    doc: DocumentExtraction,
    target_tokens: int = TARGET_CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    min_chunk_tokens: int = MIN_CHUNK_TOKENS,
) -> list[DocumentChunk]:
    """
    Split a DocumentExtraction into a list of DocumentChunk objects,
    ready for embedding and insertion into the FAISS vector index.

    How it works, step by step:
        1. Split the full document text into paragraphs (at blank lines and
           page separators).
        2. Walk through the paragraphs, accumulating them into a "current
           chunk buffer" until adding the next paragraph would exceed
           target_tokens.
        3. When the buffer is full, flush it as a DocumentChunk, prepend
           the last overlap_tokens of that chunk to the next buffer, and
           continue.
        4. At the end, flush whatever remains in the buffer as the final
           chunk (even if it's smaller than target_tokens).
        5. Any chunk smaller than min_chunk_tokens is merged into the
           previous one rather than stored alone.

    Parameters
    ----------
    doc : DocumentExtraction
        The full extraction result from pdf_parser.py.

    target_tokens : int
        Soft maximum token count per chunk. A single paragraph that is
        larger than this limit is NOT split further (splitting mid-paragraph
        would break clause continuity). It is kept as its own oversized
        chunk and a warning is logged.

    overlap_tokens : int
        Number of tokens from the end of the previous chunk to prepend
        to the start of the next chunk.

    min_chunk_tokens : int
        Minimum token count for a chunk to be stored independently.
        Chunks below this are merged into the adjacent chunk.

    Returns
    -------
    list[DocumentChunk]
        Ordered list of chunks (chunk_index 0, 1, 2, ...). Returns an
        empty list if the document has no extractable text.
    """
    if not doc.full_text or not doc.full_text.strip():
        logger.warning(f"Document '{doc.file_name}' has no text to chunk.")
        return []

    paragraphs = _split_into_paragraphs(doc.full_text)
    if not paragraphs:
        logger.warning(f"Document '{doc.file_name}' produced no paragraphs after splitting.")
        return []

    # Collect raw chunk data as dicts first; DocumentChunk objects are built
    # at the end once total_chunks is known (avoids a ge=1 violation on the
    # total_chunks placeholder during accumulation).
    raw_chunks: list[dict] = []
    buffer_paragraphs: list[str] = []
    buffer_tokens: int = 0
    current_page: int = 1  # tracks which page we're currently reading
    chunk_start_page: int = 1
    overlap_prefix: str = ""  # tail of the previous chunk, prepended for context

    def _flush_buffer(is_final: bool = False) -> None:
        nonlocal buffer_paragraphs, buffer_tokens, chunk_start_page, overlap_prefix

        if not buffer_paragraphs:
            return

        chunk_text = "\n\n".join(buffer_paragraphs)
        token_count = _count_tokens(chunk_text)

        # Merge tiny trailing chunks into the previous one
        if not is_final and token_count < min_chunk_tokens and raw_chunks:
            prev = raw_chunks[-1]
            merged_text = prev["text"] + "\n\n" + chunk_text
            prev["text"] = merged_text
            prev["char_count"] = len(merged_text)
            prev["token_count"] = _count_tokens(merged_text)
            prev["end_page"] = current_page
            buffer_paragraphs = []
            buffer_tokens = 0
            overlap_prefix = ""
            return

        chunk_index = len(raw_chunks)
        chunk_id = f"{doc.file_hash[:12]}_chunk_{chunk_index:04d}"

        raw_chunks.append({
            "chunk_id": chunk_id,
            "document_hash": doc.file_hash,
            "document_name": doc.file_name,
            "text": chunk_text,
            "char_count": len(chunk_text),
            "token_count": token_count,
            "start_page": chunk_start_page,
            "end_page": current_page,
            "chunk_index": chunk_index,
            "overlap_tokens": _count_tokens(overlap_prefix) if overlap_prefix else 0,
        })

        # Build the overlap prefix for the next chunk: last overlap_tokens tokens
        all_tokens = _encode_tokens(chunk_text)
        overlap_slice = all_tokens[-overlap_tokens:] if len(all_tokens) > overlap_tokens else all_tokens
        overlap_prefix = _decode_tokens(overlap_slice)

        buffer_paragraphs = []
        buffer_tokens = 0
        chunk_start_page = current_page

    for para in paragraphs:
        # Detect page separators and update the current page tracker
        page_num = _parse_page_number_from_separator(para)
        if page_num is not None:
            current_page = page_num
            # Don't add the separator line itself as content
            continue

        para_tokens = _count_tokens(para)

        # A single paragraph that exceeds the target is kept as its own chunk
        if para_tokens > target_tokens:
            logger.warning(
                f"Document '{doc.file_name}': paragraph of {para_tokens} tokens "
                f"exceeds target_tokens={target_tokens}. Kept as oversized chunk."
            )
            # Flush whatever is in the buffer first
            _flush_buffer()
            # Then flush this oversized paragraph alone
            buffer_paragraphs = [para]
            buffer_tokens = para_tokens
            _flush_buffer()
            overlap_prefix = ""
            continue

        # Would adding this paragraph overflow the buffer?
        projected = buffer_tokens + para_tokens + (2 if buffer_paragraphs else 0)
        if buffer_paragraphs and projected > target_tokens:
            _flush_buffer()
            # Start the new buffer with the overlap prefix
            if overlap_prefix:
                buffer_paragraphs = [overlap_prefix, para]
                buffer_tokens = _count_tokens(overlap_prefix) + para_tokens
            else:
                buffer_paragraphs = [para]
                buffer_tokens = para_tokens
            chunk_start_page = current_page
        else:
            buffer_paragraphs.append(para)
            buffer_tokens += para_tokens + (2 if len(buffer_paragraphs) > 1 else 0)

    # Flush whatever remains
    _flush_buffer(is_final=True)

    # Build DocumentChunk objects now that total_chunks is known
    total = len(raw_chunks)
    chunks = [
        DocumentChunk(**{**raw, "total_chunks": total})
        for raw in raw_chunks
    ]

    logger.info(
        f"Chunked '{doc.file_name}': {doc.total_pages} pages -> "
        f"{total} chunks (target={target_tokens} tokens, overlap={overlap_tokens} tokens)"
    )

    return chunks
