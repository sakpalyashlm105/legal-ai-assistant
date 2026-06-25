"""
schemas/chunk.py
----------------
Pydantic data model for a single document chunk.

What is a "chunk"?
    After pdf_parser.py extracts full-page text from a PDF, chunking.py
    breaks that text into smaller, overlapping pieces called "chunks".
    Each chunk is a window of text -- typically 400-600 tokens -- that
    fits comfortably inside an embedding model's input limit and is small
    enough for an LLM to read in a single focused context window.

    Think of it like dividing a 50-page contract into 200 sticky-note
    summaries, each covering one logical section, with a little overlap
    at the edges so no sentence is split across two notes with no context.

Why do we need a schema for chunks?
    The vector store, the retrieval module, the clause extractor, and the
    report generator all need to know what a chunk contains -- not just
    the text, but which document it came from, which pages it spans, and
    where it sits in the document. A Pydantic model enforces that contract
    everywhere, the same way PageExtraction does for individual pages.

Dependencies:
    pydantic
"""

from typing import Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """
    A single chunk of text produced by chunking.py from a DocumentExtraction.

    Fields
    ------
    chunk_id : str
        Unique identifier for this chunk within its parent document.
        Format: "<file_hash>_chunk_<index>" -- e.g. "b8fc19_chunk_003".
        Using the file hash in the ID means two identical documents always
        produce the same chunk IDs, which is important for deduplication
        and caching.

    document_hash : str
        SHA-256 hash of the source PDF file. Ties this chunk back to the
        exact document it came from, even if the file is renamed.

    document_name : str
        Human-readable file name (e.g. "NDA_2023.pdf"). Used in citations
        and UI display -- never for logic (use document_hash for that).

    text : str
        The actual text content of this chunk. This is what gets embedded
        and stored in the vector index, and what the LLM reads when
        answering questions about this section.

    char_count : int
        Number of characters in the text. Quick sanity check; the embedding
        model's token count may differ (see token_count below).

    token_count : Optional[int]
        Approximate token count, computed by tiktoken if available. Stored
        so the retrieval layer can enforce token budgets without re-counting.
        None if tiktoken was not used during chunking.

    start_page : int
        1-based page number where this chunk begins. Used to build page
        citations ("See page 3 of the agreement.") in reports.

    end_page : int
        1-based page number where this chunk ends. May equal start_page
        for short documents or small chunk sizes.

    chunk_index : int
        Zero-based position of this chunk in the ordered list for its
        document. Used when you need to retrieve the chunk immediately
        before or after this one (adjacent context retrieval).

    total_chunks : int
        Total number of chunks in the parent document. Together with
        chunk_index, gives a sense of where in the document this chunk
        sits: "chunk 5 of 47" is near the beginning; "chunk 46 of 47"
        is at the end.

    overlap_tokens : int
        How many tokens of context from the previous chunk were prepended
        to this one. Stored for reference -- retrieval uses this to know
        that the first `overlap_tokens` worth of text is "shared" with the
        previous chunk and should not be double-counted in results.
    """

    chunk_id: str
    document_hash: str
    document_name: str
    text: str
    char_count: int = Field(ge=0)
    token_count: Optional[int] = None
    start_page: int = Field(ge=1)
    end_page: int = Field(ge=1)
    chunk_index: int = Field(ge=0)
    total_chunks: int = Field(ge=1)
    overlap_tokens: int = Field(default=0, ge=0)
