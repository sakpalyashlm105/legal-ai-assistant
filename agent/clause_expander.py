"""
agent/clause_expander.py
------------------------
Clause boundary expansion for the Legal AI Assistant.

Problem this solves:
    The clause extractor anchors on whatever chunk(s) the LLM uses to satisfy
    the prompt.  For a Confidentiality clause written across two sections, e.g.:

        Section 1: "Protected Information Defined; Exclusions"  (definition only)
        Section 2: "Director's Obligations"                     (non-disclosure,
                                                                 non-use, security,
                                                                 survival language)

    the LLM may return only Section 1 as the extracted clause text.  When the
    comparator then measures Section 1 against the full Confidentiality template
    (which expects obligations + survival) it produces "major deviation" → HIGH
    risk.  But the obligations ARE present -- they are in the very next chunk.

Architecture:
    extract_clauses
        → expand_clause_boundaries  (this module, new node)
            → compare_to_templates  (receives expanded_text, not snippet)
                → flag_risks

Key design decisions:
    1. FORWARD-ONLY expansion: source chunk + subsequent related chunks.
       Clause extractors typically anchor on the start of a clause; looking
       backward would pull in the tail of the previous clause.
    2. Stop conditions (checked in priority order):
         a. Next chunk starts a numbered section heading that belongs to a
            DIFFERENT legal topic → stop (e.g. "3. Governing Law" after a
            Confidentiality clause)
         b. Next chunk has no section heading AND no related keywords → stop
         c. max_extra_chunks reached (default 3) → stop without error
         d. End of document
    3. "Related" detection is keyword-based (not an LLM call) to avoid adding
       extra latency and API cost to every clause in every document run.
    4. For Confidentiality / Non-Disclosure, definition sections, obligation
       sections, exclusions, and survival language are all treated as one
       logical clause group via the keyword table.

PII-safe: this module never logs clause text content, only chunk indices and
          boundary reasons.
"""

import logging
import re
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from schemas.chunk import DocumentChunk
from schemas.clause import ExtractedClause

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Related-keyword map (clause_type -> keywords present in related content)
#
# Purpose: given the text of a candidate chunk, decide whether it is about
# the same legal concept as the clause currently being expanded.
#
# Design note: terms are lowercase substrings. Using substrings rather than
# whole-word matches catches inflections ("terminat" catches "terminate",
# "termination", "terminated") without requiring NLTK or regex complexity.
# ---------------------------------------------------------------------------

CLAUSE_RELATED_KEYWORDS: Dict[str, List[str]] = {
    "Confidentiality / Non-Disclosure": [
        "confidential",
        "nondisclosure",
        "non-disclosure",
        "protected information",
        "proprietary",
        "trade secret",
        "disclose",
        "disclosure",
        "non-use",
        "safeguard",
        "precaution",
        "survival",
        "survive",
        "exclusion",
        "excluded",
        "director",
        "employee",
        "representative",
    ],
    "Indemnification": [
        "indemnif",
        "defend",
        "hold harmless",
        "losses",
        "damages",
        "liabilities",
        "claims",
        "expenses",
        "costs",
        "third party",
        "third-party",
    ],
    "Limitation of Liability": [
        "limitation",
        "limit",
        "liability",
        "liable",
        "consequential",
        "indirect",
        "incidental",
        "punitive",
        "special",
        "damages",
        "cap",
        "ceiling",
        "in no event",
    ],
    "Governing Law / Jurisdiction": [
        "governing law",
        "jurisdiction",
        "govern",
        "construed",
        "applicable law",
        "choice of law",
        "conflict of law",
        "venue",
        "forum",
        "courts",
        "state of",
        "laws of",
    ],
    "Termination for Convenience": [
        "terminat",
        "terminate",
        "termination",
        "convenience",
        "written notice",
        "notice period",
        "days notice",
        "upon notice",
        "without cause",
    ],
    "Termination for Cause": [
        "terminat",
        "terminate",
        "termination",
        "cause",
        "breach",
        "default",
        "material breach",
        "cure period",
        "notice and cure",
    ],
    "Dispute Resolution": [
        "dispute",
        "arbitration",
        "arbitrat",
        "mediation",
        "mediator",
        "adr",
        "alternative dispute",
        "resolution",
        "settlement",
    ],
    "Renewal / Term": [
        "renew",
        "renewal",
        "term",
        "duration",
        "initial term",
        "expire",
        "expiration",
        "extend",
        "extension",
        "automatically renew",
    ],
    "Non-Compete / Non-Solicitation": [
        "non-compete",
        "noncompete",
        "non-solicitation",
        "nonsolicitation",
        "restrict",
        "restriction",
        "competitive",
        "compete",
        "solicitation",
        "solicit",
        "hire",
        "recruit",
    ],
    "Assignment": [
        "assign",
        "assignment",
        "transfer",
        "convey",
        "successors",
        "delegates",
        "binding upon",
    ],
}

# ---------------------------------------------------------------------------
# Heading detection
#
# Matches the first non-blank line if it is a numbered section heading such as:
#     "1. Protected Information Defined"
#     "2) Director's Obligations"
#     "Section 3 Term"
#     "SECTION 3. Term"
#     "Article IV  Indemnification"
# ---------------------------------------------------------------------------

_NEW_SECTION_RE = re.compile(
    r"^(?:\d+[\.\)]\s+\S"                        # 1. Heading / 2) Heading
    r"|Section\s+\d+[\.\s]\s*\S"                 # Section 3. Heading
    r"|SECTION\s+\d+[\.\s]\s*\S"                 # SECTION 3. Heading
    r"|Article\s+[IVXivx\d]+[\.\s]\s*\S"         # Article IV Heading
    r"|ARTICLE\s+[IVXivx\d]+[\.\s]\s*\S)",       # ARTICLE IV Heading
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Result model (Pydantic so orchestrator can serialize to state)
# ---------------------------------------------------------------------------

class ClauseExpansionResult(BaseModel):
    """
    Outcome of expanding a clause's extraction boundary beyond its source chunk.

    Fields
    ------
    clause_type : str
        The clause category (e.g. "Confidentiality / Non-Disclosure").

    original_text : str
        The clause text as originally extracted (= ExtractedClause.extracted_text).
        Never modified by this module.

    expanded_text : str
        Merged text of all chunks determined to belong to the same logical
        clause group.  Equal to original_text when expansion_triggered=False.

    source_chunk_ids : List[str]
        Ordered list of chunk_ids included in expanded_text.
        Length > 1 when expansion_triggered=True.

    expansion_triggered : bool
        True if at least one additional chunk was included beyond the source.

    boundary_reason : str
        Why expansion stopped.  One of:
          "absent_or_no_source"   -- clause is absent or has no source_chunk_id
          "source_not_found"      -- source_chunk_id not present in chunks list
          "new_unrelated_heading" -- next chunk opens a different legal topic
          "unrelated_content"     -- no heading, no related keywords
          "max_chunks_reached"    -- hit the max_extra_chunks ceiling
          "end_of_document"       -- no more chunks after the source chunk
          "single_chunk_only"     -- document has only one chunk

    pages_used : List[int]
        Sorted, deduplicated page numbers covered by expanded_text.
    """

    clause_type: str
    original_text: str
    expanded_text: str
    source_chunk_ids: List[str] = Field(default_factory=list)
    expansion_triggered: bool = False
    boundary_reason: str = "end_of_document"
    pages_used: List[int] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _first_nonblank_line(text: str) -> str:
    """Return the first non-blank line of text, stripped."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _starts_new_section(text: str) -> Optional[str]:
    """
    If the first non-blank line of text matches a numbered section heading
    pattern, return that line.  Otherwise return None.

    We check only the first non-blank line because headings in the middle of
    a chunk are continuations, not section boundaries.
    """
    first = _first_nonblank_line(text)
    if first and _NEW_SECTION_RE.match(first):
        return first
    return None


def _heading_belongs_to_clause_type(heading_line: str) -> Optional[str]:
    """
    Given a heading line, return the clause_type whose keywords best match
    the heading text, or None if no clause type matches.

    Matching is case-insensitive substring on the heading text.
    Returns the FIRST match found (order matches CLAUSE_RELATED_KEYWORDS dict).
    """
    heading_lower = heading_line.lower()
    for clause_type, keywords in CLAUSE_RELATED_KEYWORDS.items():
        if any(kw in heading_lower for kw in keywords):
            return clause_type
    return None


def _is_related_content(text: str, clause_type: str) -> bool:
    """
    Return True if text contains keywords associated with clause_type.

    Uses case-insensitive substring matching across the full chunk text
    (not just the heading), so continuation paragraphs without numbered
    headings are still recognized as related.
    """
    keywords = CLAUSE_RELATED_KEYWORDS.get(clause_type, [])
    if not keywords:
        return False
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def expand_clause_context(
    clause: ExtractedClause,
    chunks: List[DocumentChunk],
    max_extra_chunks: int = 3,
) -> ClauseExpansionResult:
    """
    Expand a clause's text boundary to include adjacent related chunks.

    When a clause extractor anchors on the first section of a multi-section
    legal clause, this function extends the text forward through subsequent
    chunks that belong to the same logical clause group.  The expanded text
    is then used for template comparison instead of the original snippet.

    Parameters
    ----------
    clause : ExtractedClause
        The clause to expand.  Returns immediately (no expansion) when
        is_present=False or source_chunk_id is None.
    chunks : List[DocumentChunk]
        All chunks from the document, in any order (sorted internally by
        chunk_index).
    max_extra_chunks : int
        Maximum number of additional chunks to append (default 3).
        Prevents unbounded expansion on long, topically consistent documents.

    Returns
    -------
    ClauseExpansionResult
        Always succeeds -- never raises.  When expansion is not triggered,
        expanded_text == original_text.
    """
    original_text = clause.extracted_text or ""

    # ── Absent / no-source clause ────────────────────────────────────────────
    if not clause.is_present or not clause.source_chunk_id:
        return ClauseExpansionResult(
            clause_type=clause.clause_type,
            original_text=original_text,
            expanded_text=original_text,
            source_chunk_ids=[clause.source_chunk_id] if clause.source_chunk_id else [],
            expansion_triggered=False,
            boundary_reason="absent_or_no_source",
            pages_used=[clause.page_reference] if clause.page_reference else [],
        )

    # ── Sort chunks by position ───────────────────────────────────────────────
    sorted_chunks = sorted(chunks, key=lambda c: c.chunk_index)

    # ── Find source chunk ─────────────────────────────────────────────────────
    source_chunk = next(
        (c for c in sorted_chunks if c.chunk_id == clause.source_chunk_id), None
    )
    if source_chunk is None:
        logger.warning(
            "expand_clause_context: source_chunk_id not found in %d chunks for '%s'",
            len(chunks),
            clause.clause_type,
        )
        return ClauseExpansionResult(
            clause_type=clause.clause_type,
            original_text=original_text,
            expanded_text=original_text,
            source_chunk_ids=[clause.source_chunk_id],
            expansion_triggered=False,
            boundary_reason="source_not_found",
            pages_used=[clause.page_reference] if clause.page_reference else [],
        )

    # ── Gather subsequent chunks ──────────────────────────────────────────────
    subsequent = [c for c in sorted_chunks if c.chunk_index > source_chunk.chunk_index]

    if not subsequent:
        return ClauseExpansionResult(
            clause_type=clause.clause_type,
            original_text=original_text,
            expanded_text=original_text,
            source_chunk_ids=[source_chunk.chunk_id],
            expansion_triggered=False,
            boundary_reason="end_of_document",
            pages_used=list(range(source_chunk.start_page, source_chunk.end_page + 1)),
        )

    # ── Walk forward through subsequent chunks ────────────────────────────────
    included_ids: List[str] = [source_chunk.chunk_id]
    included_pages: List[int] = list(range(source_chunk.start_page, source_chunk.end_page + 1))
    extra_count = 0
    boundary_reason = "end_of_document"

    for next_chunk in subsequent:
        if extra_count >= max_extra_chunks:
            boundary_reason = "max_chunks_reached"
            break

        heading = _starts_new_section(next_chunk.text)

        if heading:
            # ── New numbered section: check its topic ─────────────────────────
            matched_type = _heading_belongs_to_clause_type(heading)
            if matched_type and matched_type != clause.clause_type:
                # Different legal topic (e.g. "Governing Law" after Confidentiality)
                boundary_reason = "new_unrelated_heading"
                logger.debug(
                    "expand_clause_context: stop at chunk %d — heading '%s' "
                    "belongs to '%s', not '%s'",
                    next_chunk.chunk_index,
                    heading[:60],
                    matched_type,
                    clause.clause_type,
                )
                break
            # Same topic or unrecognised heading — include this chunk
            included_ids.append(next_chunk.chunk_id)
            included_pages.extend(range(next_chunk.start_page, next_chunk.end_page + 1))
            extra_count += 1
            logger.debug(
                "expand_clause_context: include chunk %d — heading '%s' same topic",
                next_chunk.chunk_index,
                heading[:60],
            )
        else:
            # ── No heading: decide by keyword content ─────────────────────────
            if _is_related_content(next_chunk.text, clause.clause_type):
                included_ids.append(next_chunk.chunk_id)
                included_pages.extend(range(next_chunk.start_page, next_chunk.end_page + 1))
                extra_count += 1
                logger.debug(
                    "expand_clause_context: include chunk %d — related content, no heading",
                    next_chunk.chunk_index,
                )
            else:
                boundary_reason = "unrelated_content"
                break

    expansion_triggered = len(included_ids) > 1

    # ── Build expanded text ───────────────────────────────────────────────────
    if expansion_triggered:
        chunk_by_id = {c.chunk_id: c for c in chunks}
        in_order = [
            chunk_by_id[cid]
            for cid in included_ids
            if cid in chunk_by_id
        ]
        expanded_text = "\n\n".join(c.text for c in in_order)
    else:
        expanded_text = original_text

    logger.info(
        "expand_clause_context: '%s' chunks 1→%d triggered=%s reason=%s",
        clause.clause_type,
        len(included_ids),
        expansion_triggered,
        boundary_reason,
    )

    return ClauseExpansionResult(
        clause_type=clause.clause_type,
        original_text=original_text,
        expanded_text=expanded_text,
        source_chunk_ids=included_ids,
        expansion_triggered=expansion_triggered,
        boundary_reason=boundary_reason,
        pages_used=sorted(set(included_pages)),
    )


def expand_all_clauses(
    clauses: List[ExtractedClause],
    chunks: List[DocumentChunk],
    max_extra_chunks: int = 3,
) -> Dict[str, ClauseExpansionResult]:
    """
    Run expand_clause_context() on every clause (present or absent).

    Returns
    -------
    dict mapping clause_type str → ClauseExpansionResult.
    Absent clauses are included with expansion_triggered=False.
    """
    return {
        clause.clause_type: expand_clause_context(clause, chunks, max_extra_chunks)
        for clause in clauses
    }
