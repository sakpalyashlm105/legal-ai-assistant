"""
agent/missing_clause_verifier.py
----------------------------------
Stricter absence verification: before finalising a "missing clause" finding,
search all document chunks for synonym terms / related headings belonging to
that clause category.

This is Step 15 in the pipeline — runs AFTER evidence verification on present
clauses and BEFORE risk scoring (flag_risks).

Why this matters:
    Legal documents frequently use non-standard headings.  A Limitation of
    Liability clause may appear under "Caps on Damages" or "Maximum Exposure";
    a Governing Law clause may appear under "Applicable Regulations".  The
    extractor can miss these if the chunk it retrieved didn't contain the
    heading phrasing it expected.  One failed extraction pass → clause marked
    absent → HIGH risk → potentially spurious escalation.

What this module does:
    For each clause that is_present=False, it searches all document chunks for
    keywords from the CLAUSE_RELATED_KEYWORDS table in clause_expander.py.
    The keyword table is NOT duplicated here — it is imported directly.

    If keyword evidence IS found:
        → Escalate to HITL with possible_clause_under_different_heading=True.
          The human confirms whether the clause is truly absent or just headed
          differently.  The absence finding stays HIGH until a human overrides it.
          REG-001 is unaffected — absence = HIGH regardless of keyword presence.

    If no keyword evidence is found:
        → Clause stays genuinely absent, HIGH, as today.  No change.

Deliberate non-action:
    This module NEVER auto-extracts text for an absent clause and NEVER auto-
    marks it as present.  If you find keywords, escalate to a human.  The human
    decides, not the system.

PII-safe: keyword match positions are not logged.  Only match count and
          clause category are logged.
"""

import logging
from typing import Dict, List, Optional

from schemas.clause import ExtractedClause
from schemas.chunk import DocumentChunk
from agent.clause_expander import CLAUSE_RELATED_KEYWORDS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public: check one absent clause against all chunks
# ---------------------------------------------------------------------------

def has_related_keywords(
    clause_type: str,
    chunks: List[DocumentChunk],
) -> tuple[bool, int]:
    """
    Search all chunk texts for keywords related to an absent clause type.

    Parameters
    ----------
    clause_type : str
        One of the 10 approved clause categories.
    chunks : list of DocumentChunk
        All chunks from the current document.

    Returns
    -------
    (found: bool, match_count: int)
        found       -- True if at least one keyword was found anywhere
        match_count -- total keyword occurrences across all chunks
    """
    keywords = CLAUSE_RELATED_KEYWORDS.get(clause_type, [])
    if not keywords:
        return False, 0

    total_matches = 0
    for chunk in chunks:
        text_lower = (chunk.text or "").lower()
        for kw in keywords:
            if kw in text_lower:
                total_matches += 1

    return total_matches > 0, total_matches


# ---------------------------------------------------------------------------
# Public: batch check for all absent clauses
# ---------------------------------------------------------------------------

def verify_missing_clauses(
    clauses: List[ExtractedClause],
    chunks: List[DocumentChunk],
) -> List[dict]:
    """
    For every absent clause, check whether related keyword evidence exists.

    Parameters
    ----------
    clauses : list of ExtractedClause
        All extracted clauses (present and absent).
    chunks  : list of DocumentChunk
        All document chunks.

    Returns
    -------
    List of escalation dicts, one per absent clause where keyword evidence WAS
    found.  Each dict has:
        clause_type                             : str
        possible_clause_under_different_heading : True
        keyword_match_count                     : int
        note                                    : str (human-readable summary)

    Absent clauses with NO keyword evidence produce no entry — they remain
    genuinely absent and the caller does not need to do anything extra.
    """
    escalations = []
    absent_clauses = [c for c in clauses if not c.is_present]

    for clause in absent_clauses:
        found, count = has_related_keywords(clause.clause_type, chunks)
        if found:
            logger.info(
                "verify_missing_clauses: '%s' absent but %d related keyword hits found — "
                "escalating to HITL for human confirmation",
                clause.clause_type, count,
            )
            escalations.append({
                "clause_type": clause.clause_type,
                "possible_clause_under_different_heading": True,
                "keyword_match_count": count,
                "note": (
                    f"'{clause.clause_type}' was not extracted, but related "
                    f"keyword terms were found {count} time(s) across document chunks. "
                    f"This clause may appear under a non-standard heading. "
                    f"A human reviewer should confirm whether this clause is truly "
                    f"absent before the HIGH risk finding is finalised."
                ),
            })
        else:
            logger.debug(
                "verify_missing_clauses: '%s' absent, no keyword evidence — genuinely missing",
                clause.clause_type,
            )

    return escalations
