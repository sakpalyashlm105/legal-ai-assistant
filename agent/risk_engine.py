"""
agent/risk_engine.py
--------------------
Risk assessment node for the Legal AI Assistant.

What this file does:
    Takes the extraction results (list[ExtractedClause]) and the template
    comparison results (list[ClauseComparison]) and produces a final
    list[RiskFinding] -- one per clause category -- with a risk level of
    HIGH, MEDIUM, or LOW.

Risk assignment rules (locked graded design decisions from CLAUDE.md):

    1. Missing clause (is_present=False)
       -> Always HIGH, no exceptions, no precedent override.
          Absence of a legal protection is categorically more dangerous than
          a modified one, because there is literally no clause to enforce.

    2. Present clause, no template available
       -> LOW. We have no baseline to measure against, so we cannot call it
          risky. The reviewer can still inspect it manually.

    3. Present clause, deviation_severity="none"
       -> LOW. Matches the standard -- no action needed.

    4. Present clause, deviation_severity="minor"
       -> MEDIUM (base level), then check precedent log.
          If precedent found: downgrade to LOW + annotate.

    5. Present clause, deviation_severity="major"
       -> HIGH (base level), then check precedent log.
          If precedent found: downgrade to MEDIUM + annotate.

Precedent-aware override (locked graded design from CLAUDE.md):
    Before finalizing a non-standard-clause risk level, check the feedback
    log (data/processed/feedback_log.json) for previously-approved clauses
    of the same type. If found, downgrade one tier and record the precedent.
    This override NEVER applies to missing clauses.

Feedback log format:
    A JSON array of objects, each representing a human-approved clause:
    [
      {
        "clause_type": "Governing Law / Jurisdiction",
        "approved_text_fragment": "laws of the State of California",
        "approval_date": "2024-03-15",
        "document_hash": "abc123...",
        "note": "California jurisdiction approved by legal counsel."
      },
      ...
    ]
    Matching is fuzzy: if the extracted text contains the approved_text_fragment
    (case-insensitive substring), the precedent applies.

Dependencies:
    schemas/clause.py, schemas/risk.py, config.py
"""

import json
import logging
from pathlib import Path
from typing import Optional

from schemas.clause import ExtractedClause
from schemas.risk import ClauseComparison, RiskFinding

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feedback log path
# ---------------------------------------------------------------------------

FEEDBACK_LOG_PATH = Path(__file__).parent.parent / "data" / "processed" / "feedback_log.json"

# ---------------------------------------------------------------------------
# Feedback log loader (cached per process run)
# ---------------------------------------------------------------------------

_feedback_log: Optional[list[dict]] = None


def _load_feedback_log() -> list[dict]:
    """
    Load the feedback log from disk. Returns an empty list if the file does
    not exist yet (normal on first run -- no approvals have been recorded).

    Why cache?
        The feedback log is read once per document processed, potentially
        many times in a batch run. Loading from disk every time would be
        slow. The cache is process-scoped (not persisted), so changes to
        the log file during a run are not picked up -- acceptable for batch.
    """
    global _feedback_log
    if _feedback_log is not None:
        return _feedback_log

    if not FEEDBACK_LOG_PATH.exists():
        logger.info(
            "Feedback log not found at %s. Starting with no precedents.",
            FEEDBACK_LOG_PATH,
        )
        _feedback_log = []
        return _feedback_log

    try:
        with open(FEEDBACK_LOG_PATH, "r", encoding="utf-8") as f:
            _feedback_log = json.load(f)
        logger.info(
            "Loaded %d precedent entries from feedback log.", len(_feedback_log)
        )
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Failed to load feedback log: %s. Using empty log.", e)
        _feedback_log = []

    return _feedback_log


def _clear_feedback_cache() -> None:
    """Reset the in-process feedback log cache. Used in tests."""
    global _feedback_log
    _feedback_log = None


# ---------------------------------------------------------------------------
# Precedent lookup
# ---------------------------------------------------------------------------

def _find_precedent(
    clause_type: str,
    extracted_text: Optional[str],
) -> Optional[dict]:
    """
    Search the feedback log for a previously-approved clause of the same type
    whose approved_text_fragment appears in the extracted text.

    Returns the matching log entry dict, or None if no precedent found.

    Why substring matching?
        We can't require exact text equality -- legal drafters make small
        edits. The approved_text_fragment captures the KEY distinguishing
        phrase (e.g. "laws of the State of California"), not the full clause.
        If that phrase appears in the extracted text, the precedent applies.
    """
    if not extracted_text:
        return None

    log = _load_feedback_log()
    extracted_lower = extracted_text.lower()

    for entry in log:
        if entry.get("clause_type") != clause_type:
            continue
        fragment = entry.get("approved_text_fragment", "")
        if fragment and fragment.lower() in extracted_lower:
            return entry

    return None


# ---------------------------------------------------------------------------
# Risk level assignment (pure logic, no I/O)
# ---------------------------------------------------------------------------

def _assign_base_risk(
    clause: ExtractedClause,
    comparison: ClauseComparison,
) -> tuple[str, str]:
    """
    Assign the base risk level and reason for a single clause, BEFORE applying
    any precedent override.

    Returns (risk_level, reason) as strings.
    """
    # Rule 1: missing clause -> always HIGH
    if not clause.is_present:
        return (
            "HIGH",
            f"{clause.clause_type} clause is absent from the document.",
        )

    # Rule 2: no template available -> LOW (nothing to compare against)
    if not comparison.template_found:
        return (
            "LOW",
            f"{clause.clause_type} clause is present; no standard template "
            "available for comparison.",
        )

    severity = comparison.deviation_severity

    # Rule 3: no deviation -> LOW
    if severity == "none":
        return (
            "LOW",
            f"{clause.clause_type} clause matches the standard template.",
        )

    # Rule 4: minor deviation -> MEDIUM
    if severity == "minor":
        summary = comparison.deviation_summary or "minor deviation from template"
        return (
            "MEDIUM",
            f"{clause.clause_type} clause has a minor deviation: {summary}",
        )

    # Rule 5: major deviation -> HIGH
    summary = comparison.deviation_summary or "major deviation from template"
    return (
        "HIGH",
        f"{clause.clause_type} clause has a major deviation: {summary}",
    )


def _apply_precedent_downgrade(
    risk_level: str,
    clause: ExtractedClause,
    comparison: ClauseComparison,
) -> tuple[str, bool, Optional[str]]:
    """
    Apply the precedent-aware override if applicable.

    Returns (final_risk_level, precedent_applied, precedent_note).

    Downgrade table:
        HIGH   -> MEDIUM  (if precedent found)
        MEDIUM -> LOW     (if precedent found)
        LOW    -> LOW     (no downgrade needed)

    This is NEVER called for missing clauses -- the caller (flag_risks) skips
    this step when clause.is_present=False.
    """
    if risk_level == "LOW":
        return "LOW", False, None

    precedent = _find_precedent(clause.clause_type, clause.extracted_text)
    if precedent is None:
        return risk_level, False, None

    downgraded = "MEDIUM" if risk_level == "HIGH" else "LOW"
    note = (
        f"Previously approved on {precedent.get('approval_date', 'unknown date')}: "
        f"{precedent.get('note', 'no note recorded')} "
        f"(document hash: {precedent.get('document_hash', 'unknown')[:12]}...)"
    )
    logger.info(
        "Precedent override applied for %s: %s -> %s. %s",
        clause.clause_type, risk_level, downgraded, note,
    )
    return downgraded, True, note


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def flag_risks(
    clauses: list[ExtractedClause],
    comparisons: list[ClauseComparison],
) -> list[RiskFinding]:
    """
    Produce a RiskFinding for each clause category.

    How it works:
        1. Pair each ExtractedClause with its ClauseComparison (by position --
           both lists are always ordered by CLAUSE_CATEGORIES).
        2. Assign a base risk level using the 5-rule table above.
        3. For non-missing clauses: check the feedback log for a precedent and
           downgrade one tier if found.
        4. Build and return a RiskFinding for each pair.

    Parameters
    ----------
    clauses : list[ExtractedClause]
        The 10 ExtractedClause objects from extract_clauses(). Must be in
        CLAUSE_CATEGORIES order.
    comparisons : list[ClauseComparison]
        The 10 ClauseComparison objects from compare_to_templates(). Must be
        in the same order as clauses.

    Returns
    -------
    list[RiskFinding]
        Always exactly 10 findings, one per clause category.

    Raises
    ------
    ValueError
        If the two input lists have different lengths (programming error,
        not a runtime condition).
    """
    if len(clauses) != len(comparisons):
        raise ValueError(
            f"clauses ({len(clauses)}) and comparisons ({len(comparisons)}) "
            "must have the same length."
        )

    findings: list[RiskFinding] = []

    for clause, comparison in zip(clauses, comparisons):
        base_risk, reason = _assign_base_risk(clause, comparison)

        # Precedent override is skipped for missing clauses (locked rule)
        if clause.is_present:
            final_risk, precedent_applied, precedent_note = _apply_precedent_downgrade(
                base_risk, clause, comparison
            )
        else:
            final_risk = base_risk
            precedent_applied = False
            precedent_note = None

        findings.append(RiskFinding(
            clause_type=clause.clause_type,
            risk_level=final_risk,
            reason=reason,
            is_missing=not clause.is_present,
            deviation_summary=comparison.deviation_summary,
            precedent_applied=precedent_applied,
            precedent_note=precedent_note,
            source_clause=clause if clause.is_present else None,
        ))

        logger.info(
            "Risk: %s | %s | precedent=%s",
            final_risk, clause.clause_type, precedent_applied,
        )

    high_count = sum(1 for f in findings if f.risk_level == "HIGH")
    missing_count = sum(1 for f in findings if f.is_missing)
    logger.info(
        "Risk summary: %d HIGH (%d missing), %d MEDIUM, %d LOW",
        high_count,
        missing_count,
        sum(1 for f in findings if f.risk_level == "MEDIUM"),
        sum(1 for f in findings if f.risk_level == "LOW"),
    )

    return findings
