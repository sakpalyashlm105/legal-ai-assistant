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
    log (data/feedback/feedback_log.jsonl) for previously-approved MEDIUM
    clauses of the same type. If found and text similarity >= 0.70, downgrade
    one tier and record the precedent.
    This override NEVER applies to missing clauses (REG-001 invariant).

REG-001 invariant (enforced at two independent levels):
    1. flag_risks() never calls _apply_precedent_downgrade() when
       clause.is_present == False.
    2. _apply_precedent_downgrade() independently checks clause.is_present
       and returns immediately without touching the log if False.
    Either guard alone is sufficient; both together are defense-in-depth.

Feedback log format (Stage 5 — JSONL):
    data/feedback/feedback_log.jsonl — one JSON object per line, each a
    full FeedbackRecord. Only records meeting ALL of:
        - approved_for_precedent == True
        - feedback_status == "approved_precedent"
        - final_risk == "MEDIUM"
    are loaded as active precedents. Everything else is read-and-skipped.

Matching (Stage 5 — windowed difflib):
    Reuses _best_window_score() from guardrails/evidence_verifier.py and
    the FUZZY_MATCH_THRESHOLD = 0.70 constant. The evidence_excerpt stored
    in each FeedbackRecord is the needle; the current clause's extracted_text
    is the haystack. Compatibility checks:
        - clause_category: exact match required.
        - document_type: null-compatible (either side None -> passes).
        - jurisdiction / template_version: always null-compatible (current
          document context is not available at flag_risks() call time).

Dependencies:
    schemas/clause.py, schemas/feedback.py, schemas/risk.py,
    guardrails/evidence_verifier.py, extraction/pdf_parser.py
"""

import json
import logging
from pathlib import Path
from typing import Optional

from schemas.clause import ExtractedClause
from schemas.feedback import FeedbackRecord
from schemas.risk import ClauseComparison, RiskFinding
from guardrails.evidence_verifier import _best_window_score, FUZZY_MATCH_THRESHOLD
from extraction.pdf_parser import normalize_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feedback log path — Stage 5: repointed to curated JSONL precedent store
# ---------------------------------------------------------------------------

FEEDBACK_LOG_PATH = Path(__file__).parent.parent / "data" / "feedback" / "feedback_log.jsonl"

# ---------------------------------------------------------------------------
# Feedback log cache (process-scoped; cleared by _clear_feedback_cache())
# ---------------------------------------------------------------------------

_feedback_log: Optional[list[FeedbackRecord]] = None


def _load_feedback_log() -> list[FeedbackRecord]:
    """
    Load approved-precedent records from the JSONL feedback log.

    Only records meeting ALL three of these criteria are treated as active:
        - approved_for_precedent == True
        - feedback_status == "approved_precedent"
        - final_risk == "MEDIUM"

    Everything else (not_eligible, pending_precedent_review, rejected_precedent)
    is read from disk, parsed, and then discarded -- never causes an error.

    A malformed or unparseable line is skipped with a WARNING log entry.
    Risk scoring must never crash because of one bad feedback log line.

    Why cache?
        The log is read once per process; changes made during a batch run
        are not picked up until the cache is cleared with _clear_feedback_cache().
        This is intentional: avoid redundant disk reads across 10 clause checks
        per document.
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

    eligible: list[FeedbackRecord] = []
    skipped_ineligible = 0
    skipped_malformed = 0

    try:
        with open(FEEDBACK_LOG_PATH, "r", encoding="utf-8") as fh:
            for lineno, raw_line in enumerate(fh, 1):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    raw = json.loads(raw_line)
                    record = FeedbackRecord(**raw)
                except Exception as exc:
                    logger.warning(
                        "risk_engine: skipping malformed feedback log line %d — %s",
                        lineno,
                        exc,
                    )
                    skipped_malformed += 1
                    continue

                if (
                    record.approved_for_precedent
                    and record.feedback_status == "approved_precedent"
                    and record.final_risk == "MEDIUM"
                ):
                    eligible.append(record)
                else:
                    skipped_ineligible += 1

    except OSError as exc:
        logger.error(
            "risk_engine: failed to open feedback log: %s. Using empty precedent set.",
            exc,
        )
        _feedback_log = []
        return _feedback_log

    logger.info(
        "Loaded %d approved precedent(s) from feedback log "
        "(%d ineligible skipped, %d malformed skipped).",
        len(eligible),
        skipped_ineligible,
        skipped_malformed,
    )
    _feedback_log = eligible
    return _feedback_log


def _clear_feedback_cache() -> None:
    """
    Reset the in-process feedback log cache.

    Called by agent/feedback_curation.py after every successful atomic
    write to feedback_log.jsonl, ensuring the risk engine picks up newly
    promoted precedents within the same process.

    Also used in tests (autouse fixture) to guarantee test isolation.
    """
    global _feedback_log
    _feedback_log = None


# ---------------------------------------------------------------------------
# Precedent lookup
# ---------------------------------------------------------------------------

def _find_precedent(
    clause_type: str,
    extracted_text: Optional[str],
    document_type: Optional[str] = None,
) -> Optional[tuple[FeedbackRecord, float]]:
    """
    Search the approved-precedent store for a record matching the given clause.

    Matching requires ALL of:
        1. clause_category exact match (case-sensitive string equality).
        2. document_type compatibility — if both the current finding and the
           precedent's scope specify a document_type, they must agree. If
           either is None, the check is skipped (null-compatible).
        3. Windowed difflib similarity >= FUZZY_MATCH_THRESHOLD (0.70) between
           the precedent's evidence_excerpt (needle) and the current clause's
           extracted_text (haystack). Uses _best_window_score() from
           evidence_verifier.py -- same algorithm, same threshold.
        4. jurisdiction / template_version: null-compatible in both directions
           (current document's context is not available at flag_risks() call
           time, so these constraints are not enforced here).

    When multiple records match, the one with the highest similarity score wins.

    Returns (FeedbackRecord, match_score) or None if no match.
    """
    if not extracted_text:
        return None

    log = _load_feedback_log()
    if not log:
        return None

    norm_extracted = normalize_text(extracted_text)
    best_record: Optional[FeedbackRecord] = None
    best_score: float = 0.0

    for record in log:
        # Check 1: clause category must match exactly
        if record.clause_category != clause_type:
            continue

        # Check 2: document_type compatibility (null-compatible on either side)
        if document_type is not None and record.precedent_scope is not None:
            scope_doc_type = record.precedent_scope.document_type
            if scope_doc_type is not None and scope_doc_type != document_type:
                continue

        # Check 3: windowed text similarity
        norm_excerpt = normalize_text(record.evidence_excerpt)
        score = _best_window_score(norm_excerpt, norm_extracted)
        if score < FUZZY_MATCH_THRESHOLD:
            continue

        if score > best_score:
            best_score = score
            best_record = record

    if best_record is None:
        return None
    return (best_record, best_score)


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
    document_type: Optional[str] = None,
) -> tuple[str, bool, Optional[str]]:
    """
    Apply the precedent-aware override if applicable.

    Returns (final_risk_level, precedent_applied, precedent_note).

    Downgrade table:
        HIGH   -> MEDIUM  (if precedent found)
        MEDIUM -> LOW     (if precedent found)
        LOW    -> LOW     (no downgrade needed)

    REG-001 independent safety re-check:
        This function must never be called for a missing clause (the caller,
        flag_risks(), already guards this). The check below is defense-in-depth:
        even if this function is called directly with a missing clause, it
        returns the original risk without touching the precedent log.

    When precedent is applied, precedent_note records:
        - matched feedback_id
        - match score (text similarity ratio)
        - match reason (category + score)
        - original risk -> downgraded risk
        - curator who approved and when
    """
    # REG-001 defense-in-depth: missing clause must never reach precedent lookup
    if not clause.is_present:
        logger.error(
            "_apply_precedent_downgrade called for missing clause '%s' — this is a bug. "
            "Returning original risk level without precedent lookup.",
            clause.clause_type,
        )
        return risk_level, False, None

    if risk_level == "LOW":
        return "LOW", False, None

    result = _find_precedent(clause.clause_type, clause.extracted_text, document_type)
    if result is None:
        return risk_level, False, None

    record, score = result
    downgraded = "MEDIUM" if risk_level == "HIGH" else "LOW"
    match_reason = (
        f"clause_category match ({record.clause_category!r}) + "
        f"{score:.2f} text similarity"
    )
    approved_date = (
        record.precedent_approved_at.date()
        if record.precedent_approved_at
        else "unknown date"
    )
    precedent_note = (
        f"Precedent {record.feedback_id} applied: {match_reason}. "
        f"Risk: {risk_level} -> {downgraded}. "
        f"Approved by {record.precedent_approved_by or 'unknown'} ({approved_date})."
    )

    logger.info(
        "Precedent override applied for %s: %s -> %s. feedback_id=%s score=%.2f",
        clause.clause_type,
        risk_level,
        downgraded,
        record.feedback_id,
        score,
    )
    return downgraded, True, precedent_note


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def flag_risks(
    clauses: list[ExtractedClause],
    comparisons: list[ClauseComparison],
    document_type: Optional[str] = None,
) -> list[RiskFinding]:
    """
    Produce a RiskFinding for each clause category.

    How it works:
        1. Pair each ExtractedClause with its ClauseComparison (by position --
           both lists are always ordered by CLAUSE_CATEGORIES).
        2. Assign a base risk level using the 5-rule table above.
        3. For non-missing clauses: check the feedback log for a precedent and
           downgrade one tier if found (text similarity >= 0.70).
        4. Build and return a RiskFinding for each pair.

    Parameters
    ----------
    clauses : list[ExtractedClause]
        The 10 ExtractedClause objects from extract_clauses(). Must be in
        CLAUSE_CATEGORIES order.
    comparisons : list[ClauseComparison]
        The 10 ClauseComparison objects from compare_to_templates(). Must be
        in the same order as clauses.
    document_type : str or None
        Optional document type for precedent scope matching (e.g. "NDA",
        "Contract"). When None, document_type compatibility check is skipped.
        Sourced from classify_document() in the full pipeline.

    Returns
    -------
    list[RiskFinding]
        Always exactly as many findings as input clauses (10 in normal use).

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

        # REG-001: precedent override is skipped for missing clauses (locked rule).
        # _apply_precedent_downgrade has its own independent guard, but we never
        # call it at all for missing clauses -- belt AND suspenders.
        if clause.is_present:
            final_risk, precedent_applied, precedent_note = _apply_precedent_downgrade(
                base_risk, clause, comparison, document_type
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
