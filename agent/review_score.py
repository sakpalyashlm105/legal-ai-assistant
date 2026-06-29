"""
agent/review_score.py
---------------------
Shadow-mode review urgency scoring.

SHADOW MODE — this module NEVER affects routing, interrupt decisions, or
auto-approval of any finding. It only computes a number and writes it to
data/metrics/review_scores_log.jsonl for later calibration.

Calling compute_review_score() or log_review_scores() from anywhere in
the pipeline has zero effect on whether a finding reaches human review.
The only thing that controls that is node_check_human_review in
orchestrator.py — this module does not touch it.

Purpose
-------
Assigns a 0.0–1.0 urgency score to each RiskFinding so that, once enough
real data has been collected in the log, the team can calibrate whether a
score-based threshold would route the same items as the current hard rules
(HIGH risk + missing clause). If calibration looks good, the score can be
promoted to an actual routing input in a future sprint — that requires a
deliberate design decision, not a code change here.

Score components (all normalized to 0.0–1.0 before weighting)
--------------------------------------------------------------
severity      (weight 0.35) — HIGH=1.0, MEDIUM=0.5, LOW=0.1
confidence    (weight 0.25) — 1.0 - extraction_confidence (lower conf → higher score)
deviation     (weight 0.20) — major=1.0, minor=0.4, none=0.0
evidence      (weight 0.10) — not_found=1.0, fuzzy=0.5, exact=0.0, unknown=0.3
novelty       (weight 0.10) — no approved precedent=1.0, precedent matched=0.0

Missing-clause floor (hard rule, non-negotiable)
------------------------------------------------
If the finding is a missing-clause finding (is_missing=True), the function
returns 1.0 regardless of all other inputs. Missing clauses must always
fall in the highest urgency bucket. This mirrors the locked design rule:
"absence is always HIGH regardless of precedent."

HITL, not RLHF
---------------
Review scores are logged for human analysis of calibration data only.
They are never used to update model weights or feed fine-tuning pipelines.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from schemas.risk import RiskFinding

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output log
# ---------------------------------------------------------------------------

REVIEW_SCORES_LOG_PATH = (
    Path(__file__).parent.parent / "data" / "metrics" / "review_scores_log.jsonl"
)

# ---------------------------------------------------------------------------
# Score weights — must sum to 1.0
# ---------------------------------------------------------------------------

_W_SEVERITY   = 0.35
_W_CONFIDENCE = 0.25
_W_DEVIATION  = 0.20
_W_EVIDENCE   = 0.10
_W_NOVELTY    = 0.10

assert abs((_W_SEVERITY + _W_CONFIDENCE + _W_DEVIATION + _W_EVIDENCE + _W_NOVELTY) - 1.0) < 1e-9

# Score at or above this value is considered the "interrupt" bucket for
# calibration analysis. NOT used for routing — purely a label in the log.
INTERRUPT_THRESHOLD = 0.50


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ReviewScoreResult:
    """
    Full breakdown of a computed review score.

    All component values are the *weighted* contributions (component × weight),
    so they sum to the total score. The raw (pre-weight) component values are
    also stored for debugging.
    """
    clause_type: str
    is_missing: bool
    total_score: float                        # 0.0–1.0; 1.0 for missing clauses
    in_interrupt_bucket: bool                 # total_score >= INTERRUPT_THRESHOLD
    missing_clause_floor_applied: bool        # True when is_missing forced score=1.0

    # Weighted contributions (sum = total_score, except when floor applied)
    severity_contribution: float
    confidence_contribution: float
    deviation_contribution: float
    evidence_contribution: float
    novelty_contribution: float

    # Raw (pre-weight) component values, for human inspection of the log
    raw_severity: float
    raw_confidence: float
    raw_deviation: float
    raw_evidence: float
    raw_novelty: float


# ---------------------------------------------------------------------------
# Component scorers (all return 0.0–1.0)
# ---------------------------------------------------------------------------

def _severity_raw(risk_level: str) -> float:
    return {"HIGH": 1.0, "MEDIUM": 0.5, "LOW": 0.1}.get(risk_level, 0.5)


def _confidence_raw(extraction_confidence: Optional[float]) -> float:
    # Inverted: lower extraction confidence → higher score contribution.
    # None means the clause was absent; caller must short-circuit before
    # reaching this for missing-clause findings.
    if extraction_confidence is None:
        return 1.0
    return max(0.0, min(1.0, 1.0 - extraction_confidence))


def _deviation_raw(deviation_severity: Optional[str]) -> float:
    return {"major": 1.0, "minor": 0.4, "none": 0.0}.get(deviation_severity or "none", 0.0)


def _evidence_raw(evidence_match_type: Optional[str]) -> float:
    return {"not_found": 1.0, "fuzzy": 0.5, "exact": 0.0}.get(evidence_match_type or "", 0.3)


def _novelty_raw(precedent_applied: bool) -> float:
    # No matching approved precedent = novel finding = higher urgency.
    return 0.0 if precedent_applied else 1.0


# ---------------------------------------------------------------------------
# Public scoring function
# ---------------------------------------------------------------------------

def compute_review_score(finding: RiskFinding) -> ReviewScoreResult:
    """
    Compute the shadow-mode review urgency score for one RiskFinding.

    Parameters
    ----------
    finding : RiskFinding
        The risk finding to score. Must have is_missing, risk_level,
        deviation_severity (via source_clause path or direct), and
        precedent_applied populated. evidence_match_type is read from
        the source_clause if available.

    Returns
    -------
    ReviewScoreResult
        Full score breakdown. Does not perform any I/O.
    """
    clause_type = str(finding.clause_type)

    # Hard floor: missing-clause findings always score 1.0 (highest urgency).
    # This mirrors the locked rule that absent clauses are always HIGH risk
    # and should always be routed to human review.
    if finding.is_missing:
        return ReviewScoreResult(
            clause_type=clause_type,
            is_missing=True,
            total_score=1.0,
            in_interrupt_bucket=True,
            missing_clause_floor_applied=True,
            severity_contribution=0.0,
            confidence_contribution=0.0,
            deviation_contribution=0.0,
            evidence_contribution=0.0,
            novelty_contribution=0.0,
            raw_severity=1.0,
            raw_confidence=1.0,
            raw_deviation=1.0,
            raw_evidence=1.0,
            raw_novelty=1.0,
        )

    # Extract inputs from the finding
    extraction_confidence: Optional[float] = None
    evidence_match_type: Optional[str] = None
    if finding.source_clause is not None:
        extraction_confidence = finding.source_clause.confidence
        # evidence_match_type is not on ExtractedClause — it lives on ReviewItem.
        # The closest proxy available on the finding itself is None here;
        # we pass None and the scorer uses the "unknown" default (0.3).

    deviation_severity: Optional[str] = None
    if finding.source_clause is not None:
        # deviation_severity comes from ClauseComparison, carried into findings
        # via risk_engine. It is not directly on RiskFinding — we approximate
        # from deviation_summary presence and risk_level as a proxy.
        pass  # handled below via finding fields

    # deviation_severity proxy: read from finding.deviation_summary existence
    # and risk_level. The comparator stores "major"/"minor"/"none" in
    # ClauseComparison.deviation_severity but RiskFinding only carries the
    # text summary. We re-derive the severity bucket from the summary text
    # and risk_level as a best proxy without coupling to comparator internals.
    if finding.deviation_summary:
        dev_text = finding.deviation_summary.lower()
        if "major" in dev_text or "significant" in dev_text or "substantial" in dev_text:
            deviation_severity = "major"
        elif "minor" in dev_text or "small" in dev_text or "slight" in dev_text:
            deviation_severity = "minor"
        else:
            # Has a deviation summary but no explicit severity word — treat as minor
            deviation_severity = "minor"
    else:
        deviation_severity = "none"

    # Raw component values
    r_severity   = _severity_raw(finding.risk_level)
    r_confidence = _confidence_raw(extraction_confidence)
    r_deviation  = _deviation_raw(deviation_severity)
    r_evidence   = _evidence_raw(evidence_match_type)
    r_novelty    = _novelty_raw(finding.precedent_applied)

    # Weighted contributions
    c_severity   = r_severity   * _W_SEVERITY
    c_confidence = r_confidence * _W_CONFIDENCE
    c_deviation  = r_deviation  * _W_DEVIATION
    c_evidence   = r_evidence   * _W_EVIDENCE
    c_novelty    = r_novelty    * _W_NOVELTY

    total = c_severity + c_confidence + c_deviation + c_evidence + c_novelty
    total = round(min(1.0, max(0.0, total)), 4)

    return ReviewScoreResult(
        clause_type=clause_type,
        is_missing=False,
        total_score=total,
        in_interrupt_bucket=total >= INTERRUPT_THRESHOLD,
        missing_clause_floor_applied=False,
        severity_contribution=round(c_severity, 4),
        confidence_contribution=round(c_confidence, 4),
        deviation_contribution=round(c_deviation, 4),
        evidence_contribution=round(c_evidence, 4),
        novelty_contribution=round(c_novelty, 4),
        raw_severity=r_severity,
        raw_confidence=round(r_confidence, 4),
        raw_deviation=r_deviation,
        raw_evidence=r_evidence,
        raw_novelty=r_novelty,
    )


# ---------------------------------------------------------------------------
# Batch scorer + logger
# ---------------------------------------------------------------------------

def log_review_scores(
    findings: list[RiskFinding],
    thread_id: str,
    document_hash: str,
) -> list[ReviewScoreResult]:
    """
    Score every finding in the list and append one JSON record per finding to
    data/metrics/review_scores_log.jsonl.

    This is the only function that performs I/O. compute_review_score() is
    pure and has no side effects.

    SHADOW MODE: the returned list is for callers that want to inspect scores
    in tests or logs. Nothing in the live pipeline branches on these values.

    Parameters
    ----------
    findings : list[RiskFinding]
    thread_id : str  — pipeline run identifier (for log correlation)
    document_hash : str  — SHA-256 of the source document

    Returns
    -------
    list[ReviewScoreResult]  — one entry per finding, same order as input
    """
    results: list[ReviewScoreResult] = []

    REVIEW_SCORES_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    records_to_write: list[dict] = []
    for finding in findings:
        try:
            result = compute_review_score(finding)
            results.append(result)
            records_to_write.append({
                "thread_id": thread_id,
                "document_hash": document_hash,
                "clause_type": result.clause_type,
                "is_missing": result.is_missing,
                "total_score": result.total_score,
                "in_interrupt_bucket": result.in_interrupt_bucket,
                "missing_clause_floor_applied": result.missing_clause_floor_applied,
                "components": {
                    "severity":   {"raw": result.raw_severity,   "weighted": result.severity_contribution},
                    "confidence": {"raw": result.raw_confidence, "weighted": result.confidence_contribution},
                    "deviation":  {"raw": result.raw_deviation,  "weighted": result.deviation_contribution},
                    "evidence":   {"raw": result.raw_evidence,   "weighted": result.evidence_contribution},
                    "novelty":    {"raw": result.raw_novelty,    "weighted": result.novelty_contribution},
                },
            })
        except Exception as exc:
            logger.error(
                "log_review_scores: failed to score finding '%s': %s",
                finding.clause_type, exc,
            )

    if records_to_write:
        try:
            with REVIEW_SCORES_LOG_PATH.open("a", encoding="utf-8") as fh:
                for rec in records_to_write:
                    fh.write(json.dumps(rec) + "\n")
        except Exception as exc:
            logger.error("log_review_scores: failed to write log: %s", exc)

    logger.info(
        "log_review_scores: scored %d findings for thread=%s (shadow mode, no routing effect)",
        len(results), thread_id,
    )
    return results
