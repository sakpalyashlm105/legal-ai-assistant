"""
tests/unit/test_review_score.py
--------------------------------
Unit tests for agent/review_score.py (shadow-mode review urgency scoring).

These tests verify:
  1. Score is deterministic for the same inputs.
  2. Missing-clause floor: is_missing=True always returns score=1.0 regardless
     of all other inputs (even "best case" confidence/risk/deviation/evidence).
  3. A high-confidence, low-risk, no-deviation, precedent-matched finding
     produces a low score — proving the model direction is correct.
  4. Score is bounded to [0.0, 1.0].
  5. in_interrupt_bucket is consistent with INTERRUPT_THRESHOLD.
  6. log_review_scores writes to disk without error and returns one result per finding.

Shadow-mode contract: none of these tests assert anything about routing,
interrupt behavior, or auto-approval. Score values are the ONLY output.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from schemas.clause import ExtractedClause
from schemas.risk import RiskFinding
from agent.review_score import (
    compute_review_score,
    log_review_scores,
    INTERRUPT_THRESHOLD,
    ReviewScoreResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_clause(
    clause_type: str = "Indemnification",
    is_present: bool = True,
    confidence: float = 0.9,
) -> ExtractedClause:
    return ExtractedClause(
        clause_type=clause_type,
        is_present=is_present,
        extracted_text="Sample clause text." if is_present else None,
        page_reference=1 if is_present else None,
        confidence=confidence,
        source_chunk_id="chunk_001" if is_present else None,
    )


def _make_finding(
    clause_type: str = "Indemnification",
    risk_level: str = "HIGH",
    is_missing: bool = False,
    deviation_summary: str = None,
    precedent_applied: bool = False,
    confidence: float = 0.9,
) -> RiskFinding:
    source = _make_clause(clause_type, is_present=not is_missing, confidence=confidence)
    return RiskFinding(
        clause_type=clause_type,
        risk_level=risk_level,
        reason="Test finding.",
        is_missing=is_missing,
        deviation_summary=deviation_summary,
        precedent_applied=precedent_applied,
        precedent_note=None,
        source_clause=None if is_missing else source,
    )


# ---------------------------------------------------------------------------
# 1. Missing-clause floor
# ---------------------------------------------------------------------------

class TestMissingClauseFloor:
    """
    is_missing=True must ALWAYS return score=1.0 regardless of other inputs.
    This is the hard non-negotiable rule mirroring "absent clauses are always HIGH".
    """

    def test_missing_clause_scores_max(self):
        finding = _make_finding(is_missing=True)
        result = compute_review_score(finding)
        assert result.total_score == 1.0

    def test_missing_clause_floor_applied_flag(self):
        finding = _make_finding(is_missing=True)
        result = compute_review_score(finding)
        assert result.missing_clause_floor_applied is True

    def test_missing_clause_in_interrupt_bucket(self):
        finding = _make_finding(is_missing=True)
        result = compute_review_score(finding)
        assert result.in_interrupt_bucket is True

    def test_missing_clause_ignores_best_case_inputs(self):
        """
        Even with HIGH risk turned to LOW, precedent matched, and high confidence,
        a missing clause must still score 1.0. The floor is unconditional.
        """
        finding = _make_finding(
            is_missing=True,
            risk_level="LOW",          # best-case risk
            precedent_applied=True,    # precedent found
            confidence=0.99,           # near-perfect confidence
            deviation_summary=None,    # no deviation
        )
        result = compute_review_score(finding)
        assert result.total_score == 1.0
        assert result.missing_clause_floor_applied is True

    def test_missing_clause_is_missing_true(self):
        finding = _make_finding(is_missing=True)
        result = compute_review_score(finding)
        assert result.is_missing is True


# ---------------------------------------------------------------------------
# 2. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_inputs_same_score(self):
        finding = _make_finding(
            risk_level="HIGH",
            deviation_summary="major deviation from template.",
            precedent_applied=False,
            confidence=0.6,
        )
        r1 = compute_review_score(finding)
        r2 = compute_review_score(finding)
        assert r1.total_score == r2.total_score
        assert r1.severity_contribution == r2.severity_contribution
        assert r1.confidence_contribution == r2.confidence_contribution

    def test_different_inputs_different_scores(self):
        high_finding = _make_finding(risk_level="HIGH", precedent_applied=False, confidence=0.4)
        low_finding  = _make_finding(risk_level="LOW",  precedent_applied=True,  confidence=0.95)
        r_high = compute_review_score(high_finding)
        r_low  = compute_review_score(low_finding)
        assert r_high.total_score > r_low.total_score


# ---------------------------------------------------------------------------
# 3. High-confidence / low-risk / no-deviation / precedent-matched → low score
# ---------------------------------------------------------------------------

class TestLowUrgencyFinding:
    """
    A finding that is: LOW risk, high confidence, no deviation, precedent matched.
    This is the "best case" for a present clause — score should be well below
    the interrupt threshold, proving the model direction is correct.
    """

    def test_low_urgency_below_interrupt_threshold(self):
        finding = _make_finding(
            risk_level="LOW",
            is_missing=False,
            deviation_summary=None,     # no deviation
            precedent_applied=True,     # precedent matched → novelty=0.0
            confidence=0.95,            # high confidence → confidence component near 0
        )
        result = compute_review_score(finding)
        assert result.total_score < INTERRUPT_THRESHOLD, (
            f"Expected score < {INTERRUPT_THRESHOLD} for best-case finding, "
            f"got {result.total_score}"
        )
        assert result.in_interrupt_bucket is False
        assert result.missing_clause_floor_applied is False

    def test_low_urgency_novelty_component_zero_when_precedent_matched(self):
        finding = _make_finding(precedent_applied=True, risk_level="LOW", confidence=0.9)
        result = compute_review_score(finding)
        assert result.raw_novelty == 0.0

    def test_high_confidence_lowers_confidence_component(self):
        high_conf = _make_finding(confidence=0.95, risk_level="LOW")
        low_conf  = _make_finding(confidence=0.30, risk_level="LOW")
        r_high = compute_review_score(high_conf)
        r_low  = compute_review_score(low_conf)
        assert r_high.confidence_contribution < r_low.confidence_contribution


# ---------------------------------------------------------------------------
# 4. High-urgency finding (HIGH risk, novel, no precedent, low confidence)
# ---------------------------------------------------------------------------

class TestHighUrgencyFinding:
    def test_high_risk_novel_above_threshold(self):
        finding = _make_finding(
            risk_level="HIGH",
            is_missing=False,
            deviation_summary="major deviation identified.",
            precedent_applied=False,
            confidence=0.45,
        )
        result = compute_review_score(finding)
        assert result.total_score >= INTERRUPT_THRESHOLD
        assert result.in_interrupt_bucket is True

    def test_novelty_component_one_when_no_precedent(self):
        finding = _make_finding(precedent_applied=False, risk_level="HIGH")
        result = compute_review_score(finding)
        assert result.raw_novelty == 1.0


# ---------------------------------------------------------------------------
# 5. Score is bounded [0.0, 1.0]
# ---------------------------------------------------------------------------

class TestScoreBounds:
    def test_worst_case_does_not_exceed_one(self):
        finding = _make_finding(
            risk_level="HIGH", confidence=0.0,
            deviation_summary="major deviation", precedent_applied=False,
        )
        result = compute_review_score(finding)
        assert 0.0 <= result.total_score <= 1.0

    def test_best_case_does_not_go_below_zero(self):
        finding = _make_finding(
            risk_level="LOW", confidence=1.0,
            deviation_summary=None, precedent_applied=True,
        )
        result = compute_review_score(finding)
        assert 0.0 <= result.total_score <= 1.0

    def test_missing_clause_exactly_one(self):
        finding = _make_finding(is_missing=True)
        result = compute_review_score(finding)
        assert result.total_score == 1.0


# ---------------------------------------------------------------------------
# 6. log_review_scores — I/O test with tmp path
# ---------------------------------------------------------------------------

class TestLogReviewScores:
    def test_writes_one_record_per_finding(self, tmp_path):
        findings = [
            _make_finding("Indemnification", risk_level="HIGH"),
            _make_finding("Governing Law / Jurisdiction", risk_level="MEDIUM", precedent_applied=True),
            _make_finding("Non-Compete / Non-Solicitation", is_missing=True),
        ]
        log_path = tmp_path / "test_scores.jsonl"
        with patch("agent.review_score.REVIEW_SCORES_LOG_PATH", log_path):
            results = log_review_scores(findings, thread_id="test-thread", document_hash="abc123")

        assert len(results) == 3
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3

    def test_log_records_are_valid_json(self, tmp_path):
        findings = [_make_finding("Indemnification", risk_level="HIGH")]
        log_path = tmp_path / "test_scores.jsonl"
        with patch("agent.review_score.REVIEW_SCORES_LOG_PATH", log_path):
            log_review_scores(findings, thread_id="t1", document_hash="hash1")
        record = json.loads(log_path.read_text(encoding="utf-8").strip())
        assert "total_score" in record
        assert "components" in record
        assert "clause_type" in record

    def test_missing_clause_record_has_floor_flag(self, tmp_path):
        findings = [_make_finding("Termination for Convenience", is_missing=True)]
        log_path = tmp_path / "test_scores.jsonl"
        with patch("agent.review_score.REVIEW_SCORES_LOG_PATH", log_path):
            results = log_review_scores(findings, thread_id="t1", document_hash="h1")
        assert results[0].missing_clause_floor_applied is True
        record = json.loads(log_path.read_text(encoding="utf-8").strip())
        assert record["missing_clause_floor_applied"] is True

    def test_empty_findings_writes_nothing(self, tmp_path):
        log_path = tmp_path / "test_scores.jsonl"
        with patch("agent.review_score.REVIEW_SCORES_LOG_PATH", log_path):
            results = log_review_scores([], thread_id="t1", document_hash="h1")
        assert results == []
        assert not log_path.exists()
