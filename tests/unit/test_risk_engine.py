"""
tests/unit/test_risk_engine.py
--------------------------------
Automated test suite for agent/risk_engine.py.

No OpenAI calls are made -- risk_engine.py has no LLM dependency.
The feedback log is controlled via a tmp_path fixture.

What this file tests:
    1.  A missing clause always produces risk_level="HIGH", regardless of
        any precedent entry in the feedback log.
    2.  A present clause with no template found produces risk_level="LOW".
    3.  A present clause with deviation_severity="none" produces risk_level="LOW".
    4.  A present clause with deviation_severity="minor" produces risk_level="MEDIUM".
    5.  A present clause with deviation_severity="major" produces risk_level="HIGH".
    6.  A "major" deviation with a matching precedent is downgraded to "MEDIUM".
    7.  A "minor" deviation with a matching precedent is downgraded to "LOW".
    8.  A MISSING clause with a matching precedent entry is still "HIGH"
        (precedent override never applies to missing clauses -- locked rule).
    9.  flag_risks raises ValueError when clauses and comparisons differ in length.
    10. flag_risks returns exactly one RiskFinding per input clause.

How to run:
    From legal-agent/ with the venv active:
        pytest tests/unit/test_risk_engine.py -v
"""

import json
import pytest
from pathlib import Path

from schemas.clause import ExtractedClause
from schemas.risk import ClauseComparison, RiskFinding
from agent.risk_engine import flag_risks, _clear_feedback_cache, FEEDBACK_LOG_PATH
from config import CLAUSE_CATEGORIES


# ---------------------------------------------------------------------------
# Fixture: isolate feedback log for every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_feedback_cache(tmp_path, monkeypatch):
    """
    Each test gets its own feedback log path and a clean in-process cache.
    monkeypatch redirects FEEDBACK_LOG_PATH so tests never touch the real file.
    """
    _clear_feedback_cache()
    # Point the module's FEEDBACK_LOG_PATH to a tmp location
    monkeypatch.setattr("agent.risk_engine.FEEDBACK_LOG_PATH", tmp_path / "feedback_log.json")
    yield
    _clear_feedback_cache()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_clause(
    clause_type: str,
    is_present: bool = True,
    text: str = "Standard clause language here.",
) -> ExtractedClause:
    return ExtractedClause(
        clause_type=clause_type,
        is_present=is_present,
        extracted_text=text if is_present else None,
        page_reference=2 if is_present else None,
        confidence=0.9 if is_present else 0.85,
        source_chunk_id=None,
    )


def _make_comparison(
    clause_type: str,
    template_found: bool = True,
    matches_template: bool = True,
    deviation_severity: str = "none",
    deviation_summary: str | None = None,
) -> ClauseComparison:
    return ClauseComparison(
        clause_type=clause_type,
        template_found=template_found,
        matches_template=matches_template,
        deviation_severity=deviation_severity,
        deviation_summary=deviation_summary,
        template_path="/data/templates/test.txt" if template_found else None,
    )


def _write_feedback_log(tmp_path, entries: list[dict]):
    """Write a feedback log JSON file into the test's tmp_path."""
    log_path = tmp_path / "feedback_log.json"
    log_path.write_text(json.dumps(entries), encoding="utf-8")


# ---------------------------------------------------------------------------
# Test 1: Missing clause -> always HIGH
# ---------------------------------------------------------------------------

def test_missing_clause_is_always_high():
    clause = _make_clause("Indemnification", is_present=False)
    comparison = _make_comparison("Indemnification", deviation_severity="none")

    findings = flag_risks([clause], [comparison])

    assert findings[0].risk_level == "HIGH"
    assert findings[0].is_missing is True


# ---------------------------------------------------------------------------
# Test 2: Present clause, no template -> LOW
# ---------------------------------------------------------------------------

def test_present_clause_no_template_is_low():
    clause = _make_clause("Dispute Resolution")
    comparison = _make_comparison("Dispute Resolution", template_found=False)

    findings = flag_risks([clause], [comparison])

    assert findings[0].risk_level == "LOW"
    assert findings[0].is_missing is False


# ---------------------------------------------------------------------------
# Test 3: No deviation -> LOW
# ---------------------------------------------------------------------------

def test_no_deviation_is_low():
    clause = _make_clause("Governing Law / Jurisdiction")
    comparison = _make_comparison(
        "Governing Law / Jurisdiction",
        matches_template=True,
        deviation_severity="none",
    )

    findings = flag_risks([clause], [comparison])

    assert findings[0].risk_level == "LOW"


# ---------------------------------------------------------------------------
# Test 4: Minor deviation -> MEDIUM
# ---------------------------------------------------------------------------

def test_minor_deviation_is_medium():
    clause = _make_clause("Termination for Convenience")
    comparison = _make_comparison(
        "Termination for Convenience",
        matches_template=False,
        deviation_severity="minor",
        deviation_summary="Notice period extended from 30 to 60 days.",
    )

    findings = flag_risks([clause], [comparison])

    assert findings[0].risk_level == "MEDIUM"


# ---------------------------------------------------------------------------
# Test 5: Major deviation -> HIGH
# ---------------------------------------------------------------------------

def test_major_deviation_is_high():
    clause = _make_clause("Indemnification")
    comparison = _make_comparison(
        "Indemnification",
        matches_template=False,
        deviation_severity="major",
        deviation_summary="Indemnification scope extended to cover third-party IP claims.",
    )

    findings = flag_risks([clause], [comparison])

    assert findings[0].risk_level == "HIGH"
    assert findings[0].is_missing is False


# ---------------------------------------------------------------------------
# Test 6: Major deviation + precedent -> downgraded to MEDIUM
# ---------------------------------------------------------------------------

def test_major_deviation_with_precedent_is_medium(tmp_path):
    clause = _make_clause(
        "Governing Law / Jurisdiction",
        text="This Agreement shall be governed by the laws of the State of California.",
    )
    comparison = _make_comparison(
        "Governing Law / Jurisdiction",
        matches_template=False,
        deviation_severity="major",
        deviation_summary="Jurisdiction changed from New York to California.",
    )

    _write_feedback_log(tmp_path, [{
        "clause_type": "Governing Law / Jurisdiction",
        "approved_text_fragment": "laws of the State of California",
        "approval_date": "2024-03-15",
        "document_hash": "abc123def456",
        "note": "California jurisdiction approved by legal counsel for West Coast vendors.",
    }])

    findings = flag_risks([clause], [comparison])

    f = findings[0]
    assert f.risk_level == "MEDIUM"
    assert f.precedent_applied is True
    assert f.precedent_note is not None
    assert "California" in f.precedent_note or "2024-03-15" in f.precedent_note


# ---------------------------------------------------------------------------
# Test 7: Minor deviation + precedent -> downgraded to LOW
# ---------------------------------------------------------------------------

def test_minor_deviation_with_precedent_is_low(tmp_path):
    clause = _make_clause(
        "Termination for Convenience",
        text="Either party may terminate upon sixty (60) days written notice.",
    )
    comparison = _make_comparison(
        "Termination for Convenience",
        matches_template=False,
        deviation_severity="minor",
        deviation_summary="Notice period extended from 30 to 60 days.",
    )

    _write_feedback_log(tmp_path, [{
        "clause_type": "Termination for Convenience",
        "approved_text_fragment": "sixty (60) days",
        "approval_date": "2024-01-10",
        "document_hash": "deadbeef1234",
        "note": "60-day notice period approved for enterprise contracts.",
    }])

    findings = flag_risks([clause], [comparison])

    f = findings[0]
    assert f.risk_level == "LOW"
    assert f.precedent_applied is True


# ---------------------------------------------------------------------------
# Test 8: MISSING clause + matching precedent -> still HIGH (locked rule)
# ---------------------------------------------------------------------------

def test_missing_clause_ignores_precedent(tmp_path):
    """
    This is the most important invariant in the risk engine.
    A missing clause must ALWAYS be HIGH, even if a precedent entry exists
    for that clause type. Precedent override is only for present-but-deviating
    clauses, never for absent ones.
    """
    clause = _make_clause("Indemnification", is_present=False)
    comparison = _make_comparison("Indemnification", deviation_severity="none")

    # Add a precedent that would match if the clause were present
    _write_feedback_log(tmp_path, [{
        "clause_type": "Indemnification",
        "approved_text_fragment": "indemnify",
        "approval_date": "2024-06-01",
        "document_hash": "cafebabe5678",
        "note": "Limited indemnification approved.",
    }])

    findings = flag_risks([clause], [comparison])

    f = findings[0]
    assert f.risk_level == "HIGH", (
        "Missing clause must remain HIGH even when a precedent entry exists"
    )
    assert f.precedent_applied is False
    assert f.is_missing is True


# ---------------------------------------------------------------------------
# Test 9: Mismatched list lengths -> ValueError
# ---------------------------------------------------------------------------

def test_mismatched_lengths_raises_value_error():
    clauses = [_make_clause("Indemnification")]
    comparisons = []  # empty -- wrong length

    with pytest.raises(ValueError, match="same length"):
        flag_risks(clauses, comparisons)


# ---------------------------------------------------------------------------
# Test 10: Returns one finding per input clause
# ---------------------------------------------------------------------------

def test_returns_one_finding_per_clause():
    """flag_risks must return exactly as many findings as input clauses."""
    clauses = [
        _make_clause("Indemnification"),
        _make_clause("Governing Law / Jurisdiction", is_present=False),
        _make_clause("Dispute Resolution"),
    ]
    comparisons = [
        _make_comparison("Indemnification", deviation_severity="none"),
        _make_comparison("Governing Law / Jurisdiction"),
        _make_comparison("Dispute Resolution", template_found=False),
    ]

    findings = flag_risks(clauses, comparisons)

    assert len(findings) == 3
    for finding in findings:
        assert isinstance(finding, RiskFinding)
