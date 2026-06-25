"""
tests/unit/test_comparator.py
------------------------------
Automated test suite for agent/comparator.py.

All OpenAI API calls are mocked -- no real API key or network access needed.
Template file reads use real files from data/templates/ where they exist,
and a tmp_path fixture for tests that need custom template content.

What this file tests:
    1.  A clause type with a matching template and no deviation returns
        ClauseComparison(matches_template=True, deviation_severity="none").
    2.  A minor deviation returns deviation_severity="minor" and a non-None
        deviation_summary.
    3.  A major deviation returns deviation_severity="major".
    4.  A clause type with no template file returns template_found=False
        and skips the LLM call entirely.
    5.  An absent clause (is_present=False) returns a ClauseComparison with
        deviation_severity="none" without calling the LLM.
    6.  An API exception returns a safe fallback (matches_template=True,
        deviation_severity="none") without crashing.
    7.  Malformed JSON from the LLM returns the safe fallback without crashing.
    8.  compare_to_templates returns exactly one result per input clause.
    9.  The template_path field is set when a template file is found.
    10. _clause_type_to_filename converts clause type names correctly.

How to run:
    From legal-agent/ with the venv active:
        pytest tests/unit/test_comparator.py -v
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from schemas.clause import ExtractedClause
from schemas.risk import ClauseComparison
from agent.comparator import (
    compare_to_templates,
    _clause_type_to_filename,
    TEMPLATES_DIR,
)
from config import CLAUSE_CATEGORIES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_HASH = "ab" * 32


def _make_clause(
    clause_type: str,
    is_present: bool = True,
    text: str = "Standard clause text here.",
) -> ExtractedClause:
    return ExtractedClause(
        clause_type=clause_type,
        is_present=is_present,
        extracted_text=text if is_present else None,
        page_reference=3 if is_present else None,
        confidence=0.9 if is_present else 0.85,
        source_chunk_id=f"{FAKE_HASH[:12]}_chunk_0001" if is_present else None,
    )


def _mock_llm_response(
    matches_template: bool,
    deviation_severity: str,
    deviation_summary: str | None,
) -> MagicMock:
    content = json.dumps({
        "matches_template": matches_template,
        "deviation_severity": deviation_severity,
        "deviation_summary": deviation_summary,
    })
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content=content))]
    return mock_resp


# ---------------------------------------------------------------------------
# Test 1: No deviation -- matches template
# ---------------------------------------------------------------------------

def test_no_deviation_returns_matches_template():
    """
    When LLM says deviation_severity="none", ClauseComparison.matches_template
    must be True and deviation_summary must be None.
    """
    clause = _make_clause("Indemnification")

    with patch("agent.comparator._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_llm_response(
            matches_template=True,
            deviation_severity="none",
            deviation_summary=None,
        )

        results = compare_to_templates([clause])

    assert len(results) == 1
    r = results[0]
    assert r.matches_template is True
    assert r.deviation_severity == "none"
    assert r.deviation_summary is None
    assert r.template_found is True


# ---------------------------------------------------------------------------
# Test 2: Minor deviation
# ---------------------------------------------------------------------------

def test_minor_deviation_is_captured():
    """
    deviation_severity="minor" must propagate to the ClauseComparison, and
    deviation_summary must contain the LLM's description.
    """
    clause = _make_clause("Indemnification")
    summary = "The notice period for claims was extended from 30 to 60 days."

    with patch("agent.comparator._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_llm_response(
            matches_template=False,
            deviation_severity="minor",
            deviation_summary=summary,
        )

        results = compare_to_templates([clause])

    r = results[0]
    assert r.matches_template is False
    assert r.deviation_severity == "minor"
    assert r.deviation_summary == summary


# ---------------------------------------------------------------------------
# Test 3: Major deviation
# ---------------------------------------------------------------------------

def test_major_deviation_is_captured():
    """deviation_severity="major" must propagate correctly."""
    clause = _make_clause("Governing Law / Jurisdiction")
    summary = "Jurisdiction changed from New York to California."

    with patch("agent.comparator._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_llm_response(
            matches_template=False,
            deviation_severity="major",
            deviation_summary=summary,
        )

        results = compare_to_templates([clause])

    r = results[0]
    assert r.deviation_severity == "major"
    assert r.deviation_summary == summary


# ---------------------------------------------------------------------------
# Test 4: No template file -> template_found=False, no LLM call
# ---------------------------------------------------------------------------

def test_missing_template_file_skips_llm():
    """
    If no template .txt file exists for a clause type, template_found must
    be False and the LLM must NOT be called.
    """
    # "Dispute Resolution" has no template file in data/templates/
    clause = _make_clause("Dispute Resolution")

    with patch("agent.comparator._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        results = compare_to_templates([clause])

    r = results[0]
    assert r.template_found is False
    assert r.deviation_severity == "none"
    mock_client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# Test 5: Absent clause -> no LLM call, deviation_severity="none"
# ---------------------------------------------------------------------------

def test_absent_clause_skips_llm_call():
    """
    An absent clause (is_present=False) must not trigger any LLM call.
    The comparison result should have deviation_severity="none".
    """
    clause = _make_clause("Termination for Convenience", is_present=False)

    with patch("agent.comparator._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        results = compare_to_templates([clause])

    r = results[0]
    assert r.deviation_severity == "none"
    assert r.matches_template is False
    mock_client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# Test 6: API exception -> safe fallback, no crash
# ---------------------------------------------------------------------------

def test_api_exception_returns_safe_fallback():
    """
    If the OpenAI API raises an exception during comparison, compare_to_templates
    must return a safe default (matches_template=True, deviation_severity="none")
    rather than propagating the error.
    """
    clause = _make_clause("Indemnification")

    with patch("agent.comparator._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("Timeout")

        results = compare_to_templates([clause])

    r = results[0]
    assert r.template_found is True
    assert r.matches_template is True
    assert r.deviation_severity == "none"


# ---------------------------------------------------------------------------
# Test 7: Malformed JSON -> safe fallback, no crash
# ---------------------------------------------------------------------------

def test_malformed_json_returns_safe_fallback():
    bad_resp = MagicMock()
    bad_resp.choices = [MagicMock(message=MagicMock(content="not json at all"))]

    clause = _make_clause("Indemnification")

    with patch("agent.comparator._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = bad_resp

        results = compare_to_templates([clause])

    r = results[0]
    assert r.deviation_severity == "none"
    assert r.matches_template is True


# ---------------------------------------------------------------------------
# Test 8: One result per input clause
# ---------------------------------------------------------------------------

def test_returns_one_result_per_clause():
    """compare_to_templates must return the same number of results as inputs."""
    clauses = [
        _make_clause("Indemnification"),
        _make_clause("Governing Law / Jurisdiction"),
        _make_clause("Dispute Resolution", is_present=False),
    ]

    with patch("agent.comparator._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_llm_response(
            True, "none", None
        )

        results = compare_to_templates(clauses)

    assert len(results) == 3


# ---------------------------------------------------------------------------
# Test 9: template_path is set when a template file exists
# ---------------------------------------------------------------------------

def test_template_path_is_set_when_file_exists():
    """
    When a template file is found and the comparison runs, template_path must
    be set to a non-None string pointing to that file.
    """
    clause = _make_clause("Indemnification")

    with patch("agent.comparator._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_llm_response(
            True, "none", None
        )

        results = compare_to_templates([clause])

    r = results[0]
    assert r.template_path is not None
    assert "Indemnification" in r.template_path


# ---------------------------------------------------------------------------
# Test 10: _clause_type_to_filename converts names correctly
# ---------------------------------------------------------------------------

def test_clause_type_to_filename_conversion():
    """
    Verify the naming convention for all clause types that have templates,
    and spot-check a few others.
    """
    cases = {
        "Indemnification": "Indemnification.txt",
        "Governing Law / Jurisdiction": "Governing_Law_Jurisdiction.txt",
        "Confidentiality / Non-Disclosure": "Confidentiality_Non-Disclosure.txt",
        "Non-Compete / Non-Solicitation": "Non-Compete_Non-Solicitation.txt",
        "Termination for Convenience": "Termination_for_Convenience.txt",
        "Limitation of Liability": "Limitation_of_Liability.txt",
        "Dispute Resolution": "Dispute_Resolution.txt",
        "Assignment": "Assignment.txt",
        "Renewal / Term": "Renewal_Term.txt",
        "Termination for Cause": "Termination_for_Cause.txt",
    }
    for clause_type, expected_filename in cases.items():
        result = _clause_type_to_filename(clause_type)
        assert result == expected_filename, (
            f"_clause_type_to_filename({clause_type!r}) = {result!r}, "
            f"expected {expected_filename!r}"
        )
