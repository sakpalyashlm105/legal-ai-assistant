"""
tests/unit/test_evidence_verifier.py
--------------------------------------
Unit tests for guardrails/evidence_verifier.py.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from guardrails.evidence_verifier import verify_evidence, FUZZY_MATCH_THRESHOLD


SOURCE_TEXT = (
    "This Non-Disclosure Agreement ('Agreement') is entered into as of January 1, 2024, "
    "by and between Acme Corp. ('Disclosing Party') and Beta LLC ('Receiving Party'). "
    "The Receiving Party agrees to maintain the confidentiality of all Proprietary Information "
    "disclosed by the Disclosing Party and shall not disclose such information to any third party "
    "without the prior written consent of the Disclosing Party. "
    "Either party may terminate this Agreement upon thirty (30) days written notice."
)


class TestExactMatch:
    def test_exact_substring_found(self):
        extracted = "Either party may terminate this Agreement upon thirty (30) days written notice."
        result = verify_evidence(extracted, SOURCE_TEXT, source_page=3)
        assert result.found_in_source is True
        assert result.match_type == "exact"
        assert result.match_score is None
        assert result.source_page_checked == 3

    def test_exact_match_partial_sentence(self):
        extracted = "shall not disclose such information to any third party"
        result = verify_evidence(extracted, SOURCE_TEXT, source_page=1)
        assert result.found_in_source is True
        assert result.match_type == "exact"

    def test_exact_match_on_short_source(self):
        """
        When source text is short (e.g., a single paragraph), exact match
        works cleanly without windowing concerns.
        """
        source = "The parties agree to maintain confidentiality of all proprietary information."
        extracted = "maintain confidentiality of all proprietary information"
        result = verify_evidence(extracted, source, source_page=1)
        assert result.found_in_source is True
        assert result.match_type == "exact"


class TestFuzzyMatch:
    def test_fuzzy_match_above_threshold(self):
        """
        One word inserted into extracted text: 'the' added before 'confidentiality'.
        The extracted text is no longer an exact substring, but difflib windowed
        matching should score above FUZZY_MATCH_THRESHOLD.
        """
        source = (
            "The Receiving Party agrees to maintain confidentiality of all Proprietary "
            "Information disclosed by the Disclosing Party."
        )
        # Extracted has 'the' inserted before 'confidentiality' -- no longer exact
        extracted = "The Receiving Party agrees to maintain the confidentiality of all Proprietary Information."
        result = verify_evidence(extracted, source, source_page=2)
        assert result.found_in_source is True
        assert result.match_type in ("exact", "fuzzy")

    def test_fuzzy_match_score_in_range(self):
        """A fuzzy match result should have score between threshold and 1.0."""
        extracted = "The Receiving Party agrees to maintain the confidentiality of all Information."
        result = verify_evidence(extracted, SOURCE_TEXT, source_page=1)
        if result.match_type == "fuzzy":
            assert result.match_score is not None
            assert FUZZY_MATCH_THRESHOLD <= result.match_score <= 1.0


class TestNotFound:
    def test_completely_different_text_not_found(self):
        extracted = "The parties agree to arbitration in the Netherlands under UNCITRAL rules."
        result = verify_evidence(extracted, SOURCE_TEXT, source_page=1)
        assert result.found_in_source is False
        assert result.match_type == "not_found"

    def test_not_found_still_returns_score(self):
        """Even on not_found, the fuzzy score is returned so the caller can see how close it was."""
        extracted = "Totally unrelated text about widgets and sprockets and other things."
        result = verify_evidence(extracted, SOURCE_TEXT)
        assert result.found_in_source is False
        assert result.match_type == "not_found"
        # match_score is populated even on not_found (it's just below threshold)
        assert result.match_score is not None
        assert result.match_score < FUZZY_MATCH_THRESHOLD

    def test_empty_extracted_text_returns_not_found(self):
        result = verify_evidence("", SOURCE_TEXT, source_page=1)
        assert result.found_in_source is False
        assert result.match_type == "not_found"

    def test_empty_source_text_returns_not_found(self):
        result = verify_evidence("some extracted text", "", source_page=1)
        assert result.found_in_source is False
        assert result.match_type == "not_found"


class TestPageTracking:
    def test_source_page_propagated(self):
        extracted = "Acme Corp."
        result = verify_evidence(extracted, SOURCE_TEXT, source_page=7)
        assert result.source_page_checked == 7

    def test_no_source_page_defaults_none(self):
        extracted = "Acme Corp."
        result = verify_evidence(extracted, SOURCE_TEXT)
        assert result.source_page_checked is None
