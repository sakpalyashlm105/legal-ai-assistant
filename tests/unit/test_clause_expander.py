"""
tests/unit/test_clause_expander.py
------------------------------------
Unit tests for agent/clause_expander.py.

Key scenario tested (requirement 5 from the task):
    Section 1 of an NDA defines Protected Information.
    Section 2 of the same NDA contains Director's Obligations (non-disclosure,
    non-use, reasonable security, survival language).

    The clause extractor anchors on Section 1 only.  The expander should detect
    that Section 2 is about the same Confidentiality topic and include it.
    When the expanded text is passed to the comparator the result should NOT be
    "major deviation" merely because Section 1 alone lacks obligations.

    The tests here verify the expansion logic itself; comparator behavior with
    expanded text is tested via a mock to avoid LLM API calls.
"""

import pytest
from unittest.mock import patch, MagicMock

from agent.clause_expander import (
    ClauseExpansionResult,
    CLAUSE_RELATED_KEYWORDS,
    _first_nonblank_line,
    _starts_new_section,
    _heading_belongs_to_clause_type,
    _is_related_content,
    expand_clause_context,
    expand_all_clauses,
)
from schemas.chunk import DocumentChunk
from schemas.clause import ExtractedClause


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(
    chunk_id: str,
    text: str,
    chunk_index: int,
    total_chunks: int = 5,
    start_page: int = 1,
    end_page: int = 1,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_hash="testhash",
        document_name="test.pdf",
        text=text,
        char_count=len(text),
        start_page=start_page,
        end_page=end_page,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
    )


def _make_clause(
    clause_type: str = "Confidentiality / Non-Disclosure",
    extracted_text: str = "some clause text",
    source_chunk_id: str = "chunk_0",
    is_present: bool = True,
    page_reference: int = 1,
) -> ExtractedClause:
    return ExtractedClause(
        clause_type=clause_type,
        is_present=is_present,
        extracted_text=extracted_text if is_present else None,
        page_reference=page_reference if is_present else None,
        confidence=0.9,
        source_chunk_id=source_chunk_id if is_present else None,
    )


# ---------------------------------------------------------------------------
# Section 1 + Section 2 NDA fixture (the core scenario from the bug report)
# ---------------------------------------------------------------------------

SECTION_1_TEXT = (
    "1. Protected Information Defined; Exclusions\n"
    'As used in this Agreement, "Protected Information" means any information '
    "disclosed by the Company to Director that is designated as confidential or "
    "proprietary, including but not limited to trade secrets, business plans, "
    "financial data, and customer lists.\n\n"
    "The following shall not constitute Protected Information: information that is "
    "or becomes publicly available through no fault of Director."
)

SECTION_2_TEXT = (
    "2. Director's Obligations\n"
    "Director shall:\n"
    "(a) not disclose Protected Information, directly or indirectly, to any third "
    "person without the express written consent of the Company;\n"
    "(b) not use Protected Information for any purpose other than as authorized "
    "by the Company;\n"
    "(c) use reasonable security precautions to protect Protected Information; and\n"
    "(d) promptly return or destroy all Protected Information upon request.\n\n"
    "The obligations of this Section 2 shall survive termination of this Agreement."
)

SECTION_3_GOVERNING_LAW = (
    "3. Governing Law\n"
    "This Agreement shall be governed by and construed in accordance with the laws "
    "of the State of Nevada, without giving effect to any choice of law provisions."
)

SECTION_3_INDEMNIFICATION = (
    "3. Indemnification\n"
    "Each party shall indemnify, defend, and hold harmless the other party from "
    "any claims, liabilities, losses, or damages arising from a breach of this Agreement."
)


# ---------------------------------------------------------------------------
# TestHelperFunctions
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    def test_first_nonblank_line_skips_blanks(self):
        text = "\n\n  \n1. My Section\nrest of text"
        assert _first_nonblank_line(text) == "1. My Section"

    def test_first_nonblank_line_empty_string(self):
        assert _first_nonblank_line("") == ""
        assert _first_nonblank_line("   \n   ") == ""

    def test_starts_new_section_numbered_dot(self):
        assert _starts_new_section("1. Protected Information") is not None
        assert _starts_new_section("2. Director's Obligations") is not None
        assert _starts_new_section("10. Miscellaneous") is not None

    def test_starts_new_section_numbered_paren(self):
        assert _starts_new_section("3) Governing Law") is not None

    def test_starts_new_section_section_keyword(self):
        assert _starts_new_section("Section 2. Term") is not None
        assert _starts_new_section("SECTION 4. Assignment") is not None

    def test_starts_new_section_mid_text_not_matched(self):
        # Heading mid-chunk (not the first non-blank line) must NOT trigger
        text = "Some preamble.\n\n2. A section that appears mid-chunk"
        # The first non-blank line is "Some preamble." which does not match
        assert _starts_new_section(text) is None

    def test_starts_new_section_plain_paragraph(self):
        assert _starts_new_section("The parties agree that...") is None
        assert _starts_new_section("  (a) not disclose...") is None

    def test_heading_belongs_to_clause_type_confidentiality(self):
        result = _heading_belongs_to_clause_type("2. Director's Obligations")
        assert result == "Confidentiality / Non-Disclosure"

    def test_heading_belongs_to_clause_type_governing_law(self):
        result = _heading_belongs_to_clause_type("3. Governing Law")
        assert result == "Governing Law / Jurisdiction"

    def test_heading_belongs_to_clause_type_indemnification(self):
        result = _heading_belongs_to_clause_type("4. Indemnification Obligations")
        assert result == "Indemnification"

    def test_heading_belongs_to_clause_type_unknown(self):
        result = _heading_belongs_to_clause_type("5. Miscellaneous Provisions")
        assert result is None

    def test_is_related_content_confidentiality(self):
        assert _is_related_content("Director shall not disclose Protected Information", "Confidentiality / Non-Disclosure")
        assert _is_related_content("The obligations survive termination", "Confidentiality / Non-Disclosure")
        assert not _is_related_content("The parties choose Nevada as the governing jurisdiction", "Confidentiality / Non-Disclosure")

    def test_is_related_content_governing_law(self):
        assert _is_related_content("This Agreement shall be governed by Nevada laws", "Governing Law / Jurisdiction")
        assert not _is_related_content("Director shall not disclose Protected Information", "Governing Law / Jurisdiction")

    def test_is_related_content_unknown_clause_type(self):
        # Unknown clause type has no keywords -> always False
        assert not _is_related_content("anything", "Totally Unknown Clause Type")


# ---------------------------------------------------------------------------
# TestExpansionAbsentAndEdgeCases
# ---------------------------------------------------------------------------

class TestExpansionAbsentAndEdgeCases:
    def test_absent_clause_no_expansion(self):
        clause = _make_clause(is_present=False, source_chunk_id=None)
        chunks = [_make_chunk("chunk_0", SECTION_1_TEXT, 0)]
        result = expand_clause_context(clause, chunks)
        assert result.expansion_triggered is False
        assert result.boundary_reason == "absent_or_no_source"
        assert result.expanded_text == ""  # original_text is None for absent

    def test_no_source_chunk_id(self):
        clause = ExtractedClause(
            clause_type="Confidentiality / Non-Disclosure",
            is_present=True,
            extracted_text="Some text",
            page_reference=1,
            confidence=0.9,
            source_chunk_id=None,
        )
        chunks = [_make_chunk("chunk_0", SECTION_1_TEXT, 0)]
        result = expand_clause_context(clause, chunks)
        assert result.expansion_triggered is False
        assert result.boundary_reason == "absent_or_no_source"

    def test_source_chunk_not_in_list(self):
        clause = _make_clause(source_chunk_id="chunk_999")
        chunks = [_make_chunk("chunk_0", SECTION_1_TEXT, 0)]
        result = expand_clause_context(clause, chunks)
        assert result.expansion_triggered is False
        assert result.boundary_reason == "source_not_found"

    def test_no_subsequent_chunks(self):
        clause = _make_clause(source_chunk_id="chunk_0", extracted_text=SECTION_1_TEXT)
        chunks = [_make_chunk("chunk_0", SECTION_1_TEXT, 0, total_chunks=1)]
        result = expand_clause_context(clause, chunks)
        assert result.expansion_triggered is False
        assert result.boundary_reason == "end_of_document"
        assert result.expanded_text == SECTION_1_TEXT

    def test_empty_chunks_list(self):
        clause = _make_clause(source_chunk_id="chunk_0")
        result = expand_clause_context(clause, [])
        assert result.expansion_triggered is False
        assert result.boundary_reason == "source_not_found"

    def test_result_is_pydantic_model(self):
        clause = _make_clause(source_chunk_id="chunk_0", extracted_text=SECTION_1_TEXT)
        chunks = [_make_chunk("chunk_0", SECTION_1_TEXT, 0, total_chunks=1)]
        result = expand_clause_context(clause, chunks)
        assert isinstance(result, ClauseExpansionResult)
        # Must be serialisable (for orchestrator state)
        d = result.model_dump(mode="json")
        assert d["clause_type"] == "Confidentiality / Non-Disclosure"
        assert isinstance(d["source_chunk_ids"], list)


# ---------------------------------------------------------------------------
# TestCoreScenario: Section 1 definition + Section 2 obligations
# ---------------------------------------------------------------------------

class TestCoreScenario:
    """
    The scenario from the bug report:
        Section 1  -- "Protected Information Defined" (definition only)
        Section 2  -- "Director's Obligations" (non-disclosure, non-use, survival)

    The extractor anchors on Section 1.  Expansion must include Section 2.
    """

    def _make_two_section_chunks(self):
        chunk0 = _make_chunk("chunk_0", SECTION_1_TEXT, 0, total_chunks=2, start_page=1, end_page=1)
        chunk1 = _make_chunk("chunk_1", SECTION_2_TEXT, 1, total_chunks=2, start_page=1, end_page=2)
        return [chunk0, chunk1]

    def test_expansion_triggered(self):
        clause = _make_clause(
            source_chunk_id="chunk_0",
            extracted_text=SECTION_1_TEXT,
        )
        chunks = self._make_two_section_chunks()
        result = expand_clause_context(clause, chunks)
        assert result.expansion_triggered is True

    def test_expanded_text_includes_section_2_obligations(self):
        clause = _make_clause(
            source_chunk_id="chunk_0",
            extracted_text=SECTION_1_TEXT,
        )
        chunks = self._make_two_section_chunks()
        result = expand_clause_context(clause, chunks)
        assert "Director's Obligations" in result.expanded_text
        assert "not disclose" in result.expanded_text
        assert "survive" in result.expanded_text.lower()

    def test_both_chunk_ids_in_source_chunks_used(self):
        clause = _make_clause(
            source_chunk_id="chunk_0",
            extracted_text=SECTION_1_TEXT,
        )
        chunks = self._make_two_section_chunks()
        result = expand_clause_context(clause, chunks)
        assert "chunk_0" in result.source_chunk_ids
        assert "chunk_1" in result.source_chunk_ids
        assert len(result.source_chunk_ids) == 2

    def test_original_text_preserved(self):
        """original_text must never be mutated."""
        clause = _make_clause(
            source_chunk_id="chunk_0",
            extracted_text=SECTION_1_TEXT,
        )
        chunks = self._make_two_section_chunks()
        result = expand_clause_context(clause, chunks)
        assert result.original_text == SECTION_1_TEXT

    def test_pages_used_includes_both_pages(self):
        clause = _make_clause(
            source_chunk_id="chunk_0",
            extracted_text=SECTION_1_TEXT,
        )
        chunks = self._make_two_section_chunks()
        result = expand_clause_context(clause, chunks)
        assert 1 in result.pages_used
        assert 2 in result.pages_used

    def test_boundary_reason_end_of_document(self):
        clause = _make_clause(
            source_chunk_id="chunk_0",
            extracted_text=SECTION_1_TEXT,
        )
        chunks = self._make_two_section_chunks()
        result = expand_clause_context(clause, chunks)
        assert result.boundary_reason == "end_of_document"

    def test_comparator_receives_expanded_text(self):
        """
        Requirement 5: the risk scorer must not mark HIGH merely because
        Section 1 alone lacks obligations.  Here we verify the architecture:
        when expansion is triggered, compare_to_templates is called with
        expanded_text, not the snippet.

        We mock the LLM call in the comparator so no API cost is incurred.
        The mock returns "none" deviation (no risk) -- verifying that if the
        expanded text is used, the comparator WOULD see the full clause and
        could return a lower deviation.
        """
        from agent.comparator import compare_to_templates
        from agent.clause_expander import ClauseExpansionResult

        # A clause that extracted only Section 1
        clause = _make_clause(
            source_chunk_id="chunk_0",
            extracted_text=SECTION_1_TEXT,
        )
        clause.is_present = True

        # Simulate what the expander produces
        expansion = ClauseExpansionResult(
            clause_type="Confidentiality / Non-Disclosure",
            original_text=SECTION_1_TEXT,
            expanded_text=SECTION_1_TEXT + "\n\n" + SECTION_2_TEXT,
            source_chunk_ids=["chunk_0", "chunk_1"],
            expansion_triggered=True,
            boundary_reason="end_of_document",
            pages_used=[1, 2],
        )
        expansions_dict = {"Confidentiality / Non-Disclosure": expansion}

        captured_texts = []

        def _mock_call_llm(clause_type, template_text, extracted_text):
            captured_texts.append(extracted_text)
            return {"matches_template": True, "deviation_severity": "none", "deviation_summary": None}

        with patch("agent.comparator._call_llm", side_effect=_mock_call_llm), \
             patch("agent.comparator._load_template", return_value=("Template text", "/fake/path")):
            compare_to_templates([clause], expansions=expansions_dict)

        assert len(captured_texts) == 1
        # Comparator must have used the EXPANDED text (Section 1 + Section 2)
        assert SECTION_2_TEXT[:50] in captured_texts[0], (
            "Comparator was not passed the expanded text -- "
            "it still sees only the original Section 1 snippet"
        )

    def test_snippet_only_without_expansion_does_not_include_section_2(self):
        """
        Baseline: without expansion the comparator only sees Section 1 text.
        This is the false-positive scenario the expander fixes.
        """
        from agent.comparator import compare_to_templates

        clause = _make_clause(
            source_chunk_id="chunk_0",
            extracted_text=SECTION_1_TEXT,
        )

        captured_texts = []

        def _mock_call_llm(clause_type, template_text, extracted_text):
            captured_texts.append(extracted_text)
            return {"matches_template": False, "deviation_severity": "major", "deviation_summary": "Missing obligations"}

        with patch("agent.comparator._call_llm", side_effect=_mock_call_llm), \
             patch("agent.comparator._load_template", return_value=("Template text", "/fake/path")):
            compare_to_templates([clause], expansions=None)

        assert len(captured_texts) == 1
        # Without expansion, only Section 1 is seen
        assert SECTION_2_TEXT[:50] not in captured_texts[0]


# ---------------------------------------------------------------------------
# TestExpansionStopConditions
# ---------------------------------------------------------------------------

class TestExpansionStopConditions:
    def test_stops_at_governing_law_heading(self):
        """Governing Law heading after Confidentiality clause must stop expansion."""
        chunk0 = _make_chunk("c0", SECTION_1_TEXT, 0, total_chunks=2)
        chunk1 = _make_chunk("c1", SECTION_3_GOVERNING_LAW, 1, total_chunks=2)
        clause = _make_clause(source_chunk_id="c0", extracted_text=SECTION_1_TEXT)

        result = expand_clause_context(clause, [chunk0, chunk1])
        assert result.expansion_triggered is False
        assert result.boundary_reason == "new_unrelated_heading"
        assert "Governing Law" not in result.expanded_text

    def test_stops_at_indemnification_heading(self):
        """Indemnification heading after Confidentiality must stop expansion."""
        chunk0 = _make_chunk("c0", SECTION_1_TEXT, 0, total_chunks=2)
        chunk1 = _make_chunk("c1", SECTION_3_INDEMNIFICATION, 1, total_chunks=2)
        clause = _make_clause(source_chunk_id="c0", extracted_text=SECTION_1_TEXT)

        result = expand_clause_context(clause, [chunk0, chunk1])
        assert result.expansion_triggered is False
        assert result.boundary_reason == "new_unrelated_heading"

    def test_stops_at_max_chunks(self):
        """max_extra_chunks=1 must not include more than 1 additional chunk."""
        chunk0 = _make_chunk("c0", SECTION_1_TEXT, 0, total_chunks=3)
        chunk1 = _make_chunk("c1", SECTION_2_TEXT, 1, total_chunks=3)
        chunk2 = _make_chunk("c2", "2b. Additional confidentiality obligations.\nDirector must protect all data.", 2, total_chunks=3)
        clause = _make_clause(source_chunk_id="c0", extracted_text=SECTION_1_TEXT)

        result = expand_clause_context(clause, [chunk0, chunk1, chunk2], max_extra_chunks=1)
        assert result.expansion_triggered is True
        assert len(result.source_chunk_ids) == 2  # source + 1 extra max
        assert result.boundary_reason == "max_chunks_reached"

    def test_stops_at_unrelated_content_no_heading(self):
        """A chunk with no heading and no related keywords terminates expansion."""
        chunk0 = _make_chunk("c0", SECTION_1_TEXT, 0, total_chunks=2)
        unrelated_text = (
            "Exhibit A -- List of Approved Vendors\n"
            "The following vendors have been pre-approved by the procurement committee."
        )
        chunk1 = _make_chunk("c1", unrelated_text, 1, total_chunks=2)
        clause = _make_clause(source_chunk_id="c0", extracted_text=SECTION_1_TEXT)

        result = expand_clause_context(clause, [chunk0, chunk1])
        assert result.expansion_triggered is False
        assert result.boundary_reason == "unrelated_content"

    def test_three_chunk_full_clause_group(self):
        """Three chunks all belonging to Confidentiality are merged correctly."""
        part3 = (
            "2b. Additional Security Requirements\n"
            "Director shall maintain confidential data only on encrypted devices and "
            "report any unauthorized disclosure immediately."
        )
        chunk0 = _make_chunk("c0", SECTION_1_TEXT, 0, total_chunks=3)
        chunk1 = _make_chunk("c1", SECTION_2_TEXT, 1, total_chunks=3, start_page=2, end_page=2)
        chunk2 = _make_chunk("c2", part3, 2, total_chunks=3, start_page=3, end_page=3)
        clause = _make_clause(source_chunk_id="c0", extracted_text=SECTION_1_TEXT)

        result = expand_clause_context(clause, [chunk0, chunk1, chunk2])
        assert result.expansion_triggered is True
        assert len(result.source_chunk_ids) == 3
        assert "encrypted devices" in result.expanded_text
        assert result.pages_used == [1, 2, 3]


# ---------------------------------------------------------------------------
# TestExpandAllClauses
# ---------------------------------------------------------------------------

class TestExpandAllClauses:
    def test_returns_dict_keyed_by_clause_type(self):
        from config import CLAUSE_CATEGORIES
        clauses = []
        for cat in CLAUSE_CATEGORIES:
            clauses.append(ExtractedClause(
                clause_type=cat,
                is_present=False,
                extracted_text=None,
                page_reference=None,
                confidence=0.0,
                source_chunk_id=None,
            ))
        result = expand_all_clauses(clauses, [])
        assert set(result.keys()) == set(CLAUSE_CATEGORIES)

    def test_absent_clauses_not_expanded(self):
        from config import CLAUSE_CATEGORIES
        clauses = []
        for cat in CLAUSE_CATEGORIES:
            clauses.append(ExtractedClause(
                clause_type=cat,
                is_present=False,
                extracted_text=None,
                page_reference=None,
                confidence=0.0,
                source_chunk_id=None,
            ))
        result = expand_all_clauses(clauses, [])
        for exp in result.values():
            assert exp.expansion_triggered is False

    def test_present_clause_expanded_when_related_chunk_follows(self):
        chunk0 = _make_chunk("c0", SECTION_1_TEXT, 0, total_chunks=2)
        chunk1 = _make_chunk("c1", SECTION_2_TEXT, 1, total_chunks=2)
        clause = _make_clause(source_chunk_id="c0", extracted_text=SECTION_1_TEXT)

        result = expand_all_clauses([clause], [chunk0, chunk1])
        exp = result["Confidentiality / Non-Disclosure"]
        assert exp.expansion_triggered is True


# ---------------------------------------------------------------------------
# TestKeywordTable
# ---------------------------------------------------------------------------

class TestKeywordTable:
    def test_all_10_clause_types_have_keywords(self):
        from config import CLAUSE_CATEGORIES
        for cat in CLAUSE_CATEGORIES:
            assert cat in CLAUSE_RELATED_KEYWORDS, f"No keywords for '{cat}'"
            assert len(CLAUSE_RELATED_KEYWORDS[cat]) >= 3, (
                f"Too few keywords for '{cat}': {CLAUSE_RELATED_KEYWORDS[cat]}"
            )

    def test_keywords_are_lowercase(self):
        for clause_type, keywords in CLAUSE_RELATED_KEYWORDS.items():
            for kw in keywords:
                assert kw == kw.lower(), (
                    f"Keyword '{kw}' for '{clause_type}' is not lowercase -- "
                    "substring matching will miss uppercase occurrences"
                )
