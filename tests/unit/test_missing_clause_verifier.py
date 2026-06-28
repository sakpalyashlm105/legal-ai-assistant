"""
tests/unit/test_missing_clause_verifier.py
-------------------------------------------
Unit tests for agent/missing_clause_verifier.py (Step 15).

What this file tests:
    1. Absent clause with NO keyword evidence stays genuinely absent — no escalation.
    2. Absent clause WITH keyword evidence produces an escalation entry.
    3. Present clause is never escalated even if keywords are found.
    4. has_related_keywords returns (False, 0) for an unknown clause type.
    5. has_related_keywords returns (False, 0) on empty chunks list.
    6. keyword_match_count reflects total occurrences, not unique keywords.
    7. verify_missing_clauses returns one entry per absent clause with evidence.
    8. Multiple absent clauses: only the ones with keyword evidence are escalated.
    9. REG-001 invariant: escalated clause keeps risk=HIGH in the node
       (verified by checking that no auto-present marking occurs — the escalation
       dict has no "is_present" field that could confuse downstream consumers).
    10. Escalation note is human-readable and includes the clause_type name.

How to run:
    From legal-agent/ with the venv active:
        pytest tests/unit/test_missing_clause_verifier.py -v
"""

import pytest
from unittest.mock import MagicMock

from schemas.clause import ExtractedClause
from schemas.chunk import DocumentChunk
from agent.missing_clause_verifier import has_related_keywords, verify_missing_clauses


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(text: str, idx: int = 0) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"chunk-{idx}",
        chunk_index=idx,
        text=text,
        token_count=len(text.split()),
        char_count=len(text),
        source_file="test.pdf",
        document_hash="a" * 64,
        document_name="test.pdf",
        start_page=1,
        end_page=1,
        total_chunks=10,
    )


def _absent_clause(clause_type: str) -> ExtractedClause:
    return ExtractedClause(
        clause_type=clause_type,
        is_present=False,
        extracted_text=None,
        confidence=0.0,
    )


def _present_clause(clause_type: str) -> ExtractedClause:
    return ExtractedClause(
        clause_type=clause_type,
        is_present=True,
        extracted_text="Some governing law text.",
        confidence=0.9,
    )


# ---------------------------------------------------------------------------
# Test 1: absent clause, NO keyword match → no escalation
# ---------------------------------------------------------------------------

def test_absent_clause_no_keywords_not_escalated():
    clause = _absent_clause("Governing Law / Jurisdiction")
    chunks = [
        _chunk("This agreement covers the supply of widgets.", 0),
        _chunk("Payment shall be made within 30 days.", 1),
    ]
    escalations = verify_missing_clauses([clause], chunks)
    assert escalations == []


# ---------------------------------------------------------------------------
# Test 2: absent clause WITH keyword evidence → escalation entry
# ---------------------------------------------------------------------------

def test_absent_clause_with_keywords_escalated():
    clause = _absent_clause("Governing Law / Jurisdiction")
    chunks = [
        _chunk("All disputes shall be resolved under the governing law of New York.", 0),
    ]
    escalations = verify_missing_clauses([clause], chunks)
    assert len(escalations) == 1
    esc = escalations[0]
    assert esc["clause_type"] == "Governing Law / Jurisdiction"
    assert esc["possible_clause_under_different_heading"] is True
    assert esc["keyword_match_count"] > 0


# ---------------------------------------------------------------------------
# Test 3: present clause is NEVER escalated
# ---------------------------------------------------------------------------

def test_present_clause_never_escalated():
    clause = _present_clause("Governing Law / Jurisdiction")
    chunks = [
        _chunk("This agreement is governed by the laws of California.", 0),
    ]
    escalations = verify_missing_clauses([clause], chunks)
    assert escalations == []


# ---------------------------------------------------------------------------
# Test 4: unknown clause type returns (False, 0)
# ---------------------------------------------------------------------------

def test_has_related_keywords_unknown_type():
    chunks = [_chunk("Some text with indemnification and arbitration.", 0)]
    found, count = has_related_keywords("Totally Unknown Clause", chunks)
    assert found is False
    assert count == 0


# ---------------------------------------------------------------------------
# Test 5: empty chunks list returns (False, 0)
# ---------------------------------------------------------------------------

def test_has_related_keywords_empty_chunks():
    found, count = has_related_keywords("Indemnification", [])
    assert found is False
    assert count == 0


# ---------------------------------------------------------------------------
# Test 6: keyword_match_count reflects multiple hits
# ---------------------------------------------------------------------------

def test_keyword_match_count_reflects_total_hits():
    chunks = [
        _chunk("The indemnification clause requires the party to indemnify.", 0),
        _chunk("Hold harmless and indemnif obligations apply.", 1),
    ]
    found, count = has_related_keywords("Indemnification", chunks)
    assert found is True
    assert count >= 3  # "indemnif" appears 3x total + "hold harmless" 1x


# ---------------------------------------------------------------------------
# Test 7: verify_missing_clauses returns one entry per escalated clause
# ---------------------------------------------------------------------------

def test_verify_missing_clauses_one_entry_per_escalated():
    clauses = [
        _absent_clause("Governing Law / Jurisdiction"),
        _absent_clause("Non-Compete / Non-Solicitation"),
    ]
    chunks = [
        _chunk("The applicable law shall be the laws of Texas, jurisdiction of Harris County.", 0),
        _chunk("The parties agree not to compete or solicit each other's employees.", 1),
    ]
    escalations = verify_missing_clauses(clauses, chunks)
    types = {e["clause_type"] for e in escalations}
    assert "Governing Law / Jurisdiction" in types
    assert "Non-Compete / Non-Solicitation" in types


# ---------------------------------------------------------------------------
# Test 8: multiple absent clauses — only those with evidence are escalated
# ---------------------------------------------------------------------------

def test_only_clauses_with_evidence_are_escalated():
    clauses = [
        _absent_clause("Governing Law / Jurisdiction"),
        _absent_clause("Renewal / Term"),           # no matching keywords in chunks
    ]
    chunks = [
        _chunk("The courts of New York shall have exclusive jurisdiction.", 0),
        _chunk("This agreement covers the purchase of raw materials.", 1),
    ]
    escalations = verify_missing_clauses(clauses, chunks)
    types = [e["clause_type"] for e in escalations]
    assert "Governing Law / Jurisdiction" in types
    assert "Renewal / Term" not in types


# ---------------------------------------------------------------------------
# Test 9: escalation dict has no is_present field (REG-001 invariant check)
# ---------------------------------------------------------------------------

def test_escalation_dict_does_not_mark_clause_present():
    clause = _absent_clause("Assignment")
    chunks = [_chunk("The agreement shall be binding upon successors and assigns.", 0)]
    escalations = verify_missing_clauses([clause], chunks)
    assert len(escalations) == 1
    esc = escalations[0]
    # The dict must never contain an is_present field — that would risk
    # confusing downstream consumers into thinking the clause was found.
    assert "is_present" not in esc
    # And must never be True even if someone checks:
    assert esc.get("is_present") is None


# ---------------------------------------------------------------------------
# Test 10: escalation note is human-readable and names the clause type
# ---------------------------------------------------------------------------

def test_escalation_note_is_human_readable():
    clause = _absent_clause("Dispute Resolution")
    chunks = [_chunk("The parties agree to submit all disputes to arbitration.", 0)]
    escalations = verify_missing_clauses([clause], chunks)
    assert len(escalations) == 1
    note = escalations[0]["note"]
    assert "Dispute Resolution" in note
    assert len(note) > 50  # not a stub
