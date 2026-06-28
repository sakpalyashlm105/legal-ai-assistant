"""
tests/unit/test_risk_engine_precedent.py
-----------------------------------------
Stage-5 precedent integration tests for agent/risk_engine.py.

These tests verify the JSONL-backed precedent matching introduced in Stage 5:
    1. not_eligible and pending records are ignored by the loader.
    2. An approved_precedent MEDIUM record IS applied when all compatibility
       checks pass (category + document_type + similarity).
    3. Missing-clause findings are never downgraded regardless of precedent
       (REG-001 -- run again here as part of Stage 5 evidence).
    4. Incompatible document_type blocks the precedent.
    5. Text similarity below 0.70 does not apply the precedent.
    6. Applied precedent records feedback_id and match reason in precedent_note.
    7. A malformed/corrupt JSONL line does not crash scoring; it is skipped
       with a logged warning and scoring continues normally for everything else.

No OpenAI calls are made. All tests use tmp_path isolation.
"""

import json
import logging
import pytest
from datetime import datetime
from pathlib import Path

from schemas.clause import ExtractedClause
from schemas.risk import ClauseComparison, RiskFinding
from agent.risk_engine import flag_risks, _clear_feedback_cache


# ---------------------------------------------------------------------------
# Fixture: redirect FEEDBACK_LOG_PATH and reset cache for every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolate_feedback_log(tmp_path, monkeypatch):
    _clear_feedback_cache()
    monkeypatch.setattr(
        "agent.risk_engine.FEEDBACK_LOG_PATH",
        tmp_path / "feedback_log.jsonl",
    )
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
    matches_template: bool = False,
    deviation_severity: str = "major",
    deviation_summary: str | None = "Test deviation.",
) -> ClauseComparison:
    return ClauseComparison(
        clause_type=clause_type,
        template_found=template_found,
        matches_template=matches_template,
        deviation_severity=deviation_severity,
        deviation_summary=deviation_summary,
    )


def _base_record(
    clause_category: str,
    evidence_excerpt: str,
    *,
    feedback_id: str = "fb_stage5-test-001",
    feedback_status: str = "approved_precedent",
    approved_for_precedent: bool = True,
    final_risk: str = "MEDIUM",
    document_type_scope: str | None = None,
) -> dict:
    """Minimal valid FeedbackRecord dict for JSONL injection."""
    return {
        "feedback_id": feedback_id,
        "document_id": "stage5testdoc",
        "document_name": "Stage5_Test_Agreement.pdf",
        "document_type": "Contract",
        "clause_category": clause_category,
        "source_page": 3,
        "source_chunk_ids": [],
        "evidence_excerpt": evidence_excerpt,
        "original_model_category": clause_category,
        "original_model_risk": "MEDIUM",
        "original_model_reason": "Stage 5 test fixture.",
        "original_confidence": 0.88,
        "review_action": "approve",
        "final_category": clause_category,
        "final_risk": final_risk,
        "reviewer_comment": "Approved.",
        "model_finding_accepted": True,
        "clause_language_accepted_as_business_precedent": True,
        "is_clause_present": True,
        "feedback_status": feedback_status,
        "approved_for_precedent": approved_for_precedent,
        "precedent_scope": {
            "clause_category": clause_category,
            "document_type": document_type_scope,
            "jurisdiction": None,
            "template_version": None,
        },
        "precedent_approved_by": "curator@company.com",
        "precedent_approved_at": "2026-06-28T10:00:00",
        "model_name": "gpt-4o-mini",
        "created_at": "2026-06-28T09:00:00",
    }


def _write_jsonl(tmp_path: Path, records: list[dict]) -> Path:
    log = tmp_path / "feedback_log.jsonl"
    log.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )
    return log


# ---------------------------------------------------------------------------
# Test 1: not_eligible and pending records are filtered out
# ---------------------------------------------------------------------------

class TestIneligibleRecordsAreIgnored:
    def test_not_eligible_record_not_applied(self, tmp_path):
        _write_jsonl(tmp_path, [_base_record(
            "Governing Law / Jurisdiction",
            "laws of the State of Illinois, without regard to conflict of law provisions",
            feedback_status="not_eligible",
            approved_for_precedent=False,
        )])

        clause = _make_clause(
            "Governing Law / Jurisdiction",
            text="This Agreement shall be governed by the laws of the State of Illinois, "
                 "without regard to its conflict of law provisions.",
        )
        comparison = _make_comparison("Governing Law / Jurisdiction", deviation_severity="major")
        findings = flag_risks([clause], [comparison])

        f = findings[0]
        assert f.risk_level == "HIGH"   # no downgrade — record was not_eligible
        assert f.precedent_applied is False

    def test_pending_record_not_applied(self, tmp_path):
        _write_jsonl(tmp_path, [_base_record(
            "Governing Law / Jurisdiction",
            "laws of the State of Illinois, without regard to conflict of law provisions",
            feedback_status="pending_precedent_review",
            approved_for_precedent=False,
        )])

        clause = _make_clause(
            "Governing Law / Jurisdiction",
            text="This Agreement shall be governed by the laws of the State of Illinois, "
                 "without regard to its conflict of law provisions.",
        )
        comparison = _make_comparison("Governing Law / Jurisdiction", deviation_severity="major")
        findings = flag_risks([clause], [comparison])

        f = findings[0]
        assert f.risk_level == "HIGH"   # no downgrade — record is pending
        assert f.precedent_applied is False


# ---------------------------------------------------------------------------
# Test 2: approved_precedent MEDIUM record IS applied when all checks pass
# ---------------------------------------------------------------------------

class TestApprovedPrecedentIsApplied:
    def test_major_deviation_downgraded_to_medium(self, tmp_path):
        # The real fb_rev-medium-report-001 record from our live log.
        # Clause text matches the Illinois Governing Law evidence_excerpt.
        _write_jsonl(tmp_path, [_base_record(
            "Governing Law / Jurisdiction",
            evidence_excerpt=(
                "This Agreement shall be governed by and construed in accordance with "
                "the laws of the State of Illinois, without regard to its conflict of "
                "law provisions."
            ),
            feedback_id="fb_rev-medium-report-001",
        )])

        clause = _make_clause(
            "Governing Law / Jurisdiction",
            text=(
                "This Agreement shall be governed by and construed in accordance with "
                "the laws of the State of Illinois, without regard to its conflict of "
                "law provisions."
            ),
        )
        comparison = _make_comparison(
            "Governing Law / Jurisdiction",
            deviation_severity="major",
            deviation_summary="Governing law is Illinois; template requires Delaware.",
        )

        findings = flag_risks([clause], [comparison])

        f = findings[0]
        assert f.risk_level == "MEDIUM"        # HIGH downgraded via precedent
        assert f.precedent_applied is True
        assert f.precedent_note is not None
        assert "fb_rev-medium-report-001" in f.precedent_note

    def test_minor_deviation_downgraded_to_low(self, tmp_path):
        _write_jsonl(tmp_path, [_base_record(
            "Termination for Convenience",
            evidence_excerpt="terminate upon sixty (60) days written notice",
            feedback_id="fb_term-sixty-days",
        )])

        clause = _make_clause(
            "Termination for Convenience",
            text="Either party may terminate upon sixty (60) days written notice to the other party.",
        )
        comparison = _make_comparison(
            "Termination for Convenience",
            deviation_severity="minor",
            deviation_summary="Notice period extended from 30 to 60 days.",
        )

        findings = flag_risks([clause], [comparison])

        f = findings[0]
        assert f.risk_level == "LOW"          # MEDIUM downgraded via precedent
        assert f.precedent_applied is True


# ---------------------------------------------------------------------------
# Test 3: REG-001 — missing clause never downgraded (Stage 5 re-run)
# ---------------------------------------------------------------------------

class TestReg001MissingClauseNeverDowngraded:
    def test_missing_clause_stays_high_despite_precedent(self, tmp_path):
        """
        REG-001 re-run as part of Stage 5 evidence.

        An approved_precedent record for "Indemnification" exists in the log.
        The current finding is a MISSING Indemnification clause.
        Result must be HIGH, precedent_applied=False.
        """
        _write_jsonl(tmp_path, [_base_record(
            "Indemnification",
            evidence_excerpt="indemnify and hold harmless against any third-party claims",
            feedback_id="fb_indem-precedent-001",
        )])

        clause = _make_clause("Indemnification", is_present=False)
        comparison = _make_comparison(
            "Indemnification",
            deviation_severity="none",
            deviation_summary=None,
        )

        findings = flag_risks([clause], [comparison])

        f = findings[0]
        # REG-001: missing clause always HIGH, no precedent override
        assert f.risk_level == "HIGH"
        assert f.precedent_applied is False
        assert f.is_missing is True
        assert f.precedent_note is None


# ---------------------------------------------------------------------------
# Test 4: incompatible document_type blocks the precedent
# ---------------------------------------------------------------------------

class TestDocumentTypeIncompatibility:
    def test_wrong_document_type_not_applied(self, tmp_path):
        # Precedent scoped to "NDA" only
        _write_jsonl(tmp_path, [_base_record(
            "Governing Law / Jurisdiction",
            evidence_excerpt="laws of the State of Illinois, without regard to conflict",
            document_type_scope="NDA",
        )])

        clause = _make_clause(
            "Governing Law / Jurisdiction",
            text=(
                "This Agreement shall be governed by the laws of the State of Illinois, "
                "without regard to its conflict of law provisions."
            ),
        )
        comparison = _make_comparison("Governing Law / Jurisdiction", deviation_severity="major")

        # Caller passes document_type="Contract" — mismatches the NDA-scoped precedent
        findings = flag_risks([clause], [comparison], document_type="Contract")

        f = findings[0]
        assert f.risk_level == "HIGH"    # not downgraded — wrong document type
        assert f.precedent_applied is False

    def test_null_scope_applies_to_any_document_type(self, tmp_path):
        # Precedent scope has document_type=None -> applies to all types
        _write_jsonl(tmp_path, [_base_record(
            "Governing Law / Jurisdiction",
            evidence_excerpt="laws of the State of Illinois, without regard to conflict",
            document_type_scope=None,   # no constraint
        )])

        clause = _make_clause(
            "Governing Law / Jurisdiction",
            text=(
                "This Agreement shall be governed by the laws of the State of Illinois, "
                "without regard to its conflict of law provisions."
            ),
        )
        comparison = _make_comparison("Governing Law / Jurisdiction", deviation_severity="major")

        # Any document_type should match when scope is null
        findings = flag_risks([clause], [comparison], document_type="Contract")

        f = findings[0]
        assert f.risk_level == "MEDIUM"   # downgraded — null scope matches all
        assert f.precedent_applied is True


# ---------------------------------------------------------------------------
# Test 5: text similarity below 0.70 does not apply precedent
# ---------------------------------------------------------------------------

class TestSimilarityThreshold:
    def test_low_similarity_not_applied(self, tmp_path):
        # evidence_excerpt is about Delaware (very different from "Illinois")
        _write_jsonl(tmp_path, [_base_record(
            "Governing Law / Jurisdiction",
            evidence_excerpt=(
                "This agreement is subject to the laws of the State of Delaware "
                "and courts in Wilmington shall have exclusive jurisdiction over "
                "all disputes arising hereunder between the contracting parties."
            ),
        )])

        # Current clause: California, completely different text
        clause = _make_clause(
            "Governing Law / Jurisdiction",
            text=(
                "This Agreement shall be governed by the laws of the State of California. "
                "Any disputes shall be resolved exclusively in the courts of San Francisco."
            ),
        )
        comparison = _make_comparison("Governing Law / Jurisdiction", deviation_severity="major")

        findings = flag_risks([clause], [comparison])

        f = findings[0]
        # Different jurisdiction text should score below 0.70
        assert f.risk_level == "HIGH"
        assert f.precedent_applied is False


# ---------------------------------------------------------------------------
# Test 6: applied precedent records feedback_id and match reason
# ---------------------------------------------------------------------------

class TestPrecedentNoteContent:
    def test_note_contains_feedback_id_and_score(self, tmp_path):
        _write_jsonl(tmp_path, [_base_record(
            "Governing Law / Jurisdiction",
            evidence_excerpt=(
                "governed by and construed in accordance with the laws of the State of Illinois"
            ),
            feedback_id="fb_note-content-test-001",
        )])

        clause = _make_clause(
            "Governing Law / Jurisdiction",
            text=(
                "This Agreement shall be governed by and construed in accordance with "
                "the laws of the State of Illinois, without regard to conflict of law provisions."
            ),
        )
        comparison = _make_comparison("Governing Law / Jurisdiction", deviation_severity="major")

        findings = flag_risks([clause], [comparison])

        f = findings[0]
        assert f.precedent_applied is True
        note = f.precedent_note
        assert note is not None
        # Note must identify the source record
        assert "fb_note-content-test-001" in note
        # Note must contain match reason (category + score)
        assert "Governing Law / Jurisdiction" in note
        assert "->" in note  # original risk -> downgraded risk


# ---------------------------------------------------------------------------
# Test 7: malformed JSONL line does not crash scoring
# ---------------------------------------------------------------------------

class TestMalformedLineHandling:
    def test_corrupt_line_skipped_scoring_continues(self, tmp_path, caplog):
        # Write: one corrupt line, then one valid approved record
        log_path = tmp_path / "feedback_log.jsonl"
        valid_record = _base_record(
            "Governing Law / Jurisdiction",
            evidence_excerpt=(
                "laws of the State of Illinois, without regard to its conflict of law provisions"
            ),
            feedback_id="fb_after-corrupt-line",
        )
        log_path.write_text(
            '{"this is": "completely broken json because it has no required fields"}\n'
            + json.dumps(valid_record) + "\n",
            encoding="utf-8",
        )

        clause = _make_clause(
            "Governing Law / Jurisdiction",
            text=(
                "This Agreement shall be governed by the laws of the State of Illinois, "
                "without regard to its conflict of law provisions."
            ),
        )
        comparison = _make_comparison("Governing Law / Jurisdiction", deviation_severity="major")

        with caplog.at_level(logging.WARNING, logger="agent.risk_engine"):
            findings = flag_risks([clause], [comparison])

        # Scoring must complete — no exception raised
        f = findings[0]
        assert isinstance(f, RiskFinding)

        # The corrupt line should produce a WARNING log entry
        assert any("malformed" in r.message.lower() for r in caplog.records), (
            "Expected a WARNING about the malformed line, got: "
            + str([r.message for r in caplog.records])
        )

        # The valid record after the corrupt line must still apply
        assert f.risk_level == "MEDIUM"
        assert f.precedent_applied is True
        assert "fb_after-corrupt-line" in (f.precedent_note or "")
