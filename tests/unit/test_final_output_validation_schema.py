"""
tests/unit/test_final_output_validation_schema.py
--------------------------------------------------
Unit tests for schemas.report.FinalOutputValidationResult (Step 11).

Tests: construction with various field combinations, immutability, defaults.
No LLM calls, no file I/O.
"""

import pytest
from schemas.report import FinalOutputValidationResult


class TestFinalOutputValidationResultConstruction:
    def test_clean_pass(self):
        result = FinalOutputValidationResult(
            passed=True,
            checks_run=["disclaimer", "clause_count", "schema_completeness", "guardrail_disconnect"],
        )
        assert result.passed is True
        assert result.checks_failed == []
        assert result.auto_fixes_applied == []
        assert result.escalations_created == []
        assert result.notes is None

    def test_with_auto_fix(self):
        result = FinalOutputValidationResult(
            passed=True,
            checks_run=["disclaimer", "clause_count"],
            checks_failed=["disclaimer"],
            auto_fixes_applied=["disclaimer_reinjected"],
        )
        assert result.passed is True
        assert "disclaimer" in result.checks_failed
        assert "disclaimer_reinjected" in result.auto_fixes_applied
        assert result.escalations_created == []

    def test_with_escalation(self):
        result = FinalOutputValidationResult(
            passed=False,
            checks_run=["disclaimer", "clause_count", "schema_completeness"],
            checks_failed=["clause_count"],
            escalations_created=["abc-123"],
            notes="total_clauses_found=3 but 4 clause_entries are present",
        )
        assert result.passed is False
        assert "clause_count" in result.checks_failed
        assert "abc-123" in result.escalations_created
        assert result.notes is not None

    def test_multiple_escalations(self):
        result = FinalOutputValidationResult(
            passed=False,
            checks_run=["disclaimer", "clause_count", "guardrail_disconnect"],
            checks_failed=["clause_count", "guardrail_disconnect"],
            escalations_created=["id-1", "id-2"],
        )
        assert len(result.escalations_created) == 2

    def test_immutability(self):
        result = FinalOutputValidationResult(
            passed=True,
            checks_run=["disclaimer"],
        )
        with pytest.raises(Exception):
            result.passed = False  # type: ignore[misc]

    def test_serialization_round_trip(self):
        result = FinalOutputValidationResult(
            passed=False,
            checks_run=["disclaimer", "clause_count"],
            checks_failed=["clause_count"],
            escalations_created=["rev-999"],
            notes="mismatch",
        )
        data = result.model_dump()
        restored = FinalOutputValidationResult(**data)
        assert restored == result
