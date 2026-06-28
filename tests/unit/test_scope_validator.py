"""
tests/unit/test_scope_validator.py
------------------------------------
Unit tests for guardrails/scope_validator.py.

Tests confirm:
  1. Normal analysis requests pass
  2. Autonomous approval requests are flagged blocking
  3. Contract execution requests are flagged blocking
  4. Definitive legal advice requests are flagged with correct detected_intent
  5. Ambiguous-but-fine requests pass (not overly aggressive)
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from guardrails.scope_validator import validate_scope


# ---------------------------------------------------------------------------
# In-scope requests — must pass
# ---------------------------------------------------------------------------

class TestInScopeRequests:
    def test_standard_analysis_request(self):
        result = validate_scope("Analyze this NDA for risk and missing clauses.")
        assert result.passed is True
        assert result.severity == "info"
        assert result.detected_intent is None

    def test_extract_clauses_request(self):
        result = validate_scope("Extract all clauses from this contract and flag any issues.")
        assert result.passed is True

    def test_risk_scoring_request(self):
        result = validate_scope("What are the high-risk findings in this supplier agreement?")
        assert result.passed is True

    def test_termination_question(self):
        """'What does this contract say about termination' is analysis, not legal advice."""
        result = validate_scope("What does this contract say about termination rights?")
        assert result.passed is True

    def test_summarize_request(self):
        result = validate_scope("Give me a summary of the key clauses in this NDA.")
        assert result.passed is True

    def test_empty_request_passes(self):
        """Empty request defaults to in-scope."""
        result = validate_scope("")
        assert result.passed is True

    def test_compare_to_template(self):
        result = validate_scope("Compare this contract against the standard template.")
        assert result.passed is True

    def test_flag_missing_clauses(self):
        result = validate_scope("Identify any missing critical clauses in this agreement.")
        assert result.passed is True


# ---------------------------------------------------------------------------
# Out-of-scope: binding_legal_approval
# ---------------------------------------------------------------------------

class TestBindingLegalApproval:
    def test_approve_this_contract_blocked(self):
        result = validate_scope("Approve this contract so we can proceed.")
        assert result.passed is False
        assert result.severity == "blocking"
        assert result.detected_intent == "binding_legal_approval"

    def test_certify_agreement_blocked(self):
        result = validate_scope("Certify this agreement as legally valid.")
        assert result.passed is False
        assert result.detected_intent == "binding_legal_approval"

    def test_give_approval_blocked(self):
        result = validate_scope("Give your approval on this NDA.")
        assert result.passed is False
        assert result.detected_intent == "binding_legal_approval"


# ---------------------------------------------------------------------------
# Out-of-scope: contract_execution
# ---------------------------------------------------------------------------

class TestContractExecution:
    def test_sign_contract_blocked(self):
        result = validate_scope("Sign this contract on our behalf.")
        assert result.passed is False
        assert result.severity == "blocking"
        assert result.detected_intent == "contract_execution"

    def test_execute_agreement_blocked(self):
        result = validate_scope("Execute this agreement so it becomes effective.")
        assert result.passed is False
        assert result.detected_intent == "contract_execution"

    def test_finalize_deal_blocked(self):
        result = validate_scope("Finalize this deal and make it binding.")
        assert result.passed is False
        assert result.detected_intent == "contract_execution"


# ---------------------------------------------------------------------------
# Out-of-scope: contract_modification
# ---------------------------------------------------------------------------

class TestContractModification:
    def test_rewrite_contract_blocked(self):
        result = validate_scope("Rewrite the indemnification clause to be more favorable.")
        assert result.passed is False
        assert result.severity == "blocking"
        assert result.detected_intent == "contract_modification"

    def test_add_clause_blocked(self):
        result = validate_scope("Add a dispute resolution clause to this contract.")
        assert result.passed is False
        assert result.detected_intent == "contract_modification"

    def test_remove_clause_blocked(self):
        result = validate_scope("Remove the non-compete clause from the agreement.")
        assert result.passed is False
        assert result.detected_intent == "contract_modification"


# ---------------------------------------------------------------------------
# Out-of-scope: definitive_legal_advice
# ---------------------------------------------------------------------------

class TestDefinitiveLegalAdvice:
    def test_is_this_enforceable_blocked(self):
        result = validate_scope("Is this contract legally enforceable?")
        assert result.passed is False
        assert result.severity == "blocking"
        assert result.detected_intent == "definitive_legal_advice"

    def test_yes_or_no_blocked(self):
        result = validate_scope("Yes or no: is this NDA valid?")
        assert result.passed is False
        assert result.detected_intent == "definitive_legal_advice"

    def test_can_we_legally_proceed_blocked(self):
        result = validate_scope("Can we legally rely on this agreement?")
        assert result.passed is False
        assert result.detected_intent == "definitive_legal_advice"

    def test_give_legal_opinion_blocked(self):
        result = validate_scope("Give us a legal opinion on whether this clause is enforceable.")
        assert result.passed is False
        assert result.detected_intent == "definitive_legal_advice"


# ---------------------------------------------------------------------------
# Not overly aggressive — ambiguous but legitimate requests pass
# ---------------------------------------------------------------------------

class TestAmbiguousButFineRequests:
    def test_what_is_the_governing_law(self):
        """Asking what the contract says about something is analysis, not advice."""
        result = validate_scope("What is the governing law in this agreement?")
        assert result.passed is True

    def test_is_the_confidentiality_clause_present(self):
        """Asking about clause presence is analysis."""
        result = validate_scope("Is the confidentiality clause present in this NDA?")
        assert result.passed is True, (
            f"Legitimate 'is X present' question incorrectly flagged: {result.detected_intent}"
        )

    def test_check_compliance_of_document(self):
        """'Check compliance' as an analysis task is fine."""
        result = validate_scope("Check this contract for compliance with our internal policy.")
        assert result.passed is True

    def test_review_for_issues(self):
        result = validate_scope("Review this supplier agreement and highlight any concerns.")
        assert result.passed is True
