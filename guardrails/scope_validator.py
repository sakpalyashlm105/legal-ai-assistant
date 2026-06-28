"""
guardrails/scope_validator.py
-------------------------------
Scope validation guardrail: checks whether an incoming user request is asking
for something this system is designed and authorized to do.

What this system IS designed to do:
  - Analyze legal documents for risk (NDAs, contracts, amendments)
  - Extract and categorize clauses
  - Flag risks and missing clauses
  - Generate analysis reports with appropriate disclaimers
  - Surface findings for human review

What this system MUST REFUSE to do (out-of-scope intents):
  - Autonomously approve, certify, execute, or sign a contract
  - Modify or rewrite contract terms
  - Provide definitive legal advice (as opposed to AI-assisted analysis)
  - Act as a binding legal authority for any party

Public function:
  validate_scope(user_request_text) -> ScopeValidationResult

IMPORTANT LIMITATION
---------------------
This guardrail performs deterministic keyword/phrase matching against free-form
request text. It is a BEST-EFFORT check given that the current codebase has no
structured intent-capture layer (the Streamlit UI / main.py have not been built
yet). When the UI is built, it should capture a structured "request_type" or
"intent" field from the user, which would allow much more reliable scope
checking than free-text pattern matching. This module's sole source of ground
truth is what the user typed, which an adversarial user could rephrase to evade
detection. This is documented here explicitly rather than hidden in code comments.

When to escalate this module:
  A future orchestrator update (when main.py is built) should:
  1. Accept a structured intent field alongside the document upload.
  2. Pass that field to validate_scope() rather than (or in addition to) raw text.
  3. Consider adding an LLM-based secondary check for high-stakes intent detection.

Detection approach:
  - Phrase-level matching (multi-word patterns, not single-keyword triggers)
  - Severity is "blocking" for all out-of-scope detections: a user asking for
    autonomous approval or definitive legal advice must be stopped immediately
    and redirected. There is no legitimate "warning" case for out-of-scope
    intent -- either the system can do what was asked (passed=True) or it
    cannot (passed=False, blocking).

Logging:
  - Log the detected_intent category and guardrail_name, NOT the raw request text.
    The raw request text may contain sensitive information (e.g. contract parties'
    names, deal terms). Log only the category of the problem.
"""

import logging
import re
from typing import List, Optional, Tuple

from schemas.guardrails import ScopeValidationResult

logger = logging.getLogger(__name__)

_GUARDRAIL_NAME = "scope_validator"


# ---------------------------------------------------------------------------
# Out-of-scope pattern definitions
# ---------------------------------------------------------------------------
# Each entry: (intent_category_label, compiled_regex)
# Intent categories map to what we think the user was actually asking for.
#
# Patterns are phrase-level (not single-word) to minimize false positives.
# "Approve" alone is far too broad; "approve this contract" is specific.

def _scope_pattern(category: str, pattern: str) -> Tuple[str, re.Pattern]:
    return (category, re.compile(pattern, re.IGNORECASE))


_OUT_OF_SCOPE_PATTERNS: List[Tuple[str, re.Pattern]] = [
    # binding_legal_approval: user wants the AI to act as an approving authority
    _scope_pattern("binding_legal_approval",
        r"(approve|certify|authorize|greenlight|sign\s+off\s+on)\s+(this|the|our)\s+(contract|agreement|nda|document|deal)"),
    _scope_pattern("binding_legal_approval",
        r"(is|this\s+(contract|agreement|nda|document)\s+is)\s+(legally\s+)?(approved|valid|enforceable|binding)\s*[,.]?\s*(yes|no|tell\s+me)"),
    _scope_pattern("binding_legal_approval",
        r"(give|provide|issue)\s+(your\s+)?(approval|authorization|sign-?off|clearance)\s+(on|for)\s+(this|the)"),

    # contract_execution: user wants the AI to execute/sign/finalize the contract
    _scope_pattern("contract_execution",
        r"(execute|sign|finalize|countersign|ratify)\s+(this|the|our)\s+(contract|agreement|nda|document|deal)"),
    _scope_pattern("contract_execution",
        r"(make|render|cause)\s+(this|the)\s+(contract|agreement)\s+(legally\s+)?(binding|effective|enforceable)"),
    _scope_pattern("contract_execution",
        r"(complete|close|seal)\s+(this|the)\s+(deal|transaction|agreement|contract)"),

    # contract_modification: user wants the AI to rewrite or change contract terms
    # Pattern uses [\w\s-]+ to allow qualified noun phrases like "indemnification clause"
    # and hyphenated words like "non-compete"
    _scope_pattern("contract_modification",
        r"(rewrite|redraft)\s+[\w\s-]*(contract|agreement|nda|clause|terms?|language)"),
    _scope_pattern("contract_modification",
        r"(modify|change|alter|amend|update)\s+(the|this|our|a)\s+[\w\s-]*(contract|agreement|nda|clause|terms?|language)\b"),
    _scope_pattern("contract_modification",
        r"(add|insert)\s+(a|the)?\s*[\w\s-]*(clause|section|term|provision)\s+(to|into)\s+(the|this|our)"),
    _scope_pattern("contract_modification",
        r"(remove|delete|replace)\s+(the|a)?\s*[\w\s-]*(clause|section|term|provision)\s+(from|in)\s+(the|this|our)"),

    # definitive_legal_advice: user wants a binding legal opinion, not AI analysis
    _scope_pattern("definitive_legal_advice",
        r"is\s+(this|the)\s+(contract|agreement|nda|clause|document)\s+(legally\s+)?(enforceable|valid|binding|compliant)\s*[\?,]"),
    _scope_pattern("definitive_legal_advice",
        r"(tell|advise|confirm)\s+(me|us)\s+(definitively|for\s+certain|with\s+certainty)\s+(that|whether|if)"),
    _scope_pattern("definitive_legal_advice",
        r"(give|provide)\s+(me|us)\s+(a\s+)?(legal\s+opinion|legal\s+advice|binding\s+analysis|definitive\s+(answer|determination))"),
    _scope_pattern("definitive_legal_advice",
        r"(yes\s+or\s+no|true\s+or\s+false)[,:]?\s+(is\s+this|does\s+this)"),
    _scope_pattern("definitive_legal_advice",
        r"can\s+(we|i)\s+(legally|lawfully)\s+(do|proceed|sign|execute|rely\s+on)\s+(this|it)"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_scope(user_request_text: str) -> ScopeValidationResult:
    """
    Check whether the user's request is within the scope of what this system
    is designed and authorized to do.

    Parameters
    ----------
    user_request_text : str
        The user's free-form request text. In the current codebase (no UI yet),
        this may be a pipeline invocation description, an API request body, or
        text from an early Streamlit prototype. When main.py is built, prefer
        passing structured intent rather than raw free-form text.

    Returns
    -------
    ScopeValidationResult
        passed=True if the request appears to be within scope.
        passed=False, severity="blocking" if an out-of-scope intent is detected.
        detected_intent names the category of out-of-scope request if detected.
    """
    if not user_request_text or not user_request_text.strip():
        # Empty request: assume in-scope (the system will proceed with default behavior)
        return ScopeValidationResult(
            guardrail_name=_GUARDRAIL_NAME,
            passed=True,
            reason="No request text provided; proceeding with default analysis scope.",
            severity="info",
            detected_intent=None,
        )

    for intent_category, pattern in _OUT_OF_SCOPE_PATTERNS:
        if pattern.search(user_request_text):
            logger.warning(
                "%s: out-of-scope request detected — intent_category=%s",
                _GUARDRAIL_NAME, intent_category,
            )
            return ScopeValidationResult(
                guardrail_name=_GUARDRAIL_NAME,
                passed=False,
                reason=(
                    f"Request appears to ask for '{intent_category}', which is outside "
                    f"the scope of this system. This AI assistant analyzes documents for "
                    f"risk and extracts clauses — it does not approve, execute, modify, or "
                    f"provide binding legal opinions on contracts."
                ),
                severity="blocking",
                detected_intent=intent_category,
            )

    logger.info("%s: request is within analysis scope", _GUARDRAIL_NAME)
    return ScopeValidationResult(
        guardrail_name=_GUARDRAIL_NAME,
        passed=True,
        reason="Request is within the analysis scope of this system.",
        severity="info",
        detected_intent=None,
    )
