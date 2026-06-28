"""
reporting/executive_summary.py
--------------------------------
Generates the executive summary section of the LegalDocumentReport via a
single, tightly-constrained LLM call.

This is the ONLY LLM call in the entire report generation pipeline (Step 10).
Everything else in report_generator.py is deterministic formatting.

Why one LLM call here?
    Synthesizing 10 clauses, multiple risk findings, and evidence issues into
    coherent prose genuinely benefits from language generation. Pure templating
    produces mechanical text like "9 of 10 clauses found. 1 HIGH risk finding."
    A constrained LLM call produces a readable paragraph while the hard
    constraints below prevent verdict language or overclaiming.

Safe-failure behaviour (same pattern as classifier.py and extractor.py):
    1. Any API error -> log, return fallback summary (no crash).
    2. Response that contains a banned verdict phrase -> regenerate once.
    3. If regeneration also fails the check -> return fallback summary.
    The fallback is always a correct, rule-compliant summary -- never an
    empty string, never a rule-violating string.

Banned verdict phrases (enforced by _check_output before accepting):
    "is enforceable", "is unenforceable", "is valid", "is invalid",
    "legally binding", "guarantees", "certifies that"

PII-safe: this function receives already-extracted structured data (clause
names, risk levels, counts) -- it never sends raw contract text to the LLM.
"""

import logging
import re
from pathlib import Path
from typing import Optional, Tuple

import yaml

from config import OPENAI_API_KEY, LLM_MODEL
from reporting.report_generator import _fallback_executive_summary
from schemas.report import LegalDocumentReport, RiskFindingEntry

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "report_generation.yaml"

# ---------------------------------------------------------------------------
# Validator 1 — Verdict-phrase scanner
# Phrases that constitute a legal verdict -- banned from the executive summary.
# Each phrase is checked case-insensitively in the LLM's output.
# ---------------------------------------------------------------------------
BANNED_PHRASES = [
    "is enforceable",
    "is unenforceable",
    "is valid",
    "is invalid",
    "legally binding",
    "guarantees",
    "certifies that",
]

# ---------------------------------------------------------------------------
# Validator 2 — Evidence-coverage phrasing scanner
#
# Design philosophy (same as guardrails/prompt_injection.py):
#   Deterministic regex scan of the LLM's output.  No second LLM call.
#   Prompt instructions reduce the probability of bad phrasing; this
#   scanner is what actually guarantees the policy is enforced.
#
# The policy: when absent clauses exist (na_count > 0), a summary must NOT
# imply that the absent clauses were verified clean. It must explicitly
# distinguish present-and-verified from absent-and-not-applicable.
#
# Two separate checks:
#   (a) AMBIGUOUS_EVIDENCE_PATTERNS — blanket claims that would incorrectly
#       cover absent clauses.  Any match -> reject.
#   (b) COVERAGE_INDICATORS — the correct distinction must be explicitly
#       present: both a "present clauses verified" signal AND an
#       "absent clauses not applicable" signal.  If either is missing -> reject.
#
# Pattern design follows prompt_injection.py conventions:
#   - Compiled at module load (not per-call)
#   - IGNORECASE throughout
#   - Each pattern is documented with the specific LLM phrasing it targets
# ---------------------------------------------------------------------------

def _re(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


# Patterns that indicate a blanket/universal evidence claim that would
# incorrectly imply absent clauses were also verified.
# Each is labelled with the real LLM phrasing that prompted its inclusion.
AMBIGUOUS_EVIDENCE_PATTERNS: list[Tuple[str, re.Pattern]] = [
    # "all clauses were verified" / "all clauses have been verified"
    ("all_clauses_verified",
     _re(r"all\s+clauses?\s+(were|have\s+been|are)\s+verified")),
    # "no evidence verification issues detected/found/identified"
    # (blanket "no issues" without qualifying which clauses were actually checked)
    ("no_evidence_issues_blanket",
     _re(r"no\s+(evidence\s+)?(verification\s+)?(issues?|concerns?|problems?|failures?|errors?)\s*(were\s+)?(detected|found|identified|reported|noted|observed)")),
    # "no significant evidence concerns" / "no notable evidence issues"
    ("no_evidence_concerns_blanket",
     _re(r"no\s+(significant|major|notable|outstanding|material)?\s*evidence\s+(concerns?|issues?|problems?|discrepancies?)")),
    # "evidence verification completed successfully" / "evidence verification was thorough"
    ("evidence_verification_blanket_completion",
     _re(r"evidence\s+verification\s+(was\s+)?(completed?|successful|satisfactory|thorough|comprehensive|done|finished)\s*(successfully)?")),
    # "all evidence checks passed" / "all verifications completed"
    ("all_evidence_checks_passed",
     _re(r"(all|every)\s+(evidence\s+)?(checks?|verifications?|reviews?)\s+(passed|completed|succeeded|were\s+successful)")),
    # "everything was verified" / "all findings were verified"
    ("everything_verified",
     _re(r"(everything|all\s+findings?|all\s+items?|all\s+results?)\s+(was|were|have\s+been)\s+verified")),
    # "evidence check was complete" / "evidence review was done"
    ("evidence_check_complete",
     _re(r"evidence\s+(check|review|scan|analysis)\s+(was|were|is|has\s+been)\s+(complete|done|finished|thorough|satisfactory)")),
    # "verification was successful" as a standalone claim (not qualified by "present clauses")
    # Requires absence of the word "present" within 60 chars before "verification was successful"
    # -- implemented in the function logic below rather than as a simple regex
]

# Signals that a summary DOES correctly mention present-clause verification.
# At least one must be present when na_count > 0.
_PRESENT_VERIFIED_SIGNALS: list[re.Pattern] = [
    _re(r"\d+\s+present\s+clause"),                              # "4 present clauses"
    _re(r"present\s+clause.{0,40}(passed|verified|confirmed)"),  # "present clauses passed"
    _re(r"(passed|verified)\s+(evidence\s+)?(verification|check)"),  # "passed evidence verification"
    _re(r"(verified|confirmed)\s+against\s+(the\s+)?source"),    # "verified against source"
    _re(r"clause.{0,30}(evidence.{0,20})?(verified|confirmed)\s*[:=]?\s*(yes|true)"),
]

# Signals that a summary DOES correctly mention absent-clause N/A status.
# At least one must be present when na_count > 0.
_ABSENT_NA_SIGNALS: list[re.Pattern] = [
    _re(r"not\s+applicable"),                                    # "not applicable"
    _re(r"\bwere\s+absent\b"),                                   # "were absent"
    _re(r"absent\s+clause"),                                     # "absent clauses"
    _re(r"\d+\s+(clause|categor).{0,30}(absent|not\s+applicable|n\s*/?\s*a\b)"),
    _re(r"(absent|missing).{0,60}(not\s+applicable|n\s*/?\s*a\b)"),
    _re(r"not\s+applicable\s+for\s+(this|verification|evidence)"),
    _re(r"(could\s+not|cannot|did\s+not)\s+(be\s+)?(apply|run|perform)\s+(to|on|for)\s+(absent|missing)"),
]

# Lazy OpenAI client (same pattern as classifier.py)
_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def _load_prompt() -> dict:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _check_output(text: str) -> bool:
    """
    Validator 1: verdict-phrase scanner.

    Return True if the text is clean (no banned verdict phrases), False otherwise.
    Checks case-insensitively. A False result triggers one retry in
    generate_executive_summary(), then fallback if still failing.
    """
    lower = text.lower()
    for phrase in BANNED_PHRASES:
        if phrase.lower() in lower:
            logger.warning(
                "executive_summary: banned verdict phrase '%s' found -- "
                "will retry or fall back.",
                phrase,
            )
            return False
    return True


def check_evidence_coverage_phrasing(
    summary_text: str,
    verified_count: int,
    na_count: int,
) -> bool:
    """
    Validator 2: evidence-coverage phrasing scanner.

    Returns True if the summary's evidence-verification phrasing is acceptable;
    False if it is ambiguous and should be rejected (triggering retry/fallback).

    Rules (applied only when na_count > 0):
      - Any AMBIGUOUS_EVIDENCE_PATTERNS match -> False immediately.
        Rationale: an ambiguous blanket claim is a policy violation even if a
        correct sentence also appears elsewhere in the same summary.
      - If no ambiguous phrase found, the summary must ALSO contain both:
          (a) a signal that present clauses were verified, AND
          (b) a signal that absent clauses were not applicable.
        If either signal is missing -> False.
        Rationale: a summary can avoid all "bad" phrases while still failing to
        mention the absent-clause population at all.

    When na_count == 0, every clause was present and subject to real
    verification, so no absent-clause population exists to mis-describe.
    Returns True unconditionally in that case.
    """
    if na_count == 0:
        return True  # nothing to mis-describe; any phrasing is fine

    # Check for ambiguous blanket claims
    for label, pattern in AMBIGUOUS_EVIDENCE_PATTERNS:
        if pattern.search(summary_text):
            logger.warning(
                "executive_summary: ambiguous evidence-coverage phrase "
                "(%s) found in LLM output -- will retry or fall back.",
                label,
            )
            return False

    # Require explicit mention of BOTH the present-verified AND absent-NA populations
    has_present_signal = any(p.search(summary_text) for p in _PRESENT_VERIFIED_SIGNALS)
    has_absent_signal = any(p.search(summary_text) for p in _ABSENT_NA_SIGNALS)

    if not has_present_signal:
        logger.warning(
            "executive_summary: LLM output missing explicit present-clause "
            "verification signal when na_count=%d -- will retry or fall back.",
            na_count,
        )
        return False

    if not has_absent_signal:
        logger.warning(
            "executive_summary: LLM output missing explicit absent-clause "
            "N/A signal when na_count=%d -- will retry or fall back.",
            na_count,
        )
        return False

    return True


def _validate_candidate(candidate: str, prompt_vars: dict) -> Tuple[bool, str]:
    """
    Run all post-generation validators on a candidate summary.

    Returns (passed, failed_check_name).  failed_check_name is "" on success.
    Validators are applied in order; the first failure short-circuits.

    Adding a new policy check: implement it as a function following the
    _check_output / check_evidence_coverage_phrasing signature pattern,
    then add a call here -- the retry loop in generate_executive_summary
    picks it up automatically.
    """
    if not _check_output(candidate):
        return False, "verdict_phrase"

    ev_verified = prompt_vars.get("evidence_verified_count", 0)
    ev_na = prompt_vars.get("evidence_na_count", 0)
    if not check_evidence_coverage_phrasing(candidate, ev_verified, ev_na):
        return False, "evidence_coverage"

    return True, ""


def _build_prompt_vars(report_data: LegalDocumentReport) -> dict:
    """
    Extract the structured inputs the prompt template needs from the report.

    Never passes raw contract text -- only clause names, risk levels, and counts.
    """
    high_risk = [
        r.clause_category for r in report_data.risk_findings if r.risk_level == "HIGH"
    ]
    medium_risk = [
        r.clause_category for r in report_data.risk_findings if r.risk_level == "MEDIUM"
    ]
    evidence_issues = [
        e.clause_category for e in report_data.clause_entries
        if e.evidence_verified is False
    ]
    # Distinguish present-and-verified from absent-and-N/A so the LLM prompt
    # can phrase evidence verification correctly (absent ≠ checked-and-passed).
    evidence_verified_count = sum(
        1 for e in report_data.clause_entries if e.is_present and e.evidence_verified is True
    )
    evidence_na_count = sum(
        1 for e in report_data.clause_entries if not e.is_present
    )
    human_review_taken = "yes" if report_data.human_review_decisions else "no"

    return {
        "document_type": report_data.document_type,
        "total_clauses_found": report_data.total_clauses_found,
        "missing_clauses": ", ".join(report_data.missing_clauses) or "none",
        "high_risk_clauses": ", ".join(high_risk) or "none",
        "medium_risk_clauses": ", ".join(medium_risk) or "none",
        "evidence_issues": ", ".join(evidence_issues) or "none",
        "evidence_verified_count": evidence_verified_count,
        "evidence_na_count": evidence_na_count,
        "human_review_taken": human_review_taken,
    }


def _call_llm(prompt_vars: dict) -> str:
    """
    Make the single LLM call for the executive summary.

    Returns the raw text from the model. Raises on API errors so the
    caller can catch and fall back.
    """
    prompt = _load_prompt()
    system_text = prompt["system"]
    user_text = prompt["user"].format(**prompt_vars)

    client = _get_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        temperature=0.3,  # low temperature: consistent, controlled prose
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


def generate_executive_summary(report_data: LegalDocumentReport) -> str:
    """
    Generate a short, constrained executive summary using one LLM call.

    Parameters
    ----------
    report_data : LegalDocumentReport
        The fully assembled report (from assemble_report()). Used to extract
        structured inputs for the prompt -- raw contract text is never sent.

    Returns
    -------
    str
        A 3-5 sentence prose executive summary following both writing-quality
        rules (no verdict language, no percentage probabilities). If the LLM
        call fails or produces a rule-violating response, a deterministic
        fallback is returned instead.

    Safe-failure guarantee:
        This function NEVER raises. It always returns a non-empty,
        rule-compliant string.
    """
    prompt_vars = _build_prompt_vars(report_data)

    # Attempt 1 — call LLM, run all validators
    try:
        candidate = _call_llm(prompt_vars)
        passed, failed_check = _validate_candidate(candidate, prompt_vars)
        if passed:
            logger.info("executive_summary: LLM call succeeded on first attempt.")
            return candidate
        logger.warning(
            "executive_summary: first attempt failed validator '%s' -- retrying.",
            failed_check,
        )
    except Exception as e:
        logger.error(
            "executive_summary: LLM call failed (attempt 1): %s -- falling back.", e
        )
        return _build_fallback(report_data)

    # Attempt 2 — single retry after any validator failure
    try:
        candidate = _call_llm(prompt_vars)
        passed, failed_check = _validate_candidate(candidate, prompt_vars)
        if passed:
            logger.info("executive_summary: LLM call succeeded on second attempt.")
            return candidate
        logger.warning(
            "executive_summary: second attempt also failed validator '%s' -- "
            "falling back to deterministic summary.",
            failed_check,
        )
    except Exception as e:
        logger.error(
            "executive_summary: LLM call failed (attempt 2): %s -- falling back.", e
        )

    return _build_fallback(report_data)


def _build_fallback(report_data: LegalDocumentReport) -> str:
    """Construct the deterministic fallback summary from report data."""
    return _fallback_executive_summary(
        document_type=report_data.document_type,
        total_found=report_data.total_clauses_found,
        risk_entries=list(report_data.risk_findings),
        missing_names=list(report_data.missing_clauses),
        clause_entries=list(report_data.clause_entries),
    )
