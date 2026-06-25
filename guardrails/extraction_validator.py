"""
guardrails/extraction_validator.py
-----------------------------------
Extraction-quality guardrail for the Legal AI Assistant.

What this file does:
    Holds the ONE decision: "is this page's extracted text good enough,
    or does it need to be routed to OCR?"

    This logic used to live directly inside extraction/pdf_parser.py as
    a simple if-statement. We pulled it out into its own guardrail module
    so that:
      1. The rule lives in exactly one place (easy to find, easy to explain).
      2. It can be tested on its own, with fake/sample text, without
         needing a real PDF file or an OpenAI API call.
      3. Later guardrails (input validation, scope validation, etc.) can
         live in this same folder, following one consistent pattern.

Why this matters for the bigger picture:
    This is a "pre-generation guardrail" per the project's three-layer
    guardrail architecture (Section 12A of the system prompt) -- it runs
    BEFORE any AI model is asked to do anything with the page, and decides
    whether the page is even ready to be analyzed.

Note on scope (Step 3 only):
    This file currently contains ONLY the extraction-quality check.
    Future steps will add separate guardrail files (input_validator.py,
    scope_validator.py, etc.) to this same folder -- we are not building
    those yet, per the project's "build only what the current step needs"
    rule.

Dependencies:
    schemas/document.py
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Minimum number of non-whitespace ("meaningful") characters a page must
# contain for us to trust PyMuPDF's direct extraction.
#
# Why 100? This number comes from the project's locked architecture
# (system prompt Section 6 and Section 32). It is deliberately a single
# constant that BOTH pdf_parser.py and this validator agree on -- in the
# full system, this value will live in config.py and both files will
# import it from there, so the threshold is never duplicated or allowed
# to drift out of sync between files.
EXTRACTION_CHAR_THRESHOLD = 100


# ---------------------------------------------------------------------------
# Result object -- what the validator hands back
# ---------------------------------------------------------------------------

@dataclass
class ExtractionQualityResult:
    """
    The outcome of checking one page's extracted text.

    What is a "dataclass"?
        A dataclass is a simpler cousin of the Pydantic BaseModel we used
        in schemas/document.py. Think of Pydantic models like a strict
        bouncer at a club door -- they actively check IDs and reject bad
        data. A dataclass is more like a labeled box -- it just neatly
        holds a few named values together, with no active checking.
        We use a plain dataclass here (not Pydantic) because this result
        never leaves this guardrail's internal logic or crosses an
        external boundary -- it's a lightweight internal helper, not a
        data contract shared across the whole system the way
        PageExtraction and DocumentExtraction are.

    Fields
    ------
    passed : bool
        True if the page's text is good enough to keep as-is.
        False if the page needs to be routed to OCR.

    char_count : int
        The meaningful character count that was measured.

    reason : str
        A short, human-readable explanation of the decision.
        Useful for logging and for showing in the Streamlit UI later
        ("why was this page flagged?" -- Section 31 explainability).
    """
    passed: bool
    char_count: int
    reason: str


# ---------------------------------------------------------------------------
# Helper: count meaningful characters
# ---------------------------------------------------------------------------

def _count_meaningful_chars(text: str) -> int:
    """
    Count non-whitespace characters in a string.

    This is the SAME logic as count_meaningful_chars() in pdf_parser.py.
    It is intentionally duplicated here (rather than imported) for the
    same reason ocr.py duplicates it: to avoid pdf_parser.py and this
    guardrail module needing to import from each other, which would risk
    a circular import as the project grows. It is four lines of simple,
    stable logic -- the small duplication cost is worth the independence.

    Parameters
    ----------
    text : str
        Raw or normalized page text.

    Returns
    -------
    int
        Count of non-whitespace characters.
    """
    return len("".join(text.split()))


# ---------------------------------------------------------------------------
# Main public function: check one page's extraction quality
# ---------------------------------------------------------------------------

def check_extraction_quality(text: str) -> ExtractionQualityResult:
    """
    Decide whether a page's extracted text is good enough to keep,
    or whether it needs to be routed to OCR.

    This is THE guardrail decision for Step 3. It is called by
    pdf_parser.py's extract_page_text() function immediately after
    PyMuPDF returns text for a page, and BEFORE any decision is made
    about calling the OCR module.

    The rule (kept deliberately simple for now, matching the locked
    architecture in the system prompt):

        meaningful_char_count >= 100   ->  PASS (keep PyMuPDF's text)
        meaningful_char_count <  100   ->  FAIL (route to OCR)

    Why character count and not something fancier (like checking for
    real English words, or running a language-detection model)?
        A blank or near-blank page from a failed extraction reliably
        produces almost no characters at all -- often literally zero,
        or just a few stray symbols. A simple character-count threshold
        catches this extremely reliably without needing any extra AI
        calls or complex logic. We only escalate to a smarter (and more
        expensive) tool -- Vision OCR -- once this cheap check fails.
        This follows the project's stated philosophy (Section 37):
        "Prefer deterministic validation before adding another LLM call."

    Parameters
    ----------
    text : str
        The (already normalized) text extracted from a single page.

    Returns
    -------
    ExtractionQualityResult
        passed=True  -> text is good, no OCR needed
        passed=False -> text is too sparse, route this page to OCR
    """
    char_count = _count_meaningful_chars(text)

    if char_count >= EXTRACTION_CHAR_THRESHOLD:
        reason = (
            f"Page has {char_count} meaningful characters "
            f"(threshold is {EXTRACTION_CHAR_THRESHOLD}) -- extraction looks complete."
        )
        logger.debug(reason)
        return ExtractionQualityResult(
            passed=True,
            char_count=char_count,
            reason=reason,
        )

    reason = (
        f"Page has only {char_count} meaningful characters "
        f"(threshold is {EXTRACTION_CHAR_THRESHOLD}) -- "
        f"likely a scanned image or extraction failure. Needs OCR."
    )
    logger.debug(reason)
    return ExtractionQualityResult(
        passed=False,
        char_count=char_count,
        reason=reason,
    )
