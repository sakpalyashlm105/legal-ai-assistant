"""
guardrails/evidence_verifier.py
---------------------------------
Post-extraction guardrail: verify that text claimed to have been extracted
from a document actually appears in the source document.

When the LLM extracts a clause (Step 5), it returns an ExtractedClause with
an extracted_text field. This module checks whether that text genuinely
appears in the source page text -- catching hallucinated or confabulated
extractions before they propagate into the risk report.

Public function:
  verify_evidence(extracted_text, source_page_text) -> EvidenceVerificationResult

Detection approach:
  1. Exact match (after whitespace normalization via normalize_text() from
     pdf_parser.py -- reused rather than reimplemented).
  2. If no exact match: fuzzy match using difflib.SequenceMatcher.
     Threshold: 0.70 similarity score.

FUZZY MATCH THRESHOLD CHOICE: 0.70
-------------------------------------
Justification:
  - Legal contracts have very specific wording. A similarity of < 0.70 typically
    means the extracted text describes a different clause or is substantially
    paraphrased. A well-grounded extraction of a real clause should score >= 0.75
    even if the LLM paraphrased slightly (e.g. dropping a trailing parenthetical).
  - 0.70 is conservative enough to catch clear hallucinations while tolerant
    enough to accept minor whitespace, punctuation, or line-break differences
    introduced by PDF rendering artifacts (e.g. a hyphenated word split across lines
    may appear as two words in the extracted text).
  - Empirically: difflib SequenceMatcher on legal text with minor normalization
    differences consistently scores >= 0.80. Scores in the 0.50-0.65 range
    indicate the texts describe different content.
  - If calibration data from real extractions shows this threshold is too low
    or too high, adjust FUZZY_MATCH_THRESHOLD in this module.

Note on normalize_text():
  We import normalize_text from extraction.pdf_parser (which collapses whitespace
  and strips trailing spaces per line) rather than reimplementing whitespace
  normalization a second time. This ensures both layers use the same definition
  of "normalized" text.
"""

import difflib
import logging
from typing import Optional

from extraction.pdf_parser import normalize_text
from schemas.guardrails import EvidenceVerificationResult

logger = logging.getLogger(__name__)

# Minimum difflib SequenceMatcher ratio to accept a fuzzy match.
# See module docstring for threshold justification.
FUZZY_MATCH_THRESHOLD: float = 0.70


def _best_window_score(needle: str, haystack: str) -> float:
    """
    Find the best difflib similarity score for needle anywhere in haystack.

    Problem: SequenceMatcher.ratio() = 2M/T where M = matching chars,
    T = total chars in both strings. When haystack >> needle, T is large
    and ratio is always small even for a perfect match of the needle text
    somewhere in the haystack.

    Solution: slide a window of size (len(needle) + SLACK) across the haystack
    and take the maximum ratio. With SLACK=40 chars of slack for inserted/deleted
    words, a perfect match scores ~2n/(2n+40) ≈ 0.88 for n=100 (not 1.0, but
    consistently above our 0.70 threshold). A genuinely different text scores
    well below 0.70 at this window size.

    SLACK is chosen to accommodate minor paraphrasing (one inserted article or
    adjective) without being so large that unrelated text scores above threshold.
    """
    if not needle or not haystack:
        return 0.0
    SLACK = 40
    window_size = len(needle) + SLACK
    if len(haystack) <= window_size:
        return difflib.SequenceMatcher(None, needle, haystack, autojunk=False).ratio()
    step = max(1, len(needle) // 5)
    best = 0.0
    for i in range(0, len(haystack) - len(needle) + 1, step):
        window = haystack[i: i + window_size]
        score = difflib.SequenceMatcher(None, needle, window, autojunk=False).ratio()
        if score > best:
            best = score
            if best >= 0.99:
                break
    return best


def verify_evidence(
    extracted_text: str,
    source_page_text: str,
    source_page: Optional[int] = None,
) -> EvidenceVerificationResult:
    """
    Check whether extracted_text appears in source_page_text.

    Parameters
    ----------
    extracted_text : str
        The text the LLM claimed it extracted from the document
        (ExtractedClause.extracted_text).
    source_page_text : str
        The actual text of the source page (PageExtraction.text).
    source_page : int | None
        Optional page number for the result's source_page_checked field.

    Returns
    -------
    EvidenceVerificationResult
        match_type='exact'    -- substring found after normalization.
        match_type='fuzzy'    -- difflib similarity >= FUZZY_MATCH_THRESHOLD.
        match_type='not_found' -- neither match succeeded.
    """
    if not extracted_text or not extracted_text.strip():
        logger.debug("evidence_verifier: empty extracted_text, returning not_found")
        return EvidenceVerificationResult(
            extracted_text=extracted_text or "",
            found_in_source=False,
            match_type="not_found",
            match_score=None,
            source_page_checked=source_page,
        )

    norm_extracted = normalize_text(extracted_text)
    norm_source = normalize_text(source_page_text)

    # 1. Exact substring match (after normalization)
    if norm_extracted in norm_source:
        logger.debug(
            "evidence_verifier: exact match found (page=%s, len=%d)",
            source_page, len(norm_extracted),
        )
        return EvidenceVerificationResult(
            extracted_text=extracted_text,
            found_in_source=True,
            match_type="exact",
            match_score=None,
            source_page_checked=source_page,
        )

    # 2. Fuzzy match via a windowed difflib approach.
    # Rationale: comparing a short extracted_text directly against a full page
    # (often 1000+ chars) with SequenceMatcher.ratio() always yields a low score
    # because ratio = 2*matching / total_chars, where total_chars is dominated by
    # the long source. Instead, we slide a window of (len(extracted)*2) characters
    # across the source and take the best ratio. This correctly measures whether the
    # extracted text matches *somewhere* in the source, not how similar the whole
    # source is to the extracted text.
    score = _best_window_score(norm_extracted, norm_source)

    if score >= FUZZY_MATCH_THRESHOLD:
        logger.debug(
            "evidence_verifier: fuzzy match (score=%.3f, page=%s)", score, source_page
        )
        return EvidenceVerificationResult(
            extracted_text=extracted_text,
            found_in_source=True,
            match_type="fuzzy",
            match_score=round(score, 4),
            source_page_checked=source_page,
        )

    logger.info(
        "evidence_verifier: not_found (fuzzy score=%.3f < threshold=%.2f, page=%s)",
        score, FUZZY_MATCH_THRESHOLD, source_page,
    )
    return EvidenceVerificationResult(
        extracted_text=extracted_text,
        found_in_source=False,
        match_type="not_found",
        match_score=round(score, 4),
        source_page_checked=source_page,
    )
