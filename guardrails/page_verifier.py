"""
guardrails/page_verifier.py
-----------------------------
Post-extraction guardrail: verify that a clause's cited page number is valid
and that the clause text can actually be found on or near that page.

Why page verification?
  LLMs sometimes hallucinate page references. An extracted clause may be
  genuinely present in the document but cited as being on page 5 when it's
  actually on page 3, or cited as page 50 when the document only has 10 pages.
  This module catches both types of error:
    1. Structural: the cited page is out of range (> total_pages or < 1).
    2. Textual: the cited page is in range but the extracted text is not
       findable on that page or on adjacent pages (± 1 page boundary tolerance).

Public function:
  verify_page_reference(cited_page, document, extracted_text) -> PageReferenceVerificationResult

Adjacent-page tolerance:
  Legal clauses frequently span page boundaries (e.g. a clause starts on
  page 4 and continues onto page 5). If extracted_text is not found on
  cited_page exactly, the verifier also checks cited_page - 1 and
  cited_page + 1 before reporting not-found. This matches the project's
  existing awareness of page-boundary spanning (per the original system spec).

Text matching:
  Delegates to verify_evidence() from evidence_verifier.py -- single source
  of truth for "does this text appear in this source text?" logic, including
  the fuzzy-match fallback and the same 0.70 threshold.
"""

import logging
from typing import Optional

from guardrails.evidence_verifier import verify_evidence
from schemas.document import DocumentExtraction
from schemas.guardrails import PageReferenceVerificationResult

logger = logging.getLogger(__name__)


def verify_page_reference(
    cited_page: int,
    document: DocumentExtraction,
    extracted_text: Optional[str] = None,
) -> PageReferenceVerificationResult:
    """
    Check whether cited_page exists in document, and optionally whether
    extracted_text is findable on or near that page.

    Parameters
    ----------
    cited_page : int
        The page number the LLM claimed the clause is on (1-based).
    document : DocumentExtraction
        The full extracted document (source of truth for total_pages and
        per-page text).
    extracted_text : str | None
        If provided, the verifier also checks whether this text is findable
        on cited_page ± 1. If None, only the structural page-existence check
        is performed.

    Returns
    -------
    PageReferenceVerificationResult
        page_exists_in_document=False if cited_page > total_pages or < 1.
        text_found_near_cited_page is set only when extracted_text is provided
        AND the page exists.
    """
    total_pages = document.total_pages

    # Structural check: is the cited page within the document?
    if cited_page < 1 or cited_page > total_pages:
        logger.info(
            "page_verifier: cited_page=%d out of range (document has %d pages)",
            cited_page, total_pages,
        )
        return PageReferenceVerificationResult(
            cited_page=cited_page,
            page_exists_in_document=False,
            text_found_near_cited_page=False,
            notes=f"Cited page {cited_page} is out of range; document has {total_pages} page(s).",
        )

    # Page exists. If no text provided, we can only confirm structural validity.
    if not extracted_text or not extracted_text.strip():
        logger.debug(
            "page_verifier: page %d exists, no extracted_text provided for text check",
            cited_page,
        )
        return PageReferenceVerificationResult(
            cited_page=cited_page,
            page_exists_in_document=True,
            text_found_near_cited_page=False,
            notes="Page exists in document; no extracted_text provided for text verification.",
        )

    # Text check: try cited_page and ± 1 adjacent pages
    pages_to_check = _adjacent_pages(cited_page, total_pages)
    for page_num in pages_to_check:
        page_text = _get_page_text(document, page_num)
        if not page_text:
            continue
        ev = verify_evidence(extracted_text, page_text, source_page=page_num)
        if ev.found_in_source:
            if page_num == cited_page:
                note = None
            else:
                note = (
                    f"Text found on page {page_num} rather than cited page {cited_page} "
                    f"(within ±1 page boundary tolerance)."
                )
            logger.debug(
                "page_verifier: text found on page %d (cited %d), match_type=%s",
                page_num, cited_page, ev.match_type,
            )
            return PageReferenceVerificationResult(
                cited_page=cited_page,
                page_exists_in_document=True,
                text_found_near_cited_page=True,
                notes=note,
            )

    logger.info(
        "page_verifier: text not found on page %d or adjacent pages (±1). cited_page=%d",
        cited_page, cited_page,
    )
    return PageReferenceVerificationResult(
        cited_page=cited_page,
        page_exists_in_document=True,
        text_found_near_cited_page=False,
        notes=(
            f"Extracted text not found on page {cited_page} or adjacent pages "
            f"({min(pages_to_check)}-{max(pages_to_check)})."
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _adjacent_pages(cited_page: int, total_pages: int) -> list[int]:
    """Return [cited_page-1, cited_page, cited_page+1] clamped to [1..total_pages]."""
    candidates = [cited_page - 1, cited_page, cited_page + 1]
    # Reorder so we check cited_page first (most likely match)
    ordered = [cited_page] + [p for p in candidates if p != cited_page]
    return [p for p in ordered if 1 <= p <= total_pages]


def _get_page_text(document: DocumentExtraction, page_num: int) -> str:
    """Return the text of the given page from the document, or '' if not found."""
    page = document.get_page(page_num)
    return page.text if page else ""
