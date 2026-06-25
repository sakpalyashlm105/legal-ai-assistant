"""
schemas/document.py
-------------------
Pydantic data models for text extraction results.

These are the data contracts between the extraction layer (pdf_parser.py, ocr.py)
and everything downstream (chunking, retrieval, guardrails, reporting).

Two models:
  - PageExtraction  : result for a single PDF page
  - DocumentExtraction : result for the full document (list of PageExtractions + metadata)

Why Pydantic?
  - Automatic type validation catches bugs at the boundary (e.g. page_number as str instead of int)
  - .model_dump() gives us a clean dict for logging and serialization
  - Every downstream module imports these instead of passing raw dicts around
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Controlled vocabulary for extraction method
# ---------------------------------------------------------------------------

class ExtractionMethod(str, Enum):
    """
    Which method produced the text for a given page.

    Using an Enum prevents typos and makes downstream logic like
    `if page.method == ExtractionMethod.OCR_VISION` explicit and safe.

    PYMUPDF    : text extracted directly from the PDF's embedded character data
    OCR_VISION : page was rendered as an image and read by GPT-4o-mini Vision
    FAILED     : both methods were attempted and neither produced usable text
    """
    PYMUPDF = "pymupdf"
    OCR_VISION = "ocr_vision"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Single-page result
# ---------------------------------------------------------------------------

class PageExtraction(BaseModel):
    """
    Extraction result for one page of a PDF.

    Fields
    ------
    page_number : int
        1-based page number (human-readable, matching what the user sees in a PDF viewer).
        PyMuPDF uses 0-based internally; the parser converts before storing here.

    text : str
        The extracted text for this page. May be empty if method == FAILED.
        Whitespace is normalized by the parser before storing.

    char_count : int
        Number of meaningful characters (non-whitespace) in `text`.
        The extraction validator uses this to decide whether OCR is needed.

    method : ExtractionMethod
        Which extraction method produced this result.

    ocr_confidence : Optional[float]
        Only populated when method == OCR_VISION.
        GPT-4o-mini Vision does not return a numeric confidence score directly,
        so this field is reserved for future use or a manual heuristic.
        Leave as None for now.

    extraction_notes : Optional[str]
        Any warning or informational note from the extraction process.
        Examples: "page appears to be a scanned image", "Vision OCR fallback used",
                  "page returned no text after both methods".
    """

    page_number: int = Field(..., ge=1, description="1-based page number")
    text: str = Field(default="", description="Extracted text for this page")
    char_count: int = Field(default=0, ge=0, description="Non-whitespace character count")
    method: ExtractionMethod = Field(..., description="Extraction method used")
    ocr_confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="OCR confidence score (reserved for future use)"
    )
    extraction_notes: Optional[str] = Field(
        default=None,
        description="Warnings or notes from the extraction process"
    )

    @field_validator("char_count", mode="before")
    @classmethod
    def compute_char_count_if_zero(cls, v, info):
        """
        If char_count wasn't explicitly set, compute it from `text`.
        This makes it safe to construct a PageExtraction without manually
        counting characters — the model self-corrects.
        """
        # info.data holds already-validated fields; text may or may not be set yet
        text = info.data.get("text", "")
        if v == 0 and text:
            return len(text.replace(" ", "").replace("\n", "").replace("\t", ""))
        return v

    model_config = ConfigDict(use_enum_values=False)


# ---------------------------------------------------------------------------
# Full-document result
# ---------------------------------------------------------------------------

class DocumentExtraction(BaseModel):
    """
    Extraction result for an entire PDF document.

    Produced by pdf_parser.py after processing all pages.
    Passed downstream to the chunker, classifier, and guardrails.

    Fields
    ------
    file_path : str
        Absolute or relative path to the source PDF.
        Used for audit logging — never logged with full content.

    file_name : str
        Just the filename (e.g. "NDA_AcmeCorp_2023.pdf").
        Used in reports and log messages.

    file_hash : str
        SHA-256 hash of the file bytes. Used for:
          - Duplicate detection (Section 12 guardrails)
          - Cache keys (Section 26 performance)
          - Audit trail

    total_pages : int
        Total page count as reported by PyMuPDF.

    pages : list[PageExtraction]
        One entry per page, in order. len(pages) == total_pages (or fewer if
        extraction was aborted early due to a critical failure).

    full_text : str
        Concatenation of all page texts, separated by page-break markers.
        Used for whole-document searches (e.g. missing-clause verification in Section 15).

    pages_failed : int
        Count of pages where method == FAILED.
        If pages_failed > 0, the guardrail logs a warning but does not stop the workflow
        unless ALL pages failed.

    pages_ocr : int
        Count of pages where method == OCR_VISION.
        Logged for observability (OCR fallback rate metric, Section 25).

    extraction_successful : bool
        True if at least one page produced usable text.
        False only if every page failed — in which case the workflow halts
        and asks the user for a clearer document.

    error_message : Optional[str]
        Populated if extraction_successful is False, or if a non-fatal
        file-level error occurred (e.g. encrypted PDF detected).
    """

    file_path: str = Field(..., description="Path to the source PDF file")
    file_name: str = Field(..., description="Filename only, for display and logging")
    file_hash: str = Field(..., description="SHA-256 hash of file bytes")
    total_pages: int = Field(..., ge=0, description="Total pages reported by PyMuPDF")

    pages: list[PageExtraction] = Field(
        default_factory=list,
        description="Per-page extraction results, in page order"
    )

    full_text: str = Field(
        default="",
        description="All page texts joined with page-break markers"
    )

    pages_failed: int = Field(
        default=0, ge=0,
        description="Number of pages where both extraction methods failed"
    )

    pages_ocr: int = Field(
        default=0, ge=0,
        description="Number of pages that required OCR Vision fallback"
    )

    extraction_successful: bool = Field(
        default=True,
        description="False only if every page in the document failed extraction"
    )

    error_message: Optional[str] = Field(
        default=None,
        description="File-level error description, if any"
    )

    def get_page(self, page_number: int) -> Optional[PageExtraction]:
        """
        Retrieve a specific page by its 1-based page number.
        Returns None if the page number is out of range.

        Usage:
            page = doc.get_page(7)
            if page:
                print(page.text)
        """
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None

    def summary(self) -> dict:
        """
        Returns a PII-safe summary dict suitable for logging.
        Never includes page text — only counts and metadata.

        This is what gets written to operational logs and LangSmith traces
        (Section 20: Privacy and PII-safe logging).
        """
        return {
            "file_name": self.file_name,
            "file_hash": self.file_hash[:12] + "...",  # truncated for log safety
            "total_pages": self.total_pages,
            "pages_extracted": len(self.pages),
            "pages_ocr": self.pages_ocr,
            "pages_failed": self.pages_failed,
            "extraction_successful": self.extraction_successful,
            "error_message": self.error_message,
        }
