"""
extraction/pdf_parser.py
------------------------
Primary text extraction module for the Legal AI Assistant.

What this file does:
    Opens a PDF file, extracts text from every page using PyMuPDF,
    checks whether each page's text is good enough, and routes
    low-quality pages to the OCR Vision fallback (ocr.py).

    Returns a DocumentExtraction object (defined in schemas/document.py)
    that holds the result for every page plus file-level metadata.

Why PyMuPDF (fitz)?
    Most legal PDFs are "digital-born" -- they were created in Word or
    a PDF editor, so the text is embedded directly as characters.
    PyMuPDF can read those characters instantly without needing AI.
    It is fast, free, and accurate for this type of document.

When does OCR get used?
    Some pages (especially exhibits, scanned attachments, or older
    contracts) are just images of text with no embedded characters.
    PyMuPDF returns almost nothing for those pages. When that happens,
    we hand the page to ocr.py which sends it to GPT-4o-mini Vision.

Dependencies:
    fitz (PyMuPDF 1.24.5)
    schemas/document.py
    guardrails/extraction_validator.py
    extraction/ocr.py
    config.py
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF -- imported as "fitz" by historical convention

from guardrails.extraction_validator import check_extraction_quality
from schemas.document import DocumentExtraction, ExtractionMethod, PageExtraction

# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

# Think of a logger like a diary that the program writes in as it runs.
# We give each module its own named diary so we know which module wrote each line.
# "__name__" automatically becomes "extraction.pdf_parser" -- the module's own name.
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants -- pulled from config.py in the full system
# ---------------------------------------------------------------------------

# Note: the character-count threshold itself now lives in
# guardrails/extraction_validator.py (EXTRACTION_CHAR_THRESHOLD), since
# that module owns the "is this page good enough?" decision. We don't
# duplicate the number here anymore -- pdf_parser.py just calls the
# guardrail function and trusts its answer.

# Maximum file size we will attempt to process (in bytes).
# 50 MB = 50 * 1024 * 1024. Larger files risk memory issues and very long
# processing times -- we reject them at this layer and ask the user to split the doc.
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB

# Maximum number of pages we will process in a single document.
# Matches config.py: MAX_PAGE_COUNT = 300
MAX_PAGE_COUNT = 300


# ---------------------------------------------------------------------------
# Helper: compute SHA-256 file hash
# ---------------------------------------------------------------------------

def compute_file_hash(file_path: str) -> str:
    """
    Compute the SHA-256 fingerprint of a file.

    What is SHA-256?
        Imagine putting a document through a special machine that produces
        a unique 64-character code based on every single byte in the file.
        Change even one letter in the document, and the code changes completely.
        That code is the SHA-256 hash.

    Why do we need it?
        1. Duplicate detection: if two uploads have the same hash, they are
           the same document. No need to process it twice.
        2. Cache keys: we use the hash to look up previously extracted text
           so we don't re-extract the same file every time.
        3. Audit trail: logs record the hash (not the content), so we can
           track which document was processed without storing sensitive text.

    Parameters
    ----------
    file_path : str
        The path to the PDF file on disk.
        Example: "C:/legal-agent/data/raw/contracts/NDA_Acme.pdf"

    Returns
    -------
    str
        A 64-character hexadecimal string.
        Example: "a3f2c1d4e5b6..." (64 chars total)
    """
    # hashlib.sha256() creates a SHA-256 calculator object.
    sha256 = hashlib.sha256()

    # We open the file in binary mode ("rb") -- read the raw bytes, not text.
    # PDFs are binary files; opening them as text would corrupt the data.
    with open(file_path, "rb") as f:

        # We read the file in chunks of 65,536 bytes (64 KB) at a time.
        # Why chunks? A 50 MB PDF would use 50 MB of RAM if loaded all at once.
        # Reading in chunks keeps memory usage low regardless of file size.
        # This is called "streaming" -- like watching a video instead of
        # downloading the whole thing before pressing play.
        while chunk := f.read(65536):
            sha256.update(chunk)  # feed each chunk into the calculator

    # .hexdigest() returns the final hash as a readable hex string.
    return sha256.hexdigest()


# Note: the "count meaningful characters" helper now lives in
# guardrails/extraction_validator.py as part of check_extraction_quality().
# pdf_parser.py no longer needs its own copy -- the guardrail's result
# object hands back the char_count we need.


# ---------------------------------------------------------------------------
# Helper: normalize page text
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Clean up raw text extracted from a PDF page.

    Why do we need to normalize?
        PyMuPDF sometimes returns text with:
        - Multiple blank lines in a row ("\\n\\n\\n\\n")
        - Trailing spaces at the end of lines
        - Weird spacing artifacts from PDF formatting
        These don't affect the meaning, but they waste space and can
        confuse the chunker and embedding model later.

    What we do NOT do:
        We do NOT remove hyphens, punctuation, clause numbers, or
        paragraph structure. Legal contracts depend on exact wording.
        "shall not" vs "shall" is a legal difference. We preserve meaning.

    Parameters
    ----------
    text : str
        Raw text as returned by PyMuPDF's page.get_text().

    Returns
    -------
    str
        Cleaned text with normalized whitespace.
    """
    if not text:
        return ""

    # Step 1: Split into lines, strip trailing spaces from each line
    lines = [line.rstrip() for line in text.splitlines()]

    # Step 2: Collapse runs of more than 2 consecutive blank lines into 2.
    # This preserves paragraph breaks (2 blank lines) but removes excessive gaps.
    normalized_lines = []
    blank_count = 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 2:
                normalized_lines.append(line)
        else:
            blank_count = 0
            normalized_lines.append(line)

    return "\n".join(normalized_lines).strip()


# ---------------------------------------------------------------------------
# Core function: extract one page
# ---------------------------------------------------------------------------

def extract_page_text(
    pdf_document: fitz.Document,
    page_index: int,
    use_ocr_fallback: bool = True,
) -> PageExtraction:
    """
    Extract text from a single page of an open PDF document.

    This is the workhorse function. It is called once per page by
    extract_text_from_pdf() below.

    How it works:
        1. Get the page object from PyMuPDF using page_index (0-based).
        2. Call page.get_text() to extract embedded text.
        3. Normalize the text.
        4. Ask the extraction_validator guardrail: is this text good enough?
        5. If the guardrail says PASS -> return PageExtraction (method=pymupdf).
        6. If the guardrail says FAIL and use_ocr_fallback is True ->
           import and call the OCR module, return its result.
        7. If the guardrail says FAIL and use_ocr_fallback is False ->
           return a FAILED PageExtraction.

    Parameters
    ----------
    pdf_document : fitz.Document
        An open PyMuPDF document object. Think of this as the already-opened
        book -- we don't re-open the file for every page, we just flip to
        the right page inside the same open book.

    page_index : int
        The 0-based page index (PyMuPDF's internal numbering).
        Page index 0 = page number 1 for the user.
        We convert to 1-based before storing in PageExtraction.

    use_ocr_fallback : bool, default True
        If True, pages below the character threshold are sent to the
        GPT-4o-mini Vision OCR module.
        If False, below-threshold pages are marked as FAILED immediately.
        Set to False during testing when you don't want to burn API calls.

    Returns
    -------
    PageExtraction
        A filled PageExtraction object for this page.
    """
    # Convert from 0-based (PyMuPDF) to 1-based (human-readable) page number.
    # If PyMuPDF is on page_index=0, the user sees page 1.
    page_number = page_index + 1

    logger.debug(f"Extracting page {page_number} (index {page_index})")

    try:
        # --- Step 1: Get the page object ---
        # pdf_document[page_index] is how PyMuPDF gives you a specific page.
        # Think of it like pdf_document[0] = "open to page 1 of the book".
        page = pdf_document[page_index]

        # --- Step 2: Extract text ---
        # page.get_text() is PyMuPDF's main extraction call.
        # "text" mode returns plain text, preserving line breaks.
        # Other modes exist (like "blocks" for layout-aware extraction)
        # but "text" is sufficient for our use case.
        raw_text = page.get_text("text")

        # --- Step 3: Normalize ---
        cleaned_text = normalize_text(raw_text)

        # --- Step 4: Check extraction quality (guardrail decision) ---
        # This used to be an inline "if char_count >= CHAR_THRESHOLD" check
        # right here in this file. It now lives in its own guardrail module
        # (guardrails/extraction_validator.py) so the rule is centralized,
        # documented once, and testable on its own.
        quality_result = check_extraction_quality(cleaned_text)
        char_count = quality_result.char_count

        # --- Step 5: Act on the guardrail's decision ---
        if quality_result.passed:
            # The page has enough text -- PyMuPDF extraction succeeded.
            logger.debug(f"Page {page_number}: PyMuPDF OK ({char_count} chars)")
            return PageExtraction(
                page_number=page_number,
                text=cleaned_text,
                char_count=char_count,
                method=ExtractionMethod.PYMUPDF,
            )

        # --- Step 6: Below threshold -- needs OCR ---
        logger.info(
            f"Page {page_number}: {quality_result.reason} "
            + ("Routing to OCR Vision." if use_ocr_fallback else "OCR fallback disabled.")
        )

        if not use_ocr_fallback:
            # OCR is disabled (e.g. during unit tests) -- mark as failed.
            return PageExtraction(
                page_number=page_number,
                text="",
                char_count=0,
                method=ExtractionMethod.FAILED,
                extraction_notes=f"Below threshold ({char_count} chars). OCR fallback disabled.",
            )

        # --- Step 7: OCR fallback ---
        # We import here (inside the function) rather than at the top of the file.
        # Why? Because ocr.py makes API calls to OpenAI. If we import it at the
        # top level, any script that imports pdf_parser also loads the OCR module
        # even when OCR is never needed. Lazy import = only load when actually used.
        from extraction.ocr import extract_page_with_vision

        # Render the page as an image so Vision can read it.
        # fitz.Matrix(2, 2) creates a 2x zoom matrix -- doubles the resolution.
        # Higher resolution = better OCR accuracy, at the cost of a larger image.
        # 2x is a good balance for legal documents.
        zoom_matrix = fitz.Matrix(2, 2)

        # page.get_pixmap() renders the page to a pixel image.
        # A "pixmap" is just an in-memory image (pixels map).
        pixmap = page.get_pixmap(matrix=zoom_matrix)

        # Convert the pixmap to raw PNG bytes.
        # We pass bytes (not a file path) to the OCR function so we don't
        # need to write a temporary file to disk.
        image_bytes = pixmap.tobytes("png")

        # Call the OCR Vision module and get back a PageExtraction.
        ocr_result = extract_page_with_vision(
            image_bytes=image_bytes,
            page_number=page_number,
        )

        return ocr_result

    except Exception as e:
        # Something unexpected went wrong on this page (corrupted page data,
        # rendering error, etc.). Log the error and return a FAILED result
        # rather than crashing the entire extraction.
        # This follows the principle: one bad page should not kill the whole document.
        logger.error(f"Page {page_number}: unexpected error during extraction: {e}")
        return PageExtraction(
            page_number=page_number,
            text="",
            char_count=0,
            method=ExtractionMethod.FAILED,
            extraction_notes=f"Extraction error: {str(e)}",
        )


# ---------------------------------------------------------------------------
# Main public function: extract the full document
# ---------------------------------------------------------------------------

def extract_text_from_pdf(
    file_path: str,
    use_ocr_fallback: bool = True,
) -> DocumentExtraction:
    """
    Extract text from every page of a PDF file.

    This is the main function that the rest of the system calls.
    Everything else in this file supports this function.

    The overall flow:
        1. Validate the file exists, is readable, is not too large,
           is not password-protected, and is not empty.
        2. Compute the file hash (for deduplication and caching).
        3. Open the PDF with PyMuPDF.
        4. Loop through every page, calling extract_page_text() for each.
        5. Build the full_text string from all page texts.
        6. Count how many pages used OCR vs PyMuPDF vs failed.
        7. Determine whether the overall extraction was successful.
        8. Return a DocumentExtraction object.

    Parameters
    ----------
    file_path : str
        Path to the PDF file on disk.
        Example: "data/raw/contracts/NDA_Acme_2023.pdf"

    use_ocr_fallback : bool, default True
        Whether to use GPT-4o-mini Vision for pages below the character
        threshold. Set to False during testing to avoid OpenAI API calls.

    Returns
    -------
    DocumentExtraction
        Complete extraction result for the document.
        Always returns a DocumentExtraction -- never raises an exception.
        If something goes wrong, extraction_successful=False and
        error_message explains what happened.
    """
    # Convert to a Path object for clean cross-platform file handling.
    # Path("C:\\Users\\Yash\\file.pdf").name -> "file.pdf"
    path = Path(file_path)
    file_name = path.name  # just the filename, no directory part

    logger.info(f"Starting extraction: {file_name}")

    # -----------------------------------------------------------------------
    # Pre-flight checks -- validate the file before we try to open it
    # -----------------------------------------------------------------------

    # Check 1: Does the file exist?
    if not path.exists():
        logger.error(f"File not found: {file_path}")
        return DocumentExtraction(
            file_path=file_path,
            file_name=file_name,
            file_hash="unknown",
            total_pages=0,
            extraction_successful=False,
            error_message=f"File not found: {file_path}",
        )

    # Check 2: Is it actually a file (not a folder)?
    if not path.is_file():
        logger.error(f"Path is not a file: {file_path}")
        return DocumentExtraction(
            file_path=file_path,
            file_name=file_name,
            file_hash="unknown",
            total_pages=0,
            extraction_successful=False,
            error_message=f"Path is not a file: {file_path}",
        )

    # Check 3: Is the file empty? (0 bytes)
    file_size = path.stat().st_size  # .stat() gives file metadata; .st_size is bytes
    if file_size == 0:
        logger.error(f"File is empty (0 bytes): {file_name}")
        return DocumentExtraction(
            file_path=file_path,
            file_name=file_name,
            file_hash="unknown",
            total_pages=0,
            extraction_successful=False,
            error_message="File is empty (0 bytes).",
        )

    # Check 4: Is the file too large?
    if file_size > MAX_FILE_SIZE_BYTES:
        size_mb = file_size / (1024 * 1024)
        logger.error(f"File too large: {size_mb:.1f} MB (max {MAX_FILE_SIZE_BYTES // (1024*1024)} MB)")
        return DocumentExtraction(
            file_path=file_path,
            file_name=file_name,
            file_hash="unknown",
            total_pages=0,
            extraction_successful=False,
            error_message=f"File too large: {size_mb:.1f} MB. Maximum allowed is {MAX_FILE_SIZE_BYTES // (1024*1024)} MB.",
        )

    # -----------------------------------------------------------------------
    # Compute file hash
    # -----------------------------------------------------------------------
    try:
        file_hash = compute_file_hash(file_path)
        logger.debug(f"File hash computed: {file_hash[:12]}...")
    except Exception as e:
        logger.error(f"Could not hash file: {e}")
        file_hash = "hash_error"

    # -----------------------------------------------------------------------
    # Open the PDF with PyMuPDF
    # -----------------------------------------------------------------------
    try:
        # fitz.open() opens the PDF and returns a Document object.
        # This does NOT load all pages into memory -- it opens the file handle
        # and lets you access pages on demand (like opening a book to its cover
        # without reading every page yet).
        pdf_doc = fitz.open(file_path)

    except fitz.FileDataError as e:
        # FileDataError means PyMuPDF could not parse the file as a PDF.
        # This happens with corrupted files or files that look like PDFs
        # but aren't (e.g. a .pdf extension on an image file).
        logger.error(f"Corrupted or unreadable PDF: {file_name} -- {e}")
        return DocumentExtraction(
            file_path=file_path,
            file_name=file_name,
            file_hash=file_hash,
            total_pages=0,
            extraction_successful=False,
            error_message=f"Could not open PDF -- file may be corrupted: {e}",
        )

    except Exception as e:
        logger.error(f"Unexpected error opening PDF: {e}")
        return DocumentExtraction(
            file_path=file_path,
            file_name=file_name,
            file_hash=file_hash,
            total_pages=0,
            extraction_successful=False,
            error_message=f"Unexpected error opening PDF: {e}",
        )

    # -----------------------------------------------------------------------
    # Password-protected PDF check
    # -----------------------------------------------------------------------
    # pdf_doc.needs_pass is True if the PDF requires a password to read.
    # We do not handle passwords -- we reject the document and ask the user
    # to provide an unlocked version.
    if pdf_doc.needs_pass:
        logger.error(f"Password-protected PDF: {file_name}")
        pdf_doc.close()
        return DocumentExtraction(
            file_path=file_path,
            file_name=file_name,
            file_hash=file_hash,
            total_pages=0,
            extraction_successful=False,
            error_message="PDF is password-protected. Please provide an unlocked version.",
        )

    # -----------------------------------------------------------------------
    # Page count check
    # -----------------------------------------------------------------------
    total_pages = pdf_doc.page_count
    # pdf_doc.page_count is a property (not a function call) that returns
    # how many pages PyMuPDF found in the file.

    if total_pages == 0:
        logger.error(f"PDF has 0 pages: {file_name}")
        pdf_doc.close()
        return DocumentExtraction(
            file_path=file_path,
            file_name=file_name,
            file_hash=file_hash,
            total_pages=0,
            extraction_successful=False,
            error_message="PDF has 0 pages.",
        )

    if total_pages > MAX_PAGE_COUNT:
        logger.error(f"PDF has {total_pages} pages (max {MAX_PAGE_COUNT}): {file_name}")
        pdf_doc.close()
        return DocumentExtraction(
            file_path=file_path,
            file_name=file_name,
            file_hash=file_hash,
            total_pages=total_pages,
            extraction_successful=False,
            error_message=f"PDF has {total_pages} pages. Maximum allowed is {MAX_PAGE_COUNT}.",
        )

    logger.info(f"PDF opened: {file_name} -- {total_pages} pages")

    # -----------------------------------------------------------------------
    # Page-by-page extraction loop
    # -----------------------------------------------------------------------
    extracted_pages: list[PageExtraction] = []
    pages_failed = 0
    pages_ocr = 0

    # range(total_pages) produces 0, 1, 2, ..., total_pages-1
    # These are the 0-based page indices that PyMuPDF expects.
    for page_index in range(total_pages):
        page_result = extract_page_text(
            pdf_document=pdf_doc,
            page_index=page_index,
            use_ocr_fallback=use_ocr_fallback,
        )

        extracted_pages.append(page_result)

        # Tally results for summary counters
        if page_result.method == ExtractionMethod.FAILED:
            pages_failed += 1
        elif page_result.method == ExtractionMethod.OCR_VISION:
            pages_ocr += 1

    # Always close the PDF document when done.
    # This releases the file handle so other processes can access the file.
    # Think of it like closing the book after you're done reading.
    pdf_doc.close()

    # -----------------------------------------------------------------------
    # Build full_text -- all pages joined with page markers
    # -----------------------------------------------------------------------
    # We join all page texts with a clear separator so downstream modules
    # can search the full document as one string.
    # The separator looks like:
    #   \n\n--- Page 3 ---\n\n
    # This makes it human-readable if we ever print it, and lets the
    # missing-clause verifier know which page a match came from.

    page_separator = "\n\n{separator}\n\n"
    page_texts = []
    for p in extracted_pages:
        if p.text:
            separator = f"--- Page {p.page_number} ---"
            page_texts.append(f"{page_separator.format(separator=separator)}{p.text}")

    full_text = "".join(page_texts).strip()

    # -----------------------------------------------------------------------
    # Determine overall success
    # -----------------------------------------------------------------------
    # The extraction is successful if at least one page produced usable text.
    # All pages failing = no content to analyze = extraction_successful=False.
    extraction_successful = pages_failed < total_pages

    if not extraction_successful:
        error_message = (
            f"All {total_pages} pages failed extraction. "
            "The document may be a low-quality scan. "
            "Please provide a clearer version."
        )
        logger.error(f"Complete extraction failure: {file_name}")
    elif pages_failed > 0:
        error_message = None  # partial failure is a warning, not an error
        logger.warning(
            f"Partial extraction: {pages_failed}/{total_pages} pages failed -- {file_name}"
        )
    else:
        error_message = None
        logger.info(f"Extraction complete: {file_name} -- all {total_pages} pages OK")

    # -----------------------------------------------------------------------
    # Assemble and return the DocumentExtraction
    # -----------------------------------------------------------------------
    result = DocumentExtraction(
        file_path=str(path.resolve()),   # absolute path for audit logging
        file_name=file_name,
        file_hash=file_hash,
        total_pages=total_pages,
        pages=extracted_pages,
        full_text=full_text,
        pages_failed=pages_failed,
        pages_ocr=pages_ocr,
        extraction_successful=extraction_successful,
        error_message=error_message,
    )

    # Log a PII-safe summary (no text content)
    logger.info(f"DocumentExtraction summary: {result.summary()}")

    return result
