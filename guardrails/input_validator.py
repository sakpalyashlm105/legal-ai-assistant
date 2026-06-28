"""
guardrails/input_validator.py
------------------------------
Pre-pipeline input validation guardrail.

This module runs BEFORE pdf_parser.py is ever called. It answers the question:
"Is this file even worth trying to extract?" If the answer is no, the pipeline
stops here rather than wasting extraction work (and potential API costs) on a
file that was never going to succeed.

Two public functions:

  validate_input(file_path)
      File-level checks: exists, is a file, non-empty, size within limit,
      page count within limit, not password-protected. Returns a GuardrailResult.
      severity is always "blocking" on failure -- an unreadable file cannot
      proceed at all.

  check_duplicate_document(file_hash, processed_hashes_path)
      Checks whether a file_hash has been seen before in a persisted JSON
      record. Returns passed=False, severity="warning" (not "blocking") if
      a duplicate is detected. See the DUPLICATE SEVERITY JUDGMENT CALL note
      below for the reasoning behind "warning" vs "blocking".

  record_processed_document(file_hash, processed_hashes_path, metadata)
      Adds a hash to the persisted record once a document finishes processing
      successfully. NATURAL CALL SITE: this should be called by the orchestrator
      (orchestrator.py) after extract_text_from_pdf() returns successfully and
      before any subsequent LLM nodes run -- or equivalently, by the
      save_feedback / record_metrics nodes at the end of the pipeline (whichever
      runs last unconditionally). The call site wiring belongs to a future
      orchestrator update, not this module.

DUPLICATE SEVERITY JUDGMENT CALL
---------------------------------
Duplicate detection uses severity="warning" (not "blocking") because:
  1. A user might intentionally re-upload the same document for a fresh
     analysis (e.g. after the pipeline was aborted mid-run, or to get a
     second opinion with updated templates).
  2. Blocking re-analysis silently would be worse than surfacing a warning
     and letting the orchestrator / user decide.
  3. The pipeline's state (checkpoint, reviewed decisions) is keyed by
     thread_id, not by file hash, so re-analysis of the same file does not
     corrupt any existing state.
  If the product decision changes (e.g. duplicates should always be rejected
  without exception), flip severity to "blocking" here -- no other module
  needs to change.

RELATIONSHIP TO pdf_parser.py
------------------------------
pdf_parser.py's extract_text_from_pdf() also validates the file internally
(existence, size, page count, password). Those internal checks stay in place --
pdf_parser.py needs to defend itself when called directly (e.g. in tests or
from other call sites). THIS module is the explicit, orchestrator-level gate
that runs first; pdf_parser.py's checks are the defense-in-depth backstop.
The two are complementary, not duplicated: this module can reject a file before
pdf_parser.py is ever invoked.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import fitz  # PyMuPDF

from schemas.guardrails import GuardrailResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# These match the limits enforced inside pdf_parser.py so the two layers
# agree on what "valid" means.
MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB
MAX_PAGE_COUNT: int = 300
MIN_PAGE_COUNT: int = 1

_GUARDRAIL_NAME = "input_validator"


# ---------------------------------------------------------------------------
# validate_input
# ---------------------------------------------------------------------------

def validate_input(
    file_path: str,
    *,
    max_file_size_bytes: int = MAX_FILE_SIZE_BYTES,
    max_page_count: int = MAX_PAGE_COUNT,
) -> GuardrailResult:
    """
    Check whether a file is processable before attempting extraction.

    Parameters
    ----------
    file_path : str
        Path to the PDF file to validate.
    max_file_size_bytes : int
        Override the size limit (useful in tests -- set to a small value to
        simulate an oversized file without needing a real large file).
    max_page_count : int
        Override the page count limit (useful in tests).

    Returns
    -------
    GuardrailResult
        passed=True if all checks pass.
        passed=False, severity="blocking" if ANY check fails.
        severity is always "info" on pass, always "blocking" on fail --
        an unreadable file must never proceed to extraction.
    """
    path = Path(file_path)
    name = path.name  # PII-safe: just the filename, not full content

    # 1. Exists
    if not path.exists():
        logger.warning("%s: file does not exist: %s", _GUARDRAIL_NAME, name)
        return GuardrailResult(
            guardrail_name=f"{_GUARDRAIL_NAME}.file_exists",
            passed=False,
            reason=f"File not found: {name}",
            severity="blocking",
        )

    # 2. Is a file (not a directory or symlink to nowhere)
    if not path.is_file():
        logger.warning("%s: path is not a regular file: %s", _GUARDRAIL_NAME, name)
        return GuardrailResult(
            guardrail_name=f"{_GUARDRAIL_NAME}.is_file",
            passed=False,
            reason=f"Path is not a regular file: {name}",
            severity="blocking",
        )

    # 3. Non-empty
    size_bytes = path.stat().st_size
    if size_bytes == 0:
        logger.warning("%s: file is empty: %s", _GUARDRAIL_NAME, name)
        return GuardrailResult(
            guardrail_name=f"{_GUARDRAIL_NAME}.non_empty",
            passed=False,
            reason=f"File is empty (0 bytes): {name}",
            severity="blocking",
        )

    # 4. Size within limit
    if size_bytes > max_file_size_bytes:
        limit_mb = max_file_size_bytes / (1024 * 1024)
        actual_mb = size_bytes / (1024 * 1024)
        logger.warning(
            "%s: file too large: %s (%.1f MB, limit %.0f MB)",
            _GUARDRAIL_NAME, name, actual_mb, limit_mb,
        )
        return GuardrailResult(
            guardrail_name=f"{_GUARDRAIL_NAME}.size_limit",
            passed=False,
            reason=f"File exceeds {limit_mb:.0f} MB size limit ({actual_mb:.1f} MB): {name}",
            severity="blocking",
        )

    # 5. Page count + password-protection (cheap PyMuPDF open, no text extraction)
    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        logger.warning("%s: cannot open file: %s — %s", _GUARDRAIL_NAME, name, exc)
        return GuardrailResult(
            guardrail_name=f"{_GUARDRAIL_NAME}.openable",
            passed=False,
            reason=f"Cannot open file (corrupt or unsupported format): {name}",
            severity="blocking",
        )

    try:
        # Password-protected documents report needs_pass=True
        if doc.needs_pass:
            logger.warning("%s: file is password-protected: %s", _GUARDRAIL_NAME, name)
            return GuardrailResult(
                guardrail_name=f"{_GUARDRAIL_NAME}.password_protected",
                passed=False,
                reason=f"File is password-protected and cannot be read: {name}",
                severity="blocking",
            )

        page_count = doc.page_count
        if page_count < MIN_PAGE_COUNT:
            logger.warning("%s: file has no pages: %s", _GUARDRAIL_NAME, name)
            return GuardrailResult(
                guardrail_name=f"{_GUARDRAIL_NAME}.page_count",
                passed=False,
                reason=f"File has no readable pages: {name}",
                severity="blocking",
            )

        if page_count > max_page_count:
            logger.warning(
                "%s: file has too many pages: %s (%d pages, limit %d)",
                _GUARDRAIL_NAME, name, page_count, max_page_count,
            )
            return GuardrailResult(
                guardrail_name=f"{_GUARDRAIL_NAME}.page_count",
                passed=False,
                reason=(
                    f"File has {page_count} pages, exceeding the {max_page_count}-page limit: {name}"
                ),
                severity="blocking",
            )
    finally:
        doc.close()

    logger.info("%s: file passed all checks: %s (%d pages, %.1f KB)",
                _GUARDRAIL_NAME, name, page_count, size_bytes / 1024)
    return GuardrailResult(
        guardrail_name=f"{_GUARDRAIL_NAME}.all_checks",
        passed=True,
        reason=f"File passed all input validation checks ({page_count} pages).",
        severity="info",
    )


# ---------------------------------------------------------------------------
# Duplicate document detection
# ---------------------------------------------------------------------------

def check_duplicate_document(
    file_hash: str,
    processed_hashes_path: str = "data/processed/processed_hashes.json",
) -> GuardrailResult:
    """
    Check whether this document has been processed before.

    Parameters
    ----------
    file_hash : str
        SHA-256 hex digest of the file (from compute_file_hash in pdf_parser.py).
    processed_hashes_path : str
        Path to the JSON file storing previously processed hashes.
        Created automatically if it does not exist.

    Returns
    -------
    GuardrailResult
        passed=True if the hash has never been seen before.
        passed=False, severity="warning" if the hash matches a previous run.
        severity is "warning" (not "blocking") -- see module docstring.
    """
    record = _load_hash_record(processed_hashes_path)
    hash_prefix = file_hash[:12]  # log-safe prefix per schemas/document.py convention

    if file_hash in record:
        logger.warning(
            "%s.duplicate_check: hash %s... already in processed record",
            _GUARDRAIL_NAME, hash_prefix,
        )
        return GuardrailResult(
            guardrail_name=f"{_GUARDRAIL_NAME}.duplicate_check",
            passed=False,
            reason=(
                f"Document hash {hash_prefix}... matches a previously processed file. "
                "Re-analysis is allowed but flagged for audit purposes."
            ),
            severity="warning",
        )

    logger.info("%s.duplicate_check: hash %s... not seen before", _GUARDRAIL_NAME, hash_prefix)
    return GuardrailResult(
        guardrail_name=f"{_GUARDRAIL_NAME}.duplicate_check",
        passed=True,
        reason=f"Document hash {hash_prefix}... has not been processed before.",
        severity="info",
    )


def record_processed_document(
    file_hash: str,
    processed_hashes_path: str = "data/processed/processed_hashes.json",
    metadata: dict | None = None,
) -> None:
    """
    Record a document hash as successfully processed.

    Call this AFTER extraction succeeds and BEFORE any LLM node runs,
    so that a re-upload of the same file will be flagged as a duplicate
    on the next run. The natural call site is orchestrator.py, in the
    node that runs after extract_text succeeds (or in the save_feedback /
    record_metrics node at the end of the pipeline).

    Parameters
    ----------
    file_hash : str
        SHA-256 hex digest of the file.
    processed_hashes_path : str
        Path to the JSON persistence file.
    metadata : dict | None
        Optional dict of extra fields to record alongside the hash
        (e.g. {"file_name": "NDA_Acme.pdf", "processed_at": "..."}).
        Must not contain full document text or PII.
    """
    record = _load_hash_record(processed_hashes_path)
    hash_prefix = file_hash[:12]

    entry = {
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        # Allow callers to attach safe metadata (filename, page count, etc.)
        entry.update(metadata)

    record[file_hash] = entry

    path = Path(processed_hashes_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    logger.info(
        "%s: recorded hash %s... as processed", _GUARDRAIL_NAME, hash_prefix
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_hash_record(processed_hashes_path: str) -> dict:
    """Load the persisted hash record, returning {} if the file doesn't exist yet."""
    path = Path(processed_hashes_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "%s: could not read hash record at %s — %s; treating as empty",
            _GUARDRAIL_NAME, processed_hashes_path, exc,
        )
        return {}
