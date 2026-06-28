"""
agent/feedback_curation.py
--------------------------
Stage 4 of the two-stage feedback/precedent lifecycle.

    save_feedback()                  <- stage 1 (agent/feedback_writer.py)
    approve_feedback_as_precedent()  <- stage 4 (this file)

Purpose
-------
A human curator explicitly promotes a `pending_precedent_review` record to
`approved_precedent`, meaning it will be visible to risk_engine.py as a
valid precedent for future matching.

HITL, NOT RLHF
--------------
Promotion here never updates model weights or feeds fine-tuning. It only
marks clause language as acceptable business precedent for future MEDIUM-risk
downgrade decisions in the risk engine.

PII-safe logging
----------------
Approval log statements emit feedback_id and approved_by only — never
reviewer_comment or evidence_excerpt content.

Atomic write
------------
The feedback log is rewritten atomically: read all lines → update the target
record in memory → write to a .tmp file → os.replace(.tmp → .jsonl). This
ensures the log is never in a partially-written state if the process dies.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from schemas.feedback import FeedbackRecord, PrecedentScope
from agent.risk_engine import _clear_feedback_cache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Storage path — same file as agent/feedback_writer.py
# ---------------------------------------------------------------------------

_FEEDBACK_DIR = Path(__file__).parent.parent / "data" / "feedback"
_FEEDBACK_LOG = _FEEDBACK_DIR / "feedback_log.jsonl"
_FEEDBACK_LOG_TMP = _FEEDBACK_DIR / "feedback_log.jsonl.tmp"


# ---------------------------------------------------------------------------
# Named exception hierarchy
# ---------------------------------------------------------------------------

class FeedbackCurationError(Exception):
    """Base class for all precedent promotion errors."""


class FeedbackRecordNotFoundError(FeedbackCurationError):
    """Raised when the feedback_id does not exist in the log."""


class FeedbackNotEligibleError(FeedbackCurationError):
    """Raised when the record is not in `pending_precedent_review` status."""


class FeedbackAlreadyPromotedError(FeedbackCurationError):
    """Raised when the record is already `approved_precedent` or `rejected_precedent`."""


class FeedbackPromotionValidationError(FeedbackCurationError):
    """Raised when a field-level validation check fails during promotion."""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _read_all_records() -> list[FeedbackRecord]:
    """Read every line from the log and parse into FeedbackRecord objects."""
    if not _FEEDBACK_LOG.exists():
        return []
    records: list[FeedbackRecord] = []
    with _FEEDBACK_LOG.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(FeedbackRecord(**json.loads(line)))
            except Exception as exc:
                logger.warning("feedback_curation: skipping malformed line — %s", exc)
    return records


def _write_all_records_atomic(records: list[FeedbackRecord]) -> None:
    """
    Write all records to the log atomically.

    Writes to a .tmp file first, then renames it over the real log using
    os.replace(), which is atomic on POSIX and Windows NTFS. If anything
    fails before os.replace(), the original file is untouched.
    """
    _FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    with _FEEDBACK_LOG_TMP.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(record.model_dump_json() + "\n")
    # Atomic rename — if this line is never reached (exception above), the
    # original log is untouched. If the rename itself fails, the .tmp file
    # is left behind but the original is still intact.
    os.replace(_FEEDBACK_LOG_TMP, _FEEDBACK_LOG)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def approve_feedback_as_precedent(
    feedback_id: str,
    approved_by: str,
    final_accepted_risk: str,
    precedent_scope: PrecedentScope,
    approval_note: str,
) -> FeedbackRecord:
    """
    Promote a `pending_precedent_review` record to `approved_precedent`.

    Stage 4 of the two-stage feedback/precedent lifecycle. Only records that
    passed all eligibility checks in save_feedback() (stage 1) and are
    currently at `pending_precedent_review` can be promoted.

    Parameters
    ----------
    feedback_id : str
        The record to promote. Format: "fb_<review_id>".
    approved_by : str
        Identifier of the curator running this promotion.
    final_accepted_risk : str
        Must be "MEDIUM". Other values are rejected outright — precedent
        promotion is only valid for MEDIUM-risk findings.
    precedent_scope : PrecedentScope
        Scope metadata constraining which future documents this precedent
        applies to. `clause_category` is always required. `document_type`
        will be auto-filled from the record if not provided by the curator.
    approval_note : str
        Non-empty free-text explanation of why this clause language is being
        accepted as a business precedent. Required for audit.

    Returns
    -------
    FeedbackRecord
        The promoted record with feedback_status="approved_precedent".

    Raises
    ------
    FeedbackRecordNotFoundError
        If feedback_id is not found in the log.
    FeedbackNotEligibleError
        If the record's current status is "not_eligible".
    FeedbackAlreadyPromotedError
        If the record is already "approved_precedent" or "rejected_precedent".
    FeedbackPromotionValidationError
        If any field-level validation check fails (HIGH risk, missing clause,
        None clause_category, empty approval_note, etc.).
    pydantic.ValidationError
        If the updated FeedbackRecord violates a schema invariant. This
        indicates a bug in the promotion logic.
    """
    # --- Load all records ---
    records = _read_all_records()

    # --- Check 1: record must exist ---
    target_index: Optional[int] = None
    target: Optional[FeedbackRecord] = None
    for i, rec in enumerate(records):
        if rec.feedback_id == feedback_id:
            target_index = i
            target = rec
            break

    if target is None:
        raise FeedbackRecordNotFoundError(
            f"approve_feedback_as_precedent: feedback_id={feedback_id!r} not found in "
            f"{_FEEDBACK_LOG}. Confirm the feedback_id is correct and that save_feedback() "
            "has already been called for this review."
        )

    # --- Check 2: must be pending_precedent_review ---
    if target.feedback_status == "not_eligible":
        raise FeedbackNotEligibleError(
            f"approve_feedback_as_precedent: feedback_id={feedback_id!r} has status "
            f"'not_eligible' and cannot be promoted. not_eligible records do not meet the "
            "minimum criteria for precedent consideration (e.g. HIGH risk, missing clause, "
            "no clause_category, evidence not found, page reference invalid, or reviewer "
            "did not flag clause language). Promotion is not permitted."
        )
    if target.feedback_status in ("approved_precedent", "rejected_precedent"):
        raise FeedbackAlreadyPromotedError(
            f"approve_feedback_as_precedent: feedback_id={feedback_id!r} is already at "
            f"status={target.feedback_status!r}. Re-promotion is not permitted. "
            "Each record may only be promoted once."
        )
    if target.feedback_status != "pending_precedent_review":
        raise FeedbackNotEligibleError(
            f"approve_feedback_as_precedent: feedback_id={feedback_id!r} has unexpected "
            f"status={target.feedback_status!r}. Expected 'pending_precedent_review'."
        )

    # --- Check 3: clause_category must not be None (defense in depth) ---
    if target.clause_category is None:
        raise FeedbackPromotionValidationError(
            f"approve_feedback_as_precedent: feedback_id={feedback_id!r} has "
            "clause_category=None. A precedent without a clause category cannot be matched "
            "against future documents. This record should have been classified not_eligible "
            "by save_feedback() — this path indicates a logic inconsistency."
        )

    # --- Check 4: clause must be present ---
    if not target.is_clause_present:
        raise FeedbackPromotionValidationError(
            f"approve_feedback_as_precedent: feedback_id={feedback_id!r} has "
            "is_clause_present=False. Precedent promotion requires actual clause language "
            "to compare. Missing-clause findings have no language to promote."
        )

    # --- Check 5: final_accepted_risk must be MEDIUM ---
    if final_accepted_risk != "MEDIUM":
        raise FeedbackPromotionValidationError(
            f"approve_feedback_as_precedent: final_accepted_risk must be 'MEDIUM'. "
            f"Got {final_accepted_risk!r}. HIGH-risk findings cannot become precedents — "
            "approving a HIGH finding means the reviewer confirmed a real risk, not that "
            "the clause language is acceptable as a future benchmark. LOW-risk findings "
            "are not meaningful precedents (they already score LOW without one)."
        )

    # Cross-check against the record's own final_risk
    if target.final_risk != "MEDIUM":
        raise FeedbackPromotionValidationError(
            f"approve_feedback_as_precedent: feedback_id={feedback_id!r} has "
            f"final_risk={target.final_risk!r} on the stored record, but "
            f"final_accepted_risk='MEDIUM' was passed. These must agree. "
            "Do not attempt to promote a record whose stored final_risk is not MEDIUM."
        )

    # --- Check 7: approval_note must be non-empty ---
    if not approval_note or not approval_note.strip():
        raise FeedbackPromotionValidationError(
            f"approve_feedback_as_precedent: approval_note is required and must be "
            "non-empty. Provide a brief explanation of why this clause language is being "
            "accepted as a business precedent (for the audit trail)."
        )

    # --- Check 8: scope completeness — auto-fill document_type if curator omitted it ---
    effective_scope = precedent_scope
    if effective_scope.document_type is None and target.document_type is not None:
        effective_scope = PrecedentScope(
            document_type=target.document_type,
            clause_category=effective_scope.clause_category,
            jurisdiction=effective_scope.jurisdiction,
            template_version=effective_scope.template_version,
        )

    # --- All checks passed — build the promoted record ---
    promoted = FeedbackRecord(
        **{
            **target.model_dump(),
            "feedback_status": "approved_precedent",
            "approved_for_precedent": True,
            "precedent_scope": effective_scope,
            "precedent_approved_by": approved_by,
            "precedent_approved_at": datetime.utcnow(),
            # Preserve all other fields unchanged
        }
    )

    # --- Atomic write ---
    records[target_index] = promoted
    _write_all_records_atomic(records)

    # --- Invalidate risk engine's in-process feedback cache ---
    _clear_feedback_cache()

    logger.info(
        "approve_feedback_as_precedent: promoted feedback_id=%s clause=%s approved_by=%s",
        feedback_id,
        target.clause_category,
        approved_by,
    )

    return promoted
