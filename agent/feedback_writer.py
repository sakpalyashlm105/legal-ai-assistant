"""
agent/feedback_writer.py
------------------------
HITL feedback persistence: records every resolved human review decision to
data/feedback/feedback_log.jsonl (one JSON object per line).

Two-stage lifecycle — this module owns stage 1 only:

    save_feedback()                  <- stage 1 (this file)
    approve_feedback_as_precedent()  <- stage 2 (agent/feedback_curation.py, Step 12 Stage 4)

HITL, NOT RLHF
--------------
Records written here never update model weights or feed into fine-tuning.
They are used only for HITL audit trails and (after curator approval via
approve_feedback_as_precedent) for precedent matching in risk_engine.py.

PII-safe logging
----------------
- evidence_excerpt is capped at MAX_EVIDENCE_EXCERPT_CHARS (500 chars).
  Full contract text is never written to this log.
- Logging statements emit review_id and clause_category only; never
  source_text content, corrected_value payloads, or reviewer notes.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from schemas.feedback import (
    FeedbackRecord,
    FeedbackStatus,
    MAX_EVIDENCE_EXCERPT_CHARS,
)
from schemas.review import ReviewDecision, ReviewItem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Storage path — new path, never touches legacy data/processed/feedback_log.json
# ---------------------------------------------------------------------------

_FEEDBACK_DIR = Path(__file__).parent.parent / "data" / "feedback"
_FEEDBACK_LOG = _FEEDBACK_DIR / "feedback_log.jsonl"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _derive_feedback_id(review_id: str) -> str:
    return f"fb_{review_id}"


def _load_existing(feedback_id: str) -> Optional[FeedbackRecord]:
    """Return the existing FeedbackRecord if feedback_id is already in the log, else None."""
    if not _FEEDBACK_LOG.exists():
        return None
    with _FEEDBACK_LOG.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get("feedback_id") == feedback_id:
                    return FeedbackRecord(**data)
            except Exception:
                continue
    return None


def _derive_document_type(item: ReviewItem) -> Optional[str]:
    """
    Map ReviewItem.document_type_context to a DocumentType-compatible value.
    Returns None when the value cannot be mapped to a known DocumentType.
    """
    raw = (item.document_type_context or "").strip()
    if not raw:
        return None
    for token in raw.replace(",", " ").split():
        t = token.strip()
        if t.upper() == "NDA":
            return "NDA"
        if t.title() in ("Contract", "Amendment", "Other"):
            return t.title()
    return None


def _derive_is_clause_present(item: ReviewItem) -> bool:
    """Determine clause presence from ReviewItem fields, without querying storage."""
    if item.trigger_reason == "missing_critical_clause":
        return False
    if item.evidence_match_type == "not_found":
        return False
    if item.fact_found and "absent" in item.fact_found.lower():
        return False
    return True


def _derive_final_risk(
    item: ReviewItem, decision: ReviewDecision
) -> Optional[str]:
    """Return the final risk level after the human decision."""
    if decision.action == "correct" and decision.corrected_risk_level:
        return decision.corrected_risk_level
    return item.risk_level


def _compute_feedback_status(
    *,
    clause_category: Optional[str],
    final_risk: Optional[str],
    is_clause_present: bool,
    evidence_match_type: Optional[str],
    page_reference_valid: Optional[bool],
    review_action: str,
    clause_language_accepted_as_business_precedent: bool,
) -> FeedbackStatus:
    """
    Classify the record synchronously before it is written.

    not_eligible if ANY of:
    - clause_category is None       (no category = cannot be matched as a precedent)
    - final_risk == "HIGH"          (safety rule: HIGH findings never become precedents)
    - is_clause_present == False    (safety rule: no language to promote)
    - evidence_match_type == "not_found"  (unverified evidence)
    - page_reference_valid == False (page reference could not be confirmed)
    - review_action == "reject"     (finding discarded — no language to promote)
    - reviewer did not flag clause language as a precedent candidate

    pending_precedent_review otherwise — awaiting the separate
    approve_feedback_as_precedent() step.
    """
    if clause_category is None:
        return "not_eligible"
    if final_risk == "HIGH":
        return "not_eligible"
    if not is_clause_present:
        return "not_eligible"
    if evidence_match_type == "not_found":
        return "not_eligible"
    if page_reference_valid is False:
        return "not_eligible"
    if review_action == "reject":
        return "not_eligible"
    if not clause_language_accepted_as_business_precedent:
        return "not_eligible"
    return "pending_precedent_review"


def _safe_excerpt(item: ReviewItem) -> str:
    """
    Extract an evidence excerpt from the ReviewItem.

    Prefers expanded_clause_text (post-boundary-expansion) over source_text.
    Truncates to MAX_EVIDENCE_EXCERPT_CHARS — full contract text is never stored.
    """
    raw = (item.expanded_clause_text or item.source_text or "").strip()
    if not raw:
        return "(no text available)"
    return raw[:MAX_EVIDENCE_EXCERPT_CHARS]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_feedback(item: ReviewItem, decision: ReviewDecision) -> FeedbackRecord:
    """
    Record a resolved HITL decision to data/feedback/feedback_log.jsonl.

    Stage 1 of the two-stage precedent lifecycle.  Classifies the record's
    eligibility status synchronously and appends one JSON line.  Idempotent:
    a second call with the same review_id returns the existing record without
    appending a duplicate line.

    Parameters
    ----------
    item : ReviewItem
        The source review item (should already be resolved).
    decision : ReviewDecision
        The human's recorded decision.

    Returns
    -------
    FeedbackRecord
        The constructed and persisted (or already-existing) record.

    Raises
    ------
    pydantic.ValidationError
        If the constructed record violates any FeedbackRecord schema invariant.
        This indicates a bug in the eligibility logic (e.g. approved_for_precedent
        set True when final_risk=="HIGH"), not a caller error.
    """
    feedback_id = _derive_feedback_id(decision.review_id)

    # Idempotency: return the existing record without writing a duplicate line.
    existing = _load_existing(feedback_id)
    if existing is not None:
        logger.info(
            "save_feedback: idempotent skip — feedback_id=%s already recorded",
            feedback_id,
        )
        return existing

    # --- Derive fields from item + decision ---
    is_clause_present = _derive_is_clause_present(item)
    final_risk = _derive_final_risk(item, decision)
    document_type = _derive_document_type(item)

    # model_finding_accepted: strictly and only from review_action == "approve"
    model_finding_accepted = decision.action == "approve"

    # clause_language_accepted_as_business_precedent: from the reviewer's explicit
    # mark_clause_language_as_precedent_candidate flag — never from action alone.
    # Stored faithfully even if the eligibility rules below classify the record
    # as not_eligible; this preserves the audit trail of the reviewer's intent.
    clause_language_accepted = decision.mark_clause_language_as_precedent_candidate

    # Eligibility classification — synchronous, before write
    feedback_status = _compute_feedback_status(
        clause_category=item.clause_category,
        final_risk=final_risk,
        is_clause_present=is_clause_present,
        evidence_match_type=item.evidence_match_type,
        page_reference_valid=item.page_reference_valid,
        review_action=decision.action,
        clause_language_accepted_as_business_precedent=clause_language_accepted,
    )

    # source_chunk_ids: prefer expansion-aware list, fall back to single source chunk
    source_chunk_ids: list[str] = (
        list(item.source_chunks_used)
        if item.source_chunks_used
        else ([item.source_chunk_id] if item.source_chunk_id else [])
    )

    # Build the record — FeedbackRecord schema validators run on construction
    record = FeedbackRecord(
        feedback_id=feedback_id,
        document_id=item.document_hash,
        # document_name comes from ReviewItem.document_name, which is threaded from
        # state["doc"]["file_name"] in node_build_review_items. Falls back to the
        # document_hash when the ReviewItem was created outside the orchestrator
        # (e.g. in tests or manual calls).
        document_name=item.document_name or item.document_hash,
        document_type=document_type,
        clause_category=item.clause_category,  # None when document-level trigger; always not_eligible
        source_page=item.source_page,
        source_chunk_ids=source_chunk_ids,
        evidence_excerpt=_safe_excerpt(item),
        original_model_category=str(item.clause_category) if item.clause_category else None,
        original_model_risk=item.risk_level,
        original_model_reason=item.risk_rationale or "",
        original_confidence=item.extraction_confidence,
        review_action=decision.action,
        final_category=str(item.clause_category) if item.clause_category else None,
        final_risk=final_risk,
        reviewer_comment=decision.reviewer_note,
        model_finding_accepted=model_finding_accepted,
        clause_language_accepted_as_business_precedent=clause_language_accepted,
        is_clause_present=is_clause_present,
        feedback_status=feedback_status,
        approved_for_precedent=False,  # always False at creation; Stage 4 promotes
    )

    # Persist
    _FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    with _FEEDBACK_LOG.open("a", encoding="utf-8") as fh:
        fh.write(record.model_dump_json() + "\n")

    logger.info(
        "save_feedback: recorded feedback_id=%s clause=%s status=%s",
        feedback_id,
        item.clause_category,
        feedback_status,
    )

    return record
