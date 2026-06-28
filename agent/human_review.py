"""
agent/human_review.py
----------------------
Human-in-the-Loop (HITL) review queue: storage, retrieval, and decision logic.

IMPORTANT: This is HITL, NOT RLHF. Human corrections here are used to
override the model's output for one specific document run. They never update
model weights, never feed back into training data, and must never be
described as reinforcement learning. That distinction is intentional.

Storage mechanism
-----------------
Two JSON files in data/review_queue/:
    pending_reviews.json  -- items waiting for a human decision
    resolved_reviews.json -- completed items (append-only audit trail)

This is an appropriate capstone-scope choice:
    - Simple to inspect and debug (plain JSON)
    - No external dependencies (no Redis, RabbitMQ, PostgreSQL)
    - Persists across process restarts within a session

Known limitations (documented, not hidden):
    - NOT safe for concurrent writes from multiple processes
    - No locking or transactions
    - Not suitable for production multi-user deployment
    These are explicitly out of scope for this capstone.

PII-safe logging policy (consistent with every other module in this project):
    - Log: review_id, document_hash, trigger_reason, action taken
    - Never log: source_text content, corrected_value payloads, reviewer notes
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from schemas.review import ReviewDecision, ReviewItem
from agent.feedback_writer import save_feedback

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------

_QUEUE_DIR = Path(__file__).parent.parent / "data" / "review_queue"
_PENDING_FILE = _QUEUE_DIR / "pending_reviews.json"
_RESOLVED_FILE = _QUEUE_DIR / "resolved_reviews.json"


def _ensure_queue_dir() -> None:
    """Create the storage directory and seed empty JSON files if needed."""
    _QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    if not _PENDING_FILE.exists():
        _PENDING_FILE.write_text("[]", encoding="utf-8")
    if not _RESOLVED_FILE.exists():
        _RESOLVED_FILE.write_text("[]", encoding="utf-8")


def _load_pending() -> List[dict]:
    _ensure_queue_dir()
    text = _PENDING_FILE.read_text(encoding="utf-8").strip()
    return json.loads(text) if text else []


def _save_pending(items: List[dict]) -> None:
    _ensure_queue_dir()
    _PENDING_FILE.write_text(json.dumps(items, indent=2, default=str), encoding="utf-8")


def _load_resolved() -> List[dict]:
    _ensure_queue_dir()
    text = _RESOLVED_FILE.read_text(encoding="utf-8").strip()
    return json.loads(text) if text else []


def _append_resolved(item_dict: dict) -> None:
    resolved = _load_resolved()
    resolved.append(item_dict)
    _RESOLVED_FILE.write_text(json.dumps(resolved, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_to_review_queue(item: ReviewItem) -> None:
    """
    Persist a new ReviewItem to the pending queue.

    If an item with the same review_id already exists in pending, it is
    replaced (idempotent re-add for retry scenarios).

    Parameters
    ----------
    item : ReviewItem
        The item to queue for human review.
    """
    pending = _load_pending()
    # Remove any existing entry with the same review_id (idempotent)
    pending = [p for p in pending if p.get("review_id") != item.review_id]
    pending.append(item.model_dump(mode="json"))
    _save_pending(pending)
    logger.info(
        "HITL queue: added review_id=%s doc_hash=%.16s trigger=%s",
        item.review_id,
        item.document_hash,
        item.trigger_reason,
    )


def get_pending_reviews(document_hash: Optional[str] = None) -> List[ReviewItem]:
    """
    Return all pending review items, optionally filtered to one document.

    Resolved items are not returned -- they live in resolved_reviews.json.

    Parameters
    ----------
    document_hash : str or None
        If provided, return only items whose document_hash matches.
        If None, return all pending items across all documents.

    Returns
    -------
    List[ReviewItem]
        Items with status="pending", ordered by created_at ascending.
    """
    pending_raw = _load_pending()
    items = [ReviewItem(**p) for p in pending_raw]
    if document_hash is not None:
        items = [i for i in items if i.document_hash == document_hash]
    return sorted(items, key=lambda i: i.created_at)


def record_review_decision(decision: ReviewDecision) -> ReviewItem:
    """
    Record a human decision for a pending ReviewItem.

    Steps:
    1. Look up the ReviewItem in pending by decision.review_id.
    2. Validate the decision against the item (cross-check for
       select_alternative: the selected_alternative_id must exist in the
       item's alternatives list).
    3. Mark the item as resolved and move it to resolved_reviews.json.
    4. Return the updated (resolved) ReviewItem.

    Parameters
    ----------
    decision : ReviewDecision
        The human's decision (action + any correction/selection).

    Returns
    -------
    ReviewItem
        The now-resolved item.

    Raises
    ------
    KeyError
        If no pending item with decision.review_id exists.
    ValueError
        If action="select_alternative" and selected_alternative_id doesn't
        match any alternative in the original item.
    """
    pending_raw = _load_pending()
    matching = [p for p in pending_raw if p.get("review_id") == decision.review_id]

    if not matching:
        raise KeyError(
            f"No pending review item found with review_id={decision.review_id!r}. "
            f"The item may have already been resolved or the ID is incorrect."
        )

    item_dict = matching[0]
    item = ReviewItem(**item_dict)

    # Cross-check: for select_alternative, the chosen ID must be in the item's alternatives
    if decision.action == "select_alternative":
        alt_ids = {alt.get("id") for alt in item.alternatives}
        if decision.selected_alternative_id not in alt_ids:
            raise ValueError(
                f"selected_alternative_id={decision.selected_alternative_id!r} "
                f"does not match any alternative in ReviewItem {item.review_id!r}. "
                f"Available IDs: {sorted(alt_ids)}"
            )

    # Mark resolved
    item = item.model_copy(update={"status": "resolved"})

    # Remove from pending, append to resolved (with decision embedded)
    remaining = [p for p in pending_raw if p.get("review_id") != decision.review_id]
    _save_pending(remaining)

    resolved_entry = item.model_dump(mode="json")
    resolved_entry["decision"] = decision.model_dump(mode="json")
    _append_resolved(resolved_entry)

    logger.info(
        "HITL queue: resolved review_id=%s doc_hash=%.16s action=%s",
        decision.review_id,
        item.document_hash,
        decision.action,
    )

    return item


def apply_human_decision(decision: ReviewDecision, original_item: ReviewItem) -> dict:
    """
    Compute the final value to use downstream, given a resolved decision.

    This is the actual override logic -- it determines what replaces the AI's
    original finding when the human's decision differs from the AI's output.

    Parameters
    ----------
    decision : ReviewDecision
        The human's decision (must have been recorded via record_review_decision
        before calling this, though this function does not enforce that).
    original_item : ReviewItem
        The original review item (for the "select_alternative" path, to look
        up the chosen alternative).

    Returns
    -------
    dict
        The value that should be used by downstream pipeline nodes:

        approve            -> {"action": "approve", "value": None, "discarded": False}
                             (None means "use the AI's original output as-is";
                              the caller retrieves the original from its own state)

        correct            -> {"action": "correct", "value": <corrected_value>,
                               "discarded": False}
                             (the caller substitutes this for the AI's original)

        select_alternative -> {"action": "select_alternative",
                               "value": <the chosen alternative dict>,
                               "discarded": False}
                             (the caller uses this alternative instead)

        reject             -> {"action": "reject", "value": None, "discarded": True}
                             (the caller should discard the finding entirely;
                              treat the clause/finding as if it was never extracted)
    """
    # Record the decision to the feedback log (HITL, not RLHF — never touches model weights)
    try:
        save_feedback(original_item, decision)
    except Exception as exc:  # noqa: BLE001 — feedback write must not block downstream pipeline
        logger.warning(
            "save_feedback failed for review_id=%s (%s: %s); continuing without feedback record",
            decision.review_id,
            type(exc).__name__,
            exc,
        )

    action = decision.action

    if action == "approve":
        return {"action": "approve", "value": None, "discarded": False}

    elif action == "correct":
        return {"action": "correct", "value": decision.corrected_value, "discarded": False}

    elif action == "select_alternative":
        # Find the chosen alternative dict from the original item
        chosen = next(
            (alt for alt in original_item.alternatives
             if alt.get("id") == decision.selected_alternative_id),
            None,
        )
        if chosen is None:
            raise ValueError(
                f"selected_alternative_id={decision.selected_alternative_id!r} "
                f"not found in original_item.alternatives"
            )
        return {"action": "select_alternative", "value": chosen, "discarded": False}

    elif action == "reject":
        return {"action": "reject", "value": None, "discarded": True}

    # Should never reach here given the Literal type on ReviewDecision.action
    raise ValueError(f"Unknown action: {action!r}")
