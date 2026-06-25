"""
schemas/review.py
-----------------
Pydantic models for the Human-in-the-Loop (HITL) review queue.

Two models are defined here:

    ReviewItem     -- one thing sitting in the review queue, waiting for a
                      human decision. Created by agent/orchestrator.py whenever
                      a clause or finding is flagged as requires_human_review=True.

    ReviewDecision -- the human reviewer's actual response: which action they
                      took and (if relevant) what they want to substitute or
                      select instead of the AI's original output.

Important design notes
----------------------
- This is HITL (Human-in-the-Loop), NOT RLHF. Human corrections never update
  model weights. That distinction is intentional and must never be confused.
- reviewer identity (decided_by) uses a placeholder string in this capstone
  scope. Real authentication / RBAC is explicitly out of scope -- this field
  is documented as a known simplification.
- The cross-check "selected_alternative_id must match one of the item's
  alternatives" is enforced in agent/human_review.py at point-of-use rather
  than purely at the schema level, because ReviewDecision does not have
  access to the original ReviewItem's alternatives list at validation time.
  The schema enforces presence/absence of the field; the module enforces value.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator

from schemas.clause import ClauseType


# ---------------------------------------------------------------------------
# Model 1: ReviewItem
# ---------------------------------------------------------------------------

class ReviewItem(BaseModel):
    """
    One item sitting in the human review queue.

    Created automatically by the orchestrator whenever the pipeline flags a
    clause or risk finding with requires_human_review=True. Each ReviewItem
    carries everything a human reviewer needs to make an informed decision,
    without having to dig into raw pipeline internals.

    Fields
    ------
    review_id : str
        Unique identifier for this review item (UUID generated at creation).
        Used by ReviewDecision to link a human answer back to the right item.

    document_hash : str
        SHA-256 hash of the source document. Used to group all review items
        from the same document run and to tie back to the original file.

    source_chunk_id : Optional[str]
        The chunk_id of the DocumentChunk that contained the flagged clause.
        None if the trigger was a document-level flag (e.g. wrong doc type)
        rather than a clause-level one.

    clause_category : Optional[ClauseType]
        Which of the 10 approved clause categories this item relates to.
        Uses the same ClauseType Literal from schemas/clause.py -- never
        redefined here. None if the trigger is not clause-specific.

    source_text : str
        The actual paragraph or clause text the reviewer needs to read.
        This is what the AI used to make its decision; the human sees
        exactly the same raw text. Never paraphrased or summarized here.

    source_page : Optional[int]
        1-based page number where the source_text appears in the document.
        None if the page cannot be determined (e.g. OCR fallback with no
        page reference).

    trigger_reason : str
        Which of the locked HITL trigger conditions caused this escalation.
        Examples: "missing_critical_clause", "tot_unresolved_tie",
        "low_confidence_after_retry", "high_risk_finding",
        "evidence_verification_failure".

    ai_finding_summary : str
        A short structured description of what the AI concluded -- e.g.
        "ABSENT | Indemnification | HIGH risk | confidence=0.43".
        Never a raw reasoning trace (PII risk, verbosity); just enough
        context for the reviewer to understand the AI's position quickly.

    confidence_signals : Dict[str, Any]
        Key confidence metrics that drove the escalation. For example:
        {"final_confidence": 0.43, "retry_count": 1, "tot_triggered": True,
         "evidence_match_score": 0.71}. The exact keys vary by trigger type.

    alternatives : List[Dict[str, Any]]
        Structured alternative interpretations available to the reviewer.
        For ToT-triggered items: the pruned ToTCandidate objects (serialized
        as dicts), so the reviewer can pick one via select_alternative.
        For risk/confidence-triggered items with no alternatives: empty list.
        Each dict should carry at least {"id": str, "summary": str} so that
        ReviewDecision.selected_alternative_id can reference a specific one.

    risk_level : Optional[Literal["LOW", "MEDIUM", "HIGH"]]
        The AI's assessed risk level for this finding. None if not applicable
        (e.g. the trigger is a classification uncertainty, not a risk flag).

    template_comparison_summary : Optional[str]
        Short plain-language summary of how the clause compares to the
        standard template, if template comparison was run. None otherwise.

    status : Literal["pending", "resolved"]
        "pending"  -- waiting for a human decision
        "resolved" -- a ReviewDecision has been recorded for this item

    created_at : datetime
        When this ReviewItem was created (UTC). Used for queue ordering
        and SLA tracking (though SLA enforcement is out of scope for this
        capstone -- this field is stored for future use).

    thread_id : str
        The LangGraph thread/session ID this review item belongs to. This is
        the key that lets the orchestrator resume the correct paused graph
        execution after the human decision is recorded.
    """

    model_config = ConfigDict(
        # serialize datetime as ISO-8601 strings in .model_dump(mode="json")
        ser_json_timedelta="iso8601",
    )

    review_id: str
    document_hash: str
    source_chunk_id: Optional[str] = None
    clause_category: Optional[ClauseType] = None
    source_text: str
    source_page: Optional[int] = Field(default=None, ge=1)
    trigger_reason: str
    ai_finding_summary: str
    confidence_signals: Dict[str, Any] = Field(default_factory=dict)
    alternatives: List[Dict[str, Any]] = Field(default_factory=list)
    risk_level: Optional[Literal["LOW", "MEDIUM", "HIGH"]] = None
    template_comparison_summary: Optional[str] = None
    status: Literal["pending", "resolved"] = "pending"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    thread_id: str


# ---------------------------------------------------------------------------
# Model 2: ReviewDecision
# ---------------------------------------------------------------------------

class ReviewDecision(BaseModel):
    """
    A human reviewer's decision on a ReviewItem.

    Exactly four actions are supported:

        approve             -- the AI's output was correct; accept it as-is.
        correct             -- the AI was wrong; reviewer provides the right value
                               via corrected_value.
        reject              -- the AI's output is unusable and should be discarded
                               entirely (e.g. hallucinated evidence, wrong document
                               type). Downstream code checks for the "discarded"
                               signal returned by apply_human_decision.
        select_alternative  -- for ToT-triggered items, the reviewer picks one of
                               the candidate interpretations by ID rather than
                               writing a free-form correction.

    Fields
    ------
    review_id : str
        Must match an existing ReviewItem's review_id. Validated at point-of-
        use in agent/human_review.record_review_decision().

    action : Literal["approve", "correct", "reject", "select_alternative"]
        The reviewer's chosen action. Exactly these four; no others.

    corrected_value : Optional[Dict | str]
        The human-provided replacement value. Required when action="correct",
        must be None for all other actions. Can be a dict (e.g. a corrected
        ExtractedClause payload) or a string (e.g. a corrected document type).

    selected_alternative_id : Optional[str]
        The ID of the chosen alternative from the original ReviewItem's
        alternatives list. Required when action="select_alternative", must
        be None for all other actions.
        Note: the cross-check that this ID actually exists in the original
        item's alternatives list is enforced in agent/human_review.py at
        point-of-use, not here, because this schema doesn't have access to
        the original ReviewItem's alternatives at validation time.

    reviewer_note : Optional[str]
        Free-text note from the reviewer explaining their decision. Optional
        for all action types. Stored for audit; never used in routing logic.

    decided_at : datetime
        When the decision was recorded (UTC).

    decided_by : Optional[str]
        Reviewer identifier. In this capstone scope this is a plain string
        (e.g. a name or role like "legal_reviewer_1"). Real authentication
        is explicitly out of scope -- this is a documented simplification.
    """

    model_config = ConfigDict(
        ser_json_timedelta="iso8601",
    )

    review_id: str
    action: Literal["approve", "correct", "reject", "select_alternative"]
    corrected_value: Optional[Any] = None
    selected_alternative_id: Optional[str] = None
    reviewer_note: Optional[str] = None
    decided_at: datetime = Field(default_factory=datetime.utcnow)
    decided_by: Optional[str] = None

    @model_validator(mode="after")
    def validate_action_fields(self) -> "ReviewDecision":
        """
        Enforce that corrected_value and selected_alternative_id are
        present exactly when they should be, based on the action chosen.

        Rules:
          action="correct"             -> corrected_value required, selected_alternative_id must be None
          action="select_alternative"  -> selected_alternative_id required, corrected_value must be None
          action="approve" or "reject" -> both must be None
        """
        action = self.action

        if action == "correct":
            if self.corrected_value is None:
                raise ValueError(
                    "corrected_value is required when action='correct'. "
                    "Provide the replacement value the AI should use."
                )
            if self.selected_alternative_id is not None:
                raise ValueError(
                    "selected_alternative_id must be None when action='correct'. "
                    "Use corrected_value to supply the replacement, not selected_alternative_id."
                )

        elif action == "select_alternative":
            if self.selected_alternative_id is None:
                raise ValueError(
                    "selected_alternative_id is required when action='select_alternative'. "
                    "Provide the ID of the alternative you are choosing."
                )
            if self.corrected_value is not None:
                raise ValueError(
                    "corrected_value must be None when action='select_alternative'. "
                    "Use selected_alternative_id to identify the chosen alternative."
                )

        elif action in ("approve", "reject"):
            if self.corrected_value is not None:
                raise ValueError(
                    f"corrected_value must be None when action='{action}'. "
                    f"corrected_value is only meaningful for action='correct'."
                )
            if self.selected_alternative_id is not None:
                raise ValueError(
                    f"selected_alternative_id must be None when action='{action}'. "
                    f"selected_alternative_id is only meaningful for action='select_alternative'."
                )

        return self
