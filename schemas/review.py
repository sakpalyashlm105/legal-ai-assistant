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
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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

    -- REVIEWER CONTEXT FIELDS (added in HITL-deepening pass) --

    template_clause_text : Optional[str]
        The standard template text for this clause category, if a template
        exists. Sourced from Step 6's ClauseComparison.template_path content.
        None when no template is available for this category.

    missing_elements : List[str]
        Named gaps vs the template identified by the comparator
        (e.g. ["mutuality", "survival period", "return/destruction obligation"]).
        Populated from ClauseComparison.deviation_summary parsed list when the
        comparator provides structured gap data; empty list otherwise.
        NOTE: ClauseComparison currently stores deviation_summary as a single
        free-text string, not a structured list. missing_elements is populated
        by parsing that string at ReviewItem construction time when available.

    extra_risky_language : List[str]
        Named risky additions vs template (e.g. ["one-sided obligation",
        "broad company discretion"]). Same source as missing_elements.
        Empty list when not available.

    fact_found : str
        Plain factual statement of what was found (no judgment).
        Example: "The Indemnification clause is absent from the document."

    deviation_found : Optional[str]
        Plain factual statement of how the clause differs from standard.
        None when no deviation exists or clause is absent.

    risk_rationale : str
        The reasoning for WHY this risk level was assigned, distinct from
        fact_found and deviation_found.
        Example: "Marked HIGH because template expected mutual indemnification."

    evidence_match_type : Optional[Literal["exact", "fuzzy", "not_found"]]
        Reused from Step 9's evidence verification result.
        None when evidence verification was not run for this item.

    evidence_match_score : Optional[float]
        Fuzzy match score from Step 9, if available. None otherwise.

    page_reference_valid : Optional[bool]
        Whether the page reference was confirmed valid by Step 9's page
        verifier. None when page verification was not run.

    extraction_confidence : Optional[float]
        The clause extractor's confidence score for this clause.
        NOTE: Only one confidence value exists per clause in this codebase
        (ExtractedClause.confidence). Both extraction_confidence and
        llm_confidence are populated from this same field. They are NOT
        independent signals -- this is documented, not fabricated.

    llm_confidence : Optional[float]
        Same source as extraction_confidence (see note above).

    previous_chunk_text : Optional[str]
        Text of the chunk immediately before the source chunk in the document.
        Fetched via get_local_context() from agent/tot_reasoner.py.
        None when at the first chunk or chunk list is unavailable.

    next_chunk_text : Optional[str]
        Text of the chunk immediately after the source chunk in the document.
        Fetched via get_local_context() from agent/tot_reasoner.py.
        None when at the last chunk or chunk list is unavailable.

    section_heading : Optional[str]
        The section heading for this chunk, if available.
        NOTE: DocumentChunk schema does NOT have a section_heading field
        (confirmed by inspection of schemas/chunk.py). This field is always
        None in the current implementation -- documented here so it is not
        silently assumed to be populated.

    document_type_context : Optional[str]
        The document type classification (e.g. "NDA", "SERVICE_AGREEMENT")
        so the reviewer sees what kind of document they are judging this
        clause in the context of.

    -- CLAUSE EXPANSION FIELDS (added in clause-boundary-expansion pass) --

    expanded_clause_text : Optional[str]
        The merged text of multiple adjacent chunks that the expander
        determined belong to the same logical clause group.  None when
        expansion was not triggered or the source chunk was not available.
        When populated this is the text that was actually sent to the
        comparator for template matching -- NOT the original snippet.

    source_chunks_used : List[str]
        Ordered list of chunk_ids that were merged to form expanded_clause_text.
        Length > 1 when expansion_triggered=True; length 1 when only the
        source chunk was used; empty when the clause is absent.

    expansion_triggered : bool
        True if clause boundary expansion found and merged at least one
        additional chunk beyond the originally extracted source chunk.
        False does NOT mean the clause text is wrong -- it means the
        extractor's snippet was already the complete clause (or the clause
        is absent).

    expansion_boundary_reason : Optional[str]
        Why expansion stopped.  Mirrors ClauseExpansionResult.boundary_reason.
        Values: "absent_or_no_source", "source_not_found",
                "new_unrelated_heading", "unrelated_content",
                "max_chunks_reached", "end_of_document".
        None when expansion was not attempted (e.g., absent clause).
    """

    model_config = ConfigDict(
        # serialize datetime as ISO-8601 strings in .model_dump(mode="json")
        ser_json_timedelta="iso8601",
    )

    review_id: str
    document_hash: str
    document_name: Optional[str] = None
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

    # Reviewer context fields
    template_clause_text: Optional[str] = None
    missing_elements: List[str] = Field(default_factory=list)
    extra_risky_language: List[str] = Field(default_factory=list)
    fact_found: str = ""
    deviation_found: Optional[str] = None
    risk_rationale: str = ""
    evidence_match_type: Optional[Literal["exact", "fuzzy", "not_found"]] = None
    evidence_match_score: Optional[float] = None
    page_reference_valid: Optional[bool] = None
    extraction_confidence: Optional[float] = None
    llm_confidence: Optional[float] = None
    previous_chunk_text: Optional[str] = None
    next_chunk_text: Optional[str] = None
    section_heading: Optional[str] = None  # Always None: DocumentChunk has no section_heading field
    document_type_context: Optional[str] = None

    # Clause expansion fields
    expanded_clause_text: Optional[str] = None
    source_chunks_used: List[str] = Field(default_factory=list)
    expansion_triggered: bool = False
    expansion_boundary_reason: Optional[str] = None


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
        Also exposed as reviewer_identifier (same field, no duplication).

    -- STRUCTURED DECISION FIELDS (added in HITL-deepening pass) --

    reason : str
        The reviewer's stated reason for their decision. REQUIRED (non-empty)
        when the original ReviewItem's risk_level is "HIGH". Optional for
        MEDIUM/LOW risk items. Enforced by validate_high_risk_requires_reason.
        NOTE: The validator checks self.risk_level_on_item (passed at
        construction time) rather than fetching the item from storage, so
        callers must supply risk_level_on_item for HIGH-risk decisions.

    risk_level_on_item : Optional[Literal["LOW", "MEDIUM", "HIGH"]]
        The risk_level of the original ReviewItem. Used ONLY by the
        validate_high_risk_requires_reason validator -- not persisted as
        a decision output. Callers should populate this from the ReviewItem
        being decided on so that the reason-required rule can be enforced.

    reject_category : Optional[Literal[...]]
        Required when action="reject". Provides a structured classification
        of why the finding is being discarded, enabling systematic audit of
        what kinds of AI errors humans are catching.
        Options:
          "duplicate_finding"         -- same finding already captured elsewhere
          "wrong_clause_category"     -- AI categorised it under the wrong type
          "hallucinated_deviation"    -- AI claimed a deviation that doesn't exist
          "evidence_mismatch"         -- AI's cited evidence doesn't match the text
          "low_materiality"           -- finding is real but legally immaterial
          "template_mismatch"         -- template is wrong for this doc type
          "not_legally_relevant"      -- finding is outside the scope of review

    corrected_risk_level : Optional[Literal["LOW","MEDIUM","HIGH"]]
        The corrected risk level when action="correct". Supplements (does not
        replace) corrected_value -- keeps corrected_value for backward compat.

    corrected_summary : Optional[str]
        The corrected finding summary when action="correct".

    corrected_rationale : Optional[str]
        The corrected rationale when action="correct".

    flag_for_regression_dataset : bool
        If True, this decision is flagged as a candidate for manual curation
        into the project's regression test dataset
        (data/evaluation/regression_cases.json).

        THIS NEVER FEEDS INTO MODEL TRAINING OR FINE-TUNING. It only marks
        this decision as something a human should consider adding to the
        regression test set, consistent with this project's HITL-not-RLHF
        philosophy. A human must still manually curate entries into that file;
        this flag does NOT automatically write anything anywhere.

    mark_clause_language_as_precedent_candidate : bool
        If True, the reviewer explicitly signals that this clause's actual
        language is acceptable as a future benchmark — independent of whether
        they agreed with the AI's finding.

        This is the ONLY permitted source of
        FeedbackRecord.clause_language_accepted_as_business_precedent.
        NEVER derived from review_action == "approve" alone.
        Default False. Setting it makes the record eligible for the separate
        approve_feedback_as_precedent() curation step but does NOT promote it
        automatically.
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

    # Structured decision fields
    reason: str = ""
    risk_level_on_item: Optional[Literal["LOW", "MEDIUM", "HIGH"]] = None
    reject_category: Optional[Literal[
        "duplicate_finding",
        "wrong_clause_category",
        "hallucinated_deviation",
        "evidence_mismatch",
        "low_materiality",
        "template_mismatch",
        "not_legally_relevant",
    ]] = None
    corrected_risk_level: Optional[Literal["LOW", "MEDIUM", "HIGH"]] = None
    corrected_summary: Optional[str] = None
    corrected_rationale: Optional[str] = None
    flag_for_regression_dataset: bool = False
    mark_clause_language_as_precedent_candidate: bool = False

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

    @model_validator(mode="after")
    def validate_high_risk_requires_reason(self) -> "ReviewDecision":
        """
        HIGH-risk items require a non-empty reason for any decision.
        MEDIUM and LOW risk items: reason is optional.
        """
        if self.risk_level_on_item == "HIGH" and not self.reason.strip():
            raise ValueError(
                "reason is required (non-empty) when the original finding has risk_level='HIGH'. "
                "Provide a brief explanation for your decision so the audit trail is complete."
            )
        return self

    @model_validator(mode="after")
    def validate_reject_requires_category(self) -> "ReviewDecision":
        """
        action="reject" requires a reject_category to be specified.
        """
        if self.action == "reject" and self.reject_category is None:
            raise ValueError(
                "reject_category is required when action='reject'. "
                "Choose from: duplicate_finding, wrong_clause_category, "
                "hallucinated_deviation, evidence_mismatch, low_materiality, "
                "template_mismatch, not_legally_relevant."
            )
        return self
