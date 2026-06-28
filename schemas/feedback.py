"""
schemas/feedback.py
-------------------
Pydantic v2 models for the Step 12 feedback-and-precedent lifecycle.

Two models:

    PrecedentScope   -- scope metadata constraining which future documents a
                        promoted precedent applies to.

    FeedbackRecord   -- one complete record of a resolved HITL decision, from
                        the moment it is saved through optional promotion to an
                        approved precedent.

Design invariants enforced at schema level (not just at function logic level):

    1. A record where final_risk == "HIGH" or the clause is missing
       (is_clause_present=False) can NEVER have approved_for_precedent=True.
       Attempting to construct such a record raises ValidationError.

    2. feedback_status == "approved_precedent" requires final_risk == "MEDIUM"
       and approved_for_precedent == True, enforced together.

    3. evidence_excerpt must be non-empty and under MAX_EVIDENCE_EXCERPT_CHARS.
       This ceiling (500 chars) prevents the full contract from entering the
       feedback log while still giving enough text for fuzzy-matching in Stage 5.

    4. clause_language_accepted_as_business_precedent is derived strictly from
       ReviewDecision.mark_clause_language_as_precedent_candidate — never from
       review_action == "approve". The schema cannot enforce provenance, but the
       field docstring and the writer (agent/feedback_writer.py) must honour this.

Lifecycle statuses:
    not_eligible               -- rule-based classification determined this
                                  record cannot become a precedent
    pending_precedent_review   -- meets eligibility criteria; awaiting the
                                  separate approve_feedback_as_precedent() step
    approved_precedent         -- human curator explicitly approved via the
                                  curation step; visible to risk_engine.py
    rejected_precedent         -- curator explicitly rejected after review

Note: "recorded" is intentionally absent — no record is ever persisted without
a status classification. save_feedback() classifies synchronously before write.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from schemas.clause import ClauseType, DocumentType

# ---------------------------------------------------------------------------
# Evidence excerpt ceiling
# ---------------------------------------------------------------------------

MAX_EVIDENCE_EXCERPT_CHARS: int = 500
"""
Maximum length of evidence_excerpt stored in a FeedbackRecord.

Why 500?
    Long enough to capture the distinguishing phrase of a typical legal clause
    (which rarely exceeds 2-3 sentences). Short enough to be clearly NOT the
    full contract document. Evidence verifier windowed matching works well on
    fragments of this length.

    Adjust here to change the ceiling globally — the validator references this
    constant, so tests can monkeypatch it if needed.
"""

# ---------------------------------------------------------------------------
# Feedback lifecycle statuses
# ---------------------------------------------------------------------------

FeedbackStatus = Literal[
    "not_eligible",
    "pending_precedent_review",
    "approved_precedent",
    "rejected_precedent",
]

# ---------------------------------------------------------------------------
# Model 1: PrecedentScope
# ---------------------------------------------------------------------------

class PrecedentScope(BaseModel):
    """
    Scope metadata that constrains which future documents an approved precedent
    applies to.

    All fields are Optional because they may not be inferable from the pipeline
    at the time the FeedbackRecord is created. They are intended to be filled in
    by the human curator during approve_feedback_as_precedent().

    Fields
    ------
    document_type : DocumentType or None
        Which document type this precedent applies to (e.g. "NDA"). Sourced
        from ReviewItem.document_type_context at record creation; can be
        refined by the curator. None means "applies across all document types"
        (use with caution).

    clause_category : ClauseType
        The clause category this precedent covers. Always required — a
        precedent without a clause category would be impossible to match.

    jurisdiction : str or None
        Optional jurisdiction constraint (e.g. "Delaware", "California").
        Not auto-populated — must be provided manually. None means the
        precedent is not jurisdiction-specific.

    template_version : str or None
        Optional template version tag (e.g. "v2.1"). Not auto-populated —
        must be provided manually. None means applies to all template versions.
    """

    model_config = ConfigDict(ser_json_timedelta="iso8601")

    document_type: Optional[DocumentType] = None
    clause_category: ClauseType
    jurisdiction: Optional[str] = None
    template_version: Optional[str] = None


# ---------------------------------------------------------------------------
# Model 2: FeedbackRecord
# ---------------------------------------------------------------------------

class FeedbackRecord(BaseModel):
    """
    One complete record of a resolved HITL decision, persisted to
    data/feedback/feedback_log.jsonl (one JSON object per line).

    Fields
    ------
    feedback_id : str
        Unique identifier derived from the source ReviewItem's review_id.
        Format: "fb_<review_id>". Used for idempotency — save_feedback()
        checks for an existing line with this ID before appending.

    document_id : str
        The document_hash of the source document (SHA-256 prefix, truncated
        for log readability). Ties this record back to the original document.

    document_name : str
        Human-readable filename of the source document (e.g.
        "NDA_2_Bakhu_Holdings.pdf"). Never a full path.

    document_type : DocumentType or None
        Document type at the time of review, sourced from
        ReviewItem.document_type_context.

    clause_category : ClauseType
        Which clause category this feedback is about.

    source_page : int or None
        1-based page number where the clause was found, from ReviewItem.

    source_chunk_ids : list[str]
        Chunk IDs that contributed to the clause text (from ReviewItem's
        source_chunk_id + expansion source_chunks_used).

    evidence_excerpt : str
        The key phrase or short snippet from the clause text used for
        future fuzzy-matching. Must be non-empty and <= MAX_EVIDENCE_EXCERPT_CHARS.
        Enforced by schema validator. Never the full contract document.

    original_model_category : str or None
        The clause_category the model originally assigned (may differ from
        clause_category if a correct action changed the category).

    original_model_risk : Literal["LOW","MEDIUM","HIGH"] or None
        The risk level the model originally assigned before any human override.

    original_model_reason : str
        The model's original reason/rationale for its finding. Sourced from
        ReviewItem.risk_rationale.

    original_confidence : float or None
        The model's extraction confidence, from ReviewItem.extraction_confidence.

    review_action : Literal["approve","correct","reject","select_alternative"]
        The action the reviewer took.

    final_category : str or None
        The clause category after the reviewer's decision (may be corrected).

    final_risk : Literal["LOW","MEDIUM","HIGH"] or None
        The risk level after the reviewer's decision. HIGH if approved with
        no correction; whatever the reviewer corrected it to otherwise.

    reviewer_comment : str or None
        The reviewer's free-text note from ReviewDecision.reviewer_note.

    model_finding_accepted : bool
        True if and only if review_action == "approve". Derived field —
        never manually set.

    clause_language_accepted_as_business_precedent : bool
        True if and only if ReviewDecision.mark_clause_language_as_precedent_candidate
        was True at decision time. This is the ONLY permitted source. It must
        NEVER be derived from review_action == "approve" alone, because
        agreeing with a finding does not imply the clause text is reusable.

    feedback_status : FeedbackStatus
        Current stage in the precedent lifecycle. Classified synchronously
        by save_feedback() before the record is persisted — never left at an
        unclassified state.

    approved_for_precedent : bool
        True only after approve_feedback_as_precedent() has run and completed
        all validation checks. Schema-enforced: cannot be True when
        final_risk == "HIGH" or is_clause_present == False.

    is_clause_present : bool
        Whether the clause was found in the document (True) or absent (False).
        Missing-clause findings (False) are always not_eligible for precedent,
        enforced at the schema level.

    precedent_scope : PrecedentScope or None
        Scope metadata, filled in by the curator during promotion. None until
        the record is approved_precedent.

    precedent_approved_by : str or None
        Identifier of the human who ran approve_feedback_as_precedent().
        None until approved.

    precedent_approved_at : datetime or None
        Timestamp of precedent approval. None until approved.

    prompt_version : str or None
        Prompt version tag at the time of the pipeline run, if available.

    model_name : str
        Name/ID of the LLM used (e.g. "gpt-4o-mini").

    template_version : str or None
        Template version tag, if available. Currently not auto-populated;
        None by default.

    created_at : datetime
        UTC timestamp when this record was first created by save_feedback().
    """

    model_config = ConfigDict(ser_json_timedelta="iso8601")

    # Identifiers
    feedback_id: str = Field(
        description="Unique ID derived from the source ReviewItem's review_id. "
                    "Format: 'fb_<review_id>'."
    )
    document_id: str
    document_name: str
    document_type: Optional[DocumentType] = None
    # Optional: None when the trigger was document-level (not clause-specific).
    # A record with no clause_category is always not_eligible for precedent promotion
    # because a precedent without a category cannot be matched. "Other" is intentionally
    # not used here — it is not a valid ClauseType (10 locked approved categories only).
    clause_category: Optional[ClauseType] = None

    # Source location
    source_page: Optional[int] = Field(default=None, ge=1)
    source_chunk_ids: list[str] = Field(default_factory=list)

    # Evidence excerpt
    evidence_excerpt: str = Field(
        description="Key phrase for fuzzy-matching in future precedent lookups. "
                    "Non-empty, <= MAX_EVIDENCE_EXCERPT_CHARS."
    )

    # Original model outputs
    original_model_category: Optional[str] = None
    original_model_risk: Optional[Literal["LOW", "MEDIUM", "HIGH"]] = None
    original_model_reason: str = ""
    original_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # Review outcome
    review_action: Literal["approve", "correct", "reject", "select_alternative"]
    final_category: Optional[str] = None
    final_risk: Optional[Literal["LOW", "MEDIUM", "HIGH"]] = None
    reviewer_comment: Optional[str] = None

    # Derived booleans
    model_finding_accepted: bool
    clause_language_accepted_as_business_precedent: bool

    # Clause presence (required for the precedent-safety invariant)
    is_clause_present: bool

    # Feedback lifecycle
    feedback_status: FeedbackStatus
    approved_for_precedent: bool = False

    # Precedent promotion metadata (None until approved_precedent)
    precedent_scope: Optional[PrecedentScope] = None
    precedent_approved_by: Optional[str] = None
    precedent_approved_at: Optional[datetime] = None

    # Provenance
    prompt_version: Optional[str] = None
    model_name: str = "gpt-4o-mini"
    template_version: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # ---------------------------------------------------------------------------
    # Safety validators
    # ---------------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_high_risk_never_approved_as_precedent(self) -> "FeedbackRecord":
        """
        A finding with final_risk == "HIGH" or is_clause_present == False must
        NEVER have approved_for_precedent = True.

        This is the REG-001 invariant expressed at the schema level: precedent
        matching only makes sense when comparing actual clause language, and
        missing-clause or HIGH-risk findings must never be downgraded via precedent.

        Enforced here — not just at the function level — so it is impossible to
        construct an unsafe record even by passing crafted data directly to the schema.
        """
        if self.approved_for_precedent:
            if self.final_risk == "HIGH":
                raise ValueError(
                    "approved_for_precedent cannot be True when final_risk='HIGH'. "
                    "A HIGH-risk finding is never eligible for precedent promotion — "
                    "approving the reviewer's agreement with a HIGH finding does not "
                    "make the clause language acceptable as a future benchmark. "
                    "(Bakhu Confidentiality is the canonical example of this case.)"
                )
            if not self.is_clause_present:
                raise ValueError(
                    "approved_for_precedent cannot be True when is_clause_present=False. "
                    "Precedent matching requires actual clause language to compare. "
                    "A missing-clause finding has no language to promote."
                )
        return self

    @model_validator(mode="after")
    def validate_approved_precedent_status_requirements(self) -> "FeedbackRecord":
        """
        feedback_status == "approved_precedent" requires:
            - final_risk == "MEDIUM"
            - approved_for_precedent == True

        Both must be true together. This prevents a record from landing at
        "approved_precedent" via any path that bypasses the promotion checks.
        """
        if self.feedback_status == "approved_precedent":
            if self.final_risk != "MEDIUM":
                raise ValueError(
                    "feedback_status='approved_precedent' requires final_risk='MEDIUM'. "
                    f"Got final_risk={self.final_risk!r}. Only MEDIUM-risk findings "
                    "can be promoted to precedent status."
                )
            if not self.approved_for_precedent:
                raise ValueError(
                    "feedback_status='approved_precedent' requires approved_for_precedent=True. "
                    "Use approve_feedback_as_precedent() to promote a record — "
                    "do not set the status directly."
                )
        return self

    @model_validator(mode="after")
    def validate_evidence_excerpt(self) -> "FeedbackRecord":
        """
        evidence_excerpt must be non-empty and must not exceed MAX_EVIDENCE_EXCERPT_CHARS.

        Why a ceiling?
            Prevents the full contract from entering the feedback log.
            500 chars is enough for the distinguishing phrase of any legal clause.

        The validator is phrased to catch both the placeholder case (empty string
        was stored) and the accidental full-document case (> 500 chars).
        """
        if not self.evidence_excerpt.strip():
            raise ValueError(
                "evidence_excerpt must be non-empty. Provide the key phrase from the "
                "clause text that distinguishes it for future fuzzy-matching."
            )
        if len(self.evidence_excerpt) > MAX_EVIDENCE_EXCERPT_CHARS:
            raise ValueError(
                f"evidence_excerpt exceeds the {MAX_EVIDENCE_EXCERPT_CHARS}-character ceiling "
                f"(got {len(self.evidence_excerpt)} chars). "
                "Store only the key distinguishing phrase, not the full clause or contract. "
                "This ceiling exists to prevent full contract text from entering the "
                "feedback log — PII-safe logging policy."
            )
        return self
