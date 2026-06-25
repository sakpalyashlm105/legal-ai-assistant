"""
schemas/clause.py
-----------------
Pydantic data models for document classification and clause extraction.

Why Pydantic here?
    classifier.py and extractor.py both call GPT-4o-mini and ask it to return
    structured JSON. Pydantic validates that JSON before any downstream code
    touches it -- so if the LLM hallucinates an extra field or sends confidence=1.5,
    we catch it immediately instead of silently propagating bad data.

    Think of this file as the "form schema" that every LLM output must match,
    the same way a GOS attachment type enforces which metadata fields are required.

Two models live here:
    DocumentClassification  -- what kind of document is this? (NDA / contract / etc.)
    ExtractedClause         -- one clause found (or confirmed absent) in the document.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Allowed values (mirrors config.py -- these are the locked, graded choices)
# ---------------------------------------------------------------------------

DocumentType = Literal["NDA", "Contract", "Amendment", "Other"]

ClauseType = Literal[
    "Confidentiality / Non-Disclosure",
    "Termination for Convenience",
    "Termination for Cause",
    "Governing Law / Jurisdiction",
    "Indemnification",
    "Limitation of Liability",
    "Non-Compete / Non-Solicitation",
    "Assignment",
    "Renewal / Term",
    "Dispute Resolution",
]


# ---------------------------------------------------------------------------
# Model 1: DocumentClassification
# ---------------------------------------------------------------------------

class DocumentClassification(BaseModel):
    """
    Output of classifier.py's classify_document() function.

    Fields
    ------
    document_type : DocumentType
        One of "NDA", "Contract", "Amendment", "Other".
        Used by route_by_type to decide which LangGraph branch to follow.

    confidence : float
        How confident GPT-4o-mini is in this classification, on a 0–1 scale.
        Routing rules (from CLAUDE.md, these are locked graded decisions):
          < 0.5  -> self-retry once; if still < 0.5, escalate to human review
          0.5–0.7 -> route to Tree-of-Thought reasoning
          > 0.7  -> proceed automatically

    reasoning : str
        One-sentence explanation of WHY this document type was chosen.
        Stored for audit purposes only -- never logged raw (PII risk).
        Example: "The preamble refers to 'Non-Disclosure Agreement' and
        lists two parties with a confidentiality obligation."

    retry_count : int
        How many times classify_document() has retried this document.
        Starts at 0; capped at 1 before escalating to human review.
    """

    document_type: DocumentType
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    retry_count: int = Field(default=0, ge=0)


# ---------------------------------------------------------------------------
# Model 2: ExtractedClause
# ---------------------------------------------------------------------------

class ExtractedClause(BaseModel):
    """
    One clause entry in the extraction result for a document.

    There will always be exactly 10 of these per document -- one per clause
    category. If a clause is absent, is_present=False and extracted_text=None.

    Fields
    ------
    clause_type : ClauseType
        The category of clause (one of the 10 approved types from CLAUDE.md).

    is_present : bool
        True if the clause was found in the retrieved context.
        False if the LLM determined this clause type is absent from the document.
        IMPORTANT: absent clauses are NEVER fabricated. is_present=False with
        extracted_text=None is the correct representation -- not a guess.

    extracted_text : Optional[str]
        The verbatim text of the clause as it appears in the document.
        None when is_present=False.
        Never paraphrased -- the raw contract language is preserved for
        legal review and evidence grounding.

    page_reference : Optional[int]
        1-based page number where this clause appears, taken from the
        DocumentChunk that contained it. None when is_present=False.

    confidence : float
        LLM's confidence in the extraction, 0–1.
        Same three-tier routing applies:
          < 0.5  -> self-retry, then human review
          0.5–0.7 -> Tree-of-Thought
          > 0.7  -> auto-proceed

    source_chunk_id : Optional[str]
        The chunk_id of the DocumentChunk this clause was extracted from.
        Stored for evidence grounding and page-reference verification
        (verify_page_references node, coming in a later step).

    retry_count : int
        How many times this individual clause's extraction was retried due
        to low confidence (< 0.5). Starts at 0; capped at 1. After one
        retry, the orchestrator escalates the clause to human review rather
        than retrying again.

    requires_human_review : bool
        True if this clause still has confidence < 0.5 after the one allowed
        retry, OR if Tree-of-Thought reasoning (0.5-0.7 band) could not
        resolve the ambiguity. When True, the orchestrator routes this clause
        to the HITL queue rather than accepting the extraction result.

    human_review_reason : Optional[str]
        Plain-language explanation of why human review was flagged. None when
        requires_human_review=False.

    tot_result : Optional[object]
        The ToTResult from Tree-of-Thought reasoning, if ToT was triggered for
        this clause. Stored as Any to avoid a circular import between schemas.
        None when ToT was not triggered (confidence > 0.7 or < 0.5).
    """

    clause_type: ClauseType
    is_present: bool
    extracted_text: Optional[str] = None
    page_reference: Optional[int] = Field(default=None, ge=1)
    confidence: float = Field(ge=0.0, le=1.0)
    source_chunk_id: Optional[str] = None
    retry_count: int = Field(default=0, ge=0)
    requires_human_review: bool = False
    human_review_reason: Optional[str] = None
    tot_result: Optional[object] = None
