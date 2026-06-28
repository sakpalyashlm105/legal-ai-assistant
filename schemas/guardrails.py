"""
schemas/guardrails.py
---------------------
Pydantic data models for all six guardrail checks used in the Legal AI
Assistant pipeline.

These models are the data contracts between the guardrail modules
(guardrails/*.py) and the orchestrator / report generator. Every guardrail
returns one of these models so the orchestrator can make a consistent
pass/warn/block decision without inspecting raw dicts.

Six models:
  GuardrailResult               -- generic result for any guardrail check
  ScopeValidationResult         -- extends GuardrailResult with detected_intent
  InjectionScanResult           -- extends GuardrailResult with technique_categories
  EvidenceVerificationResult    -- did an LLM extraction trace back to real source text?
  PageReferenceVerificationResult -- is the cited page number valid and consistent?
  ClaimVerificationResult       -- final per-claim disposition before report generation
"""

from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Severity vocabulary used across all guardrails
# ---------------------------------------------------------------------------

GuardrailSeverity = Literal["info", "warning", "blocking"]
"""
  info      -- check ran, nothing noteworthy; recorded for audit purposes only
  warning   -- check flagged something suspicious; pipeline can continue but
               the finding should be surfaced in the report and logs
  blocking  -- check failed hard; this document / clause MUST NOT proceed to
               the next stage until the issue is resolved or explicitly overridden
"""


# ---------------------------------------------------------------------------
# 1. Generic guardrail result
# ---------------------------------------------------------------------------

class GuardrailResult(BaseModel):
    """
    Generic pass/fail result returned by any guardrail check.

    Every guardrail module's public function returns this (or a subclass
    of it), so the orchestrator can apply a single uniform policy:
      - severity == "blocking" -> stop, log, surface to user
      - severity == "warning"  -> log, continue, flag in report
      - severity == "info"     -> log, continue silently
    """
    model_config = ConfigDict(frozen=True)

    guardrail_name: str = Field(
        description="Identifier for the specific check that produced this result, "
                    "e.g. 'input_validator.file_exists' or 'prompt_injection.scan'. "
                    "Used in structured logs and audit records."
    )
    passed: bool = Field(
        description="True if the check passed (no action needed beyond logging). "
                    "False if the check flagged a problem; severity indicates how serious."
    )
    reason: str = Field(
        description="Human-readable, PII-safe explanation of why the check passed or "
                    "failed. Must not contain contract text, file contents, or any "
                    "information that could identify a specific person. Use category-level "
                    "descriptions only (e.g. 'file exceeds 50 MB limit' not the file name)."
    )
    severity: GuardrailSeverity = Field(
        description="How serious a failure is. 'info' if passed=True (always). "
                    "'warning' or 'blocking' if passed=False. "
                    "A blocking failure must stop processing of this document."
    )


# ---------------------------------------------------------------------------
# 2. Scope validation result
# ---------------------------------------------------------------------------

class ScopeValidationResult(GuardrailResult):
    """
    Result of the scope guardrail: did the user's request stay within what
    this system is designed to do?

    Extends GuardrailResult with detected_intent, which names what the
    out-of-scope request appeared to be asking for (if determinable). This
    is used in log records and user-facing explanations.
    """
    model_config = ConfigDict(frozen=True)

    detected_intent: Optional[str] = Field(
        default=None,
        description="If passed=False, the category of out-of-scope intent detected, "
                    "e.g. 'binding_legal_approval', 'contract_execution', "
                    "'contract_modification', 'definitive_legal_advice'. "
                    "None if the request passed scope validation."
    )


# ---------------------------------------------------------------------------
# 3. Injection scan result
# ---------------------------------------------------------------------------

class InjectionScanResult(GuardrailResult):
    """
    Result of scanning document text for prompt-injection attempts.

    technique_categories names the *categories* of technique matched (e.g.
    'instruction_override', 'role_manipulation') rather than the literal
    matched text -- this prevents the result object from becoming a record
    of exact attack strings, which could be a minor information-leakage
    risk if serialized to long-term storage.

    flagged_segments contains short excerpts (≤ 60 chars) that triggered the
    scan, retained in the transient result object for the orchestrator's
    immediate decision-making only -- they must NOT be written to persistent
    logs (the orchestrator / caller is responsible for this; the schema
    documents the constraint but cannot enforce it at runtime).
    """
    model_config = ConfigDict(frozen=True)

    technique_categories: list[str] = Field(
        default_factory=list,
        description="Which prompt-injection technique categories were matched, "
                    "e.g. ['instruction_override', 'fake_delimiter']. "
                    "Empty list if passed=True. These are category labels only -- "
                    "not the literal text that triggered the match."
    )
    flagged_segments: list[str] = Field(
        default_factory=list,
        description="Short excerpts (truncated to ≤ 60 chars) that triggered the "
                    "scan. Retained in this transient object for the orchestrator's "
                    "immediate decision only. MUST NOT be written to persistent logs."
    )


# ---------------------------------------------------------------------------
# 4. Evidence verification result
# ---------------------------------------------------------------------------

class EvidenceVerificationResult(BaseModel):
    """
    Did the text an LLM said it extracted actually appear in the source document?

    Used by evidence_verifier.py, which runs after clause extraction (Step 5)
    but before report generation. A passed=False result here means an extracted
    clause's text cannot be traced back to any real page content.
    """
    model_config = ConfigDict(frozen=True)

    extracted_text: str = Field(
        description="The text that was claimed to have been extracted from the document. "
                    "Stored as a PII-safe summary (max 200 chars) in logs; the full "
                    "value is available here only for in-process verification."
    )
    found_in_source: bool = Field(
        description="True if the extracted_text was found (exactly or via fuzzy match "
                    "above threshold) in the source page text."
    )
    match_type: Literal["exact", "fuzzy", "not_found"] = Field(
        description="'exact' -- substring found after normalization; "
                    "'fuzzy' -- difflib similarity above threshold; "
                    "'not_found' -- neither match succeeded."
    )
    match_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Fuzzy match similarity score (0.0-1.0). Populated only when "
                    "match_type='fuzzy'. None for exact or not_found."
    )
    source_page_checked: Optional[int] = Field(
        default=None,
        ge=1,
        description="Page number of the source text that was checked against. "
                    "None if no page-level check was possible."
    )


# ---------------------------------------------------------------------------
# 5. Page reference verification result
# ---------------------------------------------------------------------------

class PageReferenceVerificationResult(BaseModel):
    """
    Is the page number cited by an extracted clause actually valid?

    A clause can cite page 5, but if the document only has 3 pages, that
    reference is hallucinated. This model records the outcome of checking
    both the page's existence and whether the claimed text is findable near
    the cited page (to handle clauses that span a page boundary).
    """
    model_config = ConfigDict(frozen=True)

    cited_page: int = Field(
        description="The page number the LLM claimed the clause appears on. "
                    "May be any integer (including 0 or negative) since it records "
                    "what the LLM reported, not a validated page number. "
                    "page_exists_in_document=False indicates the value is invalid."
    )
    page_exists_in_document: bool = Field(
        description="True if cited_page is within 1..document.total_pages."
    )
    text_found_near_cited_page: bool = Field(
        description="True if the clause text was found on cited_page OR on "
                    "cited_page ± 1 (to handle clauses spanning a page boundary). "
                    "Only meaningful when page_exists_in_document=True and "
                    "extracted_text was provided to the verifier."
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional explanation, e.g. 'text found on page N+1 rather than "
                    "cited page N' or 'cited page exceeds document length of M pages'."
    )


# ---------------------------------------------------------------------------
# 6. Claim verification result
# ---------------------------------------------------------------------------

class ClaimVerificationResult(BaseModel):
    """
    Final per-claim disposition after running all verification checks.

    Used by claim_verifier.py (the last guardrail before report generation).
    action_taken records what the verifier decided to do with the claim:

      passed                   -- all checks passed, claim is credible
      retried                  -- NOTE: retry logic belongs to extractor.py
                                  (Step 5/7); by the time a claim reaches this
                                  guardrail, retries are already exhausted.
                                  This value is reserved but not used by
                                  claim_verifier.py itself.
      removed                  -- claim failed verification AND is low-severity
                                  enough to drop without human review
                                  (only for LOW-risk findings without evidence)
      escalated_to_human_review -- claim failed verification AND is high-severity
                                  enough that a human must decide what to do
                                  (any HIGH-risk finding that cannot be verified
                                  must always take this path, never 'removed')
    """
    model_config = ConfigDict(frozen=True)

    claim_text: str = Field(
        description="PII-safe summary of the claim being verified (e.g. the clause "
                    "type and a brief description of the finding, not the full contract "
                    "text). Max 300 chars in practice; truncate in callers if needed."
    )
    has_supporting_evidence: bool = Field(
        description="True if verify_evidence() found the claim's text in the source "
                    "document above the fuzzy-match threshold."
    )
    evidence_source: Optional[str] = Field(
        default=None,
        description="Pointer to where evidence was found, e.g. a chunk_id "
                    "('chunk-042') or page reference ('page 7'). None if no "
                    "evidence was found."
    )
    action_taken: Literal["passed", "retried", "removed", "escalated_to_human_review"] = Field(
        description="What the claim verifier decided to do with this claim. "
                    "See class docstring for the decision logic."
    )
