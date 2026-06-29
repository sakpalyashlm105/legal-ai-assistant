"""
schemas/report.py
-----------------
Pydantic data models for the final LegalDocumentReport produced by Step 10.

Four models live here:

    ClauseReportEntry       -- one row in the clause-analysis table.
    RiskFindingEntry        -- one row in the risk-findings section.
    GuardrailSummaryEntry   -- one row summarising a single guardrail run.
    LegalDocumentReport     -- the assembled report containing all of the above.

Design principles enforced by schema:
    - Confidence language is always HIGH / MODERATE / LOW, never a raw float
      in any field that appears in a human-facing report. Raw floats stay
      inside the pipeline state; they are translated before reaching this schema.
    - No verdict language. deviation_summary, finding_summary, and
      recommendation fields carry observation-pattern text only (see field
      descriptions). Violation of this convention is a documentation / code-
      review concern, not a runtime error -- the schema cannot enforce prose
      style, but the field descriptions encode the contract so callers know
      what is expected.
    - All fields that might contain PII-adjacent data (source text, file paths)
      are kept optional and may be omitted when producing summary-level reports.

Relationships to existing schemas:
    - ClauseType comes from schemas/clause.py (locked 10-category Literal).
    - RiskLevel comes from schemas/risk.py (HIGH / MEDIUM / LOW).
    - This file does NOT import from agent/ -- the schema layer has no
      dependency on the orchestration layer (same pattern as all other schemas).
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from schemas.clause import ClauseType
from schemas.risk import RiskLevel


# ---------------------------------------------------------------------------
# Step 11 result schema
# ---------------------------------------------------------------------------

class FinalOutputValidationResult(BaseModel):
    """
    Result of the validate_final_output guardrail (Step 11).

    Records which checks ran, which failed, what was auto-fixed, and
    what was escalated for human review. Always produced -- even a fully
    clean report produces one of these with passed=True and empty failure lists.

    Fields
    ------
    passed : bool
        True only if all checks passed (or failures were safely auto-fixed).
        False if any escalation was created -- meaning the final output has a
        structural inconsistency a human must adjudicate.

    checks_run : list[str]
        Names of every check that executed (in order).

    checks_failed : list[str]
        Names of checks that detected a problem, whether auto-fixed or escalated.

    auto_fixes_applied : list[str]
        Short descriptions of changes made automatically to the report.
        Currently only "disclaimer_reinjected" is possible.

    escalations_created : list[str]
        review_id values of any ReviewItems enqueued in the HITL queue.
        Non-empty when a structural inconsistency requires human adjudication.

    notes : Optional[str]
        Free-text details about what was found, for debugging or audit.
    """
    model_config = ConfigDict(frozen=True)

    passed: bool
    checks_run: List[str]
    checks_failed: List[str] = Field(default_factory=list)
    auto_fixes_applied: List[str] = Field(default_factory=list)
    escalations_created: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Confidence label vocabulary
# ---------------------------------------------------------------------------

ConfidenceLabel = Literal["HIGH", "MODERATE", "LOW", "N/A"]
"""
Maps from internal 0-1 float confidence scores to human-readable labels.
Thresholds used by report_generator.confidence_to_label():
    >= 0.70  -> HIGH      (consistent with CLAUDE.md's auto-proceed threshold)
    0.50-0.69 -> MODERATE (consistent with the ToT routing band)
    < 0.50   -> LOW       (below the auto-proceed floor; requires closer review)
    N/A      -> absent clause; confidence is not applicable (no text was extracted)

These boundaries are not arbitrary -- they mirror the same thresholds used
for routing decisions in classifier.py and extractor.py. Reusing them here
keeps the confidence language consistent across the whole system.
"""

HumanReviewStatus = Literal["not_required", "pending", "resolved"]


# ---------------------------------------------------------------------------
# Model 1: ClauseReportEntry
# ---------------------------------------------------------------------------

class ClauseReportEntry(BaseModel):
    """
    One row in the clause-analysis table of the final report.

    Represents a single clause category's presence, evidence status, and
    risk level in a format readable by a human reviewer.

    Fields
    ------
    clause_category : ClauseType
        The clause category (one of the 10 approved types from CLAUDE.md).

    is_present : bool
        True if the clause was found in the document.

    confidence_label : ConfidenceLabel
        The extraction confidence translated to a human-readable label.
        HIGH / MODERATE / LOW -- never a raw float in this field.

    confidence_explanation : str
        One sentence explaining WHY this confidence level applies.
        Example: "High confidence: clause text was verified against the source
        document and matched a known template pattern."
        Must not claim calibrated statistical probabilities.

    source_page : Optional[int]
        Page number where the clause was found, if available.

    evidence_verified : Optional[bool]
        True if the evidence verifier confirmed the extracted text appears
        in the source document. None if the clause is absent (no text to verify)
        or if the evidence verifier did not run for this clause.

    risk_level : Optional[RiskLevel]
        The assessed risk level from the risk engine.
        None only if risk scoring did not produce a finding for this category.

    deviation_summary : Optional[str]
        Observation-pattern summary of how this clause differs from the
        standard template, if a template comparison was run.
        Must follow the observation pattern: "The system identified [X] on
        page [N]." NOT "This clause is invalid" or "this is enforceable."
        None if no deviation was found or no template was available.

    recommendation : Optional[str]
        Action-oriented guidance for a human reviewer.
        Example: "Qualified legal review is recommended for this clause as it
        deviates materially from the standard template."
        Must not constitute legal advice -- always recommends human review,
        never advises a specific legal position.
    """
    model_config = ConfigDict(frozen=True)

    clause_category: ClauseType
    is_present: bool
    confidence_label: ConfidenceLabel
    confidence_explanation: str
    source_page: Optional[int] = Field(default=None, ge=1)
    evidence_verified: Optional[bool] = None
    risk_level: Optional[RiskLevel] = None
    deviation_summary: Optional[str] = None
    recommendation: Optional[str] = None


# ---------------------------------------------------------------------------
# Model 2: RiskFindingEntry
# ---------------------------------------------------------------------------

class RiskFindingEntry(BaseModel):
    """
    One row in the risk-findings section of the final report.

    Represents a single risk finding in human-readable form.

    Fields
    ------
    clause_category : ClauseType
        The clause category this finding covers.

    risk_level : RiskLevel
        The assessed risk level: HIGH, MEDIUM, or LOW.

    finding_summary : str
        Observation-pattern description of the finding.
        Must use the form: "The system identified [X] on page [N].
        Qualified legal review is recommended."
        Must NOT state a definitive legal verdict (e.g. never "this clause
        makes the contract unenforceable").

    source_page : Optional[int]
        Page where the relevant clause was found. None if the clause is absent.

    precedent_applied : bool
        True if the risk level was downgraded based on a previously approved
        similar clause in the feedback log.

    human_review_status : HumanReviewStatus
        Whether this finding required human review, and if so, whether it
        has been resolved.
    """
    model_config = ConfigDict(frozen=True)

    clause_category: ClauseType
    risk_level: RiskLevel
    finding_summary: str
    source_page: Optional[int] = Field(default=None, ge=1)
    precedent_applied: bool = False
    human_review_status: HumanReviewStatus = "not_required"


# ---------------------------------------------------------------------------
# Model 3: GuardrailSummaryEntry
# ---------------------------------------------------------------------------

class GuardrailSummaryEntry(BaseModel):
    """
    One row in the guardrail summary section of the final report.

    Summarises a single guardrail check in PII-safe, human-readable form.

    Fields
    ------
    guardrail_name : str
        Identifier for the check, e.g. 'input_validator.all_checks'.

    passed : bool
        True if the guardrail passed with no action required.

    severity : str
        One of 'info', 'warning', 'blocking' -- the severity level recorded
        by the guardrail at runtime.

    reason : str
        PII-safe summary of the check outcome.
        Must NOT include contract text, injection pattern text, or any
        content that identifies a specific individual.
        Example: "File size and format are within accepted limits."
        NOT: "File contains the word 'approve' on page 3."
    """
    model_config = ConfigDict(frozen=True)

    guardrail_name: str
    passed: bool
    severity: str
    reason: str


# ---------------------------------------------------------------------------
# Model 4: LegalDocumentReport
# ---------------------------------------------------------------------------

class LegalDocumentReport(BaseModel):
    """
    The full assembled report for one document analysis run.

    This is the authoritative output of the reporting pipeline (Step 10).
    It collects all findings from Steps 3-9 into a single structured object
    that can be serialised to Markdown or JSON for delivery to the user.

    Two non-negotiable writing quality rules apply to every text field:
        1. Confidence language must never overclaim: use HIGH / MODERATE / LOW
           with a brief explanation, never a calibrated percentage probability.
        2. Never state a definitive legal verdict. Use the observation +
           recommendation pattern throughout.

    Fields
    ------
    document_name : str
        Original filename (no directory path) of the analysed document.

    document_hash : str
        SHA-256 hash of the document bytes. Used for de-duplication and audit.

    document_type : str
        Classified document type: NDA, Contract, Amendment, or Other.

    classification_confidence_label : ConfidenceLabel
        The classifier's confidence translated to HIGH / MODERATE / LOW.

    total_pages : int
        Number of pages in the document, as reported by the PDF parser.

    total_clauses_found : int
        Number of the 10 standard clause categories that were found (is_present=True).

    total_clauses_missing : int
        Number of the 10 standard clause categories that were absent.

    executive_summary : str
        Prose overview of the document's key findings. Generated by a single,
        tightly-constrained LLM call (Stage 3). If the LLM call fails or
        produces a rule-violating response, a deterministic fallback template
        is used instead. The executive_summary must respect both writing-quality
        rules -- the post-generation checker enforces this before this field
        is populated.

    clause_entries : List[ClauseReportEntry]
        One entry per clause category, always 10 entries per document.

    risk_findings : List[RiskFindingEntry]
        Risk findings filtered and sorted to show HIGH and MEDIUM prominently.
        LOW findings are included but displayed last.

    missing_clauses : List[str]
        Plain list of clause category names that were absent from the document.
        Derived from clause_entries where is_present=False.

    guardrail_summary : List[GuardrailSummaryEntry]
        Summary of every guardrail that ran during this analysis.

    human_review_decisions : List[Dict[str, Any]]
        Summarised records of any human review decisions made during this run.
        Each dict carries at least: review_id, action, clause_category,
        decided_at, reviewer_note. Full corrected values are NOT stored here
        (they remain in graph state) -- only the decision metadata.

    limitations_note : str
        Semi-templated text documenting what this analysis does NOT cover.
        Example: "Template comparison was not available for 3 of 10 clause
        categories. Risk assessments for those categories reflect only clause
        presence/absence, not deviation from a standard baseline."

    disclaimer : str
        Fixed AI-assisted-analysis disclaimer. Never omitted.

    generated_at : datetime
        UTC timestamp of when this report was assembled.

    processing_notes : Optional[str]
        Optional notes about unusual pipeline conditions (e.g. OCR was used
        on N pages, ToT reasoning was triggered for M clauses).
    """
    model_config = ConfigDict(
        ser_json_timedelta="iso8601",
    )

    document_name: str
    document_hash: str
    document_type: str
    classification_confidence_label: ConfidenceLabel
    total_pages: int = Field(ge=0)
    total_clauses_found: int = Field(ge=0)
    total_clauses_missing: int = Field(ge=0)
    executive_summary: str
    clause_entries: List[ClauseReportEntry]
    risk_findings: List[RiskFindingEntry]
    missing_clauses: List[str]
    guardrail_summary: List[GuardrailSummaryEntry]
    human_review_decisions: List[Dict[str, Any]] = Field(default_factory=list)
    limitations_note: str
    disclaimer: str = Field(min_length=1)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    processing_notes: Optional[str] = None
