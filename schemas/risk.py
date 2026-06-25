"""
schemas/risk.py
---------------
Pydantic data models for template comparison and risk assessment.

Two models live here:

    ClauseComparison  -- the output of comparator.py's compare_to_templates().
                         Answers: "how different is this clause from our template?"

    RiskFinding       -- the output of risk_engine.py's flag_risks().
                         Answers: "how serious is that difference, and what should
                         the reviewer focus on?"

Why two separate models instead of one?
    Comparison and risk are different concerns. Comparison is a factual
    observation ("the extracted text deviates from the template in X way").
    Risk is a judgment ("that deviation is HIGH risk because Y"). Keeping them
    separate means the comparator can be tested and improved independently of
    the risk logic, and the risk engine can be swapped or tuned without
    touching the comparison layer.

    This mirrors a separation you'd see in SAP: a BRF+ rule produces a
    decision output (comparison result), and a workflow step converts that
    output into an approval priority (risk level).
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field

from schemas.clause import ClauseType, ExtractedClause


# ---------------------------------------------------------------------------
# Allowed values
# ---------------------------------------------------------------------------

RiskLevel = Literal["HIGH", "MEDIUM", "LOW"]
DeviationSeverity = Literal["none", "minor", "major"]


# ---------------------------------------------------------------------------
# Model 1: ClauseComparison
# ---------------------------------------------------------------------------

class ClauseComparison(BaseModel):
    """
    Result of comparing one extracted clause against its standard template.

    Produced by agent/comparator.py and consumed by agent/risk_engine.py.

    Fields
    ------
    clause_type : ClauseType
        The category of clause being compared.

    template_found : bool
        True if a template file exists for this clause type in data/templates/.
        False means no comparison was possible -- risk_engine treats this as LOW
        risk by default (we have no baseline to measure against).

    matches_template : bool
        True if the extracted text is semantically equivalent to the template.
        Only meaningful when template_found=True.

    deviation_severity : DeviationSeverity
        "none"  -- text matches or is immaterially different
        "minor" -- one or two non-standard terms; scope or liability shift is small
        "major" -- significant rewrite; key protections added, removed, or inverted

    deviation_summary : Optional[str]
        One-sentence description of what specifically differs from the template.
        None when deviation_severity="none" or template_found=False.
        Never contains the raw clause text (PII/confidentiality risk in logs).

    template_path : Optional[str]
        Filesystem path to the template file that was used for comparison.
        Stored for audit purposes.
    """

    clause_type: ClauseType
    template_found: bool
    matches_template: bool = False
    deviation_severity: DeviationSeverity = "none"
    deviation_summary: Optional[str] = None
    template_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Model 2: RiskFinding
# ---------------------------------------------------------------------------

class RiskFinding(BaseModel):
    """
    Final risk assessment for one clause category in a document.

    Produced by agent/risk_engine.py. One RiskFinding per clause category,
    always exactly 10 per document.

    Fields
    ------
    clause_type : ClauseType
        The category of clause this finding covers.

    risk_level : RiskLevel
        The final assessed risk level after applying all rules and overrides:
          HIGH   -- missing clause, OR major deviation with no precedent override
          MEDIUM -- minor deviation, OR major deviation downgraded by precedent
          LOW    -- matches template, or no template available to compare against

        Locking rule from CLAUDE.md (graded decision, do not change):
          Missing clauses are ALWAYS HIGH, regardless of precedent.

    reason : str
        One sentence explaining why this risk level was assigned.
        Example: "Indemnification clause is absent from the document."
        Example: "Governing Law clause deviates from template: jurisdiction
                  changed from New York to California."

    is_missing : bool
        True when the clause was not found in the document at all.
        Drives the "Missing Clauses" section in the report and the HITL
        review queue priority.

    deviation_summary : Optional[str]
        Carried forward from ClauseComparison. None if clause is absent
        or matches the template.

    precedent_applied : bool
        True if a precedent from the feedback log was found and the risk
        level was downgraded one tier as a result.
        Only possible for non-missing clauses.

    precedent_note : Optional[str]
        Short description of which prior approval was matched, e.g.:
        "Previously approved in vendor agreement 2023-11-15 (hash: ab12...)."
        None when precedent_applied=False.

    source_clause : Optional[ExtractedClause]
        The original ExtractedClause this finding is based on.
        None when is_missing=True (there is no source clause to reference).
    """

    clause_type: ClauseType
    risk_level: RiskLevel
    reason: str
    is_missing: bool = False
    deviation_summary: Optional[str] = None
    precedent_applied: bool = False
    precedent_note: Optional[str] = None
    source_clause: Optional[ExtractedClause] = None
