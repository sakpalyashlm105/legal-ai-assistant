"""
scripts/restore_illinois_precedent.py
--------------------------------------
Restores the Illinois Governing Law / Jurisdiction approved_precedent record
that was removed during the post-audit cleanup.

Scenario: Bakhu Holdings NDA specifies Illinois as governing jurisdiction.
A prior human reviewer confirmed the clause language is acceptable as a
business precedent, allowing the risk engine to downgrade future similar
clauses from HIGH (major deviation) to MEDIUM when they match this text.

Run from legal-agent/ with the venv active:
    python scripts/restore_illinois_precedent.py
"""

import sys
from pathlib import Path

# Make sure the package root is on the path when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.feedback_writer import save_feedback
from agent.feedback_curation import approve_feedback_as_precedent
from schemas.review import ReviewItem, ReviewDecision
from schemas.feedback import PrecedentScope

# ---------------------------------------------------------------------------
# Realistic Governing Law clause text from the Bakhu NDA
# (Governing Law / Jurisdiction, Illinois, deviation = major from template
#  which specifies Delaware. Risk base = HIGH due to major deviation, but
#  human reviewer confirmed Illinois is acceptable for this counterparty.)
# ---------------------------------------------------------------------------

REVIEW_ID = "rev-illinois-gl-001"
DOCUMENT_HASH = "5c4fbc2cb36b4cd69362e388c61f1ca9327ac6f5d4e7088764ef16d5323dbe6a"
EVIDENCE_TEXT = (
    "This Agreement shall be governed by and construed in accordance with "
    "the laws of the State of Illinois, without regard to its conflict of "
    "law provisions. Each party consents to the exclusive jurisdiction of "
    "the state and federal courts located in Cook County, Illinois."
)

item = ReviewItem(
    review_id=REVIEW_ID,
    document_hash=DOCUMENT_HASH,
    document_name="NDA_2_Bakhu_Holdings,_Corp._(BKUH).pdf",
    document_type_context="NDA",
    source_text=EVIDENCE_TEXT,
    expanded_clause_text=EVIDENCE_TEXT,
    trigger_reason="low_confidence_after_retry",
    ai_finding_summary=(
        "PRESENT | Governing Law / Jurisdiction | MEDIUM | conf=0.82 | "
        "Major deviation: template specifies Delaware; document specifies Illinois. "
        "Human reviewer confirmed Illinois is acceptable for this counterparty."
    ),
    clause_category="Governing Law / Jurisdiction",
    risk_level="MEDIUM",
    risk_rationale=(
        "Major deviation from standard template (Delaware → Illinois). "
        "Risk downgraded to MEDIUM after human review confirmed Illinois "
        "is an acceptable jurisdiction for this counterparty relationship."
    ),
    extraction_confidence=0.82,
    evidence_match_type="exact",
    page_reference_valid=True,
    source_page=3,
    thread_id="thread-bakhu-demo-001",
)

decision = ReviewDecision(
    review_id=REVIEW_ID,
    action="approve",
    reviewer_note=(
        "Illinois governing law is acceptable for Bakhu Holdings. "
        "Cook County courts are a standard venue for this counterparty type. "
        "Approving as business precedent for similar NDA counterparties."
    ),
    mark_clause_language_as_precedent_candidate=True,
    corrected_risk_level="MEDIUM",
)

# Stage 1: save_feedback — creates the record with status=pending_precedent_review
print("Stage 1: calling save_feedback() ...")
record = save_feedback(item, decision)
print(f"  feedback_id : {record.feedback_id}")
print(f"  feedback_status : {record.feedback_status}")
assert record.feedback_status == "pending_precedent_review", (
    f"Expected pending_precedent_review, got {record.feedback_status}"
)

# Stage 2: approve_feedback_as_precedent — promotes to approved_precedent
print("\nStage 2: calling approve_feedback_as_precedent() ...")
scope = PrecedentScope(
    document_type="NDA",
    clause_category="Governing Law / Jurisdiction",
    jurisdiction="Illinois",
)
promoted = approve_feedback_as_precedent(
    feedback_id=record.feedback_id,
    approved_by="legal-ops-curator",
    final_accepted_risk="MEDIUM",
    precedent_scope=scope,
    approval_note=(
        "Illinois governing law confirmed acceptable for Bakhu Holdings-type "
        "counterparties. Cook County venue is standard for this deal profile. "
        "Approved as active precedent: future NDAs with Illinois governing law "
        "clauses matching this language pattern may be downgraded from HIGH to MEDIUM."
    ),
)
print(f"  feedback_id       : {promoted.feedback_id}")
print(f"  feedback_status   : {promoted.feedback_status}")
print(f"  approved_for_precedent : {promoted.approved_for_precedent}")
print(f"  precedent_scope   : {promoted.precedent_scope}")
print(f"  precedent_approved_by  : {promoted.precedent_approved_by}")
print("\nIllinois Governing Law precedent restored successfully.")
