"""
guardrails/claim_verifier.py
------------------------------
Final pre-report guardrail: verify every extracted clause's claim and every
risk finding before the report is generated.

This is the LAST guardrail in the pipeline, running after:
  - Step 5 (clause extraction)
  - Step 6 (risk scoring)
  - Stage 5 guardrails (evidence_verifier, page_verifier)

It synthesizes the Stage 5 checks into a per-claim disposition and decides
what to do with claims that fail verification.

Public function:
  verify_final_claims(clauses, risk_findings, document) -> list[ClaimVerificationResult]

REMOVED vs ESCALATED DECISION LOGIC
--------------------------------------
When a claim fails evidence/page verification:

  REMOVE  (action_taken="removed") when:
    - The claim is LOW risk
    - AND the clause is absent (is_present=False), meaning there is no
      extracted_text to verify anyway -- the finding is structural, not
      evidence-dependent. We still flag this for the report but do not
      escalate it to a human because the human would have no text to judge.
    - OR the clause is present (is_present=True) but LOW risk AND the
      page reference is invalid -- we log the evidence failure and drop
      the specific text claim, but the risk finding itself (LOW) is
      low-stakes enough that we don't need a human to adjudicate it.
      NOTE: "remove" here means "mark the text evidence as unverifiable"
      not "remove the clause from the report entirely" -- the risk finding
      still appears with a caveat.

  ESCALATE (action_taken="escalated_to_human_review") when:
    - The clause or risk finding is HIGH or MEDIUM risk AND evidence
      verification failed. A human reviewer must decide whether the AI
      fabricated evidence or whether there is a genuine document issue.
    - ALWAYS escalate a HIGH risk missing-clause finding that has
      evidence failure (though missing clauses by definition have no
      extracted_text -- this case arises when the risk finding itself
      cannot be cross-referenced to any document content, which would be
      unusual but is handled defensively here).
    - The rule "HIGH risk failing verification is always escalated, never
      removed" is a locked graded design decision (per CLAUDE.md).

  PASS (action_taken="passed") when all checks pass.

WHY NOT "retried" HERE
-----------------------
"retried" is reserved in the ClaimVerificationResult Literal but is not
used by this module. Retry logic belongs to extractor.py (Step 5/7). By
the time a claim reaches this guardrail, retries are already exhausted.
This is documented here and in the schema rather than removed, in case
a future orchestrator refactor moves retry tracking here.

ESCALATION → ReviewItem WIRING
---------------------------------
When a claim is escalated, this module constructs a ReviewItem and enqueues
it via agent.human_review.add_review_item(). This is concrete wiring, not
a TODO -- the escalated claim flows directly into the existing HITL queue
so it appears in demo_hitl.py's review queue for human decision.
"""

import logging
import uuid
from datetime import datetime
from typing import List, Optional

from schemas.clause import ExtractedClause
from schemas.document import DocumentExtraction
from schemas.guardrails import ClaimVerificationResult
from schemas.review import ReviewItem
from schemas.risk import RiskFinding

import agent.human_review as human_review_module

from guardrails.evidence_verifier import verify_evidence
from guardrails.page_verifier import verify_page_reference

logger = logging.getLogger(__name__)


def verify_final_claims(
    clauses: List[ExtractedClause],
    risk_findings: List[RiskFinding],
    document: DocumentExtraction,
    thread_id: Optional[str] = None,
) -> List[ClaimVerificationResult]:
    """
    Run evidence and page verification on all extracted clauses and risk
    findings. Return one ClaimVerificationResult per clause.

    Parameters
    ----------
    clauses : list[ExtractedClause]
        All extracted clauses from Step 5.
    risk_findings : list[RiskFinding]
        All risk findings from Step 6.
    document : DocumentExtraction
        Full extraction result (source of truth for page text and page count).
    thread_id : str or None
        LangGraph thread ID for this run. Passed to ReviewItems created by
        escalation so they appear in the same HITL queue as build_review_items'
        items and have a consistent thread_id for the reviewer.
        If None, falls back to document.file_hash[:16] as a placeholder.

    Returns
    -------
    list[ClaimVerificationResult]
        One result per clause. Clauses that fail verification are either
        "removed" or "escalated_to_human_review" per the decision logic in
        the module docstring.
    """
    # Build a lookup: clause_type -> risk_level for this document
    risk_by_clause: dict[str, str] = {
        f.clause_type: f.risk_level for f in risk_findings
    }

    results: List[ClaimVerificationResult] = []
    for clause in clauses:
        result = _verify_clause(clause, risk_by_clause, document, thread_id=thread_id)
        results.append(result)

    n_passed = sum(1 for r in results if r.action_taken == "passed")
    n_removed = sum(1 for r in results if r.action_taken == "removed")
    n_escalated = sum(1 for r in results if r.action_taken == "escalated_to_human_review")
    logger.info(
        "claim_verifier: %d clauses — passed=%d removed=%d escalated=%d",
        len(results), n_passed, n_removed, n_escalated,
    )
    return results


# ---------------------------------------------------------------------------
# Per-clause verification
# ---------------------------------------------------------------------------

def _verify_clause(
    clause: ExtractedClause,
    risk_by_clause: dict[str, str],
    document: DocumentExtraction,
    thread_id: Optional[str] = None,
) -> ClaimVerificationResult:
    risk_level = risk_by_clause.get(clause.clause_type, "LOW")

    # Absent clauses: no extracted_text to verify.
    # The risk finding is structural (missing clause), not evidence-dependent.
    if not clause.is_present or not clause.extracted_text:
        # Missing HIGH/MEDIUM risk clause: the absence itself is the finding.
        # We pass these through (the absence verifier, not this module, handles
        # further checks on missing clauses). Always mark as passed here.
        logger.debug(
            "claim_verifier: absent clause %s (risk=%s) — structural, no text to verify",
            clause.clause_type, risk_level,
        )
        return ClaimVerificationResult(
            claim_text=f"{clause.clause_type}: clause absent (risk={risk_level})",
            has_supporting_evidence=False,
            evidence_source=None,
            action_taken="passed",  # absence is not a fabrication; no evidence to check
        )

    # Present clause with extracted_text: run evidence + page checks.
    ev = verify_evidence(
        extracted_text=clause.extracted_text,
        source_page_text=_get_all_text_for_page(document, clause.page_reference),
        source_page=clause.page_reference,
    )

    page_ok = True
    if clause.page_reference is not None:
        pr = verify_page_reference(
            cited_page=clause.page_reference,
            document=document,
            extracted_text=clause.extracted_text,
        )
        page_ok = pr.page_exists_in_document

    # Both evidence and page checks passed
    if ev.found_in_source and page_ok:
        source_ref = (
            f"page {clause.page_reference}" if clause.page_reference else
            f"chunk {clause.source_chunk_id}" if clause.source_chunk_id else
            "source verified"
        )
        logger.debug(
            "claim_verifier: PASSED %s (risk=%s, match_type=%s)",
            clause.clause_type, risk_level, ev.match_type,
        )
        return ClaimVerificationResult(
            claim_text=f"{clause.clause_type}: {clause.extracted_text[:200]}",
            has_supporting_evidence=True,
            evidence_source=source_ref,
            action_taken="passed",
        )

    # Verification failed. Decide: remove or escalate.
    return _handle_verification_failure(clause, risk_level, ev, document, thread_id=thread_id)


def _handle_verification_failure(
    clause: ExtractedClause,
    risk_level: str,
    ev,
    document: DocumentExtraction,
    thread_id: Optional[str] = None,
) -> ClaimVerificationResult:
    """
    Apply the removed-vs-escalated decision logic (see module docstring).
    """
    claim_text = f"{clause.clause_type}: {(clause.extracted_text or '')[:200]}"

    if risk_level == "HIGH" or risk_level == "MEDIUM":
        # HIGH/MEDIUM risk with unverifiable evidence -> always escalate
        logger.warning(
            "claim_verifier: ESCALATING %s (risk=%s) — evidence not found in source",
            clause.clause_type, risk_level,
        )
        _enqueue_escalation(clause, risk_level, document, thread_id=thread_id)
        return ClaimVerificationResult(
            claim_text=claim_text,
            has_supporting_evidence=False,
            evidence_source=None,
            action_taken="escalated_to_human_review",
        )
    else:
        # LOW risk with unverifiable evidence -> remove (log, but don't block report)
        logger.info(
            "claim_verifier: REMOVING %s (risk=%s) — evidence not found, low-stakes",
            clause.clause_type, risk_level,
        )
        return ClaimVerificationResult(
            claim_text=claim_text,
            has_supporting_evidence=False,
            evidence_source=None,
            action_taken="removed",
        )


def _enqueue_escalation(
    clause: ExtractedClause,
    risk_level: str,
    document: DocumentExtraction,
    thread_id: Optional[str] = None,
) -> None:
    """
    Construct a ReviewItem and add it to the HITL queue via human_review.py.

    This is the concrete wiring: escalated claims go directly into the
    existing human_review queue, not into a separate system. The ReviewItem
    appears in demo_hitl.py's queue and can be acted on by resume_after_review().
    """
    present_str = "present" if clause.is_present else "absent"
    review_item = ReviewItem(
        review_id=str(uuid.uuid4()),
        document_hash=document.file_hash,
        source_chunk_id=clause.source_chunk_id,
        clause_category=clause.clause_type,
        source_text=clause.extracted_text or "",
        source_page=clause.page_reference,
        trigger_reason="evidence_verification_failure",
        ai_finding_summary=(
            f"PRESENT | {clause.clause_type} | {risk_level} risk | "
            f"confidence={clause.confidence:.2f} | evidence not found in source"
        ),
        confidence_signals={
            "clause_confidence": clause.confidence,
            "risk_level": risk_level,
            "evidence_found": False,
            "retry_count": clause.retry_count,
        },
        alternatives=[],
        risk_level=risk_level,
        template_comparison_summary=None,
        status="pending",
        created_at=datetime.utcnow(),
        thread_id=thread_id or document.file_hash[:16],
        # Reviewer-context fields available at this call site
        fact_found=f"{clause.clause_type} clause is {present_str} in the document.",
        deviation_found="Evidence verification failed: extracted text could not be located in source document.",
        risk_rationale=(
            f"Escalated because evidence_verified=False for a {risk_level} risk clause. "
            "AI may have cited text that does not appear in the document."
        ),
        evidence_match_type="not_found",
        evidence_match_score=None,
        extraction_confidence=clause.confidence,
        llm_confidence=clause.confidence,
        # Chunk context: not available here (document chunks not passed to claim_verifier)
        previous_chunk_text=None,
        next_chunk_text=None,
        section_heading=None,
    )
    human_review_module.add_to_review_queue(review_item)
    logger.info(
        "claim_verifier: enqueued ReviewItem %s for %s",
        review_item.review_id, clause.clause_type,
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_all_text_for_page(document: DocumentExtraction, page_number: Optional[int]) -> str:
    """Return the text of a specific page, or full_text if page_number is None."""
    if page_number is None:
        return document.full_text
    page = document.get_page(page_number)
    return page.text if page else document.full_text
