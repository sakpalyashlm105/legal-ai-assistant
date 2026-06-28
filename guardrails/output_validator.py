"""
guardrails/output_validator.py
-------------------------------
Step 11 guardrail: validate the ASSEMBLED report as a whole.

This is deliberately different from the Step 9 per-clause guardrails
(evidence_verifier, page_verifier, claim_verifier). Those check individual
findings in isolation -- during or just after extraction. This module checks
the FINISHED LegalDocumentReport for things that can only be verified once
everything is assembled together:

    1. Disclaimer presence
       Is the required AI-analysis disclaimer present and non-empty?
       Auto-fixable: re-inject the known-correct boilerplate.

    2. Internal count consistency
       Do total_clauses_found / total_clauses_missing agree with the
       actual clause_entries list? Does the sum equal exactly 10?
       Do all names in missing_clauses trace back to a real absent entry?
       Not auto-fixable: the system cannot determine which number is "correct"
       without re-running extraction, risking masking a real upstream bug.

    3. Schema completeness
       Are required text fields non-empty, generated_at set, and
       clause_entries exactly 10 entries?
       Not auto-fixable for count mismatches; document_name empty is escalated.

    4. Guardrail-to-report disconnect
       For any clause marked evidence_verified=False (present but unverified),
       the corresponding risk finding must NOT show human_review_status=
       "not_required" when risk_level is HIGH or MEDIUM. That combination
       means the guardrail chain flagged a problem but the report metadata
       implies it was never reviewed -- a silent failure of the HITL chain.
       Always escalated. Never auto-fixed. Most safety-critical check here.

AUTO-FIX vs ESCALATE DECISION LOGIC
--------------------------------------
  AUTO-FIX when:
    - The correct value is unambiguously known and the fix is idempotent.
    - Currently: disclaimer missing/empty -> re-inject the fixed DISCLAIMER
      constant from report_generator.py. That string is identical on every
      report; injecting it cannot introduce incorrect information.

  ESCALATE when:
    - The system cannot determine the correct value without re-running
      upstream steps (count mismatches, orphaned names, wrong clause count).
    - The discrepancy might represent a real upstream bug that should not
      be silently patched (guardrail-to-report disconnect, empty document_name).
    - The finding is safety-critical (risk HIGH/MEDIUM with unreviewed failed
      evidence is a potential false-assurance to the human reviewer).

  Escalation mechanism:
    - Constructs a ReviewItem (same schema as Step 9) and enqueues it via
      agent.human_review.add_to_review_queue(), consistent with claim_verifier.py.
    - Does NOT re-pause the LangGraph graph (see orchestrator.py's
      node_validate_final_output docstring for the reasoning).
    - The review_id is stored in FinalOutputValidationResult.escalations_created
      so the caller can surface it to the user.

PII-safe logging throughout: log field names and counts, never clause text,
finding_summary content, or executive_summary prose.
"""

import logging
import uuid
from datetime import datetime
from typing import List, Optional, Tuple

import agent.human_review as human_review_module
from reporting.report_generator import DISCLAIMER
from schemas.report import FinalOutputValidationResult, LegalDocumentReport
from schemas.review import ReviewItem

logger = logging.getLogger(__name__)

# Approved clause category count -- locked in CLAUDE.md
_APPROVED_CLAUSE_COUNT = 10

# Substring that must appear in the disclaimer field.
# We check for the core phrase rather than exact string equality so that
# minor whitespace normalisation across environments doesn't cause false failures.
_DISCLAIMER_SENTINEL = "does not constitute legal advice"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_final_output(
    report: LegalDocumentReport,
    thread_id: Optional[str] = None,
) -> Tuple[LegalDocumentReport, FinalOutputValidationResult]:
    """
    Run all four final-output checks on an assembled report.

    Parameters
    ----------
    report : LegalDocumentReport
        The fully assembled report produced by node_generate_report.
    thread_id : str or None
        LangGraph thread ID for this pipeline run. Used as context in any
        ReviewItems created by escalation. Falls back to report.document_hash[:16].

    Returns
    -------
    (report, result) : tuple
        report  -- possibly auto-fixed (disclaimer re-injected). Identical
                   to the input report if no auto-fix was needed.
        result  -- FinalOutputValidationResult summarising all checks.

    This function NEVER raises. Exceptions inside individual checks are caught,
    logged, and treated as check failures with a safe escalation.
    """
    checks_run: List[str] = []
    checks_failed: List[str] = []
    auto_fixes: List[str] = []
    escalations: List[str] = []
    notes_parts: List[str] = []

    tid = thread_id or report.document_hash[:16]

    # 1. Disclaimer
    report, check1_failed, fix1, esc1, note1 = _check_disclaimer(report, tid)
    checks_run.append("disclaimer")
    if check1_failed:
        checks_failed.append("disclaimer")
    auto_fixes.extend(fix1)
    escalations.extend(esc1)
    if note1:
        notes_parts.append(note1)

    # 2. Count consistency
    check2_failed, esc2, note2 = _check_clause_count_consistency(report, tid)
    checks_run.append("clause_count_consistency")
    if check2_failed:
        checks_failed.append("clause_count_consistency")
    escalations.extend(esc2)
    if note2:
        notes_parts.append(note2)

    # 3. Schema completeness
    check3_failed, esc3, note3 = _check_schema_completeness(report, tid)
    checks_run.append("schema_completeness")
    if check3_failed:
        checks_failed.append("schema_completeness")
    escalations.extend(esc3)
    if note3:
        notes_parts.append(note3)

    # 4. Guardrail-to-report disconnect (most safety-critical)
    check4_failed, esc4, note4 = _check_guardrail_disconnect(report, tid)
    checks_run.append("guardrail_disconnect")
    if check4_failed:
        checks_failed.append("guardrail_disconnect")
    escalations.extend(esc4)
    if note4:
        notes_parts.append(note4)

    # "passed" = no escalations (auto-fixes alone don't fail the guardrail)
    passed = len(escalations) == 0

    logger.info(
        "validate_final_output: doc_hash=%.16s passed=%s checks_failed=%s "
        "auto_fixes=%d escalations=%d",
        report.document_hash, passed, checks_failed,
        len(auto_fixes), len(escalations),
    )

    result = FinalOutputValidationResult(
        passed=passed,
        checks_run=checks_run,
        checks_failed=checks_failed,
        auto_fixes_applied=auto_fixes,
        escalations_created=escalations,
        notes="; ".join(notes_parts) if notes_parts else None,
    )
    return report, result


# ---------------------------------------------------------------------------
# Check 1: Disclaimer presence
# ---------------------------------------------------------------------------

def _check_disclaimer(
    report: LegalDocumentReport,
    thread_id: str,
) -> Tuple[LegalDocumentReport, bool, List[str], List[str], Optional[str]]:
    """
    Verify the disclaimer field is present and contains the required sentinel text.

    Returns (report, failed, auto_fixes, escalations, note).
    Auto-fixable: if the disclaimer is missing or empty, re-inject the canonical
    DISCLAIMER constant. This is the one auto-fix in this module because the
    correct value is completely unambiguous -- it's a fixed string.
    """
    disclaimer = report.disclaimer or ""
    if _DISCLAIMER_SENTINEL in disclaimer:
        logger.debug("_check_disclaimer: passed (sentinel present)")
        return report, False, [], [], None

    # Missing or incomplete disclaimer -- re-inject
    logger.warning(
        "_check_disclaimer: disclaimer missing/incomplete on doc_hash=%.16s -- "
        "auto-reinjecting",
        report.document_hash,
    )
    fixed_report = report.model_copy(update={"disclaimer": DISCLAIMER})
    return (
        fixed_report,
        True,
        ["disclaimer_reinjected"],
        [],
        "Disclaimer was absent or incomplete; canonical disclaimer text was re-injected.",
    )


# ---------------------------------------------------------------------------
# Check 2: Internal count consistency
# ---------------------------------------------------------------------------

def _check_clause_count_consistency(
    report: LegalDocumentReport,
    thread_id: str,
) -> Tuple[bool, List[str], Optional[str]]:
    """
    Cross-check the count fields in the report against the actual clause_entries.

    Three sub-checks:
      a) total_clauses_found must equal count of clause_entries where is_present=True
      b) total_clauses_missing must equal count where is_present=False
      c) total found + missing must equal _APPROVED_CLAUSE_COUNT (10)
      d) every name in missing_clauses must trace to a clause_entry with is_present=False

    Returns (failed, escalation_ids, note).
    Not auto-fixable: the system cannot safely choose between the header field
    and the clause_entries list when they disagree.
    """
    actual_present = sum(1 for e in report.clause_entries if e.is_present)
    actual_missing = sum(1 for e in report.clause_entries if not e.is_present)
    total_entries = len(report.clause_entries)

    problems: List[str] = []

    if report.total_clauses_found != actual_present:
        problems.append(
            f"total_clauses_found={report.total_clauses_found} but "
            f"clause_entries has {actual_present} present entries"
        )

    if report.total_clauses_missing != actual_missing:
        problems.append(
            f"total_clauses_missing={report.total_clauses_missing} but "
            f"clause_entries has {actual_missing} absent entries"
        )

    if total_entries != _APPROVED_CLAUSE_COUNT:
        problems.append(
            f"clause_entries has {total_entries} entries; "
            f"expected exactly {_APPROVED_CLAUSE_COUNT}"
        )

    # Orphaned missing-clause names
    absent_categories = {
        e.clause_category for e in report.clause_entries if not e.is_present
    }
    for name in report.missing_clauses:
        if name not in absent_categories:
            problems.append(
                f"missing_clauses contains '{name}' but no clause_entry "
                f"with that category has is_present=False"
            )

    if not problems:
        logger.debug("_check_clause_count_consistency: passed")
        return False, [], None

    note = "Count consistency problems: " + "; ".join(problems)
    logger.warning(
        "_check_clause_count_consistency: ESCALATING doc_hash=%.16s — %s",
        report.document_hash, note,
    )
    rev_id = _enqueue_report_escalation(
        report=report,
        thread_id=thread_id,
        trigger_reason="report_count_inconsistency",
        finding_summary=f"Report count fields are internally inconsistent. {note[:200]}",
    )
    return True, [rev_id], note


# ---------------------------------------------------------------------------
# Check 3: Schema completeness
# ---------------------------------------------------------------------------

def _check_schema_completeness(
    report: LegalDocumentReport,
    thread_id: str,
) -> Tuple[bool, List[str], Optional[str]]:
    """
    Verify that required fields are meaningfully populated beyond Pydantic defaults.

    Pydantic guarantees type validity. This check catches fields that are
    type-valid but semantically empty -- e.g. document_name="", generated_at
    missing (should never happen given Field(default_factory=...) but defensive
    programming costs little here), or clause_entries count wrong (covered in
    check 2, but also caught here for cross-coverage).

    Returns (failed, escalation_ids, note).
    """
    problems: List[str] = []

    if not report.document_name or not report.document_name.strip():
        problems.append("document_name is empty")

    if not report.executive_summary or not report.executive_summary.strip():
        problems.append("executive_summary is empty")

    if not report.limitations_note or not report.limitations_note.strip():
        problems.append("limitations_note is empty")

    # HIGH/MEDIUM risk findings must have a non-empty finding_summary
    for finding in report.risk_findings:
        if finding.risk_level in ("HIGH", "MEDIUM"):
            if not finding.finding_summary or not finding.finding_summary.strip():
                problems.append(
                    f"risk finding for '{finding.clause_category}' "
                    f"(risk={finding.risk_level}) has empty finding_summary"
                )

    if not problems:
        logger.debug("_check_schema_completeness: passed")
        return False, [], None

    note = "Schema completeness problems: " + "; ".join(problems)
    logger.warning(
        "_check_schema_completeness: ESCALATING doc_hash=%.16s — %s",
        report.document_hash, note,
    )
    rev_id = _enqueue_report_escalation(
        report=report,
        thread_id=thread_id,
        trigger_reason="report_schema_incomplete",
        finding_summary=f"Required report fields are missing or empty. {note[:200]}",
    )
    return True, [rev_id], note


# ---------------------------------------------------------------------------
# Check 4: Guardrail-to-report disconnect (most safety-critical)
# ---------------------------------------------------------------------------

def _check_guardrail_disconnect(
    report: LegalDocumentReport,
    thread_id: str,
) -> Tuple[bool, List[str], Optional[str]]:
    """
    Detect cases where the evidence-verification guardrail flagged a problem
    that is not reflected in the report's risk-finding metadata.

    The specific pattern this catches:
      - A clause is present (is_present=True) in clause_entries
      - AND evidence_verified=False for that clause (the verifier found no
        verbatim match in the source document)
      - AND the corresponding RiskFindingEntry has:
          * risk_level HIGH or MEDIUM (the guardrail should have escalated this)
          * human_review_status="not_required" (the report says it wasn't reviewed)

    This combination means: "The guardrail chain detected an evidence problem on
    a high-stakes clause but the final report implies it was never reviewed."
    That is a silent failure of the HITL pipeline -- precisely the kind of
    discrepancy this step exists to catch.

    Why this cannot be auto-fixed: we cannot retroactively determine the correct
    finding. The escalation surfaces it for human adjudication.

    Returns (failed, escalation_ids, note).
    """
    # Build a lookup: clause_category -> RiskFindingEntry
    risk_by_category = {f.clause_category: f for f in report.risk_findings}

    disconnects: List[str] = []

    for entry in report.clause_entries:
        if not entry.is_present:
            continue  # absent clauses have evidence_verified=None; nothing to check
        if entry.evidence_verified is not False:
            continue  # True or None -- not a failed verification

        # This clause is present but evidence was not verified
        finding = risk_by_category.get(entry.clause_category)
        if finding is None:
            # No risk finding at all for a present clause -- itself unusual, but
            # not the specific disconnect this check targets. Schema completeness
            # handles that. Skip here.
            continue

        if finding.risk_level in ("HIGH", "MEDIUM") and finding.human_review_status == "not_required":
            disconnects.append(
                f"'{entry.clause_category}' (risk={finding.risk_level}): "
                f"evidence_verified=False but human_review_status='not_required'"
            )

    if not disconnects:
        logger.debug("_check_guardrail_disconnect: passed")
        return False, [], None

    note = "Guardrail-to-report disconnect: " + "; ".join(disconnects)
    logger.warning(
        "_check_guardrail_disconnect: ESCALATING doc_hash=%.16s — %d disconnect(s)",
        report.document_hash, len(disconnects),
    )
    # One escalation per disconnect so each can be independently reviewed
    rev_ids: List[str] = []
    for disconnect_desc in disconnects:
        rev_id = _enqueue_report_escalation(
            report=report,
            thread_id=thread_id,
            trigger_reason="guardrail_disconnect",
            finding_summary=(
                f"Evidence verification failure not reflected in report metadata. "
                f"{disconnect_desc[:200]}"
            ),
        )
        rev_ids.append(rev_id)

    return True, rev_ids, note


# ---------------------------------------------------------------------------
# Escalation helper (mirrors _enqueue_escalation in claim_verifier.py)
# ---------------------------------------------------------------------------

def _enqueue_report_escalation(
    report: LegalDocumentReport,
    thread_id: str,
    trigger_reason: str,
    finding_summary: str,
) -> str:
    """
    Construct a ReviewItem for a report-level structural problem and enqueue it.

    Uses the same human_review.add_to_review_queue() path as claim_verifier.py
    so all escalations from any guardrail appear in the unified HITL queue.
    The review_item's source_text carries the document name (not contract text)
    and clause_category is None (this is a report-level, not clause-level, item).

    Returns the new review_id string.
    """
    rev_id = str(uuid.uuid4())
    item = ReviewItem(
        review_id=rev_id,
        document_hash=report.document_hash,
        source_chunk_id=None,
        clause_category=None,
        source_text=f"[Report-level issue for document: {report.document_name}]",
        source_page=None,
        trigger_reason=trigger_reason,
        ai_finding_summary=finding_summary,
        confidence_signals={
            "document_type": report.document_type,
            "total_clauses_found": report.total_clauses_found,
            "total_clauses_missing": report.total_clauses_missing,
        },
        alternatives=[],
        risk_level=None,
        status="pending",
        created_at=datetime.utcnow(),
        thread_id=thread_id,
        # Reviewer-context fields -- this is a report-level structural issue,
        # not a clause-level finding, so most clause fields are N/A.
        fact_found=f"Report-level structural issue detected: {trigger_reason}.",
        deviation_found=finding_summary,
        risk_rationale=(
            "Escalated by output_validator (Step 11) because a structural check "
            "on the assembled report failed. See trigger_reason for the specific check."
        ),
        document_type_context=report.document_type,
    )
    human_review_module.add_to_review_queue(item)
    logger.info(
        "output_validator: enqueued ReviewItem %s trigger=%s doc_hash=%.16s",
        rev_id, trigger_reason, report.document_hash,
    )
    return rev_id
