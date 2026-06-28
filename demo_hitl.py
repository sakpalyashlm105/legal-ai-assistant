"""
demo_hitl.py
-------------
Interactive demo of the LangGraph Human-in-the-Loop (HITL) interrupt/resume
flow for the Legal AI Assistant.

Runs a real PDF through the pipeline with the LLM nodes MOCKED so it:
  - Costs nothing (no OpenAI API calls in the mocked stages)
  - Runs in ~5 seconds
  - Reliably triggers a HITL interrupt (a missing critical clause is forced)

Then walks you through the human review decision interactively.

Usage (from legal-agent/ with venv active):
    python demo_hitl.py
    python demo_hitl.py --pdf data/pdf/ndas/NDA_2_Bakhu_Holdings,_Corp._(BKUH).pdf

What you will see:
    PHASE 1  -- pipeline runs, hits a HIGH-risk missing clause, PAUSES
    PHASE 2  -- the review queue is shown (what the human reviewer sees)
    PHASE 3  -- you choose an action: approve / correct / reject / select_alt
    PHASE 4  -- graph resumes, final state printed
"""

import argparse
import sys
import uuid
from pathlib import Path
from typing import Optional
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

from config import CLAUSE_CATEGORIES
from schemas.clause import DocumentClassification, ExtractedClause
from schemas.risk import RiskFinding
from schemas.review import ReviewDecision
from agent.orchestrator import run_pipeline, resume_after_review, get_graph_state
import agent.human_review as hr_module

DIVIDER = "=" * 70
SECTION  = "-" * 70

def section(title: str) -> None:
    print(f"\n{SECTION}")
    print(f"  {title}")
    print(SECTION)


# ---------------------------------------------------------------------------
# Mock helpers (make the demo fast and free)
# ---------------------------------------------------------------------------

def _mock_classification():
    return DocumentClassification(
        document_type="NDA",
        confidence=0.95,
        reasoning="Demo mock: confident NDA classification.",
        retry_count=0,
    )

def _mock_clauses():
    """9 present clauses + 1 absent (Indemnification) to force a HITL trigger.

    extracted_text values are verbatim substrings of the real extracted PDF text
    (verified to pass evidence_verifier's exact-match check after normalize_text).
    This gives the demo a realistic mix of verified/unverified results instead of
    a uniform 9/9 "Evidence Verified: No" artifact from placeholder text.
    """
    # Verbatim substrings from the real PDF — each passes exact-match verification.
    # Indemnification is intentionally absent (is_present=False, no text).
    REAL_SNIPPETS = {
        "Confidentiality / Non-Disclosure": (
            'This Confidentiality and Nondisclosure Agreement ("Agreement") is by and between Bakhu\n'
            'Holdings, Corp., a Nevada corporation',
            1,
        ),
        "Termination for Convenience": (
            "Each party acknowledges and agrees that the release of Protected\n"
            "Information in violation of this Agreement may cause irreparable harm for which the Company may not\n"
            "be fully or adequately compensated by recovery of monetary damages.",
            2,
        ),
        "Termination for Cause": (
            "the Company shall be entitled to injunctive relief from a court of\n"
            "competent jurisdiction in addition to any other remedy available at law or in equity.",
            2,
        ),
        "Governing Law / Jurisdiction": (
            "All issues and questions concerning the construction, validity,\n"
            "interpretation, and enforceability of this Agreement shall be governed by, and construed in accordance\n"
            "with, the Nevada Laws, without giving effect to any choice of law or conflict of law rules or provisions.",
            3,
        ),
        "Limitation of Liability": (
            "In no event shall either party have any right to recover from the other party any consequential damages",
            3,
        ),
        "Non-Compete / Non-Solicitation": (
            "(a) not disclose Protected Information,\n"
            "directly or indirectly, to any third person without the express written consent of the Company",
            1,
        ),
        "Assignment": (
            "All Protected Information shall remain the sole property of the Company.",
            2,
        ),
        "Renewal / Term": (
            "This Agreement shall become effective on the date of execution\n"
            "by the Company and Director and remain in effect for so long as any of the Protected Information remains\n"
            "confidential or proprietary to the Company.",
            2,
        ),
        "Dispute Resolution": (
            "Any action seeking to enforce any provision of, or based on any matter\n"
            "arising out of or in connection with, this Agreement or the transactions contemplated hereby or thereby\n"
            "shall be brought in and determined exclusively by the state courts in Los Angeles County in the State of\n"
            "California",
            3,
        ),
    }

    clauses = []
    for cat in CLAUSE_CATEGORIES:
        if cat == "Indemnification":
            clauses.append(ExtractedClause(
                clause_type=cat,
                is_present=False,
                extracted_text=None,
                page_reference=None,
                confidence=0.91,
                source_chunk_id=None,
                retry_count=0,
                requires_human_review=False,
            ))
        else:
            text, page = REAL_SNIPPETS[cat]
            clauses.append(ExtractedClause(
                clause_type=cat,
                is_present=True,
                extracted_text=text,
                page_reference=page,
                confidence=0.92,
                source_chunk_id="chunk-001",
                retry_count=0,
                requires_human_review=False,
            ))
    return clauses

def _mock_findings(clauses):
    """Generate findings: Indemnification is missing -> HIGH risk -> HITL."""
    from agent.comparator import compare_to_templates
    from agent.risk_engine import flag_risks
    # Let the real risk engine run (no LLM calls there) with real clauses
    comparisons = []  # no template calls needed for demo
    findings = []
    for clause in clauses:
        if not clause.is_present:
            findings.append(RiskFinding(
                clause_type=clause.clause_type,
                risk_level="HIGH",
                is_missing=True,
                reason="Critical clause absent from document -- no precedent override applies.",
                precedent_applied=False,
                precedent_note=None,
            ))
        else:
            findings.append(RiskFinding(
                clause_type=clause.clause_type,
                risk_level="LOW",
                is_missing=False,
                reason="Clause present, no significant deviation from standard.",
                precedent_applied=False,
                precedent_note=None,
            ))
    return findings, comparisons


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def run_demo(pdf_path: Path, real_llm: bool = False) -> None:
    # Clear any leftover queue items from previous runs/tests
    hr_module._ensure_queue_dir()
    hr_module._PENDING_FILE.write_text("[]", encoding="utf-8")

    if not pdf_path.exists():
        print(f"\n[ERROR] PDF not found: {pdf_path}")
        print("Run from the legal-agent/ directory, e.g.:")
        print("  python demo_hitl.py --pdf data/pdf/ndas/NDA_2_Bakhu_Holdings,_Corp._(BKUH).pdf")
        sys.exit(1)

    thread_id = f"demo-{uuid.uuid4().hex[:8]}"

    # -------------------------------------------------------------------------
    print(f"\n{DIVIDER}")
    print(f"  LEGAL AI ASSISTANT -- HITL INTERRUPT/RESUME DEMO")
    print(f"  Document  : {pdf_path.name}")
    print(f"  Thread ID : {thread_id}")
    if real_llm:
        print(f"  (REAL LLM MODE -- uses GPT-4o-mini, costs tokens, takes ~2-3 min)")
    else:
        print(f"  (LLM nodes are mocked so this runs instantly for free)")
    print(DIVIDER)

    # -------------------------------------------------------------------------
    if real_llm:
        section("PHASE 1 -- RUNNING PIPELINE  (ALL stages real -- GPT-4o-mini)")
        print("  Stages: validate_input -> validate_scope -> extract -> scan_prompt_injection")
        print("          -> chunk -> classify -> extract_clauses -> expand_clause_boundaries")
        print("          -> verify_clauses -> flag_risks -> verify_final_claims -> build_review_items")
        print("  NOTE: No forcing -- HITL triggers only if real LLM finds HIGH-risk findings.")
        print("  This may take 2-3 minutes and will use OpenAI API credits.")
        print()

        result = run_pipeline(
            str(pdf_path),
            thread_id=thread_id,
            request_text="Analyze this legal document for risk and missing clauses.",
            processing_notes="Full real-LLM run via --real-llm flag.",
        )
    else:
        section("PHASE 1 -- RUNNING PIPELINE  (extraction is real; LLM stages mocked)")
        print("  Stages: validate_input -> validate_scope -> extract -> scan_prompt_injection")
        print("          -> chunk -> classify -> extract_clauses -> expand_clause_boundaries")
        print("          -> verify_clauses -> flag_risks -> verify_final_claims -> build_review_items")
        print("  Forcing: Indemnification clause ABSENT -> HIGH risk -> HITL trigger")
        print()

        mock_clauses = _mock_clauses()
        mock_findings, mock_comparisons = _mock_findings(mock_clauses)

        DEMO_NOTE = (
            "NOTE (Demo Run): LLM clause-extraction stages were mocked for this run. "
            "PDF extraction, evidence verification, and guardrail checks ran for real. "
            "Evidence verification reflects whether demo mock text matched the real PDF content."
        )

        with (
            patch("agent.orchestrator.classify_document",    return_value=_mock_classification()),
            patch("agent.orchestrator.extract_clauses",      return_value=mock_clauses),
            patch("agent.orchestrator.flag_risks",           return_value=mock_findings),
            patch("agent.orchestrator.compare_to_templates", return_value=mock_comparisons),
        ):
            result = run_pipeline(
                str(pdf_path),
                thread_id=thread_id,
                request_text="Analyze this NDA document for risk and missing clauses.",
                processing_notes=DEMO_NOTE,
            )

    print(f"  Pipeline status       : {result['status'].upper()}")
    print(f"  needs_human_review    : {result['needs_human_review']}")
    print(f"  review_items queued   : {len(result['review_items'])}")

    # Show guardrail summary (new in Step 9 wiring)
    gr = result.get("guardrail_results", [])
    if gr:
        print(f"\n  Guardrails ran        : {len(gr)}")
        for g_result in gr:
            status = "PASS" if g_result["passed"] else g_result["severity"].upper()
            print(f"    [{status:8s}] {g_result['guardrail_name']}")

    if result["status"] == "completed":
        print("\n  [No HITL triggers fired -- pipeline completed directly.]")
        fs = result.get("final_state") or {}
        if fs:
            print(f"\n  +-- FINAL STATE -------------------------------------------------------")
            print(f"  |completed               : {fs.get('completed')}")
            print(f"  |document_type           : {fs.get('document_type')}")
            print(f"  |classification_conf     : {fs.get('classification_confidence')}")
            print(f"  |total_chunks            : {fs.get('total_chunks')}")
            print(f"  |clauses_present         : {fs.get('clauses_present')} / 10")
            print(f"  |total_findings          : {fs.get('total_findings')}")
            risk = fs.get("risk_counts", {})
            print(f"  |risk_counts             : HIGH={risk.get('HIGH',0)}  MEDIUM={risk.get('MEDIUM',0)}  LOW={risk.get('LOW',0)}")
            print(f"  |review_items_count      : {fs.get('review_items_count')}")
            print(f"  |human_decisions_applied : {fs.get('human_decisions_applied')}")
            print(f"  +---------------------------------------------------------------------")
            val_passed = fs.get("validation_passed")
            val_fixes  = fs.get("validation_auto_fixes", [])
            val_escs   = fs.get("validation_escalations", [])
            print(f"  [Step 11] validate_final_output: passed={val_passed}", end="")
            if val_fixes:
                print(f"  auto_fixes={val_fixes}", end="")
            if val_escs:
                print(f"  escalations={len(val_escs)} (see HITL queue)", end="")
            print()
            gr_report = result.get("generated_report")
            if gr_report:
                doc_hash = gr_report.get("document_hash", "")[:16]
                md_path = Path(__file__).parent / "data" / "processed" / f"{doc_hash}.md"
                if md_path.exists():
                    print(f"\n  Generated report saved : {md_path}")
                    print(f"  Report size            : {md_path.stat().st_size:,} bytes")
                    lines = md_path.read_text(encoding="utf-8").splitlines()
                    print(f"\n{SECTION}")
                    print(f"  GENERATED REPORT PREVIEW (first 60 lines)")
                    print(SECTION)
                    for line in lines[:60]:
                        print(f"  {line}")
                    if len(lines) > 60:
                        print(f"  ... ({len(lines) - 60} more lines, see {md_path})")
        return

    # -------------------------------------------------------------------------
    section("PHASE 2 -- GRAPH IS PAUSED  (waiting for human reviewer)")
    snap = get_graph_state(thread_id)
    print(f"  Graph paused at node  : {snap['next']}")
    print(f"  is_paused             : {snap['is_paused']}")

    pending = hr_module.get_pending_reviews()
    print(f"\n  Pending review items  : {len(pending)}")

    if not pending:
        print("  [Unexpected: no items in queue]")
        return

    # -------------------------------------------------------------------------
    # MULTI-ITEM REVIEW: collect decisions for ALL pending items, then resume once.
    # Previously the demo only handled pending[0] and resumed immediately.
    # The graph is paused at a single interrupt point; all items were queued
    # before the pause. We collect all decisions first, then call resume once.
    # -------------------------------------------------------------------------
    decisions: list[ReviewDecision] = []

    _REJECT_CATEGORIES = [
        "duplicate_finding",
        "wrong_clause_category",
        "hallucinated_deviation",
        "evidence_mismatch",
        "low_materiality",
        "template_mismatch",
        "not_legally_relevant",
    ]

    def _wrap(text: Optional[str], width: int = 70, indent: str = "  |   ") -> str:
        """Wrap long text to fit in the review card."""
        if not text:
            return "(none)"
        import textwrap
        lines = textwrap.wrap(text, width=width)
        return f"\n{indent}".join(lines) if lines else "(none)"

    def _prompt_reason(item_risk: Optional[str]) -> str:
        """Prompt for reason; re-prompt if empty and risk is HIGH."""
        while True:
            r = input("  Reason (required for HIGH-risk; press Enter to skip for MEDIUM/LOW): ").strip()
            if item_risk == "HIGH" and not r:
                print("  [This is a HIGH-risk finding. A reason is required.]")
                continue
            return r

    for idx, item in enumerate(pending, start=1):
        section(f"PHASE 2.{idx} -- REVIEW ITEM {idx} of {len(pending)}")

        # ---- Rich review card ----
        print(f"  +================ REVIEW ITEM {idx}/{len(pending)} ================+")
        print(f"  | Clause Category : {item.clause_category or '(document-level)'}")
        print(f"  | Document Type   : {item.document_type_context or '(unknown)'}")
        print(f"  | Risk Level      : {item.risk_level or 'N/A'}")
        print(f"  | Trigger Reason  : {item.trigger_reason}")
        print(f"  |")
        print(f"  | FACT FOUND:")
        print(f"  |   {_wrap(item.fact_found or item.ai_finding_summary)}")
        print(f"  |")
        print(f"  | DEVIATION FOUND:")
        print(f"  |   {_wrap(item.deviation_found or 'None -- no deviation data available')}")
        print(f"  |")
        print(f"  | RISK RATIONALE:")
        print(f"  |   {_wrap(item.risk_rationale or item.ai_finding_summary)}")
        print(f"  |")
        print(f"  | EXTRACTED CLAUSE TEXT (full):")
        src = item.source_text or "(none)"
        for line in src.splitlines():
            print(f"  |   {line}")
        print(f"  |")
        print(f"  | SOURCE CONTEXT:")
        print(f"  |   [Previous] {_wrap(item.previous_chunk_text, 60) if item.previous_chunk_text else '(none -- first chunk or unavailable)'}")
        print(f"  |   [This clause, page {item.source_page or 'unknown'}]")
        print(f"  |   [Next]     {_wrap(item.next_chunk_text, 60) if item.next_chunk_text else '(none -- last chunk or unavailable)'}")
        print(f"  |")
        print(f"  | TEMPLATE COMPARISON:")
        print(f"  |   Expected: {_wrap(item.template_clause_text or 'No template available for this category', 60)}")
        missing_str = ", ".join(item.missing_elements) if item.missing_elements else "none flagged"
        extra_str   = ", ".join(item.extra_risky_language) if item.extra_risky_language else "none flagged"
        print(f"  |   Missing elements     : {missing_str}")
        print(f"  |   Extra risky language : {extra_str}")
        print(f"  |")
        print(f"  | VERIFICATION METADATA:")
        print(f"  |   Evidence match   : {item.evidence_match_type or 'N/A'} (score: {item.evidence_match_score if item.evidence_match_score is not None else 'N/A'})")
        print(f"  |   Page ref valid   : {item.page_reference_valid if item.page_reference_valid is not None else 'N/A'}")
        print(f"  |   Confidence       : {item.extraction_confidence if item.extraction_confidence is not None else 'N/A'}")
        print(f"  |")
        print(f"  | CLAUSE BOUNDARY EXPANSION:")
        if item.expansion_triggered:
            print(f"  |   Status           : EXPANDED ({len(item.source_chunks_used)} chunks merged)")
            print(f"  |   Stopped because  : {item.expansion_boundary_reason or 'N/A'}")
            print(f"  |   Chunks used      : {', '.join(item.source_chunks_used)}")
            print(f"  |   NOTE: Template comparison was run on the EXPANDED text below,")
            print(f"  |         not on the original snippet above. This prevents a false")
            print(f"  |         HIGH-risk finding caused by partial clause extraction.")
            print(f"  |   Expanded text:")
            if item.expanded_clause_text:
                for line in item.expanded_clause_text.splitlines():
                    print(f"  |     {line}")
            else:
                print(f"  |     (none)")
        else:
            reason = item.expansion_boundary_reason or "not triggered"
            print(f"  |   Status           : NOT EXPANDED ({reason})")
            print(f"  |   Template comparison used the original extracted snippet.")
        print(f"  +===============================================+")

        section(f"PHASE 3.{idx} -- HUMAN REVIEWER DECISION  (item {idx}/{len(pending)})")
        print("  Choose an action:")
        print("    [1] approve          -- AI is correct, accept as-is")
        print("    [2] correct          -- AI is wrong, provide a correction")
        print("    [3] reject           -- Discard this finding entirely")
        print("    [4] select_alt       -- (not applicable here, no alternatives)")
        if item.risk_level == "HIGH":
            print(f"\n  NOTE: This is a HIGH-risk finding. Your decision REQUIRES a reason.")
        print()

        while True:
            choice = input("  Your choice (1/2/3): ").strip()
            if choice in ("1", "2", "3"):
                break
            print("  Please enter 1, 2, or 3.")

        if choice == "1":
            reason = _prompt_reason(item.risk_level)
            decision = ReviewDecision(
                review_id=item.review_id,
                action="approve",
                reason=reason,
                reviewer_note=reason or "Reviewed and confirmed.",
                decided_by="demo_reviewer",
                risk_level_on_item=item.risk_level,
            )
            print(f"\n  Action : APPROVE -- AI's finding accepted as-is")

        elif choice == "2":
            reason = _prompt_reason(item.risk_level)
            print()
            print("  Enter the corrected text (or press Enter for a demo value):")
            corrected_text = input("  Corrected text: ").strip()
            if not corrected_text:
                corrected_text = f"{item.clause_category or 'Clause'} located in document -- AI missed it."

            print("  Corrected risk level? [1=LOW  2=MEDIUM  3=HIGH  Enter=keep original]:")
            rl_choice = input("  Risk level: ").strip()
            corrected_risk = {"1": "LOW", "2": "MEDIUM", "3": "HIGH"}.get(rl_choice)

            print("  Brief corrected summary (or Enter to skip):")
            corrected_summary = input("  Summary: ").strip() or None

            print("  Corrected rationale (or Enter to skip):")
            corrected_rationale = input("  Rationale: ").strip() or None

            decision = ReviewDecision(
                review_id=item.review_id,
                action="correct",
                reason=reason,
                corrected_value={
                    "clause_type": item.clause_category,
                    "is_present": True,
                    "extracted_text": corrected_text,
                    "confidence": 1.0,
                    "human_overridden": True,
                },
                corrected_risk_level=corrected_risk,
                corrected_summary=corrected_summary,
                corrected_rationale=corrected_rationale,
                reviewer_note=reason or "Human correction applied.",
                decided_by="demo_reviewer",
                risk_level_on_item=item.risk_level,
            )
            print(f"\n  Action : CORRECT -- overriding AI with: {corrected_text!r:.80}")

        else:  # "3"
            reason = _prompt_reason(item.risk_level)
            print()
            print("  Select reject category:")
            for i, cat in enumerate(_REJECT_CATEGORIES, start=1):
                print(f"    [{i}] {cat}")
            while True:
                cat_choice = input("  Category (1-7): ").strip()
                if cat_choice.isdigit() and 1 <= int(cat_choice) <= 7:
                    reject_cat = _REJECT_CATEGORIES[int(cat_choice) - 1]
                    break
                print("  Please enter a number 1-7.")

            decision = ReviewDecision(
                review_id=item.review_id,
                action="reject",
                reason=reason or "Finding discarded by reviewer.",
                reviewer_note=reason or "Finding is unreliable -- discarding.",
                decided_by="demo_reviewer",
                reject_category=reject_cat,
                risk_level_on_item=item.risk_level,
            )
            print(f"\n  Action : REJECT ({reject_cat}) -- finding discarded")

        decisions.append(decision)

    # Record all decisions into the queue before resuming
    for d in decisions:
        hr_module.record_review_decision(d)

    # The graph is resumed with the LAST (or only) decision payload.
    # In the single-item case this is identical to the previous behavior.
    # In the multi-item case: ALL items are resolved before resume; the graph
    # reads human_decisions from state and sees all of them.
    last_decision = decisions[-1]

    # -------------------------------------------------------------------------
    section("PHASE 4 -- RESUMING GRAPH  (applying decisions + continuing)")
    print(f"  Decisions recorded    : {len(decisions)}")
    print(f"  Calling resume_after_review(thread_id={thread_id!r}, action={last_decision.action!r})")
    print()

    resume_result = resume_after_review(thread_id, last_decision)

    print(f"  Resume status         : {resume_result['status'].upper()}")
    fs = resume_result.get("final_state") or {}
    if fs:
        print(f"\n  +-- FINAL STATE -------------------------------------------------------")
        print(f"  |completed               : {fs.get('completed')}")
        print(f"  |document_type           : {fs.get('document_type')}")
        print(f"  |classification_conf     : {fs.get('classification_confidence')}")
        print(f"  |total_chunks            : {fs.get('total_chunks')}")
        print(f"  |clauses_present         : {fs.get('clauses_present')} / 10")
        print(f"  |total_findings          : {fs.get('total_findings')}")
        risk = fs.get("risk_counts", {})
        print(f"  |risk_counts             : HIGH={risk.get('HIGH',0)}  MEDIUM={risk.get('MEDIUM',0)}  LOW={risk.get('LOW',0)}")
        print(f"  |review_items_count      : {fs.get('review_items_count')}")
        print(f"  |human_decisions_applied : {fs.get('human_decisions_applied')}")
        print(f"  +---------------------------------------------------------------------")

    # Show what the decision injected into graph state
    snap_after = get_graph_state(thread_id)
    decisions_in_state = snap_after["values"].get("human_decisions", [])
    if decisions_in_state:
        d = decisions_in_state[-1]
        print(f"\n  Human decision recorded in graph state:")
        print(f"    action    : {d['action']}")
        print(f"    discarded : {d['discarded']}")
        if d.get("value"):
            print(f"    value     : {str(d['value'])[:120]}")

    # Show generated report location
    gr = resume_result.get("generated_report")
    if gr:
        doc_hash = gr.get("document_hash", "")[:16]
        from pathlib import Path
        md_path = Path(__file__).parent / "data" / "processed" / f"{doc_hash}.md"
        if md_path.exists():
            print(f"\n  Generated report saved : {md_path}")
            print(f"  Report size            : {md_path.stat().st_size:,} bytes")
            # Print first 60 lines of the report as a preview
            lines = md_path.read_text(encoding="utf-8").splitlines()
            print(f"\n{SECTION}")
            print(f"  GENERATED REPORT PREVIEW (first 60 lines)")
            print(SECTION)
            for line in lines[:60]:
                print(f"  {line}")
            if len(lines) > 60:
                print(f"  ... ({len(lines) - 60} more lines, see {md_path})")

    # -------------------------------------------------------------------------
    print(f"\n{DIVIDER}")
    print(f"  DEMO COMPLETE")
    print(DIVIDER)
    decision_summary = ", ".join(f"{d.action}({d.review_id[:8]})" for d in decisions)
    print(f"  What just happened:")
    print(f"    1. Pipeline ran on a real PDF (extraction was real)")
    print(f"    2. {len(decisions)} HITL trigger(s) detected -> graph PAUSED")
    print(f"    3. {len(decisions)} ReviewItem(s) created in the queue (thread_id embedded)")
    print(f"    4. Human reviewed all items: {decision_summary}")
    print(f"    5. All decisions recorded; graph resumed ONCE after all items addressed")
    print(f"    6. Graph RESUMED from the checkpoint -> ran to completion")
    print(f"    7. final_state shows human_decisions_applied={fs.get('human_decisions_applied', 0)}")
    # Step 11 output
    val_passed = fs.get("validation_passed")
    val_fixes  = fs.get("validation_auto_fixes", [])
    val_escs   = fs.get("validation_escalations", [])
    print(f"    8. [Step 11] validate_final_output: passed={val_passed}", end="")
    if val_fixes:
        print(f"  auto_fixes={val_fixes}", end="")
    if val_escs:
        print(f"  escalations={len(val_escs)} (see HITL queue)", end="")
    print()
    print()
    print(f"  Key insight: the graph used a real LangGraph MemorySaver checkpointer.")
    print(f"  The thread_id '{thread_id}' is what links the paused state to the resume.")
    print(f"  In production, steps 2-5 would happen asynchronously (queue -> UI -> decision).")
    print(f"  Multi-item behavior: ALL items are reviewed before the graph resumes.")
    print(DIVIDER)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Legal AI Assistant -- HITL demo")
    parser.add_argument(
        "--pdf",
        default="data/pdf/ndas/NDA_2_Bakhu_Holdings,_Corp._(BKUH).pdf",
        help="PDF to process (relative to legal-agent/)",
    )
    parser.add_argument(
        "--real-llm",
        action="store_true",
        default=False,
        help="Use real GPT-4o-mini for all stages (costs tokens, ~2-3 min). "
             "Without this flag, LLM stages are mocked for a free instant run.",
    )
    args = parser.parse_args()
    pdf_path = Path(args.pdf)
    if not pdf_path.is_absolute():
        pdf_path = Path(__file__).parent / pdf_path
    run_demo(pdf_path, real_llm=args.real_llm)
