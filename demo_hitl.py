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
    """9 present clauses + 1 absent (Indemnification) to force a HITL trigger."""
    clauses = []
    for i, cat in enumerate(CLAUSE_CATEGORIES):
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
            clauses.append(ExtractedClause(
                clause_type=cat,
                is_present=True,
                extracted_text=f"[Demo] Standard {cat} language found on page 2.",
                page_reference=2,
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

def run_demo(pdf_path: Path) -> None:
    # Clear any leftover queue items from previous runs/tests
    hr_module._ensure_queue_dir()
    hr_module._PENDING_FILE.write_text("[]", encoding="utf-8")

    if not pdf_path.exists():
        print(f"\n[ERROR] PDF not found: {pdf_path}")
        print("Run from the legal-agent/ directory, e.g.:")
        print("  python demo_hitl.py --pdf data/pdf/ndas/NDA_2_Bakhu_Holdings,_Corp._(BKUH).pdf")
        sys.exit(1)

    thread_id = f"demo-{uuid.uuid4().hex[:8]}"
    mock_clauses = _mock_clauses()
    mock_findings, mock_comparisons = _mock_findings(mock_clauses)

    # -------------------------------------------------------------------------
    print(f"\n{DIVIDER}")
    print(f"  LEGAL AI ASSISTANT -- HITL INTERRUPT/RESUME DEMO")
    print(f"  Document  : {pdf_path.name}")
    print(f"  Thread ID : {thread_id}")
    print(f"  (LLM nodes are mocked so this runs instantly for free)")
    print(DIVIDER)

    # -------------------------------------------------------------------------
    section("PHASE 1 -- RUNNING PIPELINE  (extraction is real; LLM stages mocked)")
    print("  Stages: extract -> chunk -> classify -> extract_clauses -> flag_risks")
    print("  Forcing: Indemnification clause ABSENT -> HIGH risk -> HITL trigger")
    print()

    with (
        patch("agent.orchestrator.classify_document",    return_value=_mock_classification()),
        patch("agent.orchestrator.extract_clauses",      return_value=mock_clauses),
        patch("agent.orchestrator.flag_risks",           return_value=mock_findings),
        patch("agent.orchestrator.compare_to_templates", return_value=mock_comparisons),
    ):
        result = run_pipeline(str(pdf_path), thread_id=thread_id)

    print(f"  Pipeline status       : {result['status'].upper()}")
    print(f"  needs_human_review    : {result['needs_human_review']}")
    print(f"  review_items queued   : {len(result['review_items'])}")

    if result["status"] == "completed":
        print("\n  [No HITL triggers fired -- pipeline completed directly.]")
        print(f"  final_state: {result['final_state']}")
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

    item = pending[0]
    print(f"\n  +-- REVIEW ITEM -------------------------------------------------------")
    print(f"  |review_id       : {item.review_id}")
    print(f"  |clause_category : {item.clause_category}")
    print(f"  |trigger_reason  : {item.trigger_reason}")
    print(f"  |risk_level      : {item.risk_level}")
    print(f"  |ai_summary      : {item.ai_finding_summary}")
    print(f"  |source_text     : {item.source_text[:100]!r}")
    print(f"  |thread_id       : {item.thread_id}")
    print(f"  +---------------------------------------------------------------------")

    # -------------------------------------------------------------------------
    section("PHASE 3 -- HUMAN REVIEWER DECISION")
    print("  Choose an action:")
    print("    [1] approve          -- AI is correct, accept as-is")
    print("    [2] correct          -- AI is wrong, provide a correction")
    print("    [3] reject           -- Discard this finding entirely")
    print("    [4] select_alt       -- (not applicable here, no alternatives)")
    print()

    while True:
        choice = input("  Your choice (1/2/3): ").strip()
        if choice in ("1", "2", "3"):
            break
        print("  Please enter 1, 2, or 3.")

    if choice == "1":
        decision = ReviewDecision(
            review_id=item.review_id,
            action="approve",
            reviewer_note="Reviewed and confirmed: clause is indeed absent.",
            decided_by="demo_reviewer",
        )
        print(f"\n  Action : APPROVE -- AI's finding accepted as-is")

    elif choice == "2":
        print()
        print("  Enter the corrected finding (what the correct clause text should be).")
        corrected_text = input("  Corrected text (or press Enter for a demo value): ").strip()
        if not corrected_text:
            corrected_text = "Indemnification clause found in Exhibit A, paragraph 3 -- AI missed it."
        decision = ReviewDecision(
            review_id=item.review_id,
            action="correct",
            corrected_value={
                "clause_type": item.clause_category,
                "is_present": True,
                "extracted_text": corrected_text,
                "confidence": 1.0,
                "human_overridden": True,
            },
            reviewer_note="Found in exhibit, AI missed it.",
            decided_by="demo_reviewer",
        )
        print(f"\n  Action : CORRECT -- overriding AI with: {corrected_text!r:.80}")

    else:  # "3"
        decision = ReviewDecision(
            review_id=item.review_id,
            action="reject",
            reviewer_note="Finding is unreliable -- discarding.",
            decided_by="demo_reviewer",
        )
        print(f"\n  Action : REJECT -- finding discarded")

    # -------------------------------------------------------------------------
    section("PHASE 4 -- RESUMING GRAPH  (applying decision + continuing)")
    print(f"  Calling resume_after_review(thread_id={thread_id!r}, action={decision.action!r})")
    print()

    resume_result = resume_after_review(thread_id, decision)

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

    # -------------------------------------------------------------------------
    print(f"\n{DIVIDER}")
    print(f"  DEMO COMPLETE")
    print(DIVIDER)
    print(f"  What just happened:")
    print(f"    1. Pipeline ran on a real PDF (extraction was real)")
    print(f"    2. A HIGH-risk missing clause was detected -> graph PAUSED")
    print(f"    3. A ReviewItem was created in the queue (thread_id embedded)")
    print(f"    4. Human chose action='{decision.action}'")
    print(f"    5. Decision was recorded, injected into graph state via update_state()")
    print(f"    6. Graph RESUMED from the checkpoint -> ran to completion")
    print(f"    7. final_state shows human_decisions_applied={fs.get('human_decisions_applied', 0)}")
    print()
    print(f"  Key insight: the graph used a real LangGraph MemorySaver checkpointer.")
    print(f"  The thread_id '{thread_id}' is what links the paused state to the resume.")
    print(f"  In production, steps 2-5 would happen asynchronously (queue -> UI -> decision).")
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
    args = parser.parse_args()
    pdf_path = Path(args.pdf)
    if not pdf_path.is_absolute():
        pdf_path = Path(__file__).parent / pdf_path
    run_demo(pdf_path)
