"""
agent/orchestrator.py
----------------------
Minimal LangGraph StateGraph that wires the core pipeline together and
implements a REAL human-in-the-loop (HITL) interrupt/resume mechanism.

SCOPE STATEMENT (important)
---------------------------
This is an intentionally MINIMAL graph built in Step 8. It implements
exactly the nodes needed to demonstrate a working interrupt/resume cycle:

    extract -> chunk -> classify -> extract_clauses -> flag_risks
        -> build_review_items -> (interrupt if needed) -> check_human_review
        -> apply_decisions -> finalize

It does NOT implement the full 26-node sequence described in the project's
overall architecture (validate_input, scan_prompt_injection, rerank_context,
verify_evidence, generate_report, record_metrics, etc.). Those nodes are
future scope and are explicitly deferred. This distinction is documented
here rather than hidden.

LangGraph version: 0.1.19
Interrupt mechanism: compile(interrupt_before=["check_human_review"])
    -- the graph pauses BEFORE check_human_review runs whenever
    build_review_items sets needs_human_review=True in state.
    Resuming is done via invoke(None, config).

Checkpointer: MemorySaver (in-memory)
    -- appropriate for capstone scope. State is lost when the process exits.
    SqliteSaver is available in this version for cross-restart persistence
    but adds complexity that is explicitly out of scope here.

HITL, not RLHF -- human decisions override AI output for one run only.
They never update model weights or feed into training data.

PII-safe logging: log thread_id, doc_hash prefix, risk levels, node names.
Never log full source_text, clause contents, or corrected_value payloads.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agent.classifier import classify_document
from agent.comparator import compare_to_templates
from agent.extractor import extract_clauses
from agent.human_review import add_to_review_queue, apply_human_decision, record_review_decision
from agent.risk_engine import flag_risks
from extraction.pdf_parser import extract_text_from_pdf
from retrieval.chunking import chunk_document
from schemas.review import ReviewDecision, ReviewItem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class PipelineState(TypedDict):
    """
    Shared state dict flowing through every node.

    Each node reads from this and returns a partial update (only changed keys).
    LangGraph merges updates into the full state.

    Fields
    ------
    pdf_path          : str -- absolute path to the PDF being processed
    _thread_id        : str -- LangGraph thread ID (injected at run start)
    document_hash     : str -- SHA-256 hash, set after extraction
    doc               : dict or None -- serialized DocumentExtraction
    chunks            : list -- serialized DocumentChunk list
    classification    : dict or None -- serialized DocumentClassification
    clauses           : list -- serialized ExtractedClause list
    comparisons       : list -- serialized TemplateComparison list
    findings          : list -- serialized RiskFinding list
    review_items      : list -- ReviewItem dicts queued for this run
    needs_human_review: bool -- True if any finding needs HITL
    human_decisions   : list -- applied ReviewDecision outcomes
    final_state       : dict or None -- assembled final output
    error             : str or None -- set if any node fails
    """
    pdf_path: str
    _thread_id: str
    document_hash: str
    doc: Optional[Dict]
    chunks: List[Dict]
    classification: Optional[Dict]
    clauses: List[Dict]
    comparisons: List[Dict]
    findings: List[Dict]
    review_items: List[Dict]
    needs_human_review: bool
    human_decisions: List[Dict]
    final_state: Optional[Dict]
    error: Optional[str]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def node_extract(state: PipelineState) -> Dict[str, Any]:
    """Extract text from the PDF using PyMuPDF (+ OCR fallback)."""
    try:
        doc = extract_text_from_pdf(state["pdf_path"])
        logger.info("node_extract: doc=%s pages=%d", doc.file_name, doc.total_pages)
        return {
            "doc": doc.model_dump(mode="json"),
            "document_hash": doc.file_hash,
            "error": None,
        }
    except Exception as exc:
        logger.error("node_extract failed: %s", exc)
        return {"error": f"extraction failed: {exc}"}


def node_chunk(state: PipelineState) -> Dict[str, Any]:
    """Chunk extracted text into token-bounded segments."""
    if state.get("error"):
        return {}
    try:
        from schemas.document import DocumentExtraction
        doc = DocumentExtraction(**state["doc"])
        chunks = chunk_document(doc)
        logger.info("node_chunk: %d chunks produced", len(chunks))
        return {"chunks": [c.model_dump(mode="json") for c in chunks]}
    except Exception as exc:
        logger.error("node_chunk failed: %s", exc)
        return {"error": f"chunking failed: {exc}"}


def node_classify(state: PipelineState) -> Dict[str, Any]:
    """Classify document type (NDA / Contract / Amendment / Other)."""
    if state.get("error"):
        return {}
    try:
        from schemas.document import DocumentExtraction
        doc = DocumentExtraction(**state["doc"])
        classification = classify_document(doc)
        logger.info(
            "node_classify: type=%s conf=%.3f",
            classification.document_type,
            classification.confidence,
        )
        return {"classification": classification.model_dump(mode="json")}
    except Exception as exc:
        logger.error("node_classify failed: %s", exc)
        return {"error": f"classification failed: {exc}"}


def node_extract_clauses(state: PipelineState) -> Dict[str, Any]:
    """Extract all 10 clause categories from the chunked document."""
    if state.get("error"):
        return {}
    try:
        from schemas.chunk import DocumentChunk
        chunks = [DocumentChunk(**c) for c in state["chunks"]]
        doc_name = state["doc"].get("file_name", "unknown.pdf")
        clauses = extract_clauses(chunks, document_name=doc_name)
        present = sum(1 for c in clauses if c.is_present)
        logger.info("node_extract_clauses: %d/10 clauses present", present)
        return {"clauses": [c.model_dump(mode="json") for c in clauses]}
    except Exception as exc:
        logger.error("node_extract_clauses failed: %s", exc)
        return {"error": f"clause extraction failed: {exc}"}


def node_flag_risks(state: PipelineState) -> Dict[str, Any]:
    """Compare clauses to templates and produce risk findings."""
    if state.get("error"):
        return {}
    try:
        from schemas.clause import ExtractedClause
        clauses = [ExtractedClause(**c) for c in state["clauses"]]
        comparisons = compare_to_templates(clauses)
        findings = flag_risks(clauses, comparisons)
        high_risk = [f for f in findings if f.risk_level == "HIGH"]
        logger.info(
            "node_flag_risks: %d findings, %d HIGH",
            len(findings), len(high_risk),
        )
        return {
            "comparisons": [c.model_dump(mode="json") for c in comparisons],
            "findings": [f.model_dump(mode="json") for f in findings],
        }
    except Exception as exc:
        logger.error("node_flag_risks failed: %s", exc)
        return {"error": f"risk scoring failed: {exc}"}


def node_build_review_items(state: PipelineState) -> Dict[str, Any]:
    """
    Inspect findings and clauses for HITL triggers. Creates ReviewItems.

    HITL triggers:
        missing_critical_clause  -- finding.is_missing=True
        high_risk_finding        -- finding.risk_level="HIGH"
        low_confidence_after_retry -- clause.requires_human_review=True
    """
    if state.get("error"):
        return {}
    try:
        from schemas.clause import ExtractedClause
        from schemas.risk import RiskFinding

        clauses = [ExtractedClause(**c) for c in state["clauses"]]
        findings = [RiskFinding(**f) for f in state["findings"]]
        thread_id = state.get("_thread_id", "unknown-thread")
        doc_hash = state.get("document_hash", "")

        review_items: List[ReviewItem] = []

        for finding in findings:
            trigger = None
            if finding.is_missing:
                trigger = "missing_critical_clause"
            elif finding.risk_level == "HIGH":
                trigger = "high_risk_finding"

            if trigger:
                matching_clause = next(
                    (c for c in clauses if c.clause_type == finding.clause_type), None
                )
                source_text = (
                    matching_clause.extracted_text
                    if matching_clause and matching_clause.extracted_text
                    else f"[Clause '{finding.clause_type}' not found in document]"
                )
                item = ReviewItem(
                    review_id=str(uuid.uuid4()),
                    document_hash=doc_hash,
                    source_chunk_id=matching_clause.source_chunk_id if matching_clause else None,
                    clause_category=finding.clause_type,
                    source_text=source_text,
                    source_page=matching_clause.page_reference if matching_clause else None,
                    trigger_reason=trigger,
                    ai_finding_summary=(
                        f"{'ABSENT' if finding.is_missing else 'PRESENT'} | "
                        f"{finding.clause_type} | {finding.risk_level} | "
                        f"{finding.reason[:80]}"
                    ),
                    confidence_signals={
                        "risk_level": finding.risk_level,
                        "is_missing": finding.is_missing,
                        "precedent_applied": finding.precedent_applied,
                    },
                    risk_level=finding.risk_level,
                    thread_id=thread_id,
                )
                review_items.append(item)
                add_to_review_queue(item)

        # Also handle extractor-level HITL flags not already covered above
        for clause in clauses:
            if clause.requires_human_review:
                already_queued = any(
                    ri.clause_category == clause.clause_type for ri in review_items
                )
                if not already_queued:
                    item = ReviewItem(
                        review_id=str(uuid.uuid4()),
                        document_hash=doc_hash,
                        source_chunk_id=clause.source_chunk_id,
                        clause_category=clause.clause_type,
                        source_text=clause.extracted_text or "[No text available]",
                        source_page=clause.page_reference,
                        trigger_reason="low_confidence_after_retry",
                        ai_finding_summary=(
                            f"{'PRESENT' if clause.is_present else 'ABSENT'} | "
                            f"{clause.clause_type} | conf={clause.confidence:.2f} | "
                            f"{clause.human_review_reason or 'low confidence'}"
                        ),
                        confidence_signals={
                            "final_confidence": clause.confidence,
                            "retry_count": clause.retry_count,
                        },
                        thread_id=thread_id,
                    )
                    review_items.append(item)
                    add_to_review_queue(item)

        needs_review = len(review_items) > 0
        logger.info(
            "node_build_review_items: %d items queued, needs_human_review=%s",
            len(review_items), needs_review,
        )
        return {
            "review_items": [ri.model_dump(mode="json") for ri in review_items],
            "needs_human_review": needs_review,
        }
    except Exception as exc:
        logger.error("node_build_review_items failed: %s", exc)
        return {"error": f"build_review_items failed: {exc}"}


def node_check_human_review(state: PipelineState) -> Dict[str, Any]:
    """
    Boundary node that sits at the interrupt point.

    The graph compiles with interrupt_before=["check_human_review"].
    When needs_human_review=True, execution pauses before this node.
    After resume (invoke(None, config)), this node runs and passes through.

    Note: LangGraph 0.1.19 requires every node to write at least one state
    key. We write needs_human_review=False to signal the interrupt phase is
    complete.
    """
    logger.info("node_check_human_review: running after resume")
    # Must return at least one key -- mark the interrupt phase done
    return {"needs_human_review": False}


def node_apply_decisions(state: PipelineState) -> Dict[str, Any]:
    """
    Summarise which human decisions have been injected into state.

    The actual decision values are injected by resume_after_review() via
    update_state() before resuming. This node reads human_decisions (already
    updated) and logs the count.

    Note: LangGraph 0.1.19 requires every node to write at least one state
    key. We pass through human_decisions unchanged so it is always written.
    """
    decisions = state.get("human_decisions", [])
    logger.info("node_apply_decisions: %d decisions applied", len(decisions))
    # Must write at least one key; pass through human_decisions as-is
    return {"human_decisions": decisions}


def node_finalize(state: PipelineState) -> Dict[str, Any]:
    """
    Assemble the final pipeline output.

    In a complete implementation this would generate the report.
    For Step 8's scope it returns a structured summary dict.
    """
    if state.get("error"):
        return {"final_state": {"error": state["error"], "completed": False}}

    findings = state.get("findings", [])
    clauses = state.get("clauses", [])
    risk_counts: Dict[str, int] = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for f in findings:
        level = f.get("risk_level", "LOW")
        risk_counts[level] = risk_counts.get(level, 0) + 1

    final = {
        "completed": True,
        "document_hash": state.get("document_hash", ""),
        "document_type": (state.get("classification") or {}).get("document_type", "Unknown"),
        "classification_confidence": (state.get("classification") or {}).get("confidence", 0.0),
        "total_chunks": len(state.get("chunks", [])),
        "clauses_present": sum(1 for c in clauses if c.get("is_present")),
        "total_findings": len(findings),
        "risk_counts": risk_counts,
        "review_items_count": len(state.get("review_items", [])),
        "human_decisions_applied": len(state.get("human_decisions", [])),
    }
    logger.info(
        "node_finalize: completed doc_hash=%.16s type=%s risks=HIGH:%d MEDIUM:%d LOW:%d",
        final["document_hash"],
        final["document_type"],
        risk_counts["HIGH"],
        risk_counts["MEDIUM"],
        risk_counts["LOW"],
    )
    return {"final_state": final}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_build_review(state: PipelineState) -> str:
    """
    Conditional edge after build_review_items.
    Routes to check_human_review (interrupt) or directly to finalize.
    """
    if state.get("error"):
        return "finalize"
    return "check_human_review" if state.get("needs_human_review") else "finalize"


# ---------------------------------------------------------------------------
# Graph assembly (module-level singleton)
# ---------------------------------------------------------------------------

_checkpointer = MemorySaver()


def _build_graph():
    g = StateGraph(PipelineState)

    g.add_node("extract", node_extract)
    g.add_node("chunk", node_chunk)
    g.add_node("classify", node_classify)
    g.add_node("extract_clauses", node_extract_clauses)
    g.add_node("flag_risks", node_flag_risks)
    g.add_node("build_review_items", node_build_review_items)
    g.add_node("check_human_review", node_check_human_review)
    g.add_node("apply_decisions", node_apply_decisions)
    g.add_node("finalize", node_finalize)

    g.set_entry_point("extract")
    g.add_edge("extract", "chunk")
    g.add_edge("chunk", "classify")
    g.add_edge("classify", "extract_clauses")
    g.add_edge("extract_clauses", "flag_risks")
    g.add_edge("flag_risks", "build_review_items")

    g.add_conditional_edges(
        "build_review_items",
        route_after_build_review,
        {"check_human_review": "check_human_review", "finalize": "finalize"},
    )

    g.add_edge("check_human_review", "apply_decisions")
    g.add_edge("apply_decisions", "finalize")
    g.add_edge("finalize", END)

    return g.compile(
        checkpointer=_checkpointer,
        interrupt_before=["check_human_review"],
    )


_app = _build_graph()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(pdf_path: str, thread_id: str = None) -> Dict[str, Any]:
    """
    Run the pipeline for a PDF. Returns when completed or interrupted.

    Parameters
    ----------
    pdf_path  : str -- absolute path to the PDF
    thread_id : str or None -- new UUID generated if None

    Returns
    -------
    dict:
        thread_id          : str
        status             : "completed" | "interrupted"
        needs_human_review : bool
        review_items       : list (populated when interrupted)
        final_state        : dict (populated when completed)
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    config = {"configurable": {"thread_id": thread_id}}

    initial: PipelineState = {
        "pdf_path": pdf_path,
        "_thread_id": thread_id,
        "document_hash": "",
        "doc": None,
        "chunks": [],
        "classification": None,
        "clauses": [],
        "comparisons": [],
        "findings": [],
        "review_items": [],
        "needs_human_review": False,
        "human_decisions": [],
        "final_state": None,
        "error": None,
    }

    state = _app.invoke(initial, config)
    snap = _app.get_state(config)
    is_interrupted = bool(snap.next)

    if is_interrupted:
        logger.info("run_pipeline INTERRUPTED thread=%s next=%s", thread_id, snap.next)
        return {
            "thread_id": thread_id,
            "status": "interrupted",
            "needs_human_review": True,
            "review_items": state.get("review_items", []),
            "final_state": None,
        }
    else:
        logger.info("run_pipeline COMPLETED thread=%s", thread_id)
        return {
            "thread_id": thread_id,
            "status": "completed",
            "needs_human_review": False,
            "review_items": [],
            "final_state": state.get("final_state"),
        }


def resume_after_review(thread_id: str, decision: ReviewDecision) -> Dict[str, Any]:
    """
    Apply a human decision and resume an interrupted graph.

    Steps:
    1. Verify the graph is actually paused on this thread_id.
    2. Record the decision (marks ReviewItem resolved).
    3. Apply it to compute the override value.
    4. Inject the outcome into the paused state via update_state.
    5. Resume via invoke(None, config).

    Parameters
    ----------
    thread_id : str -- the paused thread's ID
    decision  : ReviewDecision -- the human's answer

    Returns
    -------
    dict (same shape as run_pipeline)
    """
    config = {"configurable": {"thread_id": thread_id}}
    snap = _app.get_state(config)

    if not snap.next:
        raise ValueError(
            f"Graph thread {thread_id!r} is not paused. "
            f"Cannot resume a graph that is already completed or not started."
        )

    resolved_item = record_review_decision(decision)
    outcome = apply_human_decision(decision, resolved_item)

    logger.info(
        "resume_after_review: thread=%s review_id=%s action=%s discarded=%s",
        thread_id, decision.review_id, decision.action, outcome.get("discarded"),
    )

    # Inject the decision outcome into state so finalize can report it
    current_decisions = list(snap.values.get("human_decisions", []))
    current_decisions.append({
        "review_id": decision.review_id,
        "action": decision.action,
        "discarded": outcome.get("discarded", False),
        "value": outcome.get("value"),
    })
    _app.update_state(config, {"human_decisions": current_decisions})

    # Resume from the interrupt point
    state = _app.invoke(None, config)
    snap_after = _app.get_state(config)
    is_still_interrupted = bool(snap_after.next)

    if is_still_interrupted:
        return {
            "thread_id": thread_id,
            "status": "interrupted",
            "needs_human_review": True,
            "review_items": state.get("review_items", []),
            "final_state": None,
        }
    return {
        "thread_id": thread_id,
        "status": "completed",
        "needs_human_review": False,
        "review_items": [],
        "final_state": state.get("final_state"),
    }


def get_graph_state(thread_id: str) -> Dict[str, Any]:
    """
    Return the current state snapshot for a thread.

    Returns
    -------
    dict:
        values    : current state dict
        next      : tuple of nodes the graph is waiting at
        is_paused : bool
    """
    config = {"configurable": {"thread_id": thread_id}}
    snap = _app.get_state(config)
    return {
        "values": snap.values,
        "next": snap.next,
        "is_paused": bool(snap.next),
    }
