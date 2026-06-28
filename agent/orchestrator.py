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

LangGraph version: 1.2.6
Interrupt mechanism: interrupt() called inside node_check_human_review
    -- the graph pauses mid-node when needs_human_review=True.
    interrupt(value) raises GraphInterrupt on the first call (pausing the
    graph), and returns the resume value when the graph is resumed via
    invoke(Command(resume=<value>), config).  The node re-executes from the
    top on resume; interrupt() returns immediately with the human's payload
    instead of raising again.

    BEFORE (0.1.19):
        compile(interrupt_before=["check_human_review"])
        update_state(config, {human_decisions: ...})
        invoke(None, config)

    NOW (1.2.6):
        compile(checkpointer=...)            -- no interrupt_before
        interrupt({...}) inside the node     -- pauses here
        invoke(Command(resume={...}), config) -- resume, value returned by interrupt()

Checkpointer: InMemorySaver (in-memory, MemorySaver is now an alias)
    -- appropriate for capstone scope. State is lost when the process exits.

HITL, not RLHF -- human decisions override AI output for one run only.
They never update model weights or feed into training data.

PII-safe logging: log thread_id, doc_hash prefix, risk levels, node names.
Never log full source_text, clause contents, or corrected_value payloads.
"""

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from agent.classifier import classify_document
from agent.clause_expander import ClauseExpansionResult, expand_all_clauses
from agent.comparator import compare_to_templates
from agent.extractor import extract_clauses
from agent.human_review import add_to_review_queue, apply_human_decision, record_review_decision
from agent.risk_engine import flag_risks
from extraction.pdf_parser import extract_text_from_pdf
from guardrails.claim_verifier import verify_final_claims as _verify_final_claims
from guardrails.evidence_verifier import verify_evidence
from guardrails.input_validator import validate_input
from guardrails.output_validator import validate_final_output as _validate_final_output
from guardrails.page_verifier import verify_page_reference
from guardrails.prompt_injection import scan_for_prompt_injection
from guardrails.scope_validator import validate_scope
from retrieval.chunking import chunk_document
from reporting.report_generator import assemble_report, render_markdown, render_json
from reporting.executive_summary import generate_executive_summary
from schemas.clause import ExtractedClause
from schemas.risk import RiskFinding
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
    # request_text is the user's intent string (e.g. from API or UI).
    # validate_scope checks this before extraction starts.
    # None / empty string -> scope check passes by default (no intent detected).
    request_text: Optional[str]
    document_hash: str
    doc: Optional[Dict]
    chunks: List[Dict]
    classification: Optional[Dict]
    clauses: List[Dict]
    comparisons: List[Dict]
    findings: List[Dict]
    # guardrail_results accumulates every GuardrailResult from all six guardrail
    # nodes so the eventual report can include "what guardrails ran and what they found".
    guardrail_results: List[Dict]
    # evidence_results holds EvidenceVerificationResult + PageReferenceVerificationResult
    # dicts per present clause, produced by node_verify_clauses.
    evidence_results: List[Dict]
    # claim_verification_results holds ClaimVerificationResult dicts per clause,
    # produced by node_verify_final_claims after flag_risks.
    claim_verification_results: List[Dict]
    expansions: List[Dict]          # serialised ClauseExpansionResult list (one per clause)
    review_items: List[Dict]
    needs_human_review: bool
    human_decisions: List[Dict]
    generated_report: Optional[Dict]   # serialised LegalDocumentReport (Step 10)
    final_output_validation: Optional[Dict]  # serialised FinalOutputValidationResult (Step 11)
    processing_notes: Optional[str]     # free-text note passed into assemble_report
    final_state: Optional[Dict]
    error: Optional[str]


# ---------------------------------------------------------------------------
# Helper: accumulate a guardrail result into state
# ---------------------------------------------------------------------------

def _append_guardrail(state: PipelineState, result_dict: Dict) -> List[Dict]:
    """Return an updated guardrail_results list with result_dict appended."""
    existing = list(state.get("guardrail_results") or [])
    existing.append(result_dict)
    return existing


# ---------------------------------------------------------------------------
# Guardrail nodes (run BEFORE core pipeline stages)
# ---------------------------------------------------------------------------

def node_validate_input(state: PipelineState) -> Dict[str, Any]:
    """
    Stage 1 guardrail: validate the input file before any extraction.

    Checks: file exists, is a PDF, is within size/page-count limits, is not
    a duplicate of a previously processed document.

    If severity='blocking': sets state['error'] so all downstream nodes
    skip immediately (they all check `if state.get('error'): return {}`).
    The graph then routes directly to 'finalize', which produces a
    completed=False result with the guardrail reason.

    If severity='info' (pass): appends result to guardrail_results and
    proceeds to validate_scope.
    """
    result = validate_input(state["pdf_path"])
    result_dict = {
        "guardrail_name": result.guardrail_name,
        "passed": result.passed,
        "severity": result.severity,
        "reason": result.reason,
    }
    updated_guardrails = _append_guardrail(state, result_dict)

    if not result.passed and result.severity == "blocking":
        logger.warning(
            "node_validate_input: BLOCKED file=%s reason=%s",
            state["pdf_path"].rsplit("/", 1)[-1].rsplit("\\", 1)[-1],
            result.reason,
        )
        return {
            "guardrail_results": updated_guardrails,
            "error": f"[validate_input] {result.reason}",
        }

    logger.info("node_validate_input: passed file=%s",
                state["pdf_path"].rsplit("/", 1)[-1].rsplit("\\", 1)[-1])
    return {"guardrail_results": updated_guardrails}


def node_validate_scope(state: PipelineState) -> Dict[str, Any]:
    """
    Stage 2 guardrail: check the user's request is within scope.

    Blocks requests that ask for autonomous contract approval, legal advice,
    contract execution, or other out-of-scope actions.

    request_text is optional -- if None/empty, scope check passes by default
    (the system proceeds with standard document analysis behavior).
    """
    if state.get("error"):
        return {}

    request_text = state.get("request_text") or ""
    result = validate_scope(request_text)
    result_dict = {
        "guardrail_name": result.guardrail_name,
        "passed": result.passed,
        "severity": result.severity,
        "reason": result.reason,
        "detected_intent": result.detected_intent,
    }
    updated_guardrails = _append_guardrail(state, result_dict)

    if not result.passed and result.severity == "blocking":
        logger.warning(
            "node_validate_scope: BLOCKED intent=%s", result.detected_intent
        )
        return {
            "guardrail_results": updated_guardrails,
            "error": f"[validate_scope] {result.reason}",
        }

    logger.info("node_validate_scope: passed (intent=None or in-scope)")
    return {"guardrail_results": updated_guardrails}


def node_scan_prompt_injection(state: PipelineState) -> Dict[str, Any]:
    """
    Stage 3 guardrail: scan extracted document text for prompt injection.

    Runs on the full extracted text AFTER extraction and BEFORE classification.
    Rationale: scanning full_text catches injection patterns spread across
    page boundaries; per-page scanning would miss multi-page attacks.

    blocking -> sets error; graph routes to finalize (classification never runs).
    warning  -> logs + records in guardrail_results; processing continues.
    """
    if state.get("error"):
        return {}

    doc_dict = state.get("doc")
    if not doc_dict:
        # No document extracted -- extraction must have failed; error already set
        return {}

    full_text = doc_dict.get("full_text", "")
    result = scan_for_prompt_injection(full_text)
    result_dict = {
        "guardrail_name": result.guardrail_name,
        "passed": result.passed,
        "severity": result.severity,
        "reason": result.reason,
        # technique_categories only -- flagged_segments are NOT logged per design
        "technique_categories": result.technique_categories,
    }
    updated_guardrails = _append_guardrail(state, result_dict)

    if not result.passed and result.severity == "blocking":
        logger.warning(
            "node_scan_prompt_injection: BLOCKED categories=%s",
            result.technique_categories,
        )
        return {
            "guardrail_results": updated_guardrails,
            "error": (
                f"[scan_prompt_injection] Prompt injection detected: "
                f"{', '.join(result.technique_categories)}"
            ),
        }

    if not result.passed and result.severity == "warning":
        logger.warning(
            "node_scan_prompt_injection: WARNING (non-blocking) categories=%s",
            result.technique_categories,
        )

    logger.info("node_scan_prompt_injection: completed severity=%s", result.severity)
    return {"guardrail_results": updated_guardrails}


def node_verify_clauses(state: PipelineState) -> Dict[str, Any]:
    """
    Stage 4 guardrail: run evidence + page verification on all present clauses.

    Runs AFTER extract_clauses and BEFORE flag_risks. For every clause where
    is_present=True and extracted_text is non-empty:
      - verify_evidence: does the extracted text actually appear in the source
        page (exact or fuzzy match)?
      - verify_page_reference: does the cited page number exist in the document?

    Absent clauses are skipped -- their risk is structural (missing clause),
    not evidence-dependent.

    This node does NOT block the graph. It records results in evidence_results
    so that node_verify_final_claims (after flag_risks) can escalate HIGH/MEDIUM
    clauses whose evidence cannot be confirmed.

    Gap noted: flag_risks currently has no awareness of evidence_verified status.
    That is a future task -- see node_verify_final_claims for the escalation path.
    """
    if state.get("error"):
        return {}

    clauses_raw = state.get("clauses", [])
    doc_dict = state.get("doc")
    if not doc_dict or not clauses_raw:
        return {"evidence_results": []}

    from schemas.clause import ExtractedClause
    from schemas.document import DocumentExtraction

    try:
        doc = DocumentExtraction(**doc_dict)
        clauses = [ExtractedClause(**c) for c in clauses_raw]
    except Exception as exc:
        logger.error("node_verify_clauses: deserialization failed: %s", exc)
        return {"evidence_results": []}

    evidence_results = []
    for clause in clauses:
        if not clause.is_present or not clause.extracted_text:
            # Absent clauses have no extracted text to verify
            continue

        # Evidence check: does extracted text appear in the cited page?
        page_num = clause.page_reference
        source_text = ""
        if page_num is not None:
            page = doc.get_page(page_num)
            source_text = page.text if page else doc.full_text
        else:
            source_text = doc.full_text

        ev_result = verify_evidence(
            clause.extracted_text, source_text, source_page=page_num
        )

        # Page reference check: does the cited page number exist in the doc?
        page_result = None
        if page_num is not None:
            page_result = verify_page_reference(
                cited_page=page_num,
                document=doc,
                extracted_text=clause.extracted_text,
            )

        entry = {
            "clause_type": clause.clause_type,
            "evidence_found": ev_result.found_in_source,
            "evidence_match_type": ev_result.match_type,
            "evidence_match_score": ev_result.match_score,
            "page_cited": page_num,
            "page_exists": page_result.page_exists_in_document if page_result else None,
            "text_near_page": page_result.text_found_near_cited_page if page_result else None,
        }
        evidence_results.append(entry)
        logger.debug(
            "node_verify_clauses: %s evidence=%s page_ok=%s",
            clause.clause_type,
            ev_result.match_type,
            page_result.page_exists_in_document if page_result else "n/a",
        )

    logger.info(
        "node_verify_clauses: %d present clauses verified; %d not found in source",
        len(evidence_results),
        sum(1 for e in evidence_results if not e["evidence_found"]),
    )
    return {"evidence_results": evidence_results}


def node_verify_final_claims(state: PipelineState) -> Dict[str, Any]:
    """
    Stage 5 guardrail: verify final claims AFTER flag_risks, BEFORE HITL check.

    Calls verify_final_claims() from guardrails/claim_verifier.py on the full
    set of clauses and risk findings. For each clause whose extracted_text
    cannot be confirmed in the source document AND whose risk_level is HIGH or
    MEDIUM, the verifier:
      - Sets action_taken='escalated_to_human_review'
      - Calls add_to_review_queue() directly (same queue as build_review_items)

    This means escalations from claim_verifier surface in the same unified
    HITL queue that check_human_review reads from -- not a separate system.

    This node does NOT block the graph regardless of verification outcomes.
    Escalated items join the queue and check_human_review handles them.
    """
    if state.get("error"):
        return {}

    clauses_raw = state.get("clauses", [])
    findings_raw = state.get("findings", [])
    doc_dict = state.get("doc")
    if not doc_dict or not clauses_raw:
        return {"claim_verification_results": []}

    from schemas.clause import ExtractedClause
    from schemas.document import DocumentExtraction
    from schemas.risk import RiskFinding

    try:
        doc = DocumentExtraction(**doc_dict)
        clauses = [ExtractedClause(**c) for c in clauses_raw]
        findings = [RiskFinding(**f) for f in findings_raw]
    except Exception as exc:
        logger.error("node_verify_final_claims: deserialization failed: %s", exc)
        return {"claim_verification_results": []}

    try:
        cvr_list = _verify_final_claims(
            clauses, findings, doc, thread_id=state.get("_thread_id")
        )
        n_escalated = sum(
            1 for r in cvr_list if r.action_taken == "escalated_to_human_review"
        )
        logger.info(
            "node_verify_final_claims: %d results, %d escalated to HITL queue",
            len(cvr_list), n_escalated,
        )
        return {
            "claim_verification_results": [
                {
                    "claim_text": r.claim_text,
                    "has_supporting_evidence": r.has_supporting_evidence,
                    "evidence_source": r.evidence_source,
                    "action_taken": r.action_taken,
                }
                for r in cvr_list
            ]
        }
    except Exception as exc:
        logger.error("node_verify_final_claims failed: %s", exc)
        return {"claim_verification_results": []}


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


def node_expand_clause_boundaries(state: PipelineState) -> Dict[str, Any]:
    """
    Clause boundary expansion node -- runs after extract_clauses, before verify_clauses.

    Problem solved:
        The extractor may anchor on the FIRST section of a multi-section clause
        (e.g. Section 1: definition of Protected Information) and miss subsequent
        sections (e.g. Section 2: Director's Obligations) that are part of the
        same logical clause group.  When the comparator then measures the snippet
        against the full template, it produces a false "major deviation" → HIGH
        risk finding even though the obligations ARE present in the document.

    Fix:
        For each present clause, look forward from the source chunk and include
        adjacent chunks that belong to the same legal concept.  The merged text
        is stored in ClauseExpansionResult and passed to compare_to_templates()
        in node_flag_risks so the comparator sees the full clause, not the snippet.

    This node never raises; errors are caught and logged.  On failure the state
    key `expansions` is set to an empty list and compare_to_templates falls back
    to using clause.extracted_text directly.
    """
    if state.get("error"):
        return {}
    try:
        from schemas.clause import ExtractedClause
        from schemas.chunk import DocumentChunk

        clauses = [ExtractedClause(**c) for c in state.get("clauses", [])]
        chunks = [DocumentChunk(**c) for c in state.get("chunks", [])]

        if not clauses or not chunks:
            return {"expansions": []}

        expansions_dict = expand_all_clauses(clauses, chunks)

        triggered = sum(1 for e in expansions_dict.values() if e.expansion_triggered)
        logger.info(
            "node_expand_clause_boundaries: %d clauses checked, %d expanded",
            len(expansions_dict),
            triggered,
        )

        return {
            "expansions": [e.model_dump(mode="json") for e in expansions_dict.values()]
        }
    except Exception as exc:
        logger.error("node_expand_clause_boundaries failed: %s", exc)
        return {"expansions": []}


def node_flag_risks(state: PipelineState) -> Dict[str, Any]:
    """Compare clauses to templates and produce risk findings."""
    if state.get("error"):
        return {}
    try:
        from schemas.clause import ExtractedClause
        clauses = [ExtractedClause(**c) for c in state["clauses"]]

        # Rebuild expansions dict so the comparator can use expanded text
        expansions_dict: Optional[Dict[str, ClauseExpansionResult]] = None
        raw_expansions = state.get("expansions", [])
        if raw_expansions:
            try:
                expansions_dict = {
                    e["clause_type"]: ClauseExpansionResult(**e)
                    for e in raw_expansions
                }
            except Exception as exc:
                logger.warning("node_flag_risks: could not deserialise expansions: %s", exc)

        comparisons = compare_to_templates(clauses, expansions=expansions_dict)
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


def _get_adjacent_chunk_texts(
    chunks: List[Dict],
    target_chunk_id: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """
    Return (previous_chunk_text, next_chunk_text) for the chunk with target_chunk_id.
    Uses get_local_context() from tot_reasoner -- not a second implementation.
    Returns (None, None) if chunk_id is absent or chunks list is empty.
    """
    if not target_chunk_id or not chunks:
        return None, None
    try:
        from agent.tot_reasoner import get_local_context
        from schemas.chunk import DocumentChunk
        doc_chunks = [DocumentChunk(**c) for c in chunks]
        # Find the target chunk's chunk_index
        target = next((c for c in doc_chunks if c.chunk_id == target_chunk_id), None)
        if target is None:
            return None, None
        context = get_local_context(doc_chunks, target.chunk_index, neighbor_window=1)
        prev_text = next_text = None
        for c in context:
            if c.chunk_index < target.chunk_index:
                prev_text = c.text
            elif c.chunk_index > target.chunk_index:
                next_text = c.text
        return prev_text, next_text
    except Exception:
        return None, None


def _load_template_text(template_path: Optional[str]) -> Optional[str]:
    """Load template clause text from disk given the path stored on ClauseComparison."""
    if not template_path:
        return None
    try:
        p = Path(template_path)
        if p.exists():
            return p.read_text(encoding="utf-8").strip() or None
        return None
    except Exception:
        return None


def node_build_review_items(state: PipelineState) -> Dict[str, Any]:
    """
    Inspect findings and clauses for HITL triggers. Creates ReviewItems.

    HITL triggers:
        missing_critical_clause  -- finding.is_missing=True
        high_risk_finding        -- finding.risk_level="HIGH"
        low_confidence_after_retry -- clause.requires_human_review=True

    Fact/deviation/rationale splitting approach:
        fact_found      = "{category} clause is {absent/present}." -- simple structural fact
        deviation_found = finding.deviation_summary (comparator's free-text deviation note)
        risk_rationale  = finding.reason (risk_engine's full reason string)
        These are NOT NLP-split from a single blended string -- they come from
        distinct upstream fields. If upstream had only one blended string we would
        populate risk_rationale and leave fact_found/deviation_found as structured
        summaries rather than attempt a fragile NLP split.
    """
    if state.get("error"):
        return {}
    try:
        from schemas.clause import ExtractedClause
        from schemas.risk import ClauseComparison, RiskFinding
        from pathlib import Path

        clauses = [ExtractedClause(**c) for c in state["clauses"]]
        comparisons = [ClauseComparison(**c) for c in state.get("comparisons", [])]
        findings = [RiskFinding(**f) for f in state["findings"]]
        chunks_raw = state.get("chunks", [])
        thread_id = state.get("_thread_id", "unknown-thread")
        doc_hash = state.get("document_hash", "")
        doc_name = state.get("doc", {}).get("file_name", "unknown.pdf")
        doc_type = None
        if state.get("classification"):
            doc_type = state["classification"].get("document_type")

        # Build comparison lookup by clause_type
        comparison_by_type = {c.clause_type: c for c in comparisons}

        # Build expansion lookup by clause_type
        expansions_by_type: Dict[str, ClauseExpansionResult] = {}
        for raw_exp in state.get("expansions", []):
            try:
                exp = ClauseExpansionResult(**raw_exp)
                expansions_by_type[exp.clause_type] = exp
            except Exception:
                pass

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
                comparison = comparison_by_type.get(finding.clause_type)
                template_text = _load_template_text(
                    comparison.template_path if comparison else None
                )
                prev_text, next_text = _get_adjacent_chunk_texts(
                    chunks_raw,
                    matching_clause.source_chunk_id if matching_clause else None,
                )
                # Structured fact/deviation/rationale (see docstring for approach)
                present_str = "absent" if finding.is_missing else "present"
                fact_found = f"{finding.clause_type} clause is {present_str} in the document."
                deviation_found = finding.deviation_summary if (
                    comparison and comparison.deviation_severity != "none"
                ) else None
                risk_rationale = finding.reason

                # Expansion data for this clause (populated by expand_clause_boundaries)
                exp = expansions_by_type.get(finding.clause_type)

                item = ReviewItem(
                    review_id=str(uuid.uuid4()),
                    document_hash=doc_hash,
                    document_name=doc_name,
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
                    # Reviewer-context fields
                    template_clause_text=template_text,
                    missing_elements=[],  # ClauseComparison.deviation_summary is free-text; no structured list
                    extra_risky_language=[],  # same -- no structured list available upstream
                    fact_found=fact_found,
                    deviation_found=deviation_found,
                    risk_rationale=risk_rationale,
                    extraction_confidence=matching_clause.confidence if matching_clause else None,
                    llm_confidence=matching_clause.confidence if matching_clause else None,
                    previous_chunk_text=prev_text,
                    next_chunk_text=next_text,
                    section_heading=None,  # DocumentChunk has no section_heading field
                    document_type_context=doc_type,
                    # Clause expansion fields
                    expanded_clause_text=exp.expanded_text if (exp and exp.expansion_triggered) else None,
                    source_chunks_used=exp.source_chunk_ids if exp else [],
                    expansion_triggered=exp.expansion_triggered if exp else False,
                    expansion_boundary_reason=exp.boundary_reason if exp else None,
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
                    prev_text, next_text = _get_adjacent_chunk_texts(
                        chunks_raw, clause.source_chunk_id
                    )
                    present_str = "present" if clause.is_present else "absent"
                    item = ReviewItem(
                        review_id=str(uuid.uuid4()),
                        document_hash=doc_hash,
                        document_name=doc_name,
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
                        # New reviewer-context fields
                        fact_found=f"{clause.clause_type} clause is {present_str} in the document.",
                        risk_rationale=clause.human_review_reason or "Low extraction confidence after retry.",
                        extraction_confidence=clause.confidence,
                        llm_confidence=clause.confidence,
                        previous_chunk_text=prev_text,
                        next_chunk_text=next_text,
                        section_heading=None,
                        document_type_context=doc_type,
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
    HITL interrupt node -- pauses execution and waits for a human decision.

    On the FIRST execution (when the graph is about to pause):
        interrupt(payload) raises GraphInterrupt, which the LangGraph runtime
        catches and surfaces to the caller as {'__interrupt__': ...}.
        The graph state is checkpointed at this point.

    On RESUME (when invoke(Command(resume=decisions), config) is called):
        The node re-executes from the top. interrupt() this time returns
        the value passed as Command(resume=...) instead of raising.
        That value is the list of applied human_decisions dicts that
        resume_after_review() computed before calling invoke(Command(...)).

    This is the 1.2.6 idiom: interrupt() inside the node replaces the
    old interrupt_before=[...] compile option + update_state() + invoke(None).
    """
    logger.info("node_check_human_review: reached (first call pauses, resume returns decisions)")

    # Pass review context to the human reviewer as the interrupt payload
    payload = {
        "review_items": state.get("review_items", []),
        "document_hash": state.get("document_hash", ""),
        "thread_id": state.get("_thread_id", ""),
    }

    # On first call: raises GraphInterrupt (graph pauses here).
    # On resume:     returns the value provided in Command(resume=...).
    human_decisions = interrupt(payload)

    logger.info(
        "node_check_human_review: resumed with %d decisions",
        len(human_decisions) if isinstance(human_decisions, list) else 0,
    )
    return {
        "human_decisions": human_decisions if isinstance(human_decisions, list) else [],
        "needs_human_review": False,
    }


def node_apply_decisions(state: PipelineState) -> Dict[str, Any]:
    """
    Log and pass through the decisions injected by node_check_human_review.

    In 1.2.6 the decisions arrive directly via the interrupt() resume value,
    stored in state by node_check_human_review. This node confirms the count.
    """
    decisions = state.get("human_decisions", [])
    logger.info("node_apply_decisions: %d decisions applied", len(decisions))
    return {"human_decisions": decisions}


def node_generate_report(state: PipelineState) -> Dict[str, Any]:
    """
    Step 10: assemble and render the LegalDocumentReport.

    Positioned after apply_decisions (so human decisions are in state) and
    before finalize. Calls assemble_report() (deterministic) then
    generate_executive_summary() (one constrained LLM call, with fallback).

    The rendered markdown is written to data/processed/<document_hash[:16]>.md
    so it persists across process restarts. The full report object is also
    stored in state as generated_report (serialised dict) for inspection.

    On any error the node logs and returns without a report, so finalize
    can still complete the run.
    """
    if state.get("error"):
        return {}

    try:
        doc_dict = state.get("doc") or {}
        doc_name = doc_dict.get("file_name", "unknown.pdf")
        doc_hash = state.get("document_hash", "")
        doc_type = (state.get("classification") or {}).get("document_type", "Unknown")
        doc_conf = (state.get("classification") or {}).get("confidence", 0.0)
        total_pages = doc_dict.get("total_pages", 0)

        # Deserialise clause and risk objects from state dicts
        raw_clauses = state.get("clauses", [])
        raw_findings = state.get("findings", [])

        clauses = [ExtractedClause(**c) for c in raw_clauses if c]
        findings = [RiskFinding(**f) for f in raw_findings if f]

        evidence_results = state.get("evidence_results", [])
        guardrail_results = state.get("guardrail_results", [])

        # Summarise human decisions for report (no corrected_value -- stays in state)
        human_decisions_summary = []
        for d in state.get("human_decisions", []):
            human_decisions_summary.append({
                "action": d.get("action"),
                "clause_category": d.get("review_item", {}).get("clause_category") if isinstance(d.get("review_item"), dict) else None,
                "reviewer_note": d.get("reviewer_note"),
                "decided_at": str(d.get("decided_at", "")),
                "discarded": d.get("discarded", False),
                "reason": d.get("reason", ""),
                "corrected_risk_level": d.get("corrected_risk_level"),
                "corrected_summary": d.get("corrected_summary"),
            })

        processing_notes = state.get("processing_notes") or None

        # Assemble without executive summary first (placeholder)
        report = assemble_report(
            document_name=doc_name,
            document_hash=doc_hash,
            document_type=doc_type,
            classification_confidence=doc_conf,
            total_pages=total_pages,
            clauses=clauses,
            risk_findings=findings,
            guardrail_results=guardrail_results,
            evidence_results=evidence_results,
            human_review_decisions=human_decisions_summary,
            executive_summary="",  # filled in next
            processing_notes=processing_notes,
        )

        # Generate executive summary (LLM call with fallback)
        exec_summary = generate_executive_summary(report)

        # Re-assemble with the real executive summary
        report = assemble_report(
            document_name=doc_name,
            document_hash=doc_hash,
            document_type=doc_type,
            classification_confidence=doc_conf,
            total_pages=total_pages,
            clauses=clauses,
            risk_findings=findings,
            guardrail_results=guardrail_results,
            evidence_results=evidence_results,
            human_review_decisions=human_decisions_summary,
            executive_summary=exec_summary,
            processing_notes=processing_notes,
        )

        md = render_markdown(report)

        # Persist markdown to data/processed/
        import os
        from pathlib import Path
        processed_dir = Path(__file__).parent.parent / "data" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        hash_prefix = doc_hash[:16] if doc_hash else "unknown"
        md_path = processed_dir / f"{hash_prefix}.md"
        md_path.write_text(md, encoding="utf-8")

        logger.info(
            "node_generate_report: report written to %s (%d chars)",
            md_path, len(md),
        )

        return {"generated_report": report.model_dump(mode="json")}

    except Exception as e:
        logger.error("node_generate_report failed: %s", e, exc_info=True)
        return {}


def node_validate_final_output(state: PipelineState) -> Dict[str, Any]:
    """
    Step 11: validate the assembled report as a whole before the pipeline finalizes.

    Positioned after node_generate_report and before node_finalize. Runs
    _validate_final_output() from guardrails/output_validator.py which checks:
      1. Disclaimer presence (auto-fixable)
      2. Internal count consistency (escalated if wrong)
      3. Schema completeness (escalated if incomplete)
      4. Guardrail-to-report disconnect (escalated -- most safety-critical)

    RE-PAUSE vs LOG-ONLY for escalations:
    This node does NOT call interrupt() to pause the graph a second time, even
    when it creates escalation ReviewItems. Reasoning:
      - The interrupt/resume cycle has already completed at node_check_human_review.
      - Re-pausing here would require a second resume_after_review() call from the
        caller, complicating demo_hitl.py and the public API significantly.
      - The assembled report is already persisted to disk (data/processed/).
      - Escalations at this stage represent structural issues that a human can
        act on asynchronously -- they appear in the HITL queue for the next
        reviewer session, and the caller sees validation_passed=False in final_state.
    This is the deliberate, documented decision. If requirements change to require
    a synchronous second review cycle, add a second interrupt() here and update
    run_pipeline/resume_after_review accordingly.

    Auto-fix (disclaimer re-injection) IS applied: the corrected report dict
    is stored back into state.generated_report so downstream code (and any
    re-render in node_finalize) sees the fixed version.
    """
    if state.get("error"):
        return {}

    report_dict = state.get("generated_report")
    if not report_dict:
        # No report was generated (upstream failure). Nothing to validate.
        logger.warning("node_validate_final_output: no generated_report in state -- skipping")
        return {}

    from schemas.report import LegalDocumentReport
    try:
        report = LegalDocumentReport(**report_dict)
    except Exception as exc:
        logger.error("node_validate_final_output: cannot deserialise report: %s", exc)
        return {}

    try:
        fixed_report, val_result = _validate_final_output(
            report, thread_id=state.get("_thread_id")
        )
        logger.info(
            "node_validate_final_output: passed=%s checks_failed=%s "
            "auto_fixes=%d escalations=%d",
            val_result.passed,
            val_result.checks_failed,
            len(val_result.auto_fixes_applied),
            len(val_result.escalations_created),
        )
        return {
            "generated_report": fixed_report.model_dump(mode="json"),
            "final_output_validation": val_result.model_dump(mode="json"),
        }
    except Exception as exc:
        logger.error("node_validate_final_output failed: %s", exc, exc_info=True)
        return {}


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

    val = state.get("final_output_validation") or {}
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
        # Step 11 result summary
        "validation_passed": val.get("passed", None),
        "validation_auto_fixes": val.get("auto_fixes_applied", []),
        "validation_escalations": val.get("escalations_created", []),
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

def route_after_validate_input(state: PipelineState) -> str:
    """Route to validate_scope or finalize if input validation blocked."""
    return "finalize" if state.get("error") else "validate_scope"


def route_after_validate_scope(state: PipelineState) -> str:
    """Route to extract or finalize if scope validation blocked."""
    return "finalize" if state.get("error") else "extract"


def route_after_scan_injection(state: PipelineState) -> str:
    """Route to chunk or finalize if prompt injection was detected (blocking)."""
    return "finalize" if state.get("error") else "chunk"


def route_after_build_review(state: PipelineState) -> str:
    """
    Conditional edge after build_review_items.
    Routes to check_human_review (interrupt) or generate_report (no HITL needed).
    Errors route to finalize directly (skipping report generation).
    """
    if state.get("error"):
        return "finalize"
    return "check_human_review" if state.get("needs_human_review") else "generate_report"


# ---------------------------------------------------------------------------
# Graph assembly (module-level singleton)
# ---------------------------------------------------------------------------

_checkpointer = InMemorySaver()


def _build_graph():
    g = StateGraph(PipelineState)

    # -- Guardrail nodes (Step 9 wiring) --
    g.add_node("validate_input",         node_validate_input)
    g.add_node("validate_scope",         node_validate_scope)
    g.add_node("scan_prompt_injection",  node_scan_prompt_injection)
    g.add_node("verify_clauses",         node_verify_clauses)
    g.add_node("verify_final_claims",    node_verify_final_claims)

    # -- Core pipeline nodes (Step 8) --
    g.add_node("extract",            node_extract)
    g.add_node("chunk",              node_chunk)
    g.add_node("classify",           node_classify)
    g.add_node("extract_clauses",           node_extract_clauses)
    g.add_node("expand_clause_boundaries",  node_expand_clause_boundaries)
    g.add_node("flag_risks",                node_flag_risks)
    g.add_node("build_review_items", node_build_review_items)
    g.add_node("check_human_review", node_check_human_review)
    g.add_node("apply_decisions",          node_apply_decisions)
    g.add_node("generate_report",          node_generate_report)
    g.add_node("validate_final_output",    node_validate_final_output)
    g.add_node("finalize",                 node_finalize)

    # Entry point is now validate_input (was extract)
    g.set_entry_point("validate_input")

    # Pre-extraction guardrails -- blocking routes to finalize
    g.add_conditional_edges(
        "validate_input",
        route_after_validate_input,
        {"validate_scope": "validate_scope", "finalize": "finalize"},
    )
    g.add_conditional_edges(
        "validate_scope",
        route_after_validate_scope,
        {"extract": "extract", "finalize": "finalize"},
    )

    # Extraction
    g.add_edge("extract", "scan_prompt_injection")

    # Post-extraction injection scan -- blocking routes to finalize
    g.add_conditional_edges(
        "scan_prompt_injection",
        route_after_scan_injection,
        {"chunk": "chunk", "finalize": "finalize"},
    )

    # Core pipeline (unchanged)
    g.add_edge("chunk",          "classify")
    g.add_edge("classify",       "extract_clauses")

    # Expand clause boundaries after extraction, then evidence-verify, then risk-score
    g.add_edge("extract_clauses",          "expand_clause_boundaries")
    g.add_edge("expand_clause_boundaries", "verify_clauses")
    g.add_edge("verify_clauses",           "flag_risks")

    # Final claim verification after risk scoring, before HITL check
    g.add_edge("flag_risks",           "verify_final_claims")
    g.add_edge("verify_final_claims",  "build_review_items")

    g.add_conditional_edges(
        "build_review_items",
        route_after_build_review,
        {"check_human_review": "check_human_review", "generate_report": "generate_report", "finalize": "finalize"},
    )

    g.add_edge("check_human_review",      "apply_decisions")
    g.add_edge("apply_decisions",         "generate_report")
    g.add_edge("generate_report",         "validate_final_output")
    g.add_edge("validate_final_output",   "finalize")
    g.add_edge("finalize",                END)

    # 1.2.6: no interrupt_before -- the interrupt() call inside
    # node_check_human_review handles pausing directly.
    return g.compile(checkpointer=_checkpointer)


_app = _build_graph()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(
    pdf_path: str,
    thread_id: str = None,
    request_text: str = None,
    processing_notes: str = None,
) -> Dict[str, Any]:
    """
    Run the pipeline for a PDF. Returns when completed or interrupted.

    Parameters
    ----------
    pdf_path     : str -- absolute path to the PDF
    thread_id    : str or None -- new UUID generated if None
    request_text : str or None -- user's intent/request, checked by validate_scope.
                   Pass None or empty string to skip scope checking (default analysis).

    Returns
    -------
    dict:
        thread_id          : str
        status             : "completed" | "interrupted" | "blocked"
        needs_human_review : bool
        review_items       : list (populated when interrupted)
        final_state        : dict (populated when completed or blocked)
        guardrail_results  : list (all guardrail results from this run)
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    config = {"configurable": {"thread_id": thread_id}}

    initial: PipelineState = {
        "pdf_path": pdf_path,
        "_thread_id": thread_id,
        "request_text": request_text,
        "document_hash": "",
        "doc": None,
        "chunks": [],
        "classification": None,
        "clauses": [],
        "comparisons": [],
        "findings": [],
        "expansions": [],
        "guardrail_results": [],
        "evidence_results": [],
        "claim_verification_results": [],
        "review_items": [],
        "needs_human_review": False,
        "human_decisions": [],
        "generated_report": None,
        "final_output_validation": None,
        "processing_notes": processing_notes,
        "final_state": None,
        "error": None,
    }

    state = _app.invoke(initial, config)
    snap = _app.get_state(config)
    # In 1.2.6, snap.next is non-empty when paused at an interrupt() call.
    # The state dict returned by invoke() reflects the last successfully
    # written state before the interrupt raised -- so review_items are there.
    is_interrupted = bool(snap.next)

    guardrail_results = snap.values.get("guardrail_results", [])

    if is_interrupted:
        logger.info("run_pipeline INTERRUPTED thread=%s next=%s", thread_id, snap.next)
        review_items = snap.values.get("review_items", [])
        return {
            "thread_id": thread_id,
            "status": "interrupted",
            "needs_human_review": True,
            "review_items": review_items,
            "final_state": None,
            "guardrail_results": guardrail_results,
            "generated_report": snap.values.get("generated_report"),
        }

    # Detect a guardrail-blocked run: finalize set completed=False due to error
    final = state.get("final_state") or {}
    if not final.get("completed", True) and state.get("error"):
        logger.warning("run_pipeline BLOCKED thread=%s error=%s", thread_id, state.get("error"))
        return {
            "thread_id": thread_id,
            "status": "blocked",
            "needs_human_review": False,
            "review_items": [],
            "final_state": final,
            "guardrail_results": guardrail_results,
            "generated_report": None,
        }

    logger.info("run_pipeline COMPLETED thread=%s", thread_id)
    return {
        "thread_id": thread_id,
        "status": "completed",
        "needs_human_review": False,
        "review_items": [],
        "final_state": final,
        "guardrail_results": guardrail_results,
        "generated_report": snap.values.get("generated_report"),
    }


def resume_after_review(thread_id: str, decision: ReviewDecision) -> Dict[str, Any]:
    """
    Apply a human decision and resume an interrupted graph.

    Steps:
    1. Verify the graph is actually paused on this thread_id.
    2. Record the decision (marks ReviewItem resolved in the queue).
    3. Apply it to compute the override value (approve/correct/reject outcome).
    4. Resume via invoke(Command(resume=decisions), config).
       -- Command(resume=...) is the 1.2.6 idiom: the value is returned by
          interrupt() when the node re-executes, replacing the old pattern of
          update_state() + invoke(None, ...).

    BEFORE (0.1.19):
        _app.update_state(config, {"human_decisions": decisions})
        _app.invoke(None, config)

    NOW (1.2.6):
        _app.invoke(Command(resume=decisions), config)
        -- node_check_human_review re-executes; interrupt() returns decisions
           instead of raising; the node stores them in state and continues.

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

    # Build the decisions list to pass as the resume value.
    # This is returned by interrupt() inside node_check_human_review on re-execution.
    current_decisions = list(snap.values.get("human_decisions", []))
    current_decisions.append({
        "review_id": decision.review_id,
        "action": decision.action,
        "discarded": outcome.get("discarded", False),
        "value": outcome.get("value"),
    })

    # 1.2.6 resume: Command(resume=...) carries the payload into interrupt()
    state = _app.invoke(Command(resume=current_decisions), config)
    snap_after = _app.get_state(config)
    is_still_interrupted = bool(snap_after.next)

    if is_still_interrupted:
        return {
            "thread_id": thread_id,
            "status": "interrupted",
            "needs_human_review": True,
            "review_items": snap_after.values.get("review_items", []),
            "final_state": None,
            "generated_report": snap_after.values.get("generated_report"),
        }
    return {
        "thread_id": thread_id,
        "status": "completed",
        "needs_human_review": False,
        "review_items": [],
        "final_state": state.get("final_state"),
        "generated_report": snap_after.values.get("generated_report"),
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
