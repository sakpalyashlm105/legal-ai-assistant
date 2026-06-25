"""
agent/extractor.py
------------------
Clause extraction node for the Legal AI Assistant.

What this file does:
    Receives the top-3 retrieved document chunks (already ranked by the FAISS
    vector store + LLM reranker) and asks GPT-4o-mini to find and extract all
    10 approved clause types. Returns a list of 10 ExtractedClause objects --
    one per category, with is_present=False for clauses that are absent.

Why extract all 10 clauses in one call?
    Sending one prompt per clause would mean 10 separate API calls per document,
    multiplied by however many documents are in a batch. One batched prompt
    covering all 10 categories is ~10x faster and costs the same token budget.
    The tradeoff is a longer prompt, but GPT-4o-mini handles it well and the
    structured JSON output format keeps the response parseable.

How missing clauses are handled:
    If a clause is absent, the LLM is instructed to return is_present=False
    and extracted_text=null. The risk engine (Step 6) treats any missing
    clause as a HIGH risk finding, with no precedent override. This is a
    locked graded design decision -- do not change without updating the
    design document.

Dependencies:
    openai, pydantic, yaml, config.py, schemas/clause.py, schemas/chunk.py
"""

import json
import logging
from pathlib import Path
from typing import Optional

import yaml
from openai import OpenAI

from config import OPENAI_API_KEY, LLM_MODEL, CLAUSE_CATEGORIES
from schemas.chunk import DocumentChunk
from schemas.clause import ExtractedClause
from agent.tot_reasoner import run_tree_of_thought, get_local_context

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Confidence thresholds -- mirror CLAUDE.md, do not change.
CONFIDENCE_AUTO_PROCEED = 0.7
CONFIDENCE_TOT_FLOOR = 0.5

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "extract_clauses.yaml"

# ---------------------------------------------------------------------------
# Lazy OpenAI client
# ---------------------------------------------------------------------------

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


# ---------------------------------------------------------------------------
# Prompt loader
# ---------------------------------------------------------------------------

def _load_prompt() -> dict:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_context_text(chunks: list[DocumentChunk]) -> str:
    """
    Format the retrieved chunks into a single context string for the prompt.

    Each chunk gets a header line so the LLM knows the source chunk_id and
    page range -- this is what lets it fill in page_reference and
    source_chunk_id accurately.

    Example output:
        [Source: chunk_id=abc123_chunk_0002, pages 3-5]
        The indemnifying party shall defend, indemnify, and hold harmless...

        [Source: chunk_id=abc123_chunk_0005, pages 8-8]
        This Agreement shall be governed by the laws of the State of...
    """
    parts = []
    for chunk in chunks:
        # Quoting the ID and using a pipe separator reduces the chance the LLM
        # echoes back label text (e.g. "chunk_id=...") as part of source_chunk_id.
        header = (
            f'[chunk_id: "{chunk.chunk_id}" | '
            f"pages: {chunk.start_page}-{chunk.end_page}]"
        )
        parts.append(f"{header}\n{chunk.text}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Core LLM call
# ---------------------------------------------------------------------------

def _call_llm(chunks: list[DocumentChunk]) -> list[ExtractedClause]:
    """
    Send retrieved chunks to GPT-4o-mini and parse the clause extraction result.

    Raises
    ------
    ValueError
        If the LLM returns invalid JSON, the array has the wrong length, or
        Pydantic validation fails on any individual clause entry.
    """
    prompt = _load_prompt()
    context_text = _build_context_text(chunks)
    clause_categories_str = "\n".join(f"- {c}" for c in CLAUSE_CATEGORIES)

    user_message = prompt["user"].format(
        context_text=context_text,
        clause_categories=clause_categories_str,
        num_categories=len(CLAUSE_CATEGORIES),
    )

    client = _get_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    raw_content = response.choices[0].message.content

    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned non-JSON content: {raw_content!r}") from e

    # The prompt asks for a JSON object with a list under a key.
    # GPT-4o-mini with json_object mode sometimes wraps the array; unwrap it.
    if isinstance(data, dict):
        # Find the first list value in the response object
        list_values = [v for v in data.values() if isinstance(v, list)]
        if not list_values:
            raise ValueError(f"LLM response has no list field: {data}")
        raw_list = list_values[0]
    elif isinstance(data, list):
        raw_list = data
    else:
        raise ValueError(f"Unexpected LLM response structure: {type(data)}")

    if len(raw_list) != len(CLAUSE_CATEGORIES):
        raise ValueError(
            f"Expected {len(CLAUSE_CATEGORIES)} clause entries, got {len(raw_list)}"
        )

    clauses = []
    for entry in raw_list:
        try:
            # Defensive fallback: even with the clarified prompt format above
            # (chunk_id: "..." | pages: ...), strip any stray "chunk_id=" label
            # text the model might still echo back, in case of model drift or an
            # edge case the prompt wording doesn't fully prevent.
            if isinstance(entry.get("source_chunk_id"), str):
                entry["source_chunk_id"] = entry["source_chunk_id"].removeprefix("chunk_id=")
            clauses.append(ExtractedClause(**entry))
        except Exception as e:
            raise ValueError(f"Invalid clause entry {entry!r}: {e}") from e

    return clauses


# ---------------------------------------------------------------------------
# Three-tier confidence routing (per-clause)
# ---------------------------------------------------------------------------

def _retry_single_clause(
    clause: ExtractedClause,
    chunks: list[DocumentChunk],
) -> ExtractedClause:
    """
    Retry extraction for a single low-confidence (< 0.5) clause by adding
    local document context from the chunks adjacent to its source chunk.

    Uses get_local_context() from tot_reasoner so both the <0.5 retry path
    and the 0.5-0.7 ToT path use the same adjacent-chunk mechanism -- no
    duplicate context-lookup logic.

    Returns a new ExtractedClause with retry_count=1. If the retry also
    fails or returns low confidence, requires_human_review is set on the
    result but the extraction is not retried again -- capped at 1 retry.
    """
    # Find source chunk index for get_local_context().
    source_idx: int | None = None
    for c in chunks:
        if c.chunk_id == clause.source_chunk_id:
            source_idx = c.chunk_index
            break

    if source_idx is not None:
        local = get_local_context(chunks, source_idx)
    else:
        local = chunks  # fall back to the full chunk list if chunk not found

    try:
        # Re-run the full LLM extraction on the local context window.
        retry_clauses = _call_llm(local)
        # Find the matching clause_type in the retry result.
        match = next(
            (rc for rc in retry_clauses if rc.clause_type == clause.clause_type),
            None,
        )
        if match is None:
            match = clause  # defensive -- should not happen

        # Stamp retry_count=1; mark for human review if still low.
        needs_review = match.confidence < CONFIDENCE_TOT_FLOOR
        return match.model_copy(update={
            "retry_count": 1,
            "requires_human_review": needs_review,
            "human_review_reason": (
                f"Confidence {match.confidence:.2f} still below {CONFIDENCE_TOT_FLOOR} "
                "after one retry. Escalating to human review."
            ) if needs_review else None,
        })
    except Exception as e:
        logger.error(
            f"Clause retry failed for '{clause.clause_type}': {type(e).__name__}: {e}"
        )
        return clause.model_copy(update={
            "retry_count": 1,
            "requires_human_review": True,
            "human_review_reason": f"Retry LLM call failed: {type(e).__name__}: {e}",
        })


def _apply_confidence_routing(
    clause: ExtractedClause,
    chunks: list[DocumentChunk],
    document_name: str,
) -> ExtractedClause:
    """
    Apply the three-tier confidence routing rule to a single extracted clause.

    Tier 1 -- confidence > 0.7: proceed automatically. No change.
    Tier 2 -- 0.5 <= confidence <= 0.7 (exclusive of 0.7): route to ToT.
    Tier 3 -- confidence < 0.5: retry once with local context, then HITL.

    Boundary semantics (matching classifier.py exactly):
        CONFIDENCE_AUTO_PROCEED = 0.7  (strictly greater than -> auto)
        CONFIDENCE_TOT_FLOOR    = 0.5  (>= 0.5 and < 0.7 -> ToT)
        < 0.5                          -> retry, then HITL

    This function is only called on present clauses that have a non-None
    source_chunk_id -- absent clauses (is_present=False) are structural gaps
    handled by the risk engine, not ambiguous classifications.
    """
    if not clause.is_present or clause.source_chunk_id is None:
        # Absent clauses are not ambiguous -- no routing needed.
        return clause

    conf = clause.confidence

    if conf > CONFIDENCE_AUTO_PROCEED:
        # Tier 1: auto-proceed. Nothing to do.
        return clause

    if conf >= CONFIDENCE_TOT_FLOOR:
        # Tier 2: genuine ambiguity -- run Tree-of-Thought.
        logger.info(
            f"Clause '{clause.clause_type}' (chunk={clause.source_chunk_id}) "
            f"in 0.5-0.7 band (conf={conf:.2f}). Routing to ToT."
        )
        tot = run_tree_of_thought(
            clause_category=clause.clause_type,
            clause_text=clause.extracted_text or "",
            source_chunk_id=clause.source_chunk_id,
            document_chunks=chunks,
        )
        if tot.requires_human_review:
            return clause.model_copy(update={
                "requires_human_review": True,
                "human_review_reason": tot.human_review_reason,
                "tot_result": tot,
            })
        # Use the winning candidate's category (may be different if ToT
        # determined the original category was wrong).
        winner = tot.winning_candidates[0]
        return clause.model_copy(update={
            "clause_type": winner.clause_category,
            "confidence": winner.composite_score,
            "tot_result": tot,
        })

    # Tier 3: confidence < 0.5 -- retry once.
    if clause.retry_count == 0:
        logger.warning(
            f"Clause '{clause.clause_type}' (chunk={clause.source_chunk_id}) "
            f"has confidence {conf:.2f} < {CONFIDENCE_TOT_FLOOR}. Retrying once."
        )
        return _retry_single_clause(clause, chunks)

    # Already retried (retry_count >= 1) -- escalate.
    logger.warning(
        f"Clause '{clause.clause_type}' confidence still low after retry "
        f"(conf={conf:.2f}). Escalating to human review."
    )
    return clause.model_copy(update={
        "requires_human_review": True,
        "human_review_reason": (
            f"Confidence {conf:.2f} still below {CONFIDENCE_TOT_FLOOR} after retry."
        ),
    })


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def extract_clauses(
    chunks: list[DocumentChunk],
    document_name: str = "",
) -> list[ExtractedClause]:
    """
    Extract all 10 approved clause types from the retrieved document chunks.

    How it works:
        1. Format the chunks into a single context string with source headers.
        2. Send to GPT-4o-mini with the extract_clauses prompt.
        3. Parse and validate the 10-element JSON array into ExtractedClause objects.
        4. For each clause with confidence < CONFIDENCE_AUTO_PROCEED, log a
           warning -- the orchestrator (Step 8) will route low-confidence
           clauses to ToT or HITL based on the threshold.

    Parameters
    ----------
    chunks : list[DocumentChunk]
        The top-3 (or fewer) chunks returned by VectorStore.search_batch().
        These should already be reranked.
    document_name : str
        Used only in log messages (never logged content -- PII-safe).

    Returns
    -------
    list[ExtractedClause]
        Always exactly 10 entries -- one per CLAUSE_CATEGORIES entry.
        On total failure, returns 10 absent clauses with confidence=0.0
        rather than crashing, so the pipeline can continue to the risk
        engine which will flag them all as missing.
    """
    if not chunks:
        logger.warning(
            f"extract_clauses called with no chunks for '{document_name}'. "
            "Returning all clauses as absent."
        )
        return _all_absent_clauses()

    try:
        clauses = _call_llm(chunks)
    except (ValueError, Exception) as e:
        logger.error(
            f"extract_clauses failed for '{document_name}': {e}. "
            "Returning all clauses as absent."
        )
        return _all_absent_clauses()

    # Apply three-tier confidence routing to each clause.
    clauses = [_apply_confidence_routing(c, chunks, document_name) for c in clauses]

    present = sum(1 for c in clauses if c.is_present)
    logger.info(
        f"'{document_name}': extracted {present}/{len(CLAUSE_CATEGORIES)} "
        "clause categories as present."
    )

    return clauses


def _all_absent_clauses() -> list[ExtractedClause]:
    """
    Return 10 absent ExtractedClause objects -- one per category.
    Used as a safe fallback when extraction fails entirely.
    """
    return [
        ExtractedClause(
            clause_type=category,
            is_present=False,
            extracted_text=None,
            page_reference=None,
            confidence=0.0,
            source_chunk_id=None,
        )
        for category in CLAUSE_CATEGORIES
    ]
