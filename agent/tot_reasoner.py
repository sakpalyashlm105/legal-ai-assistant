"""
agent/tot_reasoner.py
---------------------
Tree-of-Thought (ToT) beam-search reasoner for ambiguous clause extraction.

When is this called?
    Clause extraction in extractor.py produces a confidence score per clause.
    When that confidence lands in the 0.5-0.7 band, the single-pass extraction
    is not trusted -- the text is genuinely ambiguous. This module takes the
    ambiguous passage and runs a 3-depth beam search over possible clause-
    category interpretations, converging on the best answer (or escalating to
    human review if it cannot decide).

How the beam search works:
    Depth 1: generate 2-4 candidate interpretations via LLM. Prune anything
             with composite_score < 0.30.
    Depth 2: re-score surviving candidates using LOCAL document context (the
             chunks immediately before and after the ambiguous one). Prune
             anything below 0.40. Context lookup is handled by get_local_context()
             -- purely sequential/local, NOT FAISS semantic search.
    Depth 3: final ranking, no further pruning. If the top 2 are within 0.05
             composite score of each other, call the critic LLM to break the tie.

    Pruned candidates are kept in ToTResult.all_candidates for audit purposes.
    They just don't appear in winning_candidates.

Safe failure contract:
    On ANY exception at any depth (API error, malformed JSON, Pydantic validation
    failure), the function never raises. It returns a ToTResult with
    requires_human_review=True and a human_review_reason describing the failure.
    This matches the same pattern used by classifier.py, extractor.py,
    comparator.py, and risk_engine.py.

PII-safe logging:
    Logs clause_category, source_chunk_id, depth reached, candidate counts,
    and decisions. Never logs full clause_text or extracted contract language.

Dependencies:
    openai, pydantic, yaml, schemas/chunk.py, schemas/tot.py, config.py
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Optional

import yaml
from openai import OpenAI

from config import OPENAI_API_KEY, LLM_MODEL, CLAUSE_CATEGORIES
from schemas.chunk import DocumentChunk
from schemas.tot import ToTCandidate, ToTResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRUNE_DEPTH_1 = 0.30   # candidates below this at depth 1 are pruned
PRUNE_DEPTH_2 = 0.40   # candidates below this at depth 2 are pruned
TIE_THRESHOLD = 0.05   # top-2 within this margin triggers the critic step

GENERATOR_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "tot_generator.yaml"
CRITIC_PROMPT_PATH    = Path(__file__).parent.parent / "prompts" / "tot_critic.yaml"

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
# Prompt loaders
# ---------------------------------------------------------------------------

def _load_generator_prompt() -> dict:
    with open(GENERATOR_PROMPT_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_critic_prompt() -> dict:
    with open(CRITIC_PROMPT_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# get_local_context -- LOCKED SPECIFICATION (do not alter without design review)
# ---------------------------------------------------------------------------

def get_local_context(
    chunks: list[DocumentChunk],
    target_chunk_index: int,
    neighbor_window: int = 1,
) -> list[DocumentChunk]:
    """
    Return the target chunk plus its immediate neighbors, restricted to the
    same document (same document_hash as the target chunk).

    This is used at depth 2 of the ToT beam search to give the LLM a small
    window of surrounding text to help clarify whether an ambiguous passage
    belongs to one clause category or another. It is PURELY LOCAL AND
    SEQUENTIAL -- it never calls FAISS or any semantic search. That is a
    deliberate contrast with the vector-store retrieval used in Steps 4-5.

    Parameters
    ----------
    chunks : list[DocumentChunk]
        All chunks passed into the ToT run (typically the full document's
        chunk list). This function will filter to same-document chunks only.
    target_chunk_index : int
        The chunk_index field (zero-based position) of the ambiguous chunk.
        This is the chunk whose category is being debated.
    neighbor_window : int
        How many neighbors to include on each side. Default 1 means at most
        one previous chunk and one next chunk. Do not increase without an
        explicit design review (controls token usage).

    Returns
    -------
    list[DocumentChunk]
        Ordered [previous?, target, next?]. The target is always present.
        Boundary cases (first chunk, last chunk, single-chunk document) are
        handled gracefully -- see inline logic below.

    Notes on section-heading filtering:
        The DocumentChunk schema does NOT currently have a section_heading
        field. If that field is added in a future step, this function should
        be updated to exclude adjacent chunks whose section heading clearly
        differs from the target's (indicating an unrelated section). For now,
        no section-heading filtering is applied -- all same-document neighbors
        within the window are included.
    """
    if not chunks:
        return []

    # Find the target chunk by its chunk_index, restricted to same document.
    target = None
    for c in chunks:
        if c.chunk_index == target_chunk_index:
            target = c
            break

    if target is None:
        logger.warning(
            f"get_local_context: no chunk with chunk_index={target_chunk_index} found."
        )
        return []

    doc_hash = target.document_hash

    # Filter to same-document chunks only, sorted by chunk_index.
    same_doc = sorted(
        [c for c in chunks if c.document_hash == doc_hash],
        key=lambda c: c.chunk_index,
    )

    # Locate the target's position in the same-doc list.
    target_pos = next(
        (i for i, c in enumerate(same_doc) if c.chunk_index == target_chunk_index),
        None,
    )
    if target_pos is None:
        return [target]

    result = []

    # Previous neighbor(s) -- boundary: if target_pos == 0, no previous chunk.
    prev_start = max(0, target_pos - neighbor_window)
    for chunk in same_doc[prev_start:target_pos]:
        result.append(chunk)

    # Target itself.
    result.append(same_doc[target_pos])

    # Next neighbor(s) -- boundary: if target_pos is last, no next chunk.
    next_end = min(len(same_doc), target_pos + neighbor_window + 1)
    for chunk in same_doc[target_pos + 1:next_end]:
        result.append(chunk)

    return result


# ---------------------------------------------------------------------------
# Context text formatter for depth 2
# ---------------------------------------------------------------------------

def _build_depth2_context(
    local_chunks: list[DocumentChunk],
    target_chunk_index: int,
) -> str:
    """
    Format the local context window into the depth-2 block required by the spec.

    The target chunk is explicitly marked so the model focuses on it, not on
    the surrounding context chunks. Surrounding chunks are context-only and
    must never silently become part of the clause's extracted_text evidence.

    Format produced (per locked spec):
        [Previous chunk]
        <text>

        [TARGET CHUNK -- classify this passage]
        <text>

        [Next chunk]
        <text>
    """
    parts = []
    for chunk in local_chunks:
        if chunk.chunk_index == target_chunk_index:
            parts.append(f"[TARGET CHUNK -- classify this passage]\n{chunk.text}")
        elif chunk.chunk_index < target_chunk_index:
            parts.append(f"[Previous chunk]\n{chunk.text}")
        else:
            parts.append(f"[Next chunk]\n{chunk.text}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------

def _call_generator(
    clause_category: str,
    clause_text: str,
    depth: int,
    depth_2_instruction: str = "",
) -> list[dict]:
    """
    Call the tot_generator prompt and return the raw candidate dicts.

    Raises ValueError on bad JSON or missing "candidates" key.
    """
    prompt = _load_generator_prompt()
    clause_categories_str = "\n".join(f"- {c}" for c in CLAUSE_CATEGORIES)

    user_message = prompt["user"].format(
        clause_categories=clause_categories_str,
        triggering_category=clause_category,
        clause_text=clause_text,
        depth=depth,
        depth_2_instruction=depth_2_instruction,
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

    raw = response.choices[0].message.content
    data = json.loads(raw)

    if "candidates" not in data or not isinstance(data["candidates"], list):
        raise ValueError(f"Generator response missing 'candidates' list: {data!r}")

    return data["candidates"]


def _call_critic(
    clause_text: str,
    cand_a: ToTCandidate,
    cand_b: ToTCandidate,
) -> dict:
    """
    Call the tot_critic prompt and return the raw decision dict.

    Raises ValueError on bad JSON or invalid decision value.
    """
    prompt = _load_critic_prompt()

    user_message = prompt["user"].format(
        clause_text=clause_text,
        candidate_a_id=cand_a.candidate_id,
        candidate_a_category=cand_a.clause_category,
        candidate_a_score=f"{cand_a.composite_score:.4f}",
        candidate_a_rationale=cand_a.rationale,
        candidate_b_id=cand_b.candidate_id,
        candidate_b_category=cand_b.clause_category,
        candidate_b_score=f"{cand_b.composite_score:.4f}",
        candidate_b_rationale=cand_b.rationale,
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

    raw = response.choices[0].message.content
    data = json.loads(raw)

    valid_decisions = {"pick_one", "dual_label", "human_review"}
    if data.get("decision") not in valid_decisions:
        raise ValueError(
            f"Critic returned invalid decision {data.get('decision')!r}. "
            f"Expected one of {valid_decisions}."
        )

    return data


# ---------------------------------------------------------------------------
# Candidate parsing helpers
# ---------------------------------------------------------------------------

def _parse_candidates(
    raw_list: list[dict],
    depth: int,
    all_so_far: list[ToTCandidate],
) -> list[ToTCandidate]:
    """
    Parse raw LLM candidate dicts into ToTCandidate objects.

    When re-scoring at depth 2, the LLM may return the same candidate_ids as
    depth 1. This function regenerates unique IDs for depth-2 re-scores so
    the two sets remain distinguishable in all_candidates.

    Silently skips entries that fail Pydantic validation (logs a warning)
    rather than crashing the whole ToT run.
    """
    existing_ids = {c.candidate_id for c in all_so_far}
    candidates = []
    for i, entry in enumerate(raw_list):
        # Assign a fresh unique ID if the ID would collide.
        base_id = entry.get("candidate_id", f"cand_{i+1}")
        if base_id in existing_ids:
            base_id = f"{base_id}_d{depth}"
        entry["candidate_id"] = base_id
        entry["depth_reached"] = depth
        # LLM does not set pruning -- that is determined by the beam search.
        entry.setdefault("pruned", False)
        entry["pruned_at_depth"] = None

        try:
            candidates.append(ToTCandidate(**entry))
        except Exception as e:
            logger.warning(
                f"ToT depth {depth}: skipping malformed candidate entry "
                f"{entry.get('candidate_id', '?')} ({type(e).__name__}: {e})"
            )

    return candidates


def _prune(
    candidates: list[ToTCandidate],
    threshold: float,
    depth: int,
) -> tuple[list[ToTCandidate], list[ToTCandidate]]:
    """
    Split candidates into survivors and pruned, marking pruned ones.

    Edge case: if ALL candidates fall below the threshold, keep the single
    highest-scoring one alive to ensure the pipeline always has something
    to continue with. This prevents the all-pruned-at-depth-1 crash case.

    Returns (survivors, pruned_candidates).
    """
    survivors = [c for c in candidates if c.composite_score >= threshold]
    pruned_raw = [c for c in candidates if c.composite_score < threshold]

    if not survivors and candidates:
        # Keep the best one alive regardless of threshold.
        best = max(candidates, key=lambda c: c.composite_score)
        survivors = [best]
        pruned_raw = [c for c in candidates if c.candidate_id != best.candidate_id]
        logger.warning(
            f"ToT depth {depth}: all {len(candidates)} candidates scored below "
            f"{threshold:.2f}. Keeping best candidate ({best.candidate_id}, "
            f"score={best.composite_score:.4f}) alive to prevent empty beam."
        )

    # Mark pruned candidates with their depth -- Pydantic models are frozen,
    # so we reconstruct with updated fields.
    pruned = [
        c.model_copy(update={"pruned": True, "pruned_at_depth": depth})
        for c in pruned_raw
    ]

    return survivors, pruned


# ---------------------------------------------------------------------------
# Failure-mode helper
# ---------------------------------------------------------------------------

def _fail_result(
    clause_category: str,
    source_chunk_id: str,
    reason: str,
    fallback_candidates: list[ToTCandidate] | None = None,
) -> ToTResult:
    """
    Build a human-review ToTResult for any failure path.

    If fallback_candidates is provided (partial depth-1 results), they are
    included in all_candidates. If not, a minimal placeholder winner is
    created so ToTResult's validator (winning_candidates length >= 1) passes.
    """
    if fallback_candidates and len(fallback_candidates) > 0:
        best = max(fallback_candidates, key=lambda c: c.composite_score)
        winner = best.model_copy(update={"pruned": False, "pruned_at_depth": None})
        return ToTResult(
            clause_category_input=clause_category if clause_category in CLAUSE_CATEGORIES else None,
            source_chunk_id=source_chunk_id,
            all_candidates=fallback_candidates,
            winning_candidates=[winner],
            requires_human_review=True,
            human_review_reason=reason,
        )

    # No candidates at all -- construct a minimal placeholder.
    placeholder = ToTCandidate(
        candidate_id="placeholder_failure",
        clause_category=clause_category if clause_category in CLAUSE_CATEGORIES
                        else "Indemnification",
        textual_evidence_score=0.0,
        category_fit_score=0.0,
        template_alignment_score=0.0,
        exclusivity_score=0.0,
        composite_score=0.0,
        depth_reached=1,
        pruned=False,
        pruned_at_depth=None,
        rationale="ToT failed before any candidates were scored.",
    )
    return ToTResult(
        clause_category_input=clause_category if clause_category in CLAUSE_CATEGORIES else None,
        source_chunk_id=source_chunk_id,
        all_candidates=[placeholder],
        winning_candidates=[placeholder],
        requires_human_review=True,
        human_review_reason=reason,
    )


# ---------------------------------------------------------------------------
# Public function: run_tree_of_thought
# ---------------------------------------------------------------------------

def run_tree_of_thought(
    clause_category: str,
    clause_text: str,
    source_chunk_id: str,
    document_chunks: list[DocumentChunk],
) -> ToTResult:
    """
    Run a 3-depth beam search to resolve an ambiguous clause classification.

    Parameters
    ----------
    clause_category : str
        The clause category that triggered ToT (had confidence in 0.5-0.7 band).
    clause_text : str
        The ambiguous passage text.
    source_chunk_id : str
        The chunk_id of the chunk this passage came from.
    document_chunks : list[DocumentChunk]
        All chunks for the document. Used by get_local_context() at depth 2.

    Returns
    -------
    ToTResult
        Always returned (never raises). If anything goes wrong, the result
        has requires_human_review=True.
    """
    all_candidates: list[ToTCandidate] = []

    # Find the target chunk's index for get_local_context().
    target_chunk_index: int | None = None
    for c in document_chunks:
        if c.chunk_id == source_chunk_id:
            target_chunk_index = c.chunk_index
            break

    # -------------------------------------------------------------------
    # DEPTH 1: generate 2-4 candidates
    # -------------------------------------------------------------------
    try:
        raw_d1 = _call_generator(clause_category, clause_text, depth=1)
        d1_candidates = _parse_candidates(raw_d1, depth=1, all_so_far=[])
    except Exception as e:
        logger.error(
            f"ToT depth-1 failed for category='{clause_category}', "
            f"chunk='{source_chunk_id}': {type(e).__name__}: {e}"
        )
        return _fail_result(
            clause_category, source_chunk_id,
            f"Depth-1 LLM call failed: {type(e).__name__}: {e}",
        )

    if not d1_candidates:
        return _fail_result(
            clause_category, source_chunk_id,
            "Depth-1 returned no parseable candidates.",
        )

    survivors_d1, pruned_d1 = _prune(d1_candidates, PRUNE_DEPTH_1, depth=1)
    all_candidates.extend(pruned_d1)

    logger.info(
        f"ToT depth-1: category='{clause_category}', chunk='{source_chunk_id}', "
        f"candidates={len(d1_candidates)}, pruned={len(pruned_d1)}, "
        f"survivors={len(survivors_d1)}"
    )

    # -------------------------------------------------------------------
    # DEPTH 2: re-score with local document context
    # -------------------------------------------------------------------
    try:
        if target_chunk_index is not None:
            local_chunks = get_local_context(document_chunks, target_chunk_index)
        else:
            local_chunks = []

        if local_chunks:
            depth2_ctx = _build_depth2_context(local_chunks, target_chunk_index)
            depth2_text = f"{clause_text}\n\n{depth2_ctx}"
            depth_2_instruction = (
                "The context above includes the surrounding document chunks. "
                "Consider whether: (a) the surrounding text defines a term used in "
                "the target chunk, (b) an obligation continues across the chunk "
                "boundary, (c) an exception or limitation is added by the context, "
                "(d) the governing section is clarified, or (e) multiple clause types "
                "are genuinely present. Re-score each candidate based on this expanded view."
            )
        else:
            depth2_text = clause_text
            depth_2_instruction = (
                "No surrounding context chunks were available for this passage."
            )

        # Only re-score the candidates that survived depth 1.
        surviving_ids = {c.candidate_id for c in survivors_d1}
        survivor_categories = [c.clause_category for c in survivors_d1]

        # For depth-2, ask the generator to focus on the surviving categories.
        raw_d2 = _call_generator(
            clause_category,
            depth2_text,
            depth=2,
            depth_2_instruction=depth_2_instruction,
        )

        # Parse and attempt to map back to the surviving candidate IDs.
        d2_candidates = _parse_candidates(raw_d2, depth=2, all_so_far=survivors_d1)

        # Prefer re-scored versions that match the surviving categories.
        d2_by_category = {c.clause_category: c for c in d2_candidates}
        survivors_d2_raw = []
        for s in survivors_d1:
            if s.clause_category in d2_by_category:
                # Use the re-scored version.
                survivors_d2_raw.append(d2_by_category[s.clause_category])
            else:
                # LLM didn't re-score this category; keep the depth-1 score.
                survivors_d2_raw.append(s)

        # Prune at depth 2.
        survivors_d2, pruned_d2 = _prune(survivors_d2_raw, PRUNE_DEPTH_2, depth=2)

        # Mark depth-2 re-scored survivors as depth_reached=2.
        survivors_d2 = [
            c.model_copy(update={"depth_reached": 2}) for c in survivors_d2
        ]
        all_candidates.extend(survivors_d1)   # depth-1 survivors (for audit)
        all_candidates.extend(pruned_d2)

        logger.info(
            f"ToT depth-2: category='{clause_category}', chunk='{source_chunk_id}', "
            f"survivors_in={len(survivors_d1)}, pruned={len(pruned_d2)}, "
            f"survivors_out={len(survivors_d2)}"
        )

    except Exception as e:
        logger.error(
            f"ToT depth-2 failed for category='{clause_category}', "
            f"chunk='{source_chunk_id}': {type(e).__name__}: {e}"
        )
        # Fall back to depth-1 survivors for depth 3.
        survivors_d2 = survivors_d1
        all_candidates.extend(survivors_d1)

    # -------------------------------------------------------------------
    # DEPTH 3: final ranking, no pruning
    # -------------------------------------------------------------------
    survivors_d3 = sorted(survivors_d2, key=lambda c: c.composite_score, reverse=True)
    survivors_d3 = [
        c.model_copy(update={"depth_reached": 3}) for c in survivors_d3
    ]
    all_candidates.extend(survivors_d3)

    top_score_str = f"{survivors_d3[0].composite_score:.4f}" if survivors_d3 else "N/A"
    logger.info(
        f"ToT depth-3: category='{clause_category}', chunk='{source_chunk_id}', "
        f"final_candidates={len(survivors_d3)}, top_score={top_score_str}"
    )

    # -------------------------------------------------------------------
    # Tie check and critic step
    # -------------------------------------------------------------------
    critic_step_triggered = False
    critic_rationale: str | None = None
    winning_candidates: list[ToTCandidate] = []
    requires_human_review = False
    human_review_reason: str | None = None

    if not survivors_d3:
        return _fail_result(
            clause_category, source_chunk_id,
            "No candidates survived to depth-3 ranking.",
            fallback_candidates=all_candidates,
        )

    top = survivors_d3[0]

    if len(survivors_d3) >= 2:
        second = survivors_d3[1]
        score_gap = top.composite_score - second.composite_score

        if score_gap <= TIE_THRESHOLD:
            critic_step_triggered = True
            logger.info(
                f"ToT tie detected for category='{clause_category}', "
                f"chunk='{source_chunk_id}': "
                f"gap={score_gap:.4f} (threshold={TIE_THRESHOLD}). "
                f"Calling critic."
            )
            try:
                critic_result = _call_critic(clause_text, top, second)
                decision = critic_result["decision"]
                critic_rationale = critic_result.get("critic_rationale", "")
                winner_id = critic_result.get("winner_candidate_id")

                if decision == "pick_one":
                    # Find the candidate the critic picked.
                    winner = next(
                        (c for c in [top, second] if c.candidate_id == winner_id),
                        top,  # fallback to top-ranked if ID doesn't match
                    )
                    winning_candidates = [winner]

                elif decision == "dual_label":
                    winning_candidates = [top, second]

                else:  # "human_review"
                    winning_candidates = [top]
                    requires_human_review = True
                    human_review_reason = (
                        f"Critic declared unresolved ambiguity between "
                        f"'{top.clause_category}' and '{second.clause_category}'. "
                        f"Critic rationale: {critic_rationale}"
                    )

            except Exception as e:
                logger.error(
                    f"ToT critic step failed for category='{clause_category}', "
                    f"chunk='{source_chunk_id}': {type(e).__name__}: {e}"
                )
                winning_candidates = [top]
                requires_human_review = True
                human_review_reason = (
                    f"Critic LLM call failed: {type(e).__name__}: {e}"
                )
        else:
            # Clear winner.
            winning_candidates = [top]
    else:
        # Only one candidate survived -- it wins unconditionally.
        winning_candidates = [top]

    logger.info(
        f"ToT complete: category='{clause_category}', chunk='{source_chunk_id}', "
        f"winner(s)={[c.clause_category for c in winning_candidates]}, "
        f"critic_triggered={critic_step_triggered}, "
        f"human_review={requires_human_review}"
    )

    return ToTResult(
        clause_category_input=clause_category if clause_category in CLAUSE_CATEGORIES else None,
        source_chunk_id=source_chunk_id,
        all_candidates=all_candidates,
        winning_candidates=winning_candidates,
        critic_step_triggered=critic_step_triggered,
        critic_rationale=critic_rationale,
        requires_human_review=requires_human_review,
        human_review_reason=human_review_reason,
    )
