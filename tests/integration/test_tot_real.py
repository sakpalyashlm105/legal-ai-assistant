"""
tests/integration/test_tot_real.py
------------------------------------
Real end-to-end integration test for the Tree-of-Thought (ToT) reasoner.

Why Option A (constructed ambiguous clause) rather than Option B (real document)?
    The previous integration test run showed all 3 real documents consistently
    returned extraction confidence at 0.90+, meaning the ToT path never triggered
    naturally. There's no reliable way to force the pipeline to produce a 0.5-0.7
    confidence score on real documents without rewriting the prompt -- and changing
    the prompt to artificially lower confidence would invalidate the test. Instead
    we construct a clause text that is *genuinely* ambiguous between two categories
    (Indemnification + Limitation of Liability in the same paragraph), feed it
    directly to run_tree_of_thought(), and observe the real LLM's beam-search
    behavior with no mocking.

How to run (from legal-agent/ with venv active):
    pytest tests/integration/test_tot_real.py -v -s
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pathlib import Path
import pytest

from schemas.chunk import DocumentChunk
from schemas.tot import ToTResult, ToTCandidate
from agent.tot_reasoner import run_tree_of_thought

# ---------------------------------------------------------------------------
# Ambiguous clause text: combines indemnification obligations AND a liability cap
# in the same paragraph, which is a genuine category ambiguity.
# ---------------------------------------------------------------------------

AMBIGUOUS_CLAUSE_TEXT = (
    "Each party (the 'Indemnifying Party') shall defend, indemnify, and hold harmless "
    "the other party (the 'Indemnified Party') from and against any third-party claims, "
    "damages, liabilities, costs, and expenses (including reasonable attorneys' fees) "
    "arising out of or related to the Indemnifying Party's breach of this Agreement or "
    "its gross negligence or willful misconduct; provided, however, that in no event "
    "shall either party's aggregate liability under this Section exceed the total fees "
    "paid or payable by Customer to Vendor in the twelve (12) months immediately "
    "preceding the event giving rise to such claim."
)

# ---------------------------------------------------------------------------
# Minimal document chunk context for depth-2 local-context lookup.
# ---------------------------------------------------------------------------

DOC_HASH = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"  # 64 hex chars

TARGET_CHUNK = DocumentChunk(
    chunk_id=f"{DOC_HASH[:12]}_chunk_0002",
    document_hash=DOC_HASH,
    document_name="synthetic_ambiguous.pdf",
    text=AMBIGUOUS_CLAUSE_TEXT,
    char_count=len(AMBIGUOUS_CLAUSE_TEXT),
    token_count=len(AMBIGUOUS_CLAUSE_TEXT) // 4,
    start_page=3,
    end_page=3,
    chunk_index=2,
    total_chunks=5,
    overlap_tokens=0,
)

PREV_CHUNK = DocumentChunk(
    chunk_id=f"{DOC_HASH[:12]}_chunk_0001",
    document_hash=DOC_HASH,
    document_name="synthetic_ambiguous.pdf",
    text=(
        "ARTICLE 7. INDEMNIFICATION AND LIABILITY.\n"
        "The following provisions govern each party's obligations to defend and "
        "indemnify the other, and the limits on each party's financial exposure "
        "under this Agreement."
    ),
    char_count=200,
    token_count=50,
    start_page=2,
    end_page=3,
    chunk_index=1,
    total_chunks=5,
    overlap_tokens=20,
)

NEXT_CHUNK = DocumentChunk(
    chunk_id=f"{DOC_HASH[:12]}_chunk_0003",
    document_hash=DOC_HASH,
    document_name="synthetic_ambiguous.pdf",
    text=(
        "The indemnification obligations set forth herein shall survive the "
        "expiration or termination of this Agreement for a period of three (3) years."
    ),
    char_count=150,
    token_count=38,
    start_page=3,
    end_page=4,
    chunk_index=3,
    total_chunks=5,
    overlap_tokens=15,
)

ALL_CHUNKS = [PREV_CHUNK, TARGET_CHUNK, NEXT_CHUNK]


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

def test_tot_real_ambiguous_clause():
    """
    Feed a genuinely ambiguous clause (Indemnification + Limitation of Liability
    combined in one paragraph) to run_tree_of_thought() with real API calls.

    Prints full diagnostic output:
      - All candidates generated at depth 1 (before pruning)
      - Which candidates survived to depth 2 (and their re-scored values)
      - Which candidates survived to depth 3
      - Whether a critic step fired
      - The final winning candidate(s)
    """
    print(f"\n{'='*70}")
    print("TREE-OF-THOUGHT INTEGRATION TEST")
    print("Ambiguous clause: Indemnification + Limitation of Liability combined")
    print(f"{'='*70}")

    print(f"\nCLAUSE TEXT:\n{AMBIGUOUS_CLAUSE_TEXT}\n")

    result = run_tree_of_thought(
        clause_category="Indemnification",   # the category that triggered ToT
        clause_text=AMBIGUOUS_CLAUSE_TEXT,
        source_chunk_id=TARGET_CHUNK.chunk_id,
        document_chunks=ALL_CHUNKS,
    )

    # -------------------------------------------------------------------
    # Print all candidates (includes pruned)
    # -------------------------------------------------------------------
    print(f"\n--- ALL CANDIDATES (including pruned) [{len(result.all_candidates)} total] ---")
    for c in result.all_candidates:
        pruned_str = f"  [PRUNED at depth {c.pruned_at_depth}]" if c.pruned else ""
        print(
            f"  [{c.candidate_id}] depth={c.depth_reached} "
            f"category={c.clause_category}"
            f"{pruned_str}"
        )
        print(
            f"    textual_evidence={c.textual_evidence_score:.2f}  "
            f"category_fit={c.category_fit_score:.2f}  "
            f"template_alignment={c.template_alignment_score:.2f}  "
            f"exclusivity={c.exclusivity_score:.2f}"
        )
        print(f"    composite_score={c.composite_score:.4f}")
        print(f"    rationale: {c.rationale}")

    # -------------------------------------------------------------------
    # Print winning candidates
    # -------------------------------------------------------------------
    print(f"\n--- WINNING CANDIDATES [{len(result.winning_candidates)}] ---")
    for w in result.winning_candidates:
        print(
            f"  [{w.candidate_id}] category={w.clause_category} "
            f"composite_score={w.composite_score:.4f}"
        )
        print(f"    rationale: {w.rationale}")

    # -------------------------------------------------------------------
    # Print critic and human-review status
    # -------------------------------------------------------------------
    print(f"\n--- OUTCOME ---")
    print(f"  critic_step_triggered : {result.critic_step_triggered}")
    if result.critic_rationale:
        print(f"  critic_rationale      : {result.critic_rationale}")
    print(f"  requires_human_review : {result.requires_human_review}")
    if result.human_review_reason:
        print(f"  human_review_reason   : {result.human_review_reason}")

    # -------------------------------------------------------------------
    # Assertions: structural invariants regardless of what the LLM decided
    # -------------------------------------------------------------------
    assert isinstance(result, ToTResult), "run_tree_of_thought must always return ToTResult"
    assert len(result.all_candidates) >= 1, "all_candidates must not be empty"
    assert len(result.winning_candidates) in (1, 2), "winning_candidates must be 1 or 2"

    # All pruned candidates must have pruned_at_depth set.
    for c in result.all_candidates:
        if c.pruned:
            assert c.pruned_at_depth is not None, (
                f"Pruned candidate {c.candidate_id} is missing pruned_at_depth"
            )

    # At least one candidate must have reached depth >= 2 (local context was used).
    max_depth = max(c.depth_reached for c in result.all_candidates)
    assert max_depth >= 2, (
        f"Expected at least one candidate to reach depth 2, but max depth was {max_depth}"
    )

    # Winning candidates must be a subset of all_candidates.
    all_ids = {c.candidate_id for c in result.all_candidates}
    for w in result.winning_candidates:
        assert w.candidate_id in all_ids, (
            f"Winning candidate {w.candidate_id} not found in all_candidates"
        )

    print(f"\n{'='*70}")
    print("ALL STRUCTURAL ASSERTIONS PASSED")
    print(f"{'='*70}")
