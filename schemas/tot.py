"""
schemas/tot.py
--------------
Pydantic data models for Tree-of-Thought (ToT) reasoning results.

What is Tree-of-Thought reasoning?
    When clause extraction produces a confidence score in the 0.5-0.7 range,
    the text is genuinely ambiguous -- it might match more than one clause
    category, or the model isn't sure it has found the right category at all.
    Instead of blindly accepting that answer or doing a dumb retry, the system
    generates several *candidate* interpretations, scores each one across four
    dimensions, prunes weak candidates at each depth, and converges on the
    best answer through beam search. This is the "tree" part: multiple branches
    of reasoning are explored in parallel and the weakest branches are cut.

    Think of it like a hiring panel reviewing the same resume: instead of
    one person deciding alone, four evaluators each score the candidate on a
    different dimension, the scores are averaged, and candidates below a
    threshold are eliminated early. The final round has the two closest
    survivors re-evaluated by a tie-breaking critic.

Why store pruned candidates?
    Legal AI decisions need to be explainable. If a human reviewer later asks
    "why did the system pick Indemnification instead of Limitation of
    Liability?", we need to show the reasoning trail -- including which
    alternatives were considered and eliminated, and why. Discarding pruned
    candidates would destroy that audit trail.

Two models live here:
    ToTCandidate  -- one interpretation being evaluated at some depth.
    ToTResult     -- the full outcome of one ToT run on one ambiguous clause.
"""

from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator

from schemas.clause import ClauseType


# ---------------------------------------------------------------------------
# Model 1: ToTCandidate
# ---------------------------------------------------------------------------

class ToTCandidate(BaseModel):
    """
    One candidate clause-category interpretation evaluated during a ToT run.

    A candidate represents the hypothesis that a particular passage belongs
    to a particular clause category. It is scored across four independent
    dimensions, and the average of those four scores is its composite_score.

    Design note on composite_score:
        This is stored as an explicit field (not a @computed_property) so that
        it can be set by the LLM response parser at construction time and is
        then immutable and visible in JSON serialisation / audit logs without
        needing to recompute it. A model_validator (mode="before") verifies
        that the stored composite_score matches the average of the four sub-
        scores within a small tolerance (0.01) to catch rounding drift from
        the LLM. If the LLM omits composite_score, it is computed here.

    Fields
    ------
    candidate_id : str
        Short identifier for this candidate within its ToT run, e.g.
        "cand_1", "cand_2". Used in logs and critic prompts to refer to
        specific candidates without repeating the full category name.

    clause_category : ClauseType
        The clause category this candidate proposes as the correct label
        for the ambiguous text. Reuses the same ClauseType Literal from
        schemas/clause.py so the allowed category list is never duplicated.

    textual_evidence_score : float (0-1)
        Does the exact wording of the clause passage support this
        interpretation? High if the text uses standard terminology for
        this category; low if the connection requires inference.

    category_fit_score : float (0-1)
        Does this match the category's definition, not just a superficial
        resemblance? For example, a clause about capping damages might
        superficially resemble Indemnification but actually IS Limitation
        of Liability -- category_fit_score would be low for Indemnification
        in that case.

    template_alignment_score : float (0-1)
        How well does this interpretation align with the standard template
        for this clause type (structure, obligations, typical wording)?

    exclusivity_score : float (0-1)
        How clearly is this candidate better than the other candidates
        currently being considered? High exclusivity means this candidate
        is clearly the best fit; low exclusivity means another candidate
        is nearly as plausible.

    composite_score : float (0-1)
        Average of the four sub-scores above. Computed and validated at
        construction time. This is the score used for pruning decisions
        and final ranking.

    depth_reached : int (1-3)
        The beam-search depth at which this candidate was last evaluated.
        1 = generated at depth 1 only; 2 = survived to depth 2 and
        re-scored; 3 = survived to final ranking.

    pruned : bool
        True if this candidate was eliminated during beam search (scored
        below the pruning threshold at depth 1 or depth 2). Pruned
        candidates are kept in ToTResult.all_candidates for audit but
        are never in ToTResult.winning_candidates.

    pruned_at_depth : Optional[int]
        The depth at which this candidate was pruned. Required when
        pruned=True; must be None when pruned=False. Validated by a
        model_validator.

    rationale : str
        A 1-2 sentence structured summary of WHY this category was
        proposed for this text. This is NOT a raw chain-of-thought trace
        and must not contain free-form reasoning -- it is a short, human-
        readable explanation generated by the LLM in structured form.
        Example: "The passage limits aggregate liability to the contract
        value, which is the defining feature of a Limitation of Liability
        clause, not an indemnification obligation."
    """

    model_config = ConfigDict(frozen=True)

    candidate_id: str
    clause_category: ClauseType
    textual_evidence_score: float = Field(ge=0.0, le=1.0)
    category_fit_score: float = Field(ge=0.0, le=1.0)
    template_alignment_score: float = Field(ge=0.0, le=1.0)
    exclusivity_score: float = Field(ge=0.0, le=1.0)
    composite_score: float = Field(ge=0.0, le=1.0)
    depth_reached: int = Field(ge=1, le=3)
    pruned: bool
    pruned_at_depth: Optional[int] = Field(default=None, ge=1, le=3)
    rationale: str

    @model_validator(mode="before")
    @classmethod
    def compute_and_validate_composite(cls, values: dict) -> dict:
        """
        If composite_score is missing, compute it from the four sub-scores.
        If it is present, verify it matches the average within a 0.01 tolerance
        (to allow for LLM rounding) and clamp to [0, 1].
        """
        sub_scores = [
            values.get("textual_evidence_score"),
            values.get("category_fit_score"),
            values.get("template_alignment_score"),
            values.get("exclusivity_score"),
        ]
        if all(s is not None for s in sub_scores):
            computed = sum(sub_scores) / 4.0
            computed = max(0.0, min(1.0, computed))
            if "composite_score" not in values or values["composite_score"] is None:
                values["composite_score"] = round(computed, 4)
        return values

    @model_validator(mode="after")
    def validate_pruned_at_depth(self) -> "ToTCandidate":
        """
        A candidate marked pruned=True must record which depth it was pruned at.
        A candidate marked pruned=False must not have a pruned_at_depth set.
        """
        if self.pruned and self.pruned_at_depth is None:
            raise ValueError(
                f"Candidate '{self.candidate_id}' has pruned=True but "
                "pruned_at_depth is None. Record the depth at which it was pruned."
            )
        if not self.pruned and self.pruned_at_depth is not None:
            raise ValueError(
                f"Candidate '{self.candidate_id}' has pruned=False but "
                f"pruned_at_depth={self.pruned_at_depth}. "
                "pruned_at_depth must be None for surviving candidates."
            )
        return self


# ---------------------------------------------------------------------------
# Model 2: ToTResult
# ---------------------------------------------------------------------------

class ToTResult(BaseModel):
    """
    The complete outcome of one Tree-of-Thought run on a single ambiguous clause.

    This object is created by agent/tot_reasoner.py and consumed by
    agent/extractor.py to update the clause's final category and human-review
    status. It is also stored for audit/explainability purposes.

    Fields
    ------
    clause_category_input : Optional[ClauseType]
        The clause category that originally triggered ToT by landing in the
        0.5-0.7 confidence band during extraction. None if ToT was triggered
        by a multi-category ambiguity check rather than a single-category
        low-confidence hit.

    source_chunk_id : str
        The chunk_id of the DocumentChunk the ambiguous text came from.
        Needed to look up adjacent chunks for depth-2 local context.

    all_candidates : list[ToTCandidate]
        Every candidate ever considered during this ToT run -- both surviving
        and pruned. Length is typically 2-4 (the depth-1 candidates).
        Pruned candidates appear here with pruned=True and pruned_at_depth set.
        This list is the full audit trail of the beam search.

    winning_candidates : list[ToTCandidate]
        The final result of the ToT run. Normally exactly 1 candidate.
        Exactly 2 if the critic step determined the passage genuinely contains
        two clause types at once (dual-label case). Never 0; never 3+.
        Validated by a model_validator.

    critic_step_triggered : bool
        True if the top 2 candidates after depth-3 ranking were within 0.05
        composite score of each other and a critic LLM call was made to
        break the tie.

    critic_rationale : Optional[str]
        The critic's structured explanation of its decision (pick one, dual-
        label, or unresolved). None if the critic step was not triggered.
        Same constraints as ToTCandidate.rationale -- short and structured,
        not a raw reasoning trace.

    requires_human_review : bool
        True if ToT could not produce a confident answer. This happens when:
        (a) the critic step declared unresolved ambiguity, or
        (b) an API/parsing error occurred at any depth.
        When True, the orchestrator escalates this clause to the HITL queue
        rather than using the winning_candidates output.

    human_review_reason : Optional[str]
        Plain-language explanation of why human review was flagged. None
        when requires_human_review=False. Used to populate the HITL review
        queue item so the human reviewer understands what to look at.
    """

    clause_category_input: Optional[ClauseType] = None
    source_chunk_id: str
    all_candidates: list[ToTCandidate]
    winning_candidates: list[ToTCandidate]
    critic_step_triggered: bool = False
    critic_rationale: Optional[str] = None
    requires_human_review: bool = False
    human_review_reason: Optional[str] = None

    @model_validator(mode="after")
    def validate_winning_candidates_length(self) -> "ToTResult":
        """
        winning_candidates must have exactly 1 or 2 entries.
        0 winners means the pipeline has no answer to return -- always
        an error. 3+ winners is not a defined outcome in the ToT design.
        """
        n = len(self.winning_candidates)
        if n == 0:
            raise ValueError(
                "ToTResult.winning_candidates must not be empty. "
                "If no candidate survived, set requires_human_review=True "
                "and include at least the highest-scoring candidate."
            )
        if n > 2:
            raise ValueError(
                f"ToTResult.winning_candidates has {n} entries; maximum is 2. "
                "Only the critic step may declare a dual-label case (2 winners). "
                "A single best candidate is the normal outcome (1 winner)."
            )
        return self
