"""
schemas/metrics.py
------------------
Pydantic v2 schema for one pipeline-run metrics record.

One record is written per _app.invoke() leg — not per logical document
session. When a run is paused by HITL (run_status="interrupted"), the
record captures the real partial numbers accumulated up to that pause.
A follow-up leg (after the human decides) produces a second record
correlated by thread_id. Aggregating total cost/latency per document is
a downstream sum-by-thread_id query, not something the pipeline reconciles
in real time.

Honesty rule: every counter must be a real, measured value or 0/None if
it cannot be wired up. No fabricated or estimated values are ever written.

Pricing constants (source: OpenAI API pricing page, verified 2026-06-28):
    GPT-4o-mini input:          $0.000150 per 1K tokens
    GPT-4o-mini output:         $0.000600 per 1K tokens
    text-embedding-3-small:     $0.000020 per 1K tokens
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Pricing constants — one place to update if OpenAI changes rates
# Source: OpenAI API pricing page, verified 2026-06-28
# ---------------------------------------------------------------------------

GPT4O_MINI_INPUT_COST_PER_1K: float = 0.000150   # $ per 1K prompt tokens
GPT4O_MINI_OUTPUT_COST_PER_1K: float = 0.000600   # $ per 1K completion tokens
EMBEDDING_SMALL_COST_PER_1K: float = 0.000020      # $ per 1K embedding tokens

RunStatus = Literal["completed", "interrupted", "blocked", "failed"]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class PipelineRunMetrics(BaseModel):
    """
    One metrics record per _app.invoke() leg in run_pipeline().

    Fields are grouped by category:
        Identity        — run_id, thread_id, pdf_path, started_at
        Outcome         — run_status, error_message
        Latency         — total_latency_seconds
        LLM tokens      — prompt_tokens, completion_tokens, embedding_tokens
        Cost            — estimated_cost_usd
        Counters        — ocr_fallback_*, low_confidence_retry_count,
                          tot_invocation_count, hitl_escalation_count,
                          guardrail_block_count
    """

    model_config = ConfigDict(ser_json_timedelta="iso8601")

    # -- Identity -------------------------------------------------------------

    run_id: str = Field(
        description="Unique identifier for this metrics record (UUID4)."
    )
    thread_id: str = Field(
        description=(
            "LangGraph thread UUID from run_pipeline(). Correlates multiple "
            "invoke legs belonging to the same logical document session."
        )
    )
    pdf_path: str = Field(
        description="Absolute path to the PDF document processed in this leg."
    )
    started_at: datetime = Field(
        description="Wall-clock time when _app.invoke() was called."
    )

    # -- Outcome --------------------------------------------------------------

    run_status: RunStatus = Field(
        default="completed",
        description=(
            "Exit status of this leg: 'completed' (normal success), "
            "'interrupted' (HITL pause), 'blocked' (guardrail stop), "
            "'failed' (uncaught exception)."
        ),
    )
    error_message: Optional[str] = Field(
        default=None,
        description=(
            "Error or block reason. Populated for 'blocked' and 'failed'; "
            "None for 'completed' and 'interrupted'."
        ),
    )

    # -- Latency --------------------------------------------------------------

    total_latency_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Wall-clock seconds from _app.invoke() call to return. "
            "Partial for 'interrupted' legs — reflects real elapsed time "
            "up to the HITL pause, not the full logical session."
        ),
    )

    # -- LLM token counts -----------------------------------------------------

    prompt_tokens: int = Field(
        default=0,
        ge=0,
        description="Total input (prompt) tokens sent to gpt-4o-mini in this leg.",
    )
    completion_tokens: int = Field(
        default=0,
        ge=0,
        description="Total output (completion) tokens received from gpt-4o-mini.",
    )
    embedding_tokens: int = Field(
        default=0,
        ge=0,
        description="Total tokens sent to text-embedding-3-small.",
    )

    # -- Cost -----------------------------------------------------------------

    estimated_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Estimated USD cost for this leg. "
            "= (prompt_tokens/1000 * 0.000150) "
            "+ (completion_tokens/1000 * 0.000600) "
            "+ (embedding_tokens/1000 * 0.000020). "
            "Only accurate when token fields are fully wired up."
        ),
    )

    # -- Activity counters ----------------------------------------------------

    ocr_fallback_triggered: bool = Field(
        default=False,
        description="True if at least one page required Vision OCR fallback.",
    )
    ocr_fallback_page_count: int = Field(
        default=0,
        ge=0,
        description="Number of pages that went through Vision OCR fallback.",
    )
    low_confidence_retry_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Total low-confidence retries across classifier + all clauses. "
            "Sum of DocumentClassification.retry_count and each ExtractedClause.retry_count."
        ),
    )
    tot_invocation_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of Tree-of-Thought invocations in this leg. "
            "Derived from count of clauses where tot_result is not None."
        ),
    )
    hitl_escalation_count: int = Field(
        default=0,
        ge=0,
        description="Number of review_items that triggered HITL escalation.",
    )
    guardrail_block_count: int = Field(
        default=0,
        ge=0,
        description="Number of guardrail checks that did not pass (passed=False).",
    )
