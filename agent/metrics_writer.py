"""
agent/metrics_writer.py
-----------------------
Writes one PipelineRunMetrics record per _app.invoke() leg to a JSONL log.

Output: data/metrics/metrics_log.jsonl — one JSON object per line,
append-only, mirroring the feedback_log.jsonl pattern.

Public API (used by agent/orchestrator.py):
    write_metrics(metrics: PipelineRunMetrics) -> None

The writer is intentionally thin: it validates the record (Pydantic model),
serializes it, and appends it atomically. Cost calculation belongs in
build_metrics(), which orchestrator.py calls before write_metrics().

build_metrics() is a pure function (no I/O) that constructs the record
from the raw counters collected during a run_pipeline() leg.

Honesty rule: every counter must be a real, measured value. build_metrics()
never substitutes a plausible-looking default for a counter that isn't wired
up yet — it uses 0 explicitly, and callers must pass 0 rather than None for
integer fields that haven't been instrumented.

PII rule: pdf_path is the only path-like field written. Full contract text
is never logged here.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from schemas.metrics import (
    GPT4O_MINI_INPUT_COST_PER_1K,
    GPT4O_MINI_OUTPUT_COST_PER_1K,
    EMBEDDING_SMALL_COST_PER_1K,
    PipelineRunMetrics,
    RunStatus,
)

logger = logging.getLogger(__name__)

METRICS_LOG_PATH = Path(__file__).parent.parent / "data" / "metrics" / "metrics_log.jsonl"


# ---------------------------------------------------------------------------
# Cost calculation (pure, testable)
# ---------------------------------------------------------------------------

def calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    embedding_tokens: int,
) -> float:
    """
    Compute estimated USD cost from token counts.

    Uses pricing constants from schemas/metrics.py (verified 2026-06-28).
    Result is rounded to 8 decimal places to avoid float noise in the log.
    """
    cost = (
        (prompt_tokens / 1_000) * GPT4O_MINI_INPUT_COST_PER_1K
        + (completion_tokens / 1_000) * GPT4O_MINI_OUTPUT_COST_PER_1K
        + (embedding_tokens / 1_000) * EMBEDDING_SMALL_COST_PER_1K
    )
    return round(cost, 8)


# ---------------------------------------------------------------------------
# Record builder (pure, testable)
# ---------------------------------------------------------------------------

def build_metrics(
    *,
    thread_id: str,
    pdf_path: str,
    started_at: datetime,
    run_status: RunStatus,
    total_latency_seconds: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    embedding_tokens: int = 0,
    ocr_fallback_triggered: bool = False,
    ocr_fallback_page_count: int = 0,
    low_confidence_retry_count: int = 0,
    tot_invocation_count: int = 0,
    hitl_escalation_count: int = 0,
    guardrail_block_count: int = 0,
    error_message: Optional[str] = None,
) -> PipelineRunMetrics:
    """
    Construct a PipelineRunMetrics record from raw run counters.

    All integer counters default to 0 — callers must pass real values.
    Cost is derived here from the token counts; callers must not pre-compute it.
    run_id is generated fresh (UUID4) on each call.
    """
    return PipelineRunMetrics(
        run_id=str(uuid.uuid4()),
        thread_id=thread_id,
        pdf_path=pdf_path,
        started_at=started_at,
        run_status=run_status,
        error_message=error_message,
        total_latency_seconds=total_latency_seconds,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        embedding_tokens=embedding_tokens,
        estimated_cost_usd=calculate_cost(prompt_tokens, completion_tokens, embedding_tokens),
        ocr_fallback_triggered=ocr_fallback_triggered,
        ocr_fallback_page_count=ocr_fallback_page_count,
        low_confidence_retry_count=low_confidence_retry_count,
        tot_invocation_count=tot_invocation_count,
        hitl_escalation_count=hitl_escalation_count,
        guardrail_block_count=guardrail_block_count,
    )


# ---------------------------------------------------------------------------
# Writer (I/O)
# ---------------------------------------------------------------------------

def write_metrics(metrics: PipelineRunMetrics, log_path: Path = METRICS_LOG_PATH) -> None:
    """
    Append one PipelineRunMetrics record to the JSONL metrics log.

    Creates the log directory if it does not exist (first run).
    Each record is serialized as a single JSON line. Datetime fields are
    written as ISO 8601 strings. The file is opened in append mode so
    concurrent writes from separate processes do not truncate existing data.

    Errors are logged but never re-raised — a metrics write failure must
    never bring down the pipeline.
    """
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        line = metrics.model_dump_json() + "\n"
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line)
        logger.info(
            "metrics: wrote run_id=%s thread_id=%s status=%s latency=%.2fs cost=$%.6f",
            metrics.run_id,
            metrics.thread_id,
            metrics.run_status,
            metrics.total_latency_seconds,
            metrics.estimated_cost_usd,
        )
    except Exception as exc:
        logger.error(
            "metrics: failed to write record run_id=%s — %s",
            getattr(metrics, "run_id", "unknown"),
            exc,
        )
