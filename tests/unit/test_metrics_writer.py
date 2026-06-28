"""
tests/unit/test_metrics_writer.py
----------------------------------
Unit tests for schemas/metrics.py and agent/metrics_writer.py.

No OpenAI calls are made. The metrics log is redirected to tmp_path.

What this file tests:
    1. PipelineRunMetrics validates and round-trips through JSON correctly.
    2. All four run_status values are accepted; invalid values are rejected.
    3. calculate_cost() produces correct USD values from known token counts.
    4. build_metrics() constructs a valid record with the correct cost.
    5. write_metrics() appends a valid JSONL line; directory is auto-created.
    6. A second write_metrics() appends a second line (append, not overwrite).
    7. write_metrics() does not raise on I/O error — logs instead.
    8. 'interrupted' leg: partial numbers (non-zero latency/tokens) are written
       accurately; run_status is 'interrupted'.
    9. 'blocked' leg: error_message is populated, run_status is 'blocked'.
    10. Token/cost floor: zero-token run produces cost=0.0 exactly.

How to run:
    From legal-agent/ with the venv active:
        pytest tests/unit/test_metrics_writer.py -v
"""

import json
import logging
import pytest
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from schemas.metrics import (
    PipelineRunMetrics,
    GPT4O_MINI_INPUT_COST_PER_1K,
    GPT4O_MINI_OUTPUT_COST_PER_1K,
    EMBEDDING_SMALL_COST_PER_1K,
)
from agent.metrics_writer import (
    build_metrics,
    calculate_cost,
    write_metrics,
    METRICS_LOG_PATH,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STARTED = datetime(2026, 6, 28, 10, 0, 0, tzinfo=timezone.utc)


def _minimal_metrics(**overrides) -> PipelineRunMetrics:
    defaults = dict(
        run_id="run-test-001",
        thread_id="thread-test-001",
        pdf_path="/data/raw/Test.pdf",
        started_at=_STARTED,
        run_status="completed",
    )
    defaults.update(overrides)
    return PipelineRunMetrics(**defaults)


# ---------------------------------------------------------------------------
# Test 1: Schema round-trip
# ---------------------------------------------------------------------------

def test_schema_round_trip():
    m = _minimal_metrics()
    dumped = m.model_dump_json()
    loaded = PipelineRunMetrics.model_validate_json(dumped)

    assert loaded.run_id == m.run_id
    assert loaded.thread_id == m.thread_id
    assert loaded.run_status == "completed"
    assert loaded.prompt_tokens == 0
    assert loaded.estimated_cost_usd == 0.0


# ---------------------------------------------------------------------------
# Test 2: All four run_status values accepted; invalid value rejected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", ["completed", "interrupted", "blocked", "failed"])
def test_valid_run_status(status):
    m = _minimal_metrics(run_status=status)
    assert m.run_status == status


def test_invalid_run_status_rejected():
    with pytest.raises(ValidationError):
        _minimal_metrics(run_status="unknown_status")


# ---------------------------------------------------------------------------
# Test 3: calculate_cost() accuracy
# ---------------------------------------------------------------------------

def test_calculate_cost_known_values():
    # 1000 prompt + 500 completion + 2000 embedding
    cost = calculate_cost(1000, 500, 2000)
    expected = (
        (1000 / 1_000) * GPT4O_MINI_INPUT_COST_PER_1K
        + (500 / 1_000) * GPT4O_MINI_OUTPUT_COST_PER_1K
        + (2000 / 1_000) * EMBEDDING_SMALL_COST_PER_1K
    )
    assert abs(cost - expected) < 1e-9


def test_calculate_cost_zero_tokens():
    assert calculate_cost(0, 0, 0) == 0.0


def test_calculate_cost_embedding_only():
    cost = calculate_cost(0, 0, 1000)
    assert abs(cost - EMBEDDING_SMALL_COST_PER_1K) < 1e-9


# ---------------------------------------------------------------------------
# Test 4: build_metrics() constructs correct record
# ---------------------------------------------------------------------------

def test_build_metrics_constructs_record():
    m = build_metrics(
        thread_id="thread-build-001",
        pdf_path="/data/raw/Bakhu_NDA.pdf",
        started_at=_STARTED,
        run_status="completed",
        total_latency_seconds=12.5,
        prompt_tokens=800,
        completion_tokens=300,
        embedding_tokens=1500,
        ocr_fallback_triggered=True,
        ocr_fallback_page_count=2,
        low_confidence_retry_count=1,
        tot_invocation_count=3,
        hitl_escalation_count=2,
        guardrail_block_count=0,
    )

    assert m.thread_id == "thread-build-001"
    assert m.run_status == "completed"
    assert m.total_latency_seconds == 12.5
    assert m.prompt_tokens == 800
    assert m.completion_tokens == 300
    assert m.embedding_tokens == 1500
    assert m.ocr_fallback_triggered is True
    assert m.ocr_fallback_page_count == 2
    assert m.low_confidence_retry_count == 1
    assert m.tot_invocation_count == 3
    assert m.hitl_escalation_count == 2
    assert m.guardrail_block_count == 0
    assert m.error_message is None

    expected_cost = calculate_cost(800, 300, 1500)
    assert abs(m.estimated_cost_usd - expected_cost) < 1e-9

    # run_id is a fresh UUID4 each call
    assert len(m.run_id) == 36


def test_build_metrics_generates_unique_run_ids():
    m1 = build_metrics(
        thread_id="t1", pdf_path="/x.pdf", started_at=_STARTED,
        run_status="completed", total_latency_seconds=1.0,
    )
    m2 = build_metrics(
        thread_id="t1", pdf_path="/x.pdf", started_at=_STARTED,
        run_status="completed", total_latency_seconds=1.0,
    )
    assert m1.run_id != m2.run_id


# ---------------------------------------------------------------------------
# Test 5: write_metrics() appends a valid JSONL line; directory auto-created
# ---------------------------------------------------------------------------

def test_write_metrics_creates_file_and_line(tmp_path):
    log = tmp_path / "metrics" / "metrics_log.jsonl"
    m = build_metrics(
        thread_id="thread-write-001",
        pdf_path="/data/raw/Test.pdf",
        started_at=_STARTED,
        run_status="completed",
        total_latency_seconds=5.0,
    )

    write_metrics(m, log_path=log)

    assert log.exists()
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["thread_id"] == "thread-write-001"
    assert record["run_status"] == "completed"
    assert record["total_latency_seconds"] == 5.0


# ---------------------------------------------------------------------------
# Test 6: Second write appends (does not overwrite)
# ---------------------------------------------------------------------------

def test_write_metrics_appends(tmp_path):
    log = tmp_path / "metrics_log.jsonl"

    m1 = build_metrics(
        thread_id="t-first", pdf_path="/a.pdf", started_at=_STARTED,
        run_status="completed", total_latency_seconds=3.0,
    )
    m2 = build_metrics(
        thread_id="t-second", pdf_path="/b.pdf", started_at=_STARTED,
        run_status="completed", total_latency_seconds=4.0,
    )

    write_metrics(m1, log_path=log)
    write_metrics(m2, log_path=log)

    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["thread_id"] == "t-first"
    assert json.loads(lines[1])["thread_id"] == "t-second"


# ---------------------------------------------------------------------------
# Test 7: write_metrics() does not raise on I/O error
# ---------------------------------------------------------------------------

def test_write_metrics_does_not_raise_on_error(tmp_path, caplog):
    # Point to a path inside a non-existent path hierarchy that cannot be
    # created because a file already occupies the parent slot.
    blocker = tmp_path / "blocker"
    blocker.write_text("I am a file, not a directory")
    bad_path = blocker / "subdir" / "metrics_log.jsonl"

    m = _minimal_metrics()

    with caplog.at_level(logging.ERROR, logger="agent.metrics_writer"):
        write_metrics(m, log_path=bad_path)  # must not raise

    assert any("failed to write" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Test 8: 'interrupted' leg records partial numbers accurately
# ---------------------------------------------------------------------------

def test_interrupted_leg_records_partial_numbers(tmp_path):
    log = tmp_path / "metrics_log.jsonl"
    m = build_metrics(
        thread_id="thread-hitl-001",
        pdf_path="/data/raw/NDA.pdf",
        started_at=_STARTED,
        run_status="interrupted",
        total_latency_seconds=3.7,
        prompt_tokens=200,
        completion_tokens=80,
        embedding_tokens=500,
        hitl_escalation_count=1,
    )

    write_metrics(m, log_path=log)

    record = json.loads(log.read_text(encoding="utf-8").strip())
    assert record["run_status"] == "interrupted"
    assert record["total_latency_seconds"] == 3.7
    assert record["prompt_tokens"] == 200
    assert record["hitl_escalation_count"] == 1
    assert record["error_message"] is None


# ---------------------------------------------------------------------------
# Test 9: 'blocked' leg populates error_message
# ---------------------------------------------------------------------------

def test_blocked_leg_has_error_message(tmp_path):
    log = tmp_path / "metrics_log.jsonl"
    m = build_metrics(
        thread_id="thread-blocked-001",
        pdf_path="/data/raw/Blocked.pdf",
        started_at=_STARTED,
        run_status="blocked",
        total_latency_seconds=1.2,
        guardrail_block_count=1,
        error_message="Injection attempt detected in page 3.",
    )

    write_metrics(m, log_path=log)

    record = json.loads(log.read_text(encoding="utf-8").strip())
    assert record["run_status"] == "blocked"
    assert record["guardrail_block_count"] == 1
    assert "Injection" in record["error_message"]


# ---------------------------------------------------------------------------
# Test 10: Zero-token run produces cost=0.0
# ---------------------------------------------------------------------------

def test_zero_token_run_cost_is_zero():
    m = build_metrics(
        thread_id="t-zero", pdf_path="/x.pdf", started_at=_STARTED,
        run_status="completed", total_latency_seconds=0.5,
    )
    assert m.estimated_cost_usd == 0.0
    assert m.prompt_tokens == 0
    assert m.completion_tokens == 0
    assert m.embedding_tokens == 0
