"""
tests/integration/conftest.py
------------------------------
Session-scoped fixtures shared across all integration tests.

Isolation rule: integration tests run the real pipeline on real PDFs but must
never write to the production data files. This conftest redirects the feedback
log so that resume_after_review() → record_review_decision() → save_feedback()
calls go to a temp file, not to data/feedback/feedback_log.jsonl.
"""

import pytest
import agent.feedback_writer as fw_module
import agent.human_review as hr_module


@pytest.fixture(autouse=True)
def isolate_feedback_log(tmp_path, monkeypatch):
    """
    Redirect the feedback writer to a temp path for every integration test.

    Why: resume_after_review() calls record_review_decision(), which calls
    save_feedback() internally. Without this redirect, every integration test
    that resumes the pipeline writes a record to the real feedback_log.jsonl.
    """
    feedback_dir = tmp_path / "feedback"
    feedback_log = feedback_dir / "feedback_log.jsonl"
    monkeypatch.setattr(fw_module, "_FEEDBACK_DIR", feedback_dir)
    monkeypatch.setattr(fw_module, "_FEEDBACK_LOG", feedback_log)
