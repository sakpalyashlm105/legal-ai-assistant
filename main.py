"""
main.py — Legal AI Assistant Streamlit UI

Thin display/trigger layer over the existing pipeline. No new business logic.

Run from legal-agent/:
    streamlit run main.py
"""

import tempfile
from pathlib import Path

import streamlit as st

from agent.orchestrator import run_pipeline, resume_after_review
from reporting.report_generator import render_markdown
from schemas.report import LegalDocumentReport
from schemas.review import ReviewDecision, ReviewItem

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
)

st.title("Legal AI Assistant")
st.caption("Upload a contract PDF to extract clauses, assess risk, and generate a review report.")

# ---------------------------------------------------------------------------
# Session state keys
# ---------------------------------------------------------------------------
# thread_id     : str | None   — active pipeline thread
# status        : str | None   — "completed" | "interrupted" | "blocked"
# report_md     : str | None   — rendered markdown for the current report
# review_items  : list         — pending ReviewItem objects when interrupted

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "status" not in st.session_state:
    st.session_state.status = None
if "report_md" not in st.session_state:
    st.session_state.report_md = None
if "review_items" not in st.session_state:
    st.session_state.review_items = []
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# ---------------------------------------------------------------------------
# Helper: run pipeline and store result in session state
# ---------------------------------------------------------------------------

def _to_report_md(report_obj) -> str | None:
    """Convert a generated_report (dict or LegalDocumentReport) to markdown."""
    if report_obj is None:
        return None
    if isinstance(report_obj, dict):
        report_obj = LegalDocumentReport(**report_obj)
    return render_markdown(report_obj)


def _to_review_items(raw: list) -> list[ReviewItem]:
    """Convert the orchestrator's dict list to ReviewItem objects."""
    out = []
    for r in raw:
        if isinstance(r, ReviewItem):
            out.append(r)
        elif isinstance(r, dict):
            out.append(ReviewItem(**r))
    return out


def _run(pdf_path: str, filename: str) -> None:
    st.session_state.pdf_name = filename
    with st.spinner("Running pipeline — this may take 30-90 seconds…"):
        result = run_pipeline(pdf_path)

    st.session_state.thread_id = result["thread_id"]
    st.session_state.status = result["status"]
    st.session_state.review_items = _to_review_items(result.get("review_items", []))

    # generated_report is a LegalDocumentReport object or None
    st.session_state.report_md = _to_report_md(result.get("generated_report"))


def _resume(decision: ReviewDecision) -> None:
    with st.spinner("Applying decision and resuming pipeline…"):
        result = resume_after_review(st.session_state.thread_id, decision)

    st.session_state.status = result["status"]
    st.session_state.review_items = _to_review_items(result.get("review_items", []))

    st.session_state.report_md = _to_report_md(result.get("generated_report"))


# ---------------------------------------------------------------------------
# Section 1: Upload + run
# ---------------------------------------------------------------------------

uploaded = st.file_uploader("Upload contract PDF", type=["pdf"])

if uploaded is not None:
    # Only trigger a new run if this is a fresh upload (different filename or
    # no run yet). This prevents re-running on every Streamlit rerun while the
    # user is interacting with the review panel.
    if uploaded.name != st.session_state.pdf_name:
        # Save to a temp file so the pipeline can read it by path
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        _run(tmp_path, uploaded.name)
        st.rerun()

# ---------------------------------------------------------------------------
# Section 2: HITL review panel (shown when interrupted)
# ---------------------------------------------------------------------------

if st.session_state.status == "interrupted" and st.session_state.review_items:
    st.divider()

    # Show partial report if available (pre-interrupt clauses may already be done)
    if st.session_state.report_md:
        with st.expander("Partial report (pre-interrupt)", expanded=False):
            st.markdown(st.session_state.report_md, unsafe_allow_html=False)

    items: list[ReviewItem] = st.session_state.review_items
    item: ReviewItem = items[0]  # Always resolve the first pending item

    remaining = len(items)
    st.subheader(f"Human Review Required ({remaining} item{'s' if remaining > 1 else ''} pending)")
    st.info(
        f"**Clause:** {item.clause_category or 'N/A'}  \n"
        f"**Risk level:** {item.risk_level or 'N/A'}  \n"
        f"**AI finding:** {item.ai_finding_summary or '(none)'}  \n"
        f"**Source text:** {(item.source_text or '')[:400] or '(not available)'}"
    )

    action_col, detail_col = st.columns([1, 2])

    with action_col:
        action = st.radio(
            "Decision",
            ["approve", "correct", "reject", "select_alternative"],
            format_func=lambda x: {
                "approve": "Approve AI finding",
                "correct": "Correct AI finding",
                "reject": "Reject AI finding",
                "select_alternative": "Select alternative interpretation",
            }[x],
        )

    with detail_col:
        corrected_value = None
        selected_alt_id = None
        raw_correction = ""
        reviewer_note = st.text_input("Reviewer note (optional)")

        if action == "correct":
            st.caption("Provide the corrected clause details as JSON, e.g. {\"clause_type\": \"Governing Law / Jurisdiction\", \"is_present\": true}")
            raw_correction = st.text_area("Corrected value (JSON)", height=80)
            if raw_correction.strip():
                import json as _json
                try:
                    corrected_value = _json.loads(raw_correction)
                except ValueError:
                    st.error("Invalid JSON — please check the corrected value format.")
                    corrected_value = None

        elif action == "select_alternative":
            if item.alternatives:
                alt_labels = {a["id"]: a.get("summary", a["id"]) for a in item.alternatives}
                selected_alt_id = st.radio(
                    "Choose alternative",
                    options=list(alt_labels.keys()),
                    format_func=lambda k: alt_labels[k],
                )
            else:
                st.warning("No alternatives available for this item — use Approve or Correct instead.")

    # Block submission when select_alternative chosen but no alternatives exist
    can_submit = not (action == "select_alternative" and not item.alternatives)
    # Block submission when correct chosen but the JSON field is non-empty yet invalid
    if action == "correct" and raw_correction.strip() and corrected_value is None:
        can_submit = False

    if st.button("Submit decision", disabled=not can_submit, type="primary"):
        decision = ReviewDecision(
            review_id=item.review_id,
            action=action,
            corrected_value=corrected_value if action == "correct" else None,
            selected_alternative_id=selected_alt_id if action == "select_alternative" else None,
            reviewer_note=reviewer_note or None,
        )
        _resume(decision)
        st.rerun()

# ---------------------------------------------------------------------------
# Section 3: Completed report
# ---------------------------------------------------------------------------

elif st.session_state.status == "completed" and st.session_state.report_md:
    st.divider()
    st.success(f"Pipeline completed — {st.session_state.pdf_name}")
    st.markdown(st.session_state.report_md, unsafe_allow_html=False)

elif st.session_state.status == "blocked":
    st.divider()
    st.error(
        "Pipeline blocked by guardrail. The document did not pass content safety checks. "
        "Check the application logs for details."
    )

# ---------------------------------------------------------------------------
# Reset button (sidebar)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Controls")
    if st.button("Reset / analyse new document"):
        for key in ("thread_id", "status", "report_md", "review_items", "pdf_name"):
            st.session_state[key] = None if key != "review_items" else []
        st.rerun()

    if st.session_state.thread_id:
        st.caption(f"Thread: `{st.session_state.thread_id}`")
    if st.session_state.status:
        st.caption(f"Status: `{st.session_state.status}`")
