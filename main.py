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
from agent.human_review import record_review_decision, apply_human_decision
from datetime import datetime, timezone
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
if "is_submitting" not in st.session_state:
    st.session_state.is_submitting = False
if "collected_decisions" not in st.session_state:
    # Decisions gathered across all pending items before the single graph resume.
    st.session_state.collected_decisions = []

# ---------------------------------------------------------------------------
# Helper: run pipeline and store result in session state
# ---------------------------------------------------------------------------

def _to_report_md(report_obj) -> str | None:
    """Convert a generated_report (dict or LegalDocumentReport) to markdown."""
    if report_obj is None:
        return None
    if isinstance(report_obj, dict):
        # Amendment reports store pre-rendered markdown directly
        if report_obj.get("_is_amendment_report"):
            return report_obj.get("_markdown", "")
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
    st.session_state.collected_decisions = []   # reset for the new document
    with st.spinner("Running pipeline — this may take 30-90 seconds…"):
        result = run_pipeline(pdf_path)

    st.session_state.thread_id = result["thread_id"]
    st.session_state.status = result["status"]
    st.session_state.review_items = _to_review_items(result.get("review_items", []))

    # generated_report is a LegalDocumentReport object or None
    st.session_state.report_md = _to_report_md(result.get("generated_report"))


def _resume(decision: ReviewDecision, prior_decisions: list[dict] | None = None) -> None:
    # Optimistically clear items so a mid-rerun cannot re-render the form
    # and attempt a second resume on the same thread.
    st.session_state.review_items = []
    st.session_state.is_submitting = True
    try:
        with st.spinner("Applying decision and resuming pipeline…"):
            result = resume_after_review(
                st.session_state.thread_id, decision, prior_decisions=prior_decisions
            )
    except ValueError as exc:
        # Thread already completed or was never paused — most commonly caused
        # by a Streamlit double-rerun clicking Submit twice in quick succession.
        st.session_state.is_submitting = False
        st.error(
            f"Could not apply decision: the pipeline thread is no longer paused. "
            f"This usually means the decision was already submitted. "
            f"Check the report below, or reset and re-upload to start a new run.\n\n"
            f"Detail: {exc}"
        )
        return

    st.session_state.is_submitting = False
    st.session_state.collected_decisions = []   # all decisions consumed by resume
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

_PRESENCE_ICON = {
    "missing_critical_clause": "❌ Absent",
    "possible_clause_under_different_heading": "⚠️ Uncertain — may be under a different heading",
}
_RISK_COLOR = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
_EVIDENCE_LABEL = {"exact": "✅ Exact match", "fuzzy": "〜 Fuzzy match", "not_found": "❌ Not found"}

_REJECT_CATEGORIES = [
    "duplicate_finding",
    "wrong_clause_category",
    "hallucinated_deviation",
    "evidence_mismatch",
    "low_materiality",
    "template_mismatch",
    "not_legally_relevant",
]


def _presence_label(item: ReviewItem) -> str:
    if item.trigger_reason in _PRESENCE_ICON:
        return _PRESENCE_ICON[item.trigger_reason]
    if item.fact_found and "absent" in item.fact_found.lower():
        return "❌ Absent"
    return "✅ Present"


if st.session_state.status == "interrupted" and st.session_state.review_items and not st.session_state.is_submitting:
    st.divider()

    # Show partial report if available (pre-interrupt clauses may already be done)
    if st.session_state.report_md:
        with st.expander("Partial report (pre-interrupt)", expanded=False):
            st.markdown(st.session_state.report_md, unsafe_allow_html=False)

    items: list[ReviewItem] = st.session_state.review_items
    item: ReviewItem = items[0]  # Always show the first pending item

    total_items = len(items) + len(st.session_state.collected_decisions)
    current_item_num = len(st.session_state.collected_decisions) + 1
    remaining = len(items)

    st.subheader(f"Human Review Required — item {current_item_num} of {total_items}")
    st.progress(
        (current_item_num - 1) / total_items,
        text=f"{current_item_num - 1} of {total_items} items reviewed — {remaining} remaining",
    )

    # --- Finding card: distinct labeled fields ---
    with st.container(border=True):
        top_left, top_mid, top_right = st.columns(3)

        with top_left:
            st.markdown("**Clause**")
            st.write(str(item.clause_category) if item.clause_category else "N/A")

        with top_mid:
            st.markdown("**Document type**")
            st.write(item.document_type_context or "N/A")

        with top_right:
            st.markdown("**Page**")
            st.write(str(item.source_page) if item.source_page else "N/A")

        st.divider()

        sig_left, sig_mid, sig_right = st.columns(3)

        with sig_left:
            st.markdown("**Presence**")
            st.write(_presence_label(item))

        with sig_mid:
            st.markdown("**Extraction confidence**")
            clause_absent = _presence_label(item).startswith(("❌", "⚠️"))
            if clause_absent:
                st.write("N/A — no clause text to evaluate")
            else:
                conf = item.extraction_confidence
                st.write(f"{conf:.0%}" if conf is not None else "N/A")

        with sig_right:
            st.markdown("**Business risk**")
            icon = _RISK_COLOR.get(item.risk_level or "", "")
            st.write(f"{icon} {item.risk_level}" if item.risk_level else "N/A")

        st.divider()

        st.markdown("**What was found**")
        st.write(item.fact_found or "(none)")

        if item.deviation_found:
            st.markdown("**Deviation from benchmark**")
            st.write(item.deviation_found)

        st.markdown("**Risk rationale**")
        st.write(item.risk_rationale or "(none)")

        if item.template_comparison_summary:
            st.markdown("**Template comparison**")
            st.write(item.template_comparison_summary)

        if item.evidence_match_type:
            st.markdown("**Evidence verification**")
            label = _EVIDENCE_LABEL.get(item.evidence_match_type, item.evidence_match_type)
            score = f" (score: {item.evidence_match_score:.2f})" if item.evidence_match_score is not None else ""
            st.write(f"{label}{score}")

        with st.expander("Source text", expanded=False):
            display_text = item.expanded_clause_text or item.source_text or "(not available)"
            st.text(display_text[:2000])

    # --- Decision form ---
    st.markdown("#### Your decision")

    action_col, detail_col = st.columns([1, 2])

    with action_col:
        action = st.radio(
            "Action",
            ["approve", "correct", "reject", "select_alternative"],
            format_func=lambda x: {
                "approve": "✅ Approve AI finding",
                "correct": "✏️ Correct AI finding",
                "reject": "🗑️ Reject AI finding",
                "select_alternative": "🔀 Select alternative interpretation",
            }[x],
        )

    with detail_col:
        corrected_value = None
        selected_alt_id = None
        raw_correction = ""
        reject_category = None
        corrected_risk_level = None

        reviewer_note = st.text_input("Reviewer note (optional)")

        # Reason: required for HIGH-risk, optional otherwise
        reason_required = item.risk_level == "HIGH"
        reason = st.text_area(
            "Reason" + (" *(required for HIGH-risk findings)*" if reason_required else " (optional)"),
            height=80,
        )

        if action == "correct":
            corrected_risk_level = st.selectbox(
                "Corrected risk level (optional)",
                options=["", "LOW", "MEDIUM", "HIGH"],
                index=0,
            ) or None
            st.caption('Provide corrected clause details as JSON, e.g. {"clause_type": "Governing Law / Jurisdiction", "is_present": true}')
            raw_correction = st.text_area("Corrected value (JSON)", height=80)
            if raw_correction.strip():
                import json as _json
                try:
                    corrected_value = _json.loads(raw_correction)
                except ValueError:
                    st.error("Invalid JSON — please check the corrected value format.")
                    corrected_value = None

        elif action == "reject":
            reject_category = st.selectbox("Reject reason *(required)*", options=_REJECT_CATEGORIES)

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

        mark_precedent = st.checkbox(
            "Mark clause language as precedent candidate",
            help="Flag this clause's language as acceptable for future benchmark matching. Never feeds model training.",
        )
        flag_regression = st.checkbox(
            "Flag for regression dataset",
            help="Mark for manual curation into the regression test set. Does NOT automatically write anything or feed model training.",
        )

    # Validation gates — all checks run before ReviewDecision is constructed
    # so that pydantic never sees an invalid state and errors stay in the UI.
    can_submit = True
    if action == "select_alternative" and not item.alternatives:
        can_submit = False
        st.warning("No alternatives are available for this item — choose Approve or Correct instead.")
    if action == "correct" and not raw_correction.strip():
        can_submit = False
        st.warning("A corrected value (JSON) is required when action is 'Correct AI finding'.")
    elif action == "correct" and raw_correction.strip() and corrected_value is None:
        # Raw text present but JSON parse failed — error is already shown above the field.
        can_submit = False
    if reason_required and not reason.strip():
        can_submit = False
        st.warning("A reason is required for HIGH-risk findings before you can submit.")

    is_last_item = len(items) == 1
    btn_label = "Submit decision & generate report" if is_last_item else f"Submit & review next item ({len(items) - 1} remaining)"

    if st.button(btn_label, disabled=not can_submit, type="primary"):
        decision = ReviewDecision(
            review_id=item.review_id,
            action=action,
            corrected_value=corrected_value if action == "correct" else None,
            corrected_risk_level=corrected_risk_level if action == "correct" else None,
            selected_alternative_id=selected_alt_id if action == "select_alternative" else None,
            reject_category=reject_category if action == "reject" else None,
            reviewer_note=reviewer_note or None,
            reason=reason or "",
            risk_level_on_item=item.risk_level,
            mark_clause_language_as_precedent_candidate=mark_precedent,
            flag_for_regression_dataset=flag_regression,
        )
        if is_last_item:
            # All items reviewed — resume the graph exactly once, passing ALL
            # prior outcome dicts so every decision lands in human_decisions state
            # and appears in the final report.
            _resume(decision, prior_decisions=st.session_state.collected_decisions)
        else:
            # Not the last item — record this decision in the queue file and
            # compute its outcome dict, but do NOT resume the graph yet.
            try:
                resolved_item = record_review_decision(decision)
                outcome = apply_human_decision(decision, resolved_item)
            except Exception as exc:
                st.error(f"Failed to record decision for item {current_item_num}: {exc}")
                st.stop()
            # Store the enriched outcome dict — must include review_item.clause_category,
            # reviewer_note, reason, corrected_risk_level so node_generate_report and
            # render_markdown can match this decision to its risk finding in the report.
            # Without clause_category every finding falls into "AI Findings (Not Yet Reviewed)".
            intermediate = {
                "review_id": decision.review_id,
                "action": decision.action,
                "discarded": outcome.get("discarded", False),
                "value": outcome.get("value"),
                "reviewer_note": decision.reviewer_note,
                "reason": decision.reason or "",
                "corrected_risk_level": decision.corrected_risk_level,
                "corrected_summary": None,
                "decided_at": datetime.now(timezone.utc).isoformat(),
                "review_item": {"clause_category": resolved_item.clause_category},
            }
            print(f"[DEBUG] recording intermediate decision {current_item_num}: action={intermediate['action']} clause_category={resolved_item.clause_category}")
            st.session_state.collected_decisions.append(intermediate)
            # Advance to the next pending item by dropping the resolved one.
            st.session_state.review_items = items[1:]
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
        for key in ("thread_id", "status", "report_md", "review_items", "pdf_name", "collected_decisions"):
            st.session_state[key] = None if key not in ("review_items", "collected_decisions") else []
        st.session_state.is_submitting = False
        st.rerun()

    if st.session_state.thread_id:
        st.caption(f"Thread: `{st.session_state.thread_id}`")
    if st.session_state.status:
        st.caption(f"Status: `{st.session_state.status}`")
