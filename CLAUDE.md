# CLAUDE.md — Legal AI Assistant

This file is read automatically by Claude Code at the start of every session in this repository. It contains operational facts about the project: what exists, what's next, conventions to follow, and commands to run. For full conceptual/architectural rationale, the developer maintains a separate master prompt used in chat conversations — this file is the terse, action-oriented companion to that.

> **IMPORTANT — read this before starting any task:** If anything in this file seems inconsistent with what the developer describes as already done, **ASK for clarification** rather than trusting this file blindly or trusting your own assumption. This file is updated manually and can lag behind real progress. When in doubt, ask the developer to confirm current status before proceeding.

---

## Project Identity

Legal AI Assistant — an agentic system that analyzes legal documents (NDAs, contracts, supplier agreements, amendments), extracts clauses, compares against templates, scores risk, and generates reports with human-in-the-loop review. Built as a graded capstone project by Yash, an Intelligent Automation Analyst learning AI/ML concepts for the first time. **Treat Yash as a beginner on RAG/LangGraph/embeddings/agents — explain concepts before writing code, one step at a time, wait for confirmation before proceeding.**

This is NOT a production deployment. It's a capstone with graded checkpoints (Steps 1–10 submitted, all scored 1/1). Do not add production-scale infrastructure (Kubernetes, managed vector DBs, multi-tenant auth, etc.) — see "Explicitly Out of Scope" below.

---

## Current Status

### Steps 1–10: COMPLETE AND VERIFIED — do not recreate

**Step 1–2: Project scaffolding + data inventory**
- `requirements.txt`, `config.py`, `verify_setup.py`, `.env`, `.gitignore`, `README.md`
- `data/raw/` populated with 358 files: `ndas/` (5), `contracts/` (343), `amendments/` (5), `other/` (0), `cuad_labels/` (5)
- `data/raw/metadata.csv`, `DATA_INVENTORY.md`
- All design/scoping documents (scoping, agent design, RAG design, ToT design, multi-agent analysis) — written and graded

**Step 3: Text extraction pipeline**
- `extraction/pdf_parser.py` — PyMuPDF primary extractor with `normalize_text()`
- `extraction/ocr.py` — GPT-4o-mini Vision OCR fallback (triggered when page yields < 100 chars)

**Step 4: Chunking + FAISS vector store**
- `retrieval/chunking.py` — sliding-window chunker with configurable overlap
- `retrieval/vector_store.py` — FAISS with dense + MMR retrieval; batch embedding (one call for all 10 clause queries); LLM reranking skipped when top-1 cosine > 0.92

**Step 5: Document classification + clause extraction**
- `agent/classifier.py` — GPT-4o-mini classifies document type; three-tier confidence routing
- `agent/extractor.py` — GPT-4o-mini extracts all 10 clause categories

**Step 6: Template comparison + risk scoring**
- `agent/comparator.py` — compares extracted clauses against stored templates
- `agent/risk_engine.py` — precedent-aware risk scoring; absence is always HIGH regardless of precedent

**Step 7: Tree-of-Thought reasoning + confidence routing**
- `agent/tot_reasoner.py` — beam search width 3, max depth 3, pruning at 0.30 (depth 1) and 0.40 (depth 2); structured candidates stored in state

**Step 8: Human-in-the-Loop — real LangGraph interrupt/checkpointer**
- `agent/orchestrator.py` — full LangGraph state machine with `InMemorySaver` checkpointer; `run_pipeline()` / `resume_after_review()` / `get_graph_state()` API
- `agent/human_review.py` — HITL queue and review schema
- `demo_hitl.py` — interactive interrupt/resume demo

**Steps 8/9 deepening — HITL reviewer experience upgrade (post-Step-11 enhancement, not a new numbered step)**
- `schemas/review.py` — `ReviewItem` extended with 15 reviewer-context fields (template text, fact/deviation/rationale split, evidence match metadata, adjacent chunk context, document type context). `ReviewDecision` extended with `reason` (required for HIGH-risk), `reject_category` (required on reject), `corrected_risk_level/summary/rationale`, `flag_for_regression_dataset`.
- `agent/orchestrator.py` — `node_build_review_items` now populates all new ReviewItem fields from upstream state (comparisons, chunks via `get_local_context()`, classification type, confidence).
- `guardrails/claim_verifier.py` / `guardrails/output_validator.py` — both call sites updated to populate new context fields.
- `demo_hitl.py` — rich review card display (full clause text, source context, template comparison, verification metadata); multi-item review loop (all items reviewed before graph resumes — previously only processed first item); structured prompts for correct/reject with reason validation.
- `reporting/report_generator.py` — Human Review Decisions section now splits findings into: "AI Findings (Not Yet Reviewed)", "Human-Approved Findings", "Human-Corrected Findings", "Rejected Findings (Excluded from Risk Summary)". Rejected findings are excluded from High-Risk and Medium-Risk tables.
- Audit trail: all `ReviewDecision` fields (including new ones) are persisted to `data/review_queue/resolved_reviews.json` via `record_review_decision()` — confirmed durable.
- Tests: `tests/unit/test_review_schema_v2.py` (25 tests covering all new schema validators); call-site field-population tests added to `tests/unit/test_claim_verifier.py` (6 new tests — `TestEscalatedReviewItemNewFields`) and `tests/unit/test_output_validator.py` (7 new tests — `TestEscalatedReviewItemNewFields`, `TestRejectedFindingsCountInteraction`).
- `flag_for_regression_dataset` docstring explicitly states: NEVER feeds into model training or fine-tuning — flags for manual regression dataset curation only.
- Note on `section_heading`: `DocumentChunk` schema has no `section_heading` field — this field is always `None` in `ReviewItem` (documented, not silently absent).
- Note on `extraction_confidence` vs `llm_confidence`: only one confidence value exists per clause — both fields are populated from the same `ExtractedClause.confidence` value (not fabricated as two independent signals).
- Step 11 + rejected-findings design decision: Rejected findings still count toward `total_clauses_found`. Reject categories describe why the AI's risk analysis was wrong, not whether the clause was detected. `total_clauses_found` reflects extraction results (clause_entries), not risk assessment outcomes. No fix needed to output_validator — the count check already uses clause_entries, which is unchanged by HITL decisions. Confirmed by `TestRejectedFindingsCountInteraction`.

**Step 9: Pre/post-generation guardrails, wired as real graph nodes**
- `guardrails/input_validator.py` — file type, size, path validation
- `guardrails/scope_validator.py` — document scope pre-check
- `guardrails/prompt_injection.py` — deterministic regex scanner (6 technique categories, OWASP-aligned)
- `guardrails/extraction_validator.py` — validates extracted text before LLM processing
- `guardrails/evidence_verifier.py` — verifies extracted clause text exists verbatim in source PDF
- `guardrails/page_verifier.py` — verifies page references are valid and in-range
- `guardrails/claim_verifier.py` — verifies final report claims against source extractions

**Step 11: validate_final_output — report-level structural guardrail**
- `guardrails/output_validator.py` — four checks on the assembled report:
  1. Disclaimer presence (auto-fixable: re-injects `DISCLAIMER` constant)
  2. Internal count consistency: `total_clauses_found`/`total_clauses_missing` vs actual `clause_entries`, orphaned `missing_clauses` names (escalated)
  3. Schema completeness: empty `document_name`/`executive_summary`, HIGH/MEDIUM `RiskFindingEntry` with empty `finding_summary` (escalated)
  4. Guardrail-to-report disconnect: `evidence_verified=False` + `risk_level=HIGH/MEDIUM` + `human_review_status="not_required"` (always escalated — most safety-critical)
- Escalation mechanism: same `add_to_review_queue()` path as `claim_verifier.py`; does NOT re-pause the graph (log-only; caller sees `validation_passed` in `final_state`)
- `schemas/report.py` — added `FinalOutputValidationResult` schema
- Wired into orchestrator: `generate_report → validate_final_output → finalize`
- `FinalOutputValidationResult` stored in state as `final_output_validation`, surfaced in `final_state` via `validation_passed`, `validation_auto_fixes`, `validation_escalations`
- Tests: `tests/unit/test_final_output_validation_schema.py` (6), `tests/unit/test_output_validator.py` (30, including 7 new call-site + rejected-findings tests)

**Step 10: Report generation**
- `reporting/report_generator.py` — fully deterministic markdown report (`render_markdown`)
- `reporting/executive_summary.py` — single constrained LLM call with two post-generation validators:
  - Validator 1: verdict-phrase scanner (`_check_output`) — rejects "is enforceable", "legally binding", etc.
  - Validator 2: evidence-coverage phrasing scanner (`check_evidence_coverage_phrasing`) — rejects blanket "no evidence issues" when absent clauses exist; requires explicit present-verified AND absent-N/A distinction
  - Generalized retry loop via `_validate_candidate()` — adding a new validator is one line
- `prompts/report_generation.yaml` — includes `{evidence_verified_count}` and `{evidence_na_count}` template vars and hard constraint on N/A phrasing

**Verification:**
- 515 unit and integration tests passing (494 unit + 21 integration)
- Two real end-to-end pipeline runs on `data/pdf/ndas/NDA_2_Bakhu_Holdings,_Corp._(BKUH).pdf` with live GPT-4o-mini calls — confirmed correct missing-clause detection, evidence verification, and report generation
- LangGraph pinned at 1.2.6 (current `interrupt()` / `Command(resume=...)` API)
- `openai`, `langchain`, `langchain-openai` explicitly pinned in `requirements.txt`
- Single canonical venv (stale duplicate removed)

---

### Step 12 — feedback + precedent lifecycle: COMPLETE (all 5 stages)

**578 tests passing** (515 baseline + 27 Stage 2 + 14 Stage 3 + 1 pre-Stage-4 fix + 11 Stage 4 + 10 Stage 5).

**Completed:**
- Stage 1: `schemas/feedback.py` — `FeedbackRecord` + `PrecedentScope` with 3 Pydantic `model_validator(mode="after")` safety invariants
- Stage 2: 27 schema tests in `tests/unit/test_feedback_schema.py`
- Stage 3: `agent/feedback_writer.py` — `save_feedback()` with idempotency, eligibility classification, PII-safe logging (500-char evidence excerpt cap)
- Pre-Stage-4 fix: `clause_category: Optional[ClauseType] = None` on `FeedbackRecord`; `"Other"` is NOT a valid `ClauseType`; `None` → `not_eligible`
- Stage 4: `agent/feedback_curation.py` — `approve_feedback_as_precedent()` with 8 ordered validation checks, named exception hierarchy, atomic JSONL rewrite via `os.replace()`, `_clear_feedback_cache()` call. Committed: `9326783`.
- Stage 5: `agent/risk_engine.py` — JSONL reader, windowed difflib matching at 0.70, document_type null-compatible scope, REG-001 defense-in-depth at two independent levels. Committed: `15cd1b9`.

**Stage 5 key changes to `risk_engine.py`:**
- `FEEDBACK_LOG_PATH`: `data/processed/feedback_log.json` → `data/feedback/feedback_log.jsonl`
- `_load_feedback_log()`: flat JSON array → line-by-line `FeedbackRecord`; filters to `approved_for_precedent=True + feedback_status=approved_precedent + final_risk=MEDIUM`; malformed lines → WARNING + skip, never raise
- `_find_precedent()`: plain substring → `_best_window_score()` from `evidence_verifier.py` at `FUZZY_MATCH_THRESHOLD=0.70`; returns `(FeedbackRecord, score)` tuple. **Function name note:** Stage 1 recon notes used `_windowed_similarity` — that name is wrong. The actual function is `_best_window_score` (one implementation, no duplicate).
- `_apply_precedent_downgrade()`: REG-001 independent check (`is_clause_present` guard); `precedent_note` now records `feedback_id`, match score, match reason, risk trajectory
- `flag_risks()`: `+document_type: Optional[str] = None` for scope matching

**Real verification (both required, both done):**
- Part A direct: `_find_precedent("Governing Law / Jurisdiction", illinois_text)` → `fb_rev-medium-report-001`, score=1.00, HIGH→MEDIUM applied
- Part B pipeline: Bakhu NDA end-to-end. Governing Law / Jurisdiction downgraded HIGH→MEDIUM at 0.71 text similarity against the Illinois precedent. 6 missing clauses all `HIGH, precedent_applied=False` — **REG-001 held on real data**.

**Known limitation — jurisdiction-blind matching:** Precedent matching currently compares `clause_category` + text similarity only. `jurisdiction` is accepted as compatible whenever either side is `None`, which is always true today since jurisdiction is never extracted from real documents. This means a precedent from one jurisdiction can be applied to a clause from a different (or unknown) jurisdiction if the boilerplate phrasing is similar enough — observable in the Bakhu NDA run (Illinois precedent applied at 0.71 similarity). Not a bug (consistent with locked Step 12 design); candidate for future work alongside the already-deferred `risk_policy.yaml` / document-type-aware rules.

**Test isolation confirmed solid:** `test_feedback_writer.py` uses `autouse=True`; `test_feedback_curation.py` and `test_risk_engine_precedent.py` use explicit-parameter fixture on every test. No test writes to the real `data/feedback/feedback_log.jsonl`.

**`data/feedback/feedback_log.jsonl` state:** 2 records — `fb_bakhu-live-demo-001` (`not_eligible`) and `fb_rev-medium-report-001` (`approved_precedent`, Illinois Governing Law). Backup at `feedback_log.jsonl.bak`.

**Key design invariants (never conflate):**
- `model_finding_accepted` derived **only** from `review_action == "approve"`
- `clause_language_accepted_as_business_precedent` derived **only** from `mark_clause_language_as_precedent_candidate`
- HIGH-risk and missing-clause findings **can never be promoted to approved_precedent** — enforced at schema level (Pydantic validators) AND at logic level (both `flag_risks` and `_apply_precedent_downgrade`)

**Known environment quirk:** stale `pytest.exe` launcher (OneDrive sync artifact). Use `.\venv\Scripts\python.exe -m pytest tests/ -v --tb=short`. Also set `$env:PYTHONIOENCODING="utf-8"` for `run_pipeline.py` (arrow `→` in print statement hits cp1252 encoding on Windows console).

**Commit discipline:** commit after each stage, not batched across sessions.

**Do not start Step 13 until Step 12 (all stages) is complete and tested.**

---

### Remaining work (priority order after Step 12)

| Step | What | Notes |
|---|---|---|
| 12 | `save_feedback` — persist real HITL decisions | Risk engine's precedent-override currently reads from an empty log — **CURRENT NEXT TASK** |
| 13 | `record_metrics` — latency/cost/token/OCR-fallback-rate | Per-pipeline-run structured log |
| 14 | `route_by_type` + `retrieve_base_agreement` | Amendment-specific routing; 5 real amendment PDFs in `data/pdf/amendments/` untouched |
| 15 | `verify_missing_clauses` (stricter) | Search synonyms/related headings before concluding absence; current Step 6 treats one failed extraction as sufficient |
| 16 | Full orchestrator wiring | Complete node sequence including amendment routing |
| — | Streamlit UI (`main.py` is a stub) | After Steps 12–13 |
| — | Full evaluation framework | CUAD scoring, retrieval Precision@3, regression suite automation |
| — | Expanded observability | LangSmith tracing, audit logging, ARCHITECTURE.md, SECURITY_AND_PRIVACY.md |

**Explicitly deferred from the HITL-deepening pass (Steps 8/9 enhancement) — do NOT build in Step 12:**

| Deferred Item | What | Why Deferred |
|---|---|---|
| `request_more_context` action | A fifth HITL action allowing the reviewer to request additional document context before deciding | Changes the locked 4-action set; needs deliberate design discussion before implementation |
| `risk_policy.yaml` / document-type-aware risk rules | Document-type-specific risk scoring (e.g. amendments vs NDAs have different risk profiles) | Changes Step 6's locked risk-scoring logic; needs its own dedicated task |
| Amendment-specific "original agreement not available" handling | Detect when an amendment is being analyzed without its base agreement and surface that as a structural finding | This is Step 14 (`route_by_type` + `retrieve_base_agreement`), already in the remaining-work table above |

---

## Tech Stack (locked — do not swap without evidence of failure)

| Layer | Choice |
|---|---|
| LLM | OpenAI GPT-4o-mini (classification, extraction, comparison, reranking, risk, reports) |
| Embeddings | OpenAI text-embedding-3-small (1536-dim) |
| PDF extraction | PyMuPDF primary |
| OCR fallback | GPT-4o-mini Vision (per-page, triggered when a page yields < 100 chars) |
| Vector store | FAISS, dense + MMR, top-10 -> LLM rerank -> top-3 |
| Orchestration | LangGraph 1.2.6 (state machine, conditional edges, `InMemorySaver`) + LangChain |
| UI | Streamlit (not yet built — `main.py` is a stub) |
| Validation | Pydantic v2 schemas everywhere |
| Evaluation | CUAD dataset, scikit-learn metrics, custom regression suite |
| Observability | LangSmith (content tracing OFF by default) |

**Explicitly out of scope** (do not add): fine-tuning, Ollama, Tesseract, MCP, managed vector DB, hybrid BM25 search (unless retrieval evals prove dense-only is failing on exact-term queries), Kubernetes, multi-provider failover, RLHF, SSO/RBAC, semantic answer caching, large-scale load testing.

---

## Architecture Facts Claude Code Must Know

**10 approved clause categories** (never silently add/change): Confidentiality, Termination for Convenience, Termination for Cause, Governing Law, Indemnification, Limitation of Liability, Non-Compete, Assignment, Renewal/Term, Dispute Resolution.

**Confidence routing is three-tier, not binary** — this is a graded design commitment, implement exactly this:
- `< 0.5` -> self-retry once with refined prompt -> if still < 0.5, escalate to human review
- `0.5–0.7` -> route to Tree-of-Thought reasoning (not retry, not auto-proceed)
- `> 0.7` -> proceed automatically

**Tree-of-Thought**: beam search, width 3, max depth 3, 2–4 candidates per ambiguous paragraph, pruning at 0.30 (depth 1) and 0.40 (depth 2), preserve pruned candidates in state (don't discard them). Never expose raw chain-of-thought to logs or users — store structured candidate/evidence/score objects only.

**Risk scoring has a precedent-aware override** — before finalizing a non-standard-clause risk level, check feedback log for similar previously-approved clauses; if found, downgrade one tier (HIGH->MEDIUM) and annotate with the precedent. **This override never applies to missing-clause findings** — absence is always HIGH regardless of precedent.

**Retrieval latency mitigations** (implemented in Step 4): batch all clause queries into one embedding call rather than 10 separate ones; cache base-agreement/template retrieval per document session; skip the LLM reranking call entirely when FAISS top-1 cosine similarity > 0.92.

**This is a single LangGraph workflow with specialized nodes, not a true multi-agent system.** Don't describe it as "multi-agent" in docs or code comments unless distinct autonomous agents with separate state/communication are actually built — they are not, by design decision.

**HITL, not RLHF.** Human corrections never update model weights. Never use the term RLHF anywhere in code, comments, or docs.

**Evidence verification semantics** (critical — do not conflate):
- `evidence_verified=True` — clause is present, text was found verbatim in source PDF, passed
- `evidence_verified=False` — clause is present, text was searched but NOT found, failed
- `evidence_verified=None` — clause is absent (never extracted); N/A, not "verified clean"

The executive summary MUST distinguish "N clauses present and verified" from "M clauses absent and not applicable." The `check_evidence_coverage_phrasing` validator enforces this deterministically.

**`flagged_segments` in `InjectionScanResult` MUST NOT be written to persistent logs.** Log category labels only.

---

## Full LangGraph Node Sequence

Status key: ✅ done | ⚠️ partial | ❌ not built

```
✅ validate_input          guardrails/input_validator.py
✅ validate_scope          guardrails/scope_validator.py
✅ scan_prompt_injection   guardrails/prompt_injection.py
✅ extract_text            extraction/pdf_parser.py, extraction/ocr.py
✅ validate_extraction     guardrails/extraction_validator.py
✅ classify_document       agent/classifier.py
❌ route_by_type           inline conditional in orchestrator (amendment path not built)
❌ retrieve_base_agreement amendments only — not built; 5 amendment PDFs untouched
✅ chunk_document          retrieval/chunking.py
✅ retrieve_context        retrieval/vector_store.py
✅ rerank_context          retrieval/vector_store.py (inline, skipped when cosine > 0.92)
✅ extract_clauses         agent/extractor.py
⚠️ validate_schema         inline in extractor (no dedicated node)
✅ verify_evidence         guardrails/evidence_verifier.py
✅ verify_page_references  guardrails/page_verifier.py
⚠️ calculate_confidence    inline in classifier/extractor (no dedicated node)
✅ run_tot_if_needed       agent/tot_reasoner.py
✅ compare_to_templates    agent/comparator.py
✅ flag_risks              agent/risk_engine.py
⚠️ verify_missing_clauses  basic version in claim_verifier; stricter synonym search not built (Step 15)
✅ verify_final_claims     guardrails/claim_verifier.py
✅ human_review            agent/human_review.py + orchestrator interrupt/resume
✅ generate_report         reporting/report_generator.py + reporting/executive_summary.py
✅ validate_final_output   guardrails/output_validator.py (Step 11 — complete)
✅ save_feedback           Step 12 — all 5 stages complete; 578 tests passing
❌ record_metrics          Step 13
```

---

## Project Structure (actual files as of Step 10)

```
legal-agent/
├── main.py                    (Streamlit UI — STUB, not built)
├── config.py
├── requirements.txt
├── verify_setup.py
├── demo_hitl.py               (HITL interrupt/resume interactive demo)
├── run_pipeline.py
├── .env, .gitignore, README.md, CLAUDE.md, DATA_INVENTORY.md
│
├── agent/
│   ├── classifier.py
│   ├── extractor.py
│   ├── comparator.py
│   ├── risk_engine.py
│   ├── tot_reasoner.py
│   ├── human_review.py
│   └── orchestrator.py
│
├── extraction/
│   ├── pdf_parser.py
│   └── ocr.py
│
├── retrieval/
│   ├── chunking.py
│   └── vector_store.py
│
├── guardrails/
│   ├── input_validator.py
│   ├── scope_validator.py
│   ├── prompt_injection.py
│   ├── extraction_validator.py
│   ├── evidence_verifier.py
│   ├── page_verifier.py
│   ├── claim_verifier.py
│   └── output_validator.py       (Step 11)
│
├── schemas/
│   ├── document.py
│   ├── clause.py
│   ├── chunk.py
│   ├── risk.py
│   ├── tot.py
│   ├── review.py
│   ├── guardrails.py
│   └── report.py
│
├── prompts/
│   ├── classify_document.yaml
│   ├── extract_clauses.yaml
│   ├── compare_to_template.yaml
│   ├── tot_generator.yaml
│   ├── tot_critic.yaml
│   └── report_generation.yaml
│
├── reporting/
│   ├── report_generator.py
│   └── executive_summary.py
│
├── evaluation/
│   └── eval_cuad.py           (stub)
│
├── docs/
│   └── LIMITATIONS.md
│
├── tests/
│   ├── unit/
│   │   ├── test_extraction_pipeline.py
│   │   ├── test_chunking.py
│   │   ├── test_classifier.py
│   │   ├── test_extractor.py
│   │   ├── test_comparator.py
│   │   ├── test_risk_engine.py
│   │   ├── test_human_review.py
│   │   ├── test_input_validator.py
│   │   ├── test_prompt_injection.py
│   │   ├── test_evidence_verifier.py
│   │   ├── test_page_verifier.py
│   │   ├── test_claim_verifier.py
│   │   ├── test_guardrails_schema.py
│   │   ├── test_report_schema.py
│   │   ├── test_report_generator.py
│   │   ├── test_executive_summary.py
│   │   ├── test_review_schema.py
│   │   └── test_review_schema_v2.py   (Steps 8/9 deepening — 26 tests)
│   └── integration/
│       ├── test_hitl_interrupt.py
│       ├── test_orchestrator_guardrails.py
│       ├── test_report_generation.py
│       ├── test_tot_real.py
│       └── test_end_to_end.py
│
└── data/
    ├── raw/        POPULATED — 358 files, do not recreate or re-sort
    ├── processed/  generated reports written here
    ├── templates/
    └── vector_store/
```

**Not yet created** (build only when the step needs them):
- `feedback/feedback_manager.py` — Step 12
- `observability/` (tracing.py, metrics.py, logging_config.py) — later
- `audit/audit_logger.py` — later
- `core/` (retry.py, timeout.py, error_handler.py, cache.py, token_budget.py) — later
- `docs/ARCHITECTURE.md`, `SECURITY_AND_PRIVACY.md`, `THREAT_MODEL.md`, etc. — later

---

## Coding Conventions

- **Modular Python.** One responsibility per file/function.
- **Educational comments.** This is a learning project — comment WHY, not just WHAT.
- **Before/after diffs when editing existing files.** Don't silently rewrite whole files the developer hasn't seen change-by-change.
- **`.env` for all secrets.** Never hardcode `OPENAI_API_KEY` or any credential. Read via `config.py`.
- **Pydantic for every structured LLM output.** Confidence fields always 0–1. Risk fields always one of `LOW`/`MEDIUM`/`HIGH`. Review actions always one of `approve`/`correct`/`reject`/`select_alternative`.
- **No silent fabrication.** If extraction/OCR fails, evidence can't be verified, or a page reference is invalid — fail loudly/gracefully, never invent plausible-looking output.
- **No bare `except:`.** Classify errors (retryable vs. not): refusal, timeout, rate-limit, malformed output, etc. are distinct cases with distinct handling.
- **Prompts live in `prompts/*.yaml`**, not inline strings in Python.
- **PII-safe logging.** Never log full contract text by default. Log document ID, hash, page count, method, category, confidence, risk level — not raw content. `ENABLE_CONTENT_TRACING=False` and `ENABLE_PII_REDACTION=True` are the defaults.
- **Never state a definitive legal verdict.** Always use the observation + recommendation pattern.
- **Confidence language must never overclaim.** Never phrase anything as a calibrated statistical probability.

---

## Commands

```bash
# Activate venv (run this first, every session)
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

# Install/verify
pip install -r requirements.txt
python verify_setup.py

# Run all tests (409 passing as of Step 10)
pytest tests/ -v

# Run only unit tests (faster, no real LLM calls)
pytest tests/unit/ -v

# Run integration tests (makes real OpenAI API calls)
pytest tests/integration/ -v

# Run the HITL demo (real LLM calls on a specific PDF)
python demo_hitl.py --pdf "data/pdf/ndas/NDA_2_Bakhu_Holdings,_Corp._(BKUH).pdf"

# Run the app UI (once main.py is built — not yet)
streamlit run main.py

# Run CUAD evaluation (Phase 2+)
python -m evaluation.eval_cuad
```

---

## Working Agreement for This Session

1. Explain the concept before the code. Use SAP/Process Runner/SharePoint analogies where they fit.
2. One implementation step at a time. Don't dump unrelated files.
3. After finishing a step: remind the developer to test it, then `git add` + `git commit` before moving on.
4. If something breaks: ask for the exact error message and the relevant file content before guessing.
5. Don't recreate files listed as "Completed" above.
6. Don't silently change the tech stack, the 10 clause categories, the confidence thresholds, or the node sequence — these are settled, graded decisions. Flag a concrete reason and ask before changing any of them.
7. Distinguish clearly in any explanation: **implemented** vs. **required next** vs. **recommended later** vs. **future enterprise scope** (Priority 3 — do not build prematurely: hybrid search, CI/CD, model tiering, provider failover, semantic caching, SSO, RBAC, API gateway, job queues, managed vector DB, multi-tenancy, DR, formal compliance).
