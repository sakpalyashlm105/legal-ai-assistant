# CLAUDE.md — Legal AI Assistant

This file is read automatically by Claude Code at the start of every session in this repository. It contains operational facts about the project: what exists, what's next, conventions to follow, and commands to run. For full conceptual/architectural rationale, the developer maintains a separate master prompt used in chat conversations — this file is the terse, action-oriented companion to that.

---

## Project Identity

Legal AI Assistant — an agentic system that analyzes legal documents (NDAs, contracts, supplier agreements, amendments), extracts clauses, compares against templates, scores risk, and generates reports with human-in-the-loop review. Built as a graded capstone project by Yash, an Intelligent Automation Analyst learning AI/ML concepts for the first time. **Treat Yash as a beginner on RAG/LangGraph/embeddings/agents — explain concepts before writing code, one step at a time, wait for confirmation before proceeding.**

This is NOT a production deployment. It's a capstone with graded checkpoints (4 submitted so far, all scored 1/1). Do not add production-scale infrastructure (Kubernetes, managed vector DBs, multi-tenant auth, etc.) — see "Explicitly Out of Scope" below.

---

## Current Status

**Completed — do not recreate:**
- Full project scaffolding: `requirements.txt`, `config.py`, `verify_setup.py`, `.env`, `.gitignore`, `README.md`
- `data/raw/` populated with 358 files: `ndas/` (5), `contracts/` (343, mostly PDF), `amendments/` (5), `other/` (0), `cuad_labels/` (5 annotation files)
- `data/raw/metadata.csv` and `data/raw/DATA_INVENTORY.md`
- All design/scoping documents (scoping, agent design, RAG design, ToT design, multi-agent analysis) — written and graded, do not redesign without new test evidence

**Next task: Step 3 — text extraction pipeline.**
Build `extraction/pdf_parser.py` (PyMuPDF) and `extraction/ocr.py` (GPT-4o-mini Vision fallback). Do not jump ahead to chunking, retrieval, classification, ToT, or HITL — those are later steps. Stop and wait for the developer to test on real PDFs from `data/raw/` before moving to Step 4.

---

## Tech Stack (locked — do not swap without evidence of failure)

| Layer | Choice |
|---|---|
| LLM | OpenAI GPT-4o-mini (classification, extraction, comparison, reranking, risk, reports) |
| Embeddings | OpenAI text-embedding-3-small (1536-dim) |
| PDF extraction | PyMuPDF primary |
| OCR fallback | GPT-4o-mini Vision (per-page, triggered when a page yields <100 chars) |
| Vector store | FAISS, dense + MMR, top-10 -> LLM rerank -> top-3 |
| Orchestration | LangGraph (state machine, conditional edges) + LangChain |
| UI | Streamlit |
| Validation | Pydantic schemas everywhere |
| Evaluation | CUAD dataset, scikit-learn metrics, custom regression suite |
| Observability | LangSmith (content tracing OFF by default) |

**Explicitly out of scope** (do not add): fine-tuning, Ollama, Tesseract, MCP, managed vector DB, hybrid BM25 search (unless retrieval evals prove dense-only is failing on exact-term queries), Kubernetes, multi-provider failover, RLHF, SSO/RBAC, semantic answer caching, large-scale load testing.

---

## Architecture Facts Claude Code Must Know

**10 approved clause categories** (never silently add/change): Confidentiality, Termination for Convenience, Termination for Cause, Governing Law, Indemnification, Limitation of Liability, Non-Compete, Assignment, Renewal/Term, Dispute Resolution.

**Confidence routing is three-tier, not binary** -- this is a graded design commitment, implement exactly this:
- `< 0.5` -> self-retry once with refined prompt -> if still <0.5, escalate to human review
- `0.5-0.7` -> route to Tree-of-Thought reasoning (not retry, not auto-proceed)
- `> 0.7` -> proceed automatically

**Tree-of-Thought**: beam search, width 3, max depth 3, 2-4 candidates per ambiguous paragraph, pruning at 0.30 (depth 1) and 0.40 (depth 2), preserve pruned candidates in state (don't discard them). Never expose raw chain-of-thought to logs or users -- store structured candidate/evidence/score objects only.

**Risk scoring has a precedent-aware override** -- before finalizing a non-standard-clause risk level, check feedback log for similar previously-approved clauses; if found, downgrade one tier (HIGH->MEDIUM) and annotate with the precedent. **This override never applies to missing-clause findings** -- absence is always HIGH regardless of precedent.

**Retrieval latency mitigations** (implement when building `retrieval/`): batch all clause queries into one embedding call rather than 10 separate ones; cache base-agreement/template retrieval per document session; skip the LLM reranking call entirely when FAISS top-1 cosine similarity > 0.92.

**This is a single LangGraph workflow with specialized nodes, not a true multi-agent system.** Don't describe it as "multi-agent" in docs or code comments unless distinct autonomous agents with separate state/communication are actually built -- they are not, by design decision.

**HITL, not RLHF.** Human corrections never update model weights. Never use the term RLHF anywhere in code, comments, or docs.

---

## Full LangGraph Node Sequence (build in this order, across the project's phases)

validate_input -> validate_scope -> scan_prompt_injection -> extract_text -> validate_extraction -> classify_document -> route_by_type -> retrieve_base_agreement (amendments only) -> chunk_document -> retrieve_context -> rerank_context -> extract_clauses -> validate_schema -> verify_evidence -> verify_page_references -> calculate_confidence -> run_tot_if_needed -> compare_to_templates -> flag_risks -> verify_missing_clauses -> verify_final_claims -> human_review -> generate_report -> validate_final_output -> save_feedback -> record_metrics

Currently implementing: `extract_text` (Step 3). Everything before it in this list conceptually wraps extraction (input/scope validation); build those minimally now, fully later as their own steps.

---

## Project Structure

```
legal-agent/
|-- main.py, config.py, requirements.txt, .env, .gitignore, README.md
|-- agent/          classifier.py, extractor.py, comparator.py, risk_engine.py, human_review.py, orchestrator.py
|-- extraction/     pdf_parser.py, ocr.py            <- BUILDING NOW
|-- retrieval/      chunking.py, vector_store.py, reranker.py
|-- guardrails/     input_validator.py, scope_validator.py, prompt_injection.py, extraction_validator.py,
|                   evidence_verifier.py, page_verifier.py, absence_verifier.py, claim_verifier.py, output_validator.py
|-- schemas/        document.py, clause.py, comparison.py, risk.py, review.py, report.py, errors.py
|-- prompts/        *.yaml -- one per LLM task, versioned, never inline in Python
|-- reporting/       report_generator.py
|-- feedback/       feedback_manager.py
|-- observability/  tracing.py, metrics.py, logging_config.py
|-- audit/          audit_logger.py
|-- core/           retry.py, timeout.py, error_handler.py, cache.py, token_budget.py
|-- evaluation/      eval_cuad.py, eval_retrieval.py, eval_guardrails.py, eval_routing.py, eval_risk.py,
|                   eval_hitl.py, eval_claim_grounding.py, eval_llm_judge.py, regression_suite.py
|-- docs/           ARCHITECTURE.md, GUARDRAILS.md, SECURITY_AND_PRIVACY.md, THREAT_MODEL.md,
|                   DATA_GOVERNANCE.md, LIMITATIONS.md, etc.
|-- data/raw/       POPULATED -- 358 files, do not recreate or re-sort
|-- data/processed/, data/templates/, data/vector_store/
`-- tests/          unit/, integration/, e2e/
```

**Do not scaffold every empty module immediately.** Create files only when the implementation step needs them. Don't create `guardrails/`, `schemas/`, `prompts/` wholesale right now -- only what Step 3 needs (likely a minimal `schemas/document.py` and `guardrails/extraction_validator.py`).

---

## Coding Conventions

- **Modular Python.** One responsibility per file/function.
- **Educational comments.** This is a learning project -- comment WHY, not just WHAT.
- **Before/after diffs when editing existing files.** Don't silently rewrite whole files the developer hasn't seen change-by-change.
- **`.env` for all secrets.** Never hardcode `OPENAI_API_KEY` or any credential. Read via `config.py`.
- **Pydantic for every structured LLM output.** Confidence fields always 0-1. Risk fields always one of `LOW`/`MEDIUM`/`HIGH`. Review actions always one of `approve`/`correct`/`reject`/`select_alternative`.
- **No silent fabrication.** If extraction/OCR fails, evidence can't be verified, or a page reference is invalid -- fail loudly/gracefully, never invent plausible-looking output.
- **No bare `except:`.** Classify errors (retryable vs. not): refusal, timeout, rate-limit, malformed output, etc. are distinct cases with distinct handling.
- **Prompts live in `prompts/*.yaml`**, not inline strings in Python, once we get past Step 3's minimal needs.
- **PII-safe logging.** Never log full contract text by default. Log document ID, hash, page count, method, category, confidence, risk level -- not raw content. `ENABLE_CONTENT_TRACING=False` and `ENABLE_PII_REDACTION=True` are the defaults.

---

## Commands

```bash
# Activate venv (run this first, every session)
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

# Install/verify
pip install -r requirements.txt
python verify_setup.py

# Run the app (once main.py + UI exist)
streamlit run main.py

# Run tests (once tests/ has content)
pytest tests/ -v

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
6. Don't silently change the tech stack, the 10 clause categories, the confidence thresholds, or the node sequence -- these are settled, graded decisions. Flag a concrete reason and ask before changing any of them.
7. Distinguish clearly in any explanation: **implemented** vs. **required next** vs. **recommended later** vs. **future enterprise scope** (Priority 3 -- do not build prematurely: hybrid search, CI/CD, model tiering, provider failover, semantic caching, SSO, RBAC, API gateway, job queues, managed vector DB, multi-tenancy, DR, formal compliance).