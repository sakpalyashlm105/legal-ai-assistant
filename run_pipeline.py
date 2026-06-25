"""
run_pipeline.py
---------------
Standalone end-to-end pipeline runner for the Legal AI Assistant.

Runs a single PDF through every implemented stage and prints the results
at each step so you can verify the full flow works correctly.

Usage (from legal-agent/ with venv active):
    python run_pipeline.py
    python run_pipeline.py --pdf data/pdf/ndas/NDA_2_Bakhu_Holdings,_Corp._(BKUH).pdf
    python run_pipeline.py --pdf data/pdf/contracts/ServiceAgreement_9_Coeur_Mining,_Inc._(CDE).pdf

Stages covered:
    1. PDF Text Extraction   (extraction/pdf_parser.py)
    2. Document Chunking     (retrieval/chunking.py)
    3. Vector Store Build    (retrieval/vector_store.py)
    4. Clause Retrieval      (retrieval/vector_store.py  -- search_batch)
    5. Document Classification (agent/classifier.py)
    6. Clause Extraction     (agent/extractor.py)
    7. Template Comparison   (agent/comparator.py)
    8. Risk Flagging         (agent/risk_engine.py)
"""

import argparse
import sys
import os
from pathlib import Path

# Make sure Python can find the project modules regardless of where you run from
sys.path.insert(0, str(Path(__file__).parent))

from config import CLAUSE_CATEGORIES
from extraction.pdf_parser import extract_text_from_pdf
from retrieval.chunking import chunk_document
from retrieval.vector_store import VectorStore
from agent.classifier import classify_document
from agent.extractor import extract_clauses
from agent.comparator import compare_to_templates
from agent.risk_engine import flag_risks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIVIDER = "=" * 70
SECTION  = "-" * 70

def section(title: str) -> None:
    print(f"\n{SECTION}")
    print(f"  {title}")
    print(SECTION)

def confidence_tier(conf: float) -> str:
    if conf > 0.7:
        return ">0.7  → auto-proceed"
    elif conf >= 0.5:
        return "0.5-0.7 → Tree-of-Thought"
    else:
        return "<0.5  → HITL escalation"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(pdf_path: Path) -> None:
    if not pdf_path.exists():
        print(f"\n[ERROR] File not found: {pdf_path}")
        sys.exit(1)

    print(f"\n{DIVIDER}")
    print(f"  LEGAL AI ASSISTANT — END-TO-END PIPELINE RUN")
    print(f"  Document : {pdf_path.name}")
    print(DIVIDER)

    # -------------------------------------------------------------------------
    # STAGE 1: Text Extraction
    # -------------------------------------------------------------------------
    section("STAGE 1 — TEXT EXTRACTION  (PyMuPDF + OCR fallback)")
    doc = extract_text_from_pdf(str(pdf_path))

    print(f"  file_name             : {doc.file_name}")
    print(f"  file_hash (SHA-256)   : {doc.file_hash[:16]}...")
    print(f"  total_pages           : {doc.total_pages}")
    print(f"  pages_failed          : {doc.pages_failed}")
    print(f"  pages_ocr             : {doc.pages_ocr}")
    print(f"  extraction_successful : {doc.extraction_successful}")
    print(f"  full_text length      : {len(doc.full_text):,} characters")
    print(f"\n  --- First 400 characters of extracted text ---")
    print(f"  {doc.full_text[:400]!r}")

    if not doc.extraction_successful:
        print(f"\n[FATAL] Extraction failed: {doc.error_message}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # STAGE 2: Chunking
    # -------------------------------------------------------------------------
    section("STAGE 2 — CHUNKING  (token-based, 500-token target, 50-token overlap)")
    chunks = chunk_document(doc)
    total_chunks = len(chunks)
    avg_tokens = sum(c.token_count for c in chunks) / total_chunks if total_chunks else 0

    print(f"  total chunks          : {total_chunks}")
    print(f"  avg tokens per chunk  : {avg_tokens:.1f}")
    print(f"  min tokens            : {min(c.token_count for c in chunks)}")
    print(f"  max tokens            : {max(c.token_count for c in chunks)}")
    print(f"\n  --- Chunk #1 preview ---")
    if chunks:
        c0 = chunks[0]
        print(f"  chunk_id   : {c0.chunk_id}")
        print(f"  pages      : {c0.start_page}–{c0.end_page}")
        print(f"  token_count: {c0.token_count}")
        print(f"  text (first 300 chars): {c0.text[:300]!r}")

    if total_chunks == 0:
        print("[FATAL] No chunks produced.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # STAGE 3: Vector Store + Retrieval
    # -------------------------------------------------------------------------
    section("STAGE 3 — VECTOR STORE + RETRIEVAL  (FAISS, text-embedding-3-small)")
    store = VectorStore()
    store.add_chunks(chunks)
    print(f"  FAISS index size      : {store._index.ntotal} vectors")

    print(f"\n  --- Top-1 retrieved chunk per clause category ---")
    batch_results = store.search_batch(CLAUSE_CATEGORIES)
    for cat in CLAUSE_CATEGORIES:
        top = batch_results.get(cat, [])
        if top:
            snippet = top[0].text[:90].replace("\n", " ")
            print(f"  [{cat[:38]:<38}]  {snippet!r}")
        else:
            print(f"  [{cat[:38]:<38}]  NO RESULTS")

    # -------------------------------------------------------------------------
    # STAGE 4: Document Classification
    # -------------------------------------------------------------------------
    section("STAGE 4 — DOCUMENT CLASSIFICATION  (GPT-4o-mini)")
    classification = classify_document(doc)

    print(f"  document_type  : {classification.document_type}")
    print(f"  confidence     : {classification.confidence:.3f}  →  {confidence_tier(classification.confidence)}")
    print(f"  retry_count    : {classification.retry_count}")
    print(f"  reasoning      : {classification.reasoning}")

    # -------------------------------------------------------------------------
    # STAGE 5: Clause Extraction
    # -------------------------------------------------------------------------
    section("STAGE 5 — CLAUSE EXTRACTION  (GPT-4o-mini, 10 clause categories)")
    clauses = extract_clauses(chunks, document_name=doc.file_name)

    present  = [c for c in clauses if c.is_present]
    absent   = [c for c in clauses if not c.is_present]

    print(f"  Clauses found   : {len(present)} / {len(clauses)}")
    print(f"  Clauses absent  : {len(absent)}")
    print()

    for clause in clauses:
        status = "PRESENT" if clause.is_present else "ABSENT "
        print(
            f"  [{status}] {clause.clause_type:<42} "
            f"conf={clause.confidence:.2f}  "
            f"page={str(clause.page_reference):<4}  "
            f"chunk={clause.source_chunk_id}"
        )
        if clause.is_present and clause.extracted_text:
            print(f"           excerpt: {clause.extracted_text[:120]!r}")

    # -------------------------------------------------------------------------
    # STAGE 6: Template Comparison + Risk Scoring
    # -------------------------------------------------------------------------
    section("STAGE 6 — TEMPLATE COMPARISON + RISK SCORING")
    comparisons = compare_to_templates(clauses)
    findings    = flag_risks(clauses, comparisons)

    risk_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

    print(f"\n  {'Clause':<42} {'Risk':<8} {'Missing':<8} {'Precedent':<10}  Reason")
    print(f"  {'-'*42} {'-'*8} {'-'*8} {'-'*10}  {'-'*40}")

    for f in findings:
        risk_counts[f.risk_level] = risk_counts.get(f.risk_level, 0) + 1
        print(
            f"  {f.clause_type:<42} {f.risk_level:<8} "
            f"{'YES' if f.is_missing else 'no':<8} "
            f"{'YES' if f.precedent_applied else 'no':<10}  "
            f"{f.reason[:60]}"
        )
        if f.precedent_note:
            print(f"  {'':42}   precedent: {f.precedent_note}")

    # -------------------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------------------
    print(f"\n{DIVIDER}")
    print(f"  PIPELINE COMPLETE — SUMMARY")
    print(DIVIDER)
    print(f"  Document          : {doc.file_name}")
    print(f"  Pages             : {doc.total_pages}  (OCR pages: {doc.pages_ocr})")
    print(f"  Chunks            : {total_chunks}")
    print(f"  Classification    : {classification.document_type}  (conf={classification.confidence:.3f})")
    print(f"  Clauses present   : {len(present)} / 10")
    print(f"  Risk breakdown    : HIGH={risk_counts.get('HIGH',0)}  MEDIUM={risk_counts.get('MEDIUM',0)}  LOW={risk_counts.get('LOW',0)}")
    print(DIVIDER)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Legal AI Assistant — end-to-end pipeline runner")
    parser.add_argument(
        "--pdf",
        default="data/pdf/ndas/NDA_2_Bakhu_Holdings,_Corp._(BKUH).pdf",
        help="Path to the PDF to process (relative to legal-agent/)",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    # If not absolute, resolve relative to this script's directory
    if not pdf_path.is_absolute():
        pdf_path = Path(__file__).parent / pdf_path

    run(pdf_path)
