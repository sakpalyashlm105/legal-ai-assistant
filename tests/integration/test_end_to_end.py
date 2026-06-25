"""
tests/integration/test_end_to_end.py
--------------------------------------
Real end-to-end pipeline diagnostic for the Legal AI Assistant.

This is NOT a pass/fail unit test -- it is a diagnostic that runs real PDF
documents through the entire pipeline with real OpenAI API calls and prints
the actual output at every stage so a human can sanity-check the results.

Documents tested:
  - NDA_2_Bakhu_Holdings,_Corp._(BKUH).pdf        (short: ~2-3 pages)
  - SupplierAgreement_14_Brag_House_Holdings...pdf (medium: ~5-8 pages)
  - ServiceAgreement_9_Coeur_Mining,_Inc...pdf     (longer: ~10+ pages)

How to run (from legal-agent/ with venv active):
    pytest tests/integration/test_end_to_end.py -v -s
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pathlib import Path
import pytest

from config import CLAUSE_CATEGORIES
from extraction.pdf_parser import extract_text_from_pdf
from retrieval.chunking import chunk_document
from retrieval.vector_store import VectorStore
from agent.classifier import classify_document
from agent.extractor import extract_clauses
from agent.comparator import compare_to_templates
from agent.risk_engine import flag_risks

# ---------------------------------------------------------------------------
# Document paths
# ---------------------------------------------------------------------------

BASE = Path(__file__).parent.parent.parent / "data" / "pdf"

TEST_DOCS = [
    BASE / "ndas" / "NDA_2_Bakhu_Holdings,_Corp._(BKUH).pdf",
    BASE / "contracts" / "SupplierAgreement_14_Brag_House_Holdings,_Inc._(TBH).pdf",
    BASE / "contracts" / "ServiceAgreement_9_Coeur_Mining,_Inc._(CDE).pdf",
]


# ---------------------------------------------------------------------------
# Summary accumulator
# ---------------------------------------------------------------------------

_summary_rows: list[dict] = []


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _confidence_tier(conf: float) -> str:
    if conf > 0.7:
        return ">0.7 (auto-proceed)"
    elif conf >= 0.5:
        return "0.5-0.7 (Tree-of-Thought)"
    else:
        return "<0.5 (HITL escalation)"


# ---------------------------------------------------------------------------
# Core pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline_for_doc(pdf_path: Path) -> None:
    doc_name = pdf_path.name
    print(f"\n{'='*70}")
    print(f"DOCUMENT: {doc_name}")
    print(f"{'='*70}")

    row = {
        "document": doc_name,
        "extraction_ok": False,
        "chunks": 0,
        "clauses_found": 0,
        "highest_risk": "N/A",
        "cross_module_failures": [],
    }

    # -----------------------------------------------------------------------
    # STAGE 1: Extraction
    # -----------------------------------------------------------------------
    print(f"\n--- STAGE 1: EXTRACTION ---")
    doc = extract_text_from_pdf(str(pdf_path))
    print(f"  file_name             : {doc.file_name}")
    print(f"  total_pages           : {doc.total_pages}")
    print(f"  pages_failed          : {doc.pages_failed}")
    print(f"  pages_ocr             : {doc.pages_ocr}")
    print(f"  extraction_successful : {doc.extraction_successful}")
    print(f"  full_text (first 200) : {doc.full_text[:200]!r}")

    assert doc.extraction_successful, f"Extraction completely failed for {doc_name}"
    assert len(doc.full_text) > 0, "full_text is empty"
    row["extraction_ok"] = True

    # -----------------------------------------------------------------------
    # STAGE 2: Chunking
    # -----------------------------------------------------------------------
    print(f"\n--- STAGE 2: CHUNKING ---")
    chunks = chunk_document(doc)
    total_chunks = len(chunks)
    avg_tokens = (
        sum(c.token_count for c in chunks) / total_chunks if total_chunks else 0
    )
    print(f"  total chunks          : {total_chunks}")
    print(f"  avg token_count       : {avg_tokens:.1f}")
    if chunks:
        print(f"  chunk[0] text         :\n{chunks[0].text}")
    row["chunks"] = total_chunks

    # Cross-module check: page references within document total_pages
    page_check_failures = []
    for c in chunks:
        if c.start_page < 1 or c.end_page > doc.total_pages:
            page_check_failures.append(
                f"chunk {c.chunk_id}: pages {c.start_page}-{c.end_page} "
                f"outside 1-{doc.total_pages}"
            )
    if page_check_failures:
        print(f"  Page range check      : FAIL ({len(page_check_failures)} chunks)")
        for f in page_check_failures:
            print(f"    - {f}")
        row["cross_module_failures"].extend(page_check_failures)
    else:
        print(f"  Page range check      : PASS (all {total_chunks} chunks within 1-{doc.total_pages})")

    assert total_chunks > 0, "No chunks produced"

    # -----------------------------------------------------------------------
    # STAGE 3: Vector store / retrieval
    # -----------------------------------------------------------------------
    print(f"\n--- STAGE 3: VECTOR STORE + RETRIEVAL ---")
    store = VectorStore()
    store.add_chunks(chunks)
    print(f"  Index built with {store._index.ntotal} vectors")

    batch_results = store.search_batch(CLAUSE_CATEGORIES)
    print(f"  Retrieval results (top-1 per category):")
    for cat in CLAUSE_CATEGORIES:
        top_chunks = batch_results.get(cat, [])
        if top_chunks:
            c = top_chunks[0]
            print(f"    [{cat[:35]:<35}] chunk={c.chunk_id} | {c.text[:100]!r}")
        else:
            print(f"    [{cat[:35]:<35}] NO RESULTS")

    # -----------------------------------------------------------------------
    # STAGE 4: Classification
    # -----------------------------------------------------------------------
    print(f"\n--- STAGE 4: CLASSIFICATION ---")
    classification = classify_document(doc)
    print(f"  document_type  : {classification.document_type}")
    print(f"  confidence     : {classification.confidence:.3f}")
    print(f"  reasoning      : {classification.reasoning}")
    print(f"  confidence tier: {_confidence_tier(classification.confidence)}")
    print(f"  retry_count    : {classification.retry_count}")

    # -----------------------------------------------------------------------
    # STAGE 5: Clause extraction
    # -----------------------------------------------------------------------
    print(f"\n--- STAGE 5: CLAUSE EXTRACTION ---")
    clauses = extract_clauses(chunks, document_name=doc.file_name)
    print(f"  Total clauses returned: {len(clauses)}")

    chunk_ids_in_doc = {c.chunk_id for c in chunks}
    cross_module_check_failures = []

    for clause in clauses:
        present_str = "PRESENT" if clause.is_present else "ABSENT "
        print(
            f"  [{present_str}] {clause.clause_type:<40} "
            f"conf={clause.confidence:.2f} "
            f"page={clause.page_reference} "
            f"chunk={clause.source_chunk_id}"
        )
        if clause.is_present and clause.source_chunk_id is not None:
            if clause.source_chunk_id not in chunk_ids_in_doc:
                fail_msg = (
                    f"source_chunk_id={clause.source_chunk_id!r} for "
                    f"'{clause.clause_type}' does not match any chunk in this document"
                )
                cross_module_check_failures.append(fail_msg)
                print(f"    *** CROSS-MODULE FAIL: {fail_msg}")
            else:
                print(f"    source_chunk_id check: PASS")

    if cross_module_check_failures:
        print(f"\n  CROSS-MODULE CHECK: FAIL ({len(cross_module_check_failures)} failures)")
        row["cross_module_failures"].extend(cross_module_check_failures)
    else:
        print(f"\n  CROSS-MODULE CHECK: PASS (all source_chunk_ids match real chunks)")

    clauses_present = sum(1 for c in clauses if c.is_present)
    row["clauses_found"] = clauses_present
    assert len(clauses) == 10, f"Expected 10 clauses, got {len(clauses)}"

    # -----------------------------------------------------------------------
    # STAGE 6: Template comparison + risk scoring
    # -----------------------------------------------------------------------
    print(f"\n--- STAGE 6: TEMPLATE COMPARISON + RISK SCORING ---")
    comparisons = compare_to_templates(clauses)
    findings = flag_risks(clauses, comparisons)

    missing_clause_invariant_failures = []
    risk_levels = []

    for finding in findings:
        template_note = (
            "template comparison ran"
            if any(cmp.clause_type == finding.clause_type and cmp.template_found
                   for cmp in comparisons)
            else "no template available"
        )
        print(
            f"  [{finding.risk_level:<6}] {finding.clause_type:<40} "
            f"missing={finding.is_missing} "
            f"precedent={finding.precedent_applied} "
            f"({template_note})"
        )
        print(f"    reason: {finding.reason}")
        if finding.precedent_note:
            print(f"    precedent: {finding.precedent_note}")

        risk_levels.append(finding.risk_level)

        # CRITICAL CHECK: missing clause must always be HIGH, never have precedent
        if finding.is_missing:
            if finding.risk_level != "HIGH":
                fail_msg = (
                    f"INVARIANT VIOLATION: '{finding.clause_type}' is missing "
                    f"but risk_level={finding.risk_level!r} (expected HIGH)"
                )
                missing_clause_invariant_failures.append(fail_msg)
                print(f"  *** {fail_msg}")
            else:
                print(f"    missing-clause invariant: PASS (HIGH + precedent_applied=False)")

            if finding.precedent_applied:
                fail_msg = (
                    f"INVARIANT VIOLATION: '{finding.clause_type}' is missing "
                    f"but precedent_applied=True"
                )
                missing_clause_invariant_failures.append(fail_msg)
                print(f"  *** {fail_msg}")

    if missing_clause_invariant_failures:
        print(f"\n  MISSING-CLAUSE INVARIANT: FAIL")
        row["cross_module_failures"].extend(missing_clause_invariant_failures)
    else:
        print(f"\n  MISSING-CLAUSE INVARIANT: PASS (all missing clauses are HIGH with no precedent)")

    if risk_levels:
        for level in ["HIGH", "MEDIUM", "LOW"]:
            if level in risk_levels:
                row["highest_risk"] = level
                break

    _summary_rows.append(row)


# ---------------------------------------------------------------------------
# Tests (one per document)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pdf_path", TEST_DOCS, ids=[p.name for p in TEST_DOCS])
def test_end_to_end_pipeline(pdf_path: Path) -> None:
    if not pdf_path.exists():
        pytest.skip(f"Test document not found: {pdf_path}")
    run_pipeline_for_doc(pdf_path)


# ---------------------------------------------------------------------------
# Summary (runs after all documents)
# ---------------------------------------------------------------------------

def test_print_final_summary() -> None:
    """Print aggregated summary across all documents processed."""
    if not _summary_rows:
        pytest.skip("No documents were processed (all skipped or failed).")

    print(f"\n\n{'='*70}")
    print("FINAL SUMMARY TABLE")
    print(f"{'='*70}")
    header = (
        f"{'Document':<50} {'Ext?':>5} {'Chunks':>6} "
        f"{'Clauses':>7} {'MaxRisk':>7} {'Failures':>8}"
    )
    print(header)
    print("-" * 90)
    for row in _summary_rows:
        failures = len(row["cross_module_failures"])
        print(
            f"{row['document'][:50]:<50} "
            f"{'OK' if row['extraction_ok'] else 'FAIL':>5} "
            f"{row['chunks']:>6} "
            f"{row['clauses_found']:>7} "
            f"{row['highest_risk']:>7} "
            f"{'NONE' if failures == 0 else str(failures) + ' FAIL':>8}"
        )

    total_failures = sum(len(r["cross_module_failures"]) for r in _summary_rows)
    if total_failures > 0:
        print(f"\n{'='*70}")
        print("CROSS-MODULE CHECK FAILURES:")
        for row in _summary_rows:
            for fail in row["cross_module_failures"]:
                print(f"  [{row['document'][:30]}] {fail}")
    else:
        print(f"\nAll cross-module checks PASSED across all {len(_summary_rows)} documents.")
