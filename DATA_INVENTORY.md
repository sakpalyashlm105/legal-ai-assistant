# Legal AI Assistant - Data Inventory Report

**Generated:** 2026-06-13  
**Project Path:** `legal-agent/data/raw/`

---

## 📊 Summary

| Metric | Value |
|--------|-------|
| **Total Documents** | 358 |
| **Total Storage** | 185.79 MB |
| **PDF Files (ready)** | 334 ✓ |
| **HTML Files** | 19 |
| **Annotation Files** | 5 (CUAD labels) |
| **Metadata Entries** | 358 rows |

---

## 📁 Folder Breakdown

### ndas/ (5 files, 0.31 MB)
- **File Count:** 5
- **File Types:** .htm (HTML)
- **Source:** Downloaded from SEC EDGAR filings
- **Size Range:** 24 KB - 93 KB
- **Status:** ⚠ Below target (have 5, need 10-15)
- **Files:**
  - NDA_1_Hemoglobin_Oxygen_Therapeutics_LLC.htm (87 KB)
  - NDA_2_Bakhu_Holdings,_Corp._(BKUH).htm (24 KB)
  - NDA_3_Starry_Group_Holdings,_Inc..htm (51 KB)
  - NDA_4_TRxADE_HEALTH,_INC_(MEDS).htm (54 KB)
  - NDA_5_BlueRiver_Acquisition_Corp._(BLUAF,_BLUAW,_BLUVF).htm (95 KB)

### amendments/ (5 files, 1.11 MB)
- **File Count:** 5
- **File Types:** .htm (HTML)
- **Source:** Downloaded from SEC EDGAR filings
- **Size Range:** 17 KB - 1 MB (1 large file)
- **Status:** ⚠ Below target (have 5, need 10-15)
- **Files:**
  - Amendment_15_DYNAVAX_TECHNOLOGIES_CORP_(DVAX).htm (62 KB)
  - Amendment_16_Marqeta,_Inc._(MQ).htm (1 MB) ← Large
  - Amendment_17_Akerna_Corp._(GRYP).htm (18 KB)
  - Amendment_18_Olema_Pharmaceuticals,_Inc._(OLMA).htm (27 KB)
  - Amendment_19_Inari_Medical,_Inc._(NARI).htm (24 KB)

### contracts/ (343 files, 67.35 MB)
- **File Count:** 343
- **File Types:** 
  - .pdf (334 files) ← Production-ready
  - .htm (9 files) ← Need parsing
- **Source:** CUAD dataset (334 PDFs) + downloaded agreements (9 HTM)
- **Size Range:** 12 KB - 3.5 MB
- **Size Distribution:**
  - Small (<100 KB): 99 files
  - Medium (100 KB - 1 MB): 239 files
  - Large (≥1 MB): 5 files
- **Status:** ✓ Well-stocked for Phase 1-4 testing
- **Key Documents:**
  - Agency Agreements (frequent)
  - License Agreements
  - Supply/Service Agreements
  - Consulting Agreements
  - Manufacturing Agreements
  - Franchise Agreements
  - Joint Venture Agreements
  - Endorsement Agreements

### cuad_labels/ (5 files, 117.02 MB)
- **File Count:** 5
- **File Types:** .json (4), .csv (1)
- **Source:** CUAD v1 from Hugging Face
- **Size Range:** 4 KB - 38 MB
- **Status:** ✓ Ready for Phase 5 evaluation
- **Files:**
  - CUAD_v1.json (38 MB) - Main annotation file with all labels
  - CUADv1.json - HuggingFace version
  - test.json - Test set annotations
  - train_separate_questions.json - Training set
  - downloads_manifest.csv - File manifest from SEC downloads

### other/ (0 files)
- **Status:** Empty - can be used for miscellaneous documents

---

## 📋 Metadata CSV

**File:** `metadata.csv`  
**Rows:** 359 (1 header + 358 data)  
**Columns:** filename, subfolder, source, file_type, page_count, notes  

**Content Breakdown:**
- NDA entries: 5
- Amendment entries: 5
- Contract entries: 343
- Label entries: 5

---

## ✓ What's Ready for Use

### Phase 1: Document Extraction
- ✓ 334 PDF files available
- ✓ Good diversity in file sizes (12 KB to 3.5 MB)
- ✓ Ready for `pdf_parser.py` testing
- ✓ Ready for `ocr.py` fallback implementation

### Phase 2: Document Retrieval & Chunking
- ✓ Sufficient sample size for algorithm development
- ✓ Files span multiple pages and content volumes
- ✓ Ready for paragraph-aware chunking

### Phase 3: Agent & Classification
- ✓ 343 contracts for document type classification
- ✓ Clear document categories available
- ✓ Ready for `classifier.py` implementation

### Phase 4: Template Comparison
- ✓ Diverse contract types
- ✓ Ready for `comparator.py` clause extraction

### Phase 5: CUAD Evaluation
- ✓ CUAD_v1.json with 38 MB of annotations
- ✓ Test and training sets available
- ✓ Ready for `eval_cuad.py` benchmark testing

---

## ⚠ Identified Gaps

### 1. NDA Samples (Minor Impact)
- **Have:** 5 files
- **Target:** 10-15 files
- **Gap:** Missing 5-10 files
- **Impact:** Can still test Phase 1-4, but recommendations will improve with more samples
- **Solution:** Download additional NDAs from SEC EDGAR

### 2. Amendment Samples (Minor Impact)
- **Have:** 5 files
- **Target:** 10-15 files
- **Gap:** Missing 5-10 files
- **Impact:** Can still test Phase 1-4, but recommendations will improve with more samples
- **Solution:** Download additional amendments from SEC EDGAR

### 3. HTML Format Documents (Medium Impact)
- **Have:** 19 .htm files (not PDF)
- **Impact:** `pdf_parser.py` expects PDFs; these files need HTML parsing
- **Affected:** 5 NDAs + 5 amendments + 9 service/supplier agreements
- **Solution:** 
  - Option A: Convert .htm to PDF using external tool
  - Option B: Implement HTML parser in `ocr.py`
  - Option C: Extract text from HTML and convert to PDF with `reportlab`

### 4. Other/ Folder
- **Status:** Empty (not critical)
- **Use:** Can store miscellaneous documents later if needed

---

## 📊 File Format Analysis

### PDF Files (334)
- **Status:** ✓ Production-ready
- **Size Range:** 12 KB - 3.5 MB
- **Average Size:** ~200 KB
- **Parser:** Use `pdf_parser.py` with PyMuPDF
- **Fallback:** `ocr.py` with GPT-4o-mini Vision

### HTML Files (19)
- **Status:** ⚠ Requires preprocessing
- **Locations:**
  - ndas/: 5 files
  - amendments/: 5 files
  - contracts/: 9 files
- **Issue:** Not natively processable by PDF-based pipeline
- **Options:**
  1. Use BeautifulSoup to extract text and convert to PDF
  2. Implement HTML parser in extraction pipeline
  3. Use tool like wkhtmltopdf to convert HTML to PDF

### JSON Files (4)
- **Status:** ✓ For evaluation only
- **Size:** 38 MB + 3 smaller files
- **Use:** CUAD annotation labels for Phase 5
- **Format:** CUAD v1 standard format

### CSV File (1)
- **Status:** ✓ Manifest/metadata
- **Use:** Tracking downloads from SEC EDGAR

---

## 🚀 Action Items

### This Week (Required)
- [ ] Run `pdf_parser.py` on sample of 10 PDFs to verify extraction
- [ ] Test `ocr.py` on any unreadable PDFs
- [ ] Confirm all 334 PDFs can be opened successfully

### Short Term (Recommended)
- [ ] Convert 19 .htm files to PDF for pipeline consistency
- [ ] Download 10-15 additional NDA samples from SEC EDGAR
- [ ] Download 10-15 additional amendment samples
- [ ] Update `metadata.csv` with new files
- [ ] Verify total count reaches 50-75 documents per category

### Testing Phase
- [ ] Test `chunking.py` on small, medium, and large files
- [ ] Test `vector_store.py` with FAISS embeddings
- [ ] Test `classifier.py` on document categorization accuracy
- [ ] Test `comparator.py` with template matching

### Evaluation Phase
- [ ] Run `eval_cuad.py` against CUAD annotations
- [ ] Measure extraction F1 score on labeled clauses
- [ ] Document baseline performance metrics
- [ ] Generate evaluation report

---

## 📈 Project Readiness

**Overall Status:** ✓ **READY FOR PHASE 1 TESTING**

**Confidence Level:** High (334 PDFs available, annotations ready)

**Data Quality:** 
- File count tracking: 100% (358 files in metadata.csv)
- Format diversity: Good (PDF, HTML, JSON, CSV)
- Size diversity: Excellent (12 KB to 38 MB)
- Document count: Exceeds Phase 1 requirements

**Next Steps:** Start Phase 1 extraction testing with `pdf_parser.py`

---

## 📝 Notes

- All original files from Data/cuad/ and Data/downloads/ remain untouched
- 47 duplicate PDFs from CUAD (same names in different category folders) were deduplicated
- HTML files are viewable in browser but need special handling for text extraction
- CUAD annotations contain full clause labels for 334 of the contracts
- metadata.csv can be extended as new documents are added

---

**Last Updated:** 2026-06-13 20:05 UTC  
**Data Location:** `C:\Users\Hiloni Vora\Source Code\Legal AI Assistant\legal-agent\data\raw\`
