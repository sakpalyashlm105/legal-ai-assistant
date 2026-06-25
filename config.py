# All settings, API keys, paths, clause categories
# config.py
# Central configuration for the Legal Document Research Agent.
# All paths, model names, thresholds, and clause categories live here.
# If anything changes (new folder, different model), update it here only.

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (where OPENAI_API_KEY lives)
load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ── Model Names ───────────────────────────────────────────────────────────────
LLM_MODEL = "gpt-4o-mini"          # Used for extraction, classification, comparison
EMBEDDING_MODEL = "text-embedding-3-small"  # Used for vector store

# ── Project Root ──────────────────────────────────────────────────────────────
# Path(__file__) = location of this config.py file
# .parent = the legal-agent/ folder
BASE_DIR = Path(__file__).parent

# ── Data Paths ────────────────────────────────────────────────────────────────
DATA_DIR         = BASE_DIR / "data"
RAW_DIR          = DATA_DIR / "raw"
PROCESSED_DIR    = DATA_DIR / "processed"
TEMPLATES_DIR    = DATA_DIR / "templates"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

# Sub-folders inside raw/
NDA_DIR        = RAW_DIR / "ndas"
CONTRACTS_DIR  = RAW_DIR / "contracts"
AMENDMENTS_DIR = RAW_DIR / "amendments"
CUAD_DIR       = RAW_DIR / "cuad_labels"

# ── Extraction Settings ───────────────────────────────────────────────────────
# If PyMuPDF extracts fewer than this many characters per page on average,
# we assume the PDF is scanned and fall back to Vision LLM
OCR_FALLBACK_THRESHOLD = 100  # characters per page

# ── Retrieval Settings ────────────────────────────────────────────────────────
CHUNK_SIZE        = 512    # Target size of each text chunk (in characters)
CHUNK_OVERLAP     = 50     # Overlap between consecutive chunks to avoid cutting mid-sentence
MMR_FETCH_K       = 10     # How many chunks FAISS fetches before MMR diversity filter
MMR_FINAL_K       = 5      # How many chunks MMR returns after diversity filtering
RERANK_FINAL_K    = 3      # How many chunks survive GPT-4o-mini re-ranking

# ── Agent Settings ────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.7   # Below this, trigger human-in-the-loop confirmation
TOT_MAX_DEPTH        = 3     # Tree-of-Thought max exploration depth
TOT_BEAM_WIDTH       = 3     # Number of candidate interpretations kept per depth level

# ── Clause Categories (10 total) ──────────────────────────────────────────────
# These are the exact clause types the agent will look for in every document.
CLAUSE_CATEGORIES = [
    "Confidentiality / Non-Disclosure",
    "Termination for Convenience",
    "Termination for Cause",
    "Governing Law / Jurisdiction",
    "Indemnification",
    "Limitation of Liability",
    "Non-Compete / Non-Solicitation",
    "Assignment",
    "Renewal / Term",
    "Dispute Resolution",
]

# ── Document Types ────────────────────────────────────────────────────────────
DOCUMENT_TYPES = ["NDA", "Contract", "Amendment", "Other"]

# ── Risk Levels ───────────────────────────────────────────────────────────────
RISK_LEVELS = ["LOW", "MEDIUM", "HIGH"]