"""
agent/classifier.py
-------------------
Document type classification node for the Legal AI Assistant.

What this file does:
    Reads the opening ~2 pages of an extracted document and asks GPT-4o-mini
    to classify it as one of four types: NDA, Contract, Amendment, or Other.
    Returns a DocumentClassification with a type, confidence score, and
    one-sentence reasoning.

Why classify first?
    The LangGraph workflow routes to different branches depending on document
    type. Amendments need the base agreement retrieved before clause extraction;
    NDAs follow a lighter clause checklist than full contracts. Classification
    is the fork in the road -- everything downstream depends on getting it right.

Confidence routing (locked graded decisions from CLAUDE.md):
    < 0.5  -> self-retry once with a refined prompt; if still < 0.5, return
              result with retry_count=1 so the orchestrator escalates to HITL
    0.5-0.7 -> Tree-of-Thought reasoning (stubbed here, wired in Step 7)
    > 0.7  -> proceed automatically

Dependencies:
    openai, pydantic, yaml, config.py, schemas/clause.py
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

import yaml
from openai import OpenAI

from config import OPENAI_API_KEY, LLM_MODEL, DOCUMENT_TYPES
from schemas.clause import DocumentClassification
from schemas.document import DocumentExtraction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Approximate number of characters in ~2 pages of a typical legal document.
# Used to slice preamble_text for classification without sending the whole doc.
PREAMBLE_CHAR_LIMIT = 3000

# Confidence thresholds -- mirror CLAUDE.md exactly, do not change without
# updating the graded design document.
CONFIDENCE_AUTO_PROCEED = 0.7
CONFIDENCE_TOT_FLOOR = 0.5   # below this -> retry, then HITL

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "classify_document.yaml"

# ---------------------------------------------------------------------------
# Lazy OpenAI client (same pattern as ocr.py and vector_store.py)
# ---------------------------------------------------------------------------

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


# ---------------------------------------------------------------------------
# Prompt loader
# ---------------------------------------------------------------------------

def _load_prompt() -> dict:
    """Load the classify_document prompt template from YAML."""
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Core LLM call
# ---------------------------------------------------------------------------

def _call_llm(preamble_text: str) -> DocumentClassification:
    """
    Send the preamble to GPT-4o-mini and parse the JSON response.

    Why extract only the preamble (first ~3000 chars)?
        Classification signals are almost always in the title and opening
        recitals of a legal document -- "THIS NON-DISCLOSURE AGREEMENT",
        "Amendment No. 2 to the Master Services Agreement", etc.
        Sending the full document would waste tokens and slow the pipeline.

    Raises
    ------
    ValueError
        If the LLM response is not valid JSON or fails Pydantic validation.
    """
    prompt = _load_prompt()
    document_type_options = ", ".join(DOCUMENT_TYPES)

    user_message = prompt["user"].format(
        preamble_text=preamble_text[:PREAMBLE_CHAR_LIMIT],
        document_type_options=document_type_options,
    )

    client = _get_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,   # deterministic -- classification should not be creative
        response_format={"type": "json_object"},
    )

    raw_content = response.choices[0].message.content
    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned non-JSON response: {raw_content!r}") from e

    # Pydantic validates types and field constraints
    return DocumentClassification(**data)


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def classify_document(
    doc: DocumentExtraction,
    _retry_count: int = 0,
) -> DocumentClassification:
    """
    Classify a document as NDA, Contract, Amendment, or Other.

    How it works:
        1. Extract the first PREAMBLE_CHAR_LIMIT characters of the document.
        2. Send to GPT-4o-mini with the classify_document prompt.
        3. Parse and validate the JSON response into a DocumentClassification.
        4. Apply confidence routing:
           - confidence > 0.7  -> return immediately (auto-proceed)
           - 0.5 <= confidence <= 0.7 -> return for ToT routing (step 7)
           - confidence < 0.5 and no retries yet -> recurse once with retry
           - confidence < 0.5 and already retried -> return with retry_count=1
             so orchestrator can escalate to human review

    Parameters
    ----------
    doc : DocumentExtraction
        The full extraction result from pdf_parser.py.
    _retry_count : int
        Internal retry counter -- do not pass this from outside.

    Returns
    -------
    DocumentClassification
        Always returns a result. The orchestrator checks retry_count and
        confidence to decide whether to proceed or escalate.
    """
    preamble = doc.full_text[:PREAMBLE_CHAR_LIMIT] if doc.full_text else ""

    if not preamble.strip():
        logger.warning(
            f"Document '{doc.file_name}' has no text for classification. "
            "Returning 'Other' with zero confidence."
        )
        return DocumentClassification(
            document_type="Other",
            confidence=0.0,
            reasoning="Document has no extractable text; cannot classify.",
            retry_count=_retry_count,
        )

    try:
        result = _call_llm(preamble)
    except (ValueError, Exception) as e:
        logger.error(
            f"classify_document failed for '{doc.file_name}': {e}. "
            "Returning 'Other' with zero confidence."
        )
        return DocumentClassification(
            document_type="Other",
            confidence=0.0,
            reasoning=f"Classification failed due to an error: {type(e).__name__}",
            retry_count=_retry_count,
        )

    result = result.model_copy(update={"retry_count": _retry_count})

    # Confidence routing
    if result.confidence >= CONFIDENCE_AUTO_PROCEED:
        # Auto-proceed -- clear winner
        logger.info(
            f"Classified '{doc.file_name}' as {result.document_type} "
            f"(confidence={result.confidence:.2f}, auto-proceed)"
        )
        return result

    if result.confidence >= CONFIDENCE_TOT_FLOOR:
        # Return for Tree-of-Thought routing (wired in Step 7)
        logger.info(
            f"Classified '{doc.file_name}' as {result.document_type} "
            f"(confidence={result.confidence:.2f}, routing to ToT)"
        )
        return result

    # confidence < 0.5
    if _retry_count == 0:
        logger.warning(
            f"Low confidence ({result.confidence:.2f}) classifying '{doc.file_name}'. "
            "Retrying once."
        )
        return classify_document(doc, _retry_count=1)

    # Already retried -- return so orchestrator can escalate to HITL
    logger.warning(
        f"Classification confidence still low ({result.confidence:.2f}) after retry "
        f"for '{doc.file_name}'. Returning for human review."
    )
    return result
