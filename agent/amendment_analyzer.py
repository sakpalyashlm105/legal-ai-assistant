"""
agent/amendment_analyzer.py
----------------------------
Simplified amendment analysis path (Step 14 — routing + analysis only).

What this module does:
    When the document classifier identifies document_type == "Amendment",
    the pipeline skips the standard 10-clause extraction + template comparison
    path (which doesn't make sense for a document that only modifies specific
    clauses of some other agreement) and routes here instead.

What this module does NOT do:
    * Cross-referencing to the base agreement — base-agreement retrieval is
      not built.  If the base agreement is unavailable, this is flagged in the
      report as a known limitation (see FUTURE_WORK note below).
    * Risk scoring against templates — template comparison doesn't apply to
      amendments since the amendment text is intentionally different from the
      base template.

FUTURE_WORK (explicit deferred item — see CLAUDE.md):
    Retrieve the base agreement by name/counterparty match from data/pdf/ and
    cross-reference the modified clauses against the base agreement text.
    This requires a document matching step not currently built.

Output:
    Returns an AmendmentAnalysisResult dict with:
        modified_clauses   : list of clause names/headings explicitly referenced
        amendment_summary  : LLM-generated prose summary of what the amendment changes
        base_agreement_ref : string from the amendment text naming the original agreement (if any)
        analysis_confidence: 0.0-1.0

PII-safe: only clause category labels and the amendment summary are stored.
          Full amendment text is not logged.
"""

import json
import logging
from typing import Optional

from openai import OpenAI

from config import OPENAI_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a legal document analysis assistant.
You will be given the text of a contract amendment and must extract:
1. Which clauses or sections the amendment explicitly references or modifies.
2. A one-to-two sentence summary of what changes the amendment makes.
3. The name or description of the original agreement being amended, if stated.

Respond with valid JSON only, in this exact structure:
{
  "modified_clauses": ["<clause name 1>", "<clause name 2>", ...],
  "amendment_summary": "<one or two sentence summary>",
  "base_agreement_ref": "<original agreement name, or null if not stated>",
  "confidence": <0.0 to 1.0>
}"""

_USER_TEMPLATE = """Analyze this contract amendment and identify what it modifies.

Amendment text (first {char_limit} characters):
{amendment_text}

Respond with the JSON structure described."""

PREAMBLE_CHAR_LIMIT = 4000  # enough for most amendment documents


def analyze_amendment(full_text: str) -> dict:
    """
    Make one LLM call to identify what an amendment modifies.

    Parameters
    ----------
    full_text : str
        The full extracted text of the amendment document.

    Returns
    -------
    dict with keys:
        modified_clauses      : list[str]
        amendment_summary     : str
        base_agreement_ref    : str or None
        analysis_confidence   : float
        error                 : str or None (populated only on failure)
    """
    client = OpenAI(api_key=OPENAI_API_KEY)

    user_text = _USER_TEMPLATE.format(
        char_limit=PREAMBLE_CHAR_LIMIT,
        amendment_text=full_text[:PREAMBLE_CHAR_LIMIT],
    )

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        from agent.metrics_writer import accumulate_tokens
        accumulate_tokens(response.usage.prompt_tokens, response.usage.completion_tokens)

        raw = response.choices[0].message.content or ""
        data = json.loads(raw)

        return {
            "modified_clauses": data.get("modified_clauses", []),
            "amendment_summary": data.get("amendment_summary", ""),
            "base_agreement_ref": data.get("base_agreement_ref"),
            "analysis_confidence": float(data.get("confidence", 0.0)),
            "error": None,
        }

    except json.JSONDecodeError as exc:
        logger.error("analyze_amendment: LLM returned non-JSON: %s", exc)
        return {
            "modified_clauses": [],
            "amendment_summary": "[Analysis failed — LLM returned non-JSON response]",
            "base_agreement_ref": None,
            "analysis_confidence": 0.0,
            "error": f"json_decode_error: {exc}",
        }
    except Exception as exc:
        logger.error("analyze_amendment: unexpected error: %s", exc)
        return {
            "modified_clauses": [],
            "amendment_summary": f"[Analysis failed — {exc}]",
            "base_agreement_ref": None,
            "analysis_confidence": 0.0,
            "error": str(exc),
        }


def render_amendment_report(
    document_name: str,
    document_hash: str,
    total_pages: int,
    classification_confidence: float,
    analysis: dict,
) -> str:
    """
    Render a minimal markdown report for an amendment document.

    This replaces the standard 10-row clause table with a concise Amendment
    Summary section.  No template comparison or risk scoring — those are
    not applicable to amendment documents.

    Parameters
    ----------
    document_name          : str
    document_hash          : str
    total_pages            : int
    classification_confidence : float
    analysis               : dict from analyze_amendment()

    Returns
    -------
    Markdown string.
    """
    modified = analysis.get("modified_clauses") or []
    summary = analysis.get("amendment_summary") or "(No summary generated)"
    base_ref = analysis.get("base_agreement_ref")
    confidence = analysis.get("analysis_confidence", 0.0)
    error = analysis.get("error")

    lines = [
        f"# Legal Document Report — {document_name}",
        "",
        "> **DISCLAIMER:** This report is produced by an AI assistant for review purposes "
        "only. It does not constitute legal advice and should not be relied upon as a "
        "definitive legal interpretation. All findings must be reviewed by qualified "
        "legal counsel before any business or legal decision is made.",
        "",
        "---",
        "",
        "## Document Overview",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Document name | {document_name} |",
        f"| Document type | Amendment |",
        f"| Classification confidence | {classification_confidence:.0%} |",
        f"| Total pages | {total_pages} |",
        f"| Document hash (SHA-256, first 16 chars) | `{document_hash[:16]}` |",
        "",
        "---",
        "",
        "## Amendment Summary",
        "",
        summary,
        "",
    ]

    if base_ref:
        lines += [
            f"**Original agreement referenced:** {base_ref}",
            "",
        ]
    else:
        lines += [
            "**Original agreement referenced:** *(not explicitly stated in this document)*",
            "",
            "> **Note:** The base agreement could not be identified from the amendment text. "
            "Cross-referencing to the original agreement's clause set is deferred — see "
            "CLAUDE.md FUTURE_WORK: *retrieve_base_agreement*.",
            "",
        ]

    lines += [
        "---",
        "",
        "## Clauses / Sections Modified",
        "",
    ]

    if modified:
        for clause in modified:
            lines.append(f"- {clause}")
        lines.append("")
    else:
        lines += [
            "*(No specific clause names were extracted — the amendment may use "
            "non-standard headings or modify the agreement globally.)*",
            "",
        ]

    lines += [
        "---",
        "",
        "## Analysis Notes",
        "",
        f"- **Analysis confidence:** {confidence:.0%}",
        "- **Risk scoring:** Not applicable — template comparison does not apply to amendments.",
        "- **HITL review:** Not required for standard amendment analysis.",
    ]

    if error:
        lines += [
            "",
            f"> ⚠️ Analysis encountered an error: `{error}`",
        ]

    return "\n".join(lines)
