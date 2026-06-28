"""
agent/comparator.py
-------------------
Template comparison node for the Legal AI Assistant.

What this file does:
    For each clause that was successfully extracted from a document, this
    module loads the corresponding standard template from data/templates/ and
    asks GPT-4o-mini to compare the two texts semantically. The result
    describes HOW the extracted clause differs from the standard -- whether
    the difference is none, minor, or major.

    The risk engine (risk_engine.py) then converts this comparison result
    into a risk level (LOW / MEDIUM / HIGH).

Why GPT-4o-mini for comparison instead of a text diff?
    Legal clause comparison is semantic, not textual. A clause that says
    "New York" instead of "California" in a Governing Law provision looks
    like a two-word diff, but it is a major substantive change. A clause
    that has been reformatted or lightly paraphrased might look different
    character-for-character but carry identical legal meaning. An LLM
    understands the substance; a text diff does not.

Template file naming convention:
    data/templates/<ClauseType>.txt where the clause type name has spaces
    replaced by underscores and the "/" replaced by "_".
    Examples:
      "Indemnification"              -> Indemnification.txt
      "Governing Law / Jurisdiction" -> Governing_Law_Jurisdiction.txt
      "Confidentiality / Non-Disclosure" -> Confidentiality_Non-Disclosure.txt

Dependencies:
    openai, pydantic, yaml, config.py, schemas/clause.py, schemas/risk.py
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional

import yaml
from openai import OpenAI

from agent.clause_expander import ClauseExpansionResult
from config import OPENAI_API_KEY, LLM_MODEL, CLAUSE_CATEGORIES
from schemas.clause import ExtractedClause
from schemas.risk import ClauseComparison

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TEMPLATES_DIR = Path(__file__).parent.parent / "data" / "templates"
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "compare_to_template.yaml"

# ---------------------------------------------------------------------------
# Lazy OpenAI client
# ---------------------------------------------------------------------------

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


# ---------------------------------------------------------------------------
# Template file resolver
# ---------------------------------------------------------------------------

def _clause_type_to_filename(clause_type: str) -> str:
    """
    Convert a clause type string to its template filename.

    Rules:
      - Replace " / " with "_"  (e.g. "Governing Law / Jurisdiction" -> "Governing_Law_Jurisdiction")
      - Replace remaining spaces with "_"
      - Append ".txt"

    Examples:
      "Indemnification"                   -> "Indemnification.txt"
      "Governing Law / Jurisdiction"      -> "Governing_Law_Jurisdiction.txt"
      "Confidentiality / Non-Disclosure"  -> "Confidentiality_Non-Disclosure.txt"
      "Non-Compete / Non-Solicitation"    -> "Non-Compete_Non-Solicitation.txt"
    """
    name = clause_type.replace(" / ", "_").replace(" ", "_")
    return f"{name}.txt"


def _load_template(clause_type: str) -> Optional[tuple[str, str]]:
    """
    Load the template file for a clause type.

    Returns
    -------
    (template_text, template_path_str) if the file exists, None otherwise.
    """
    filename = _clause_type_to_filename(clause_type)
    path = TEMPLATES_DIR / filename
    if not path.exists():
        return None
    template_text = path.read_text(encoding="utf-8").strip()
    return template_text, str(path)


# ---------------------------------------------------------------------------
# Prompt loader
# ---------------------------------------------------------------------------

def _load_prompt() -> dict:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Core LLM call
# ---------------------------------------------------------------------------

def _call_llm(
    clause_type: str,
    template_text: str,
    extracted_text: str,
) -> dict:
    """
    Ask GPT-4o-mini to compare the extracted clause against the template.

    Returns the parsed JSON dict with keys:
        matches_template (bool), deviation_severity (str), deviation_summary (str|None)

    Raises ValueError on parse or validation failure.
    """
    prompt = _load_prompt()

    user_message = prompt["user"].format(
        clause_type=clause_type,
        template_text=template_text,
        extracted_text=extracted_text,
    )

    client = _get_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    from agent.metrics_writer import accumulate_tokens
    accumulate_tokens(response.usage.prompt_tokens, response.usage.completion_tokens)

    raw = response.choices[0].message.content
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned non-JSON: {raw!r}") from e

    # Validate required keys
    for key in ("matches_template", "deviation_severity"):
        if key not in data:
            raise ValueError(f"LLM response missing key '{key}': {data}")

    valid_severities = {"none", "minor", "major"}
    if data["deviation_severity"] not in valid_severities:
        raise ValueError(
            f"Invalid deviation_severity {data['deviation_severity']!r}. "
            f"Expected one of {valid_severities}."
        )

    return data


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def compare_to_templates(
    clauses: list[ExtractedClause],
    expansions: Optional[Dict[str, ClauseExpansionResult]] = None,
) -> list[ClauseComparison]:
    """
    Compare each present extracted clause against its standard template.

    How it works:
        1. For each ExtractedClause where is_present=True:
           a. Determine comparison text: if an expansion exists for this clause
              type AND expansion_triggered=True, use expanded_text (which merges
              the source chunk with adjacent related chunks containing the full
              clause obligations).  Otherwise use clause.extracted_text as before.
              This prevents partial extraction boundaries from causing false HIGH
              risk findings (e.g. only Section 1 definition vs. full template).
           b. Look up the template file for that clause_type.
           c. If no template exists: record template_found=False, skip LLM call.
           d. If template exists: call GPT-4o-mini to compare the texts.
           e. Wrap the result in a ClauseComparison object.
        2. For absent clauses (is_present=False): return a ClauseComparison
           with template_found based on whether a file exists, but
           deviation_severity="none" (nothing to compare -- risk_engine
           handles absent clauses separately as always-HIGH).

    Parameters
    ----------
    clauses : list[ExtractedClause]
        The 10 ExtractedClause objects from extract_clauses(). Must be in
        CLAUSE_CATEGORIES order.
    expansions : Optional[Dict[str, ClauseExpansionResult]]
        Mapping clause_type -> ClauseExpansionResult from the
        expand_clause_boundaries node.  When provided and expansion was
        triggered for a clause, the comparator uses expanded_text instead of
        the original snippet.  None preserves original behaviour (backward
        compatible with mocked pipeline tests that do not expand).

    Returns
    -------
    list[ClauseComparison]
        Exactly one ClauseComparison per input clause, in the same order.
        Never raises -- errors are caught and logged, returning a safe
        "no-template" comparison result.
    """
    results: list[ClauseComparison] = []

    for clause in clauses:
        # Absent clauses: note whether a template exists but don't compare
        if not clause.is_present:
            template_info = _load_template(clause.clause_type)
            results.append(ClauseComparison(
                clause_type=clause.clause_type,
                template_found=template_info is not None,
                matches_template=False,
                deviation_severity="none",
                deviation_summary=None,
                template_path=template_info[1] if template_info else None,
            ))
            continue

        # Choose comparison text: expanded (if triggered) or original snippet
        comparison_text = clause.extracted_text
        if expansions:
            exp = expansions.get(clause.clause_type)
            if exp and exp.expansion_triggered:
                comparison_text = exp.expanded_text
                logger.info(
                    "compare_to_templates: '%s' using expanded text (%d chunks merged)",
                    clause.clause_type,
                    len(exp.source_chunk_ids),
                )

        # Present clause: load template
        template_info = _load_template(clause.clause_type)
        if template_info is None:
            logger.info(
                f"No template found for clause type '{clause.clause_type}'. "
                "Skipping comparison -- risk_engine will default to LOW."
            )
            results.append(ClauseComparison(
                clause_type=clause.clause_type,
                template_found=False,
                matches_template=False,
                deviation_severity="none",
                deviation_summary=None,
                template_path=None,
            ))
            continue

        template_text, template_path = template_info

        try:
            llm_result = _call_llm(
                clause_type=clause.clause_type,
                template_text=template_text,
                extracted_text=comparison_text,
            )
            results.append(ClauseComparison(
                clause_type=clause.clause_type,
                template_found=True,
                matches_template=llm_result["matches_template"],
                deviation_severity=llm_result["deviation_severity"],
                deviation_summary=llm_result.get("deviation_summary"),
                template_path=template_path,
            ))
        except (ValueError, Exception) as e:
            logger.error(
                f"Template comparison failed for '{clause.clause_type}': {e}. "
                "Defaulting to no-deviation result."
            )
            # Safe fallback: treat as no deviation (conservative -- don't
            # manufacture a risk finding from a failed comparison)
            results.append(ClauseComparison(
                clause_type=clause.clause_type,
                template_found=True,
                matches_template=True,
                deviation_severity="none",
                deviation_summary=None,
                template_path=template_path,
            ))

    return results
