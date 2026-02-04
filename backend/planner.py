from __future__ import annotations

import json
from typing import Any, Dict

from backend.llm_backend import chat


PLANNER_SYSTEM_PROMPT = """You are miRAssist's planner.

Your task is to convert a user's natural-language research question into a structured JSON QuerySpec
that will be used to retrieve candidate miRNA–mRNA interactions.

IMPORTANT RULES:
- Return ONLY valid JSON. No prose. No markdown. No explanations.
- Use null for unknown fields.
- Do not invent entities not stated or strongly implied by the question.
- If the user does not specify arm (3p/5p), do NOT guess here; retrieval logic will default safely.

REQUIRED SCHEMA (all keys must be present):

{
  "original_question": string,
  "mode": "mirna_to_targets" | "gene_to_mirnas",
  "mirna": string | null,
  "gene": string | null,
  "cancer": {
    "name": string | null,
    "tcga": string | null
  },
  "phenotype_keywords": [string],
  "pathway_keywords": [string],
  "pathway_filter": {
    "enabled": boolean,
    "mode": "boost" | "filter",
    "min_gene_sets": number
  },
  "novel": boolean,
  "k": number,
  "filters": {
    "min_support": number,
    "require_binding_evidence": boolean,
    "require_expression": boolean
  },
  "needs_clarification": [string]
}

NOTES:
- "mode" depends on whether the question centers on a miRNA or a gene.
- "novel" should be true if the user asks for new, unvalidated, or exploratory targets.
- "phenotype_keywords" should capture things like proliferation, apoptosis, EMT, invasion, etc.
- "pathway_filter.enabled" should be true if phenotype or pathway context is implied.
"""


def _json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract JSON object from LLM output.
    Handles markdown fences and extra leading/trailing text.
    """
    s = text.strip()

    # Strip markdown fences if present
    if s.startswith("```"):
        lines = s.splitlines()
        lines = lines[1:]  # drop opening fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()

    # Extract outermost JSON object
    i = s.find("{")
    j = s.rfind("}")
    if i >= 0 and j > i:
        s = s[i : j + 1]

    return json.loads(s)


def _validate_and_fill(qs: Dict[str, Any], question: str) -> Dict[str, Any]:
    """
    Enforce schema completeness and sensible defaults.
    """
    qs["original_question"] = question

    qs.setdefault("mode", "mirna_to_targets")
    qs.setdefault("mirna", None)
    qs.setdefault("gene", None)

    # Cancer block
    if qs.get("cancer") is None:
        qs["cancer"] = {"name": None, "tcga": None}
    qs["cancer"].setdefault("name", None)
    qs["cancer"].setdefault("tcga", None)

    qs.setdefault("phenotype_keywords", [])
    qs.setdefault("pathway_keywords", [])

    # Pathway filter
    qs.setdefault(
        "pathway_filter",
        {"enabled": False, "mode": "boost", "min_gene_sets": 1},
    )
    qs["pathway_filter"].setdefault("enabled", False)
    qs["pathway_filter"].setdefault("mode", "boost")
    qs["pathway_filter"].setdefault("min_gene_sets", 1)

    qs.setdefault("novel", False)
    qs.setdefault("k", 50)

    # Filters
    qs.setdefault("filters", {})
    qs["filters"].setdefault("min_support", 1)
    qs["filters"].setdefault("require_binding_evidence", False)
    qs["filters"].setdefault("require_expression", False)

    qs.setdefault("needs_clarification", [])

    # Type coercion safety
    try:
        qs["k"] = int(qs["k"])
    except Exception:
        qs["k"] = 50

    try:
        qs["filters"]["min_support"] = int(qs["filters"]["min_support"])
    except Exception:
        qs["filters"]["min_support"] = 1

    if qs["mode"] not in ("mirna_to_targets", "gene_to_mirnas"):
        qs["mode"] = "mirna_to_targets"

    return qs


def run_planner(question: str) -> Dict[str, Any]:
    """
    Main entrypoint used by the FastAPI backend.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("Question is empty.")

    response = chat(
        system=PLANNER_SYSTEM_PROMPT,
        user=f"User question:\n{question}\n\nReturn JSON QuerySpec only.",
        max_new_tokens=700,
        temperature=0.0,
        top_p=1.0,
    )

    qs = _json_from_text(response)
    return _validate_and_fill(qs, question)
