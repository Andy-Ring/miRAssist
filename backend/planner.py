from __future__ import annotations

import json
import re
from typing import Any, Dict

from backend.config import get_default_k, get_default_result_count, get_planner_model, get_planner_temperature


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
  "phenotype_context": {
    "phenotype": string | null,
    "direction": "promotes" | "suppresses" | "increases" | "decreases" | "associated" | null,
    "raw_phrase": string | null
  },
  "pathway_selection_request": {
    "enabled": boolean,
    "query_terms": [string],
    "directional_query_terms": [string],
    "strict": boolean
  },
  "phenotype_keywords": [string],
  "pathway_keywords": [string],
  "pathway_filter": {
    "enabled": boolean,
    "mode": "filter",
    "min_gene_sets": number
  },
  "novel": boolean,
  "k": number,
  "result_count": number | null,
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
- "k" is the number of evidence cards/candidates passed forward to synthesis after backend retrieval and scoring.
- "result_count" is the number of final ranked results to print. If the user does not explicitly ask for a top-N, leave this null and the app will default to 5 printed results.
- "phenotype_keywords" should capture things like proliferation, apoptosis, EMT, invasion, etc.
- "phenotype_context" should only reflect the user's stated or strongly implied biological context.
- "pathway_selection_request.enabled" should be true if phenotype or pathway context is implied.
- "pathway_selection_request.directional_query_terms" should include exact phrases like "positive regulation of apoptosis" or "negative regulation of apoptosis" when relevant.
- "pathway_filter.enabled" should be true if phenotype or pathway context is implied, and if enabled its mode must always be "filter".
- Do not invent genes or gene-pathway memberships.
"""

_CANCER_TCGA_RULES = [
    (re.compile(r"\bbreast\b", re.IGNORECASE), "BRCA", "breast cancer"),
    (re.compile(r"\b(colon|colorectal)\b", re.IGNORECASE), "COAD", "colon cancer"),
    (re.compile(r"\bprostate\b", re.IGNORECASE), "PRAD", "prostate cancer"),
]


def _normalize_cancer_context(qs: Dict[str, Any]) -> None:
    cancer = qs.get("cancer") or {}
    name = str(cancer.get("name") or "").strip()
    tcga = str(cancer.get("tcga") or "").strip().upper()

    if not name and not tcga:
        return

    for pattern, mapped_tcga, canonical_name in _CANCER_TCGA_RULES:
        haystack = name or tcga
        if pattern.search(haystack):
            cancer["name"] = canonical_name
            cancer["tcga"] = mapped_tcga
            qs["cancer"] = cancer
            return

    if tcga:
        cancer["tcga"] = tcga
        qs["cancer"] = cancer


def _normalize_optional_clarifications(qs: Dict[str, Any]) -> None:
    needs = [str(item).strip() for item in (qs.get("needs_clarification") or []) if str(item).strip()]
    if not needs:
        qs["needs_clarification"] = []
        qs.setdefault("optional_clarifications", [])
        return

    is_actionable = bool(qs.get("mirna") or qs.get("gene"))
    if not is_actionable:
        qs["needs_clarification"] = needs
        qs.setdefault("optional_clarifications", [])
        return

    qs["optional_clarifications"] = list(dict.fromkeys(needs + list(qs.get("optional_clarifications") or [])))
    qs["needs_clarification"] = []


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
    qs.setdefault(
        "phenotype_context",
        {"phenotype": None, "direction": None, "raw_phrase": None},
    )
    qs["phenotype_context"].setdefault("phenotype", None)
    qs["phenotype_context"].setdefault("direction", None)
    qs["phenotype_context"].setdefault("raw_phrase", None)
    qs.setdefault(
        "pathway_selection_request",
        {"enabled": False, "query_terms": [], "directional_query_terms": [], "strict": False},
    )
    qs["pathway_selection_request"].setdefault("enabled", False)
    qs["pathway_selection_request"].setdefault("query_terms", [])
    qs["pathway_selection_request"].setdefault("directional_query_terms", [])
    qs["pathway_selection_request"].setdefault("strict", False)

    # Pathway filter
    qs.setdefault(
        "pathway_filter",
        {"enabled": False, "mode": "filter", "min_gene_sets": 1},
    )
    qs["pathway_filter"].setdefault("enabled", False)
    qs["pathway_filter"].setdefault("mode", "filter")
    qs["pathway_filter"].setdefault("min_gene_sets", 1)

    qs.setdefault("novel", False)
    qs.setdefault("k", get_default_k())
    qs.setdefault("result_count", None)

    # Filters
    qs.setdefault("filters", {})
    qs["filters"].setdefault("min_support", 1)
    qs["filters"].setdefault("require_binding_evidence", False)
    qs["filters"].setdefault("require_expression", False)

    qs.setdefault("needs_clarification", [])
    qs.setdefault("optional_clarifications", [])

    # Type coercion safety
    try:
        qs["k"] = int(qs["k"])
    except Exception:
        qs["k"] = get_default_k()

    try:
        if qs["result_count"] is not None:
            qs["result_count"] = int(qs["result_count"])
    except Exception:
        qs["result_count"] = get_default_result_count()

    try:
        qs["filters"]["min_support"] = int(qs["filters"]["min_support"])
    except Exception:
        qs["filters"]["min_support"] = 1

    if qs["mode"] not in ("mirna_to_targets", "gene_to_mirnas"):
        qs["mode"] = "mirna_to_targets"

    direction = qs["phenotype_context"].get("direction")
    if direction not in {"promotes", "suppresses", "increases", "decreases", "associated", None}:
        qs["phenotype_context"]["direction"] = None

    pathway_enabled = bool(
        qs["pathway_selection_request"].get("enabled")
        or qs["pathway_filter"].get("enabled")
        or qs.get("phenotype_keywords")
        or qs.get("pathway_keywords")
        or qs["phenotype_context"].get("phenotype")
    )
    qs["pathway_selection_request"]["enabled"] = pathway_enabled
    qs["pathway_selection_request"]["strict"] = pathway_enabled
    qs["pathway_filter"]["enabled"] = pathway_enabled
    qs["pathway_filter"]["mode"] = "filter"

    _normalize_cancer_context(qs)
    _normalize_optional_clarifications(qs)

    return qs


def run_planner(question: str) -> Dict[str, Any]:
    """
    Main entrypoint used by the FastAPI backend.
    """
    from backend.llm_backend import chat

    question = (question or "").strip()
    if not question:
        raise ValueError("Question is empty.")

    response = chat(
        system=PLANNER_SYSTEM_PROMPT,
        user=f"User question:\n{question}\n\nReturn JSON QuerySpec only.",
        model=get_planner_model(),
        max_new_tokens=700,
        temperature=get_planner_temperature(),
        top_p=1.0,
    )

    qs = _json_from_text(response)
    return _validate_and_fill(qs, question)
