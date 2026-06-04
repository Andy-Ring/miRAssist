from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

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
    "observed_change": "increased" | "decreased" | "promoted" | "suppressed" | "associated" | null,
    "miRNA_perturbation": "overexpression" | "knockdown" | "inhibition" | "unknown" | null,
    "raw_phrase": string | null,
    "direction": "promotes" | "suppresses" | "increases" | "decreases" | "associated" | null
  },
  "target_role_inference": {
    "enabled": boolean,
    "assumption": string,
    "expected_target_effect_on_phenotype": "positive_regulator" | "negative_regulator" | "unknown",
    "reasoning": string
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
- Extract separate fields for phenotype, observed phenotype change, and miRNA perturbation whenever possible.
- Do not claim that the miRNA directly activates genes.
- "target_role_inference" should apply only the default assumption that miRNAs usually repress direct targets.
- For miRNA overexpression: increased/promoted phenotype implies likely target genes are negative regulators of that phenotype; decreased/suppressed phenotype implies likely target genes are positive regulators.
- For knockdown/inhibition or unknown perturbation, use "unknown" unless the direction is truly clear from the question.
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
_LEGACY_DIRECTION_TO_OBSERVED = {
    "promotes": "promoted",
    "suppresses": "suppressed",
    "increases": "increased",
    "decreases": "decreased",
    "associated": "associated",
}
_OBSERVED_TO_LEGACY_DIRECTION = {
    "promoted": "promotes",
    "suppressed": "suppresses",
    "increased": "increases",
    "decreased": "decreases",
    "associated": "associated",
}
_OBSERVED_CHANGE_PATTERNS = [
    (re.compile(r"\b(promot(?:e|ed|es|ing))\b", re.IGNORECASE), "promoted"),
    (re.compile(r"\b(increase(?:d|s)?|enhance(?:d|s)?)\b", re.IGNORECASE), "increased"),
    (re.compile(r"\b(suppress(?:ed|es)?)\b", re.IGNORECASE), "suppressed"),
    (re.compile(r"\b(decrease(?:d|s)?|reduc(?:ed|es|tion))\b", re.IGNORECASE), "decreased"),
    (re.compile(r"\bassociated with\b", re.IGNORECASE), "associated"),
]
_PERTURBATION_PATTERNS = [
    (
        re.compile(
            r"\b(overexpress(?:ed|ion)?|ectopic expression|forced expression|mimic transfection|miRNA mimic)\b",
            re.IGNORECASE,
        ),
        "overexpression",
    ),
    (re.compile(r"\b(knockdown|knock-down|depletion)\b", re.IGNORECASE), "knockdown"),
    (re.compile(r"\b(inhibit(?:ed|ion)?|inhibitor|antagomir|anti-?mir)\b", re.IGNORECASE), "inhibition"),
]
_PHENOTYPE_PATTERNS = [
    (re.compile(r"\b(apoptosis|apoptotic process|apoptotic)\b", re.IGNORECASE), "apoptosis"),
    (re.compile(r"\b(proliferation|cell proliferation|cell cycle)\b", re.IGNORECASE), "proliferation"),
    (re.compile(r"\b(migration)\b", re.IGNORECASE), "migration"),
    (re.compile(r"\b(invasion|invasive)\b", re.IGNORECASE), "invasion"),
    (re.compile(r"\b(emt|epithelial[- ]mesenchymal transition)\b", re.IGNORECASE), "EMT"),
    (re.compile(r"\b(energy metabolism|metabolism|metabolic reprogramming)\b", re.IGNORECASE), "energy metabolism"),
]
_DIRECTIONAL_PATHWAY_TERMS = {
    "apoptosis": {
        "negative_regulator": [
            "negative regulation of apoptosis",
            "negative regulation of apoptotic process",
            "anti apoptotic",
            "apoptosis inhibitor",
        ],
        "positive_regulator": [
            "positive regulation of apoptosis",
            "positive regulation of apoptotic process",
            "activation of apoptotic signaling pathway",
            "pro apoptotic",
        ],
    },
    "proliferation": {
        "negative_regulator": [
            "negative regulation of cell proliferation",
            "negative regulation of proliferation",
            "cell cycle arrest",
            "growth suppression",
            "tumor suppressor",
        ],
        "positive_regulator": [
            "positive regulation of cell proliferation",
            "cell cycle",
            "mitotic cell cycle",
            "growth factor signaling",
            "proliferation",
        ],
    },
    "migration": {
        "negative_regulator": [
            "negative regulation of cell migration",
            "cell migration suppressor",
            "migration inhibition",
        ],
        "positive_regulator": [
            "positive regulation of cell migration",
            "cell migration",
            "motility",
        ],
    },
    "invasion": {
        "negative_regulator": [
            "negative regulation of cell invasion",
            "invasion suppression",
            "anti invasive",
        ],
        "positive_regulator": [
            "positive regulation of cell invasion",
            "cell invasion",
            "invasive growth",
        ],
    },
    "emt": {
        "negative_regulator": [
            "negative regulation of epithelial mesenchymal transition",
            "EMT suppression",
            "mesenchymal transition inhibitor",
        ],
        "positive_regulator": [
            "positive regulation of epithelial mesenchymal transition",
            "EMT",
            "mesenchymal transition",
        ],
    },
}


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


def _canonicalize_observed_change(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in _LEGACY_DIRECTION_TO_OBSERVED:
        return _LEGACY_DIRECTION_TO_OBSERVED[text]
    if text in {"promoted", "increased", "suppressed", "decreased", "associated"}:
        return text
    if text == "associated":
        return "associated"
    for pattern, canonical in _OBSERVED_CHANGE_PATTERNS:
        if pattern.search(text):
            return canonical
    return None


def _legacy_direction_from_observed_change(value: Any) -> Optional[str]:
    observed = _canonicalize_observed_change(value)
    if observed is None:
        return None
    return _OBSERVED_TO_LEGACY_DIRECTION.get(observed)


def _canonicalize_perturbation(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in {"overexpression", "knockdown", "inhibition", "unknown"}:
        return text
    for pattern, canonical in _PERTURBATION_PATTERNS:
        if pattern.search(text):
            return canonical
    return None


def _infer_phenotype_name(text: str) -> Optional[str]:
    for pattern, phenotype in _PHENOTYPE_PATTERNS:
        if pattern.search(text):
            return phenotype
    return None


def _enrich_phenotype_context(question: str, context: Dict[str, Any]) -> Dict[str, Any]:
    question_text = str(question or "").strip()
    context = dict(context or {})

    phenotype = str(context.get("phenotype") or "").strip() or None
    observed_change = _canonicalize_observed_change(
        context.get("observed_change") or context.get("direction")
    )
    perturbation = _canonicalize_perturbation(context.get("miRNA_perturbation"))
    raw_phrase = str(context.get("raw_phrase") or "").strip() or None

    lowered_question = question_text.lower()
    if phenotype is None:
        phenotype = _infer_phenotype_name(question_text)
    if observed_change is None:
        observed_change = _canonicalize_observed_change(lowered_question)
    if perturbation is None:
        perturbation = _canonicalize_perturbation(lowered_question)

    if raw_phrase is None and (phenotype or observed_change or perturbation):
        raw_phrase = question_text or None

    if perturbation is None and (phenotype or observed_change):
        perturbation = "unknown"

    return {
        "phenotype": phenotype,
        "observed_change": observed_change,
        "miRNA_perturbation": perturbation,
        "raw_phrase": raw_phrase,
        "direction": _legacy_direction_from_observed_change(observed_change),
    }


def build_directional_query_terms(phenotype: Any, expected_target_role: Any) -> List[str]:
    phenotype_text = str(phenotype or "").strip()
    role_text = str(expected_target_role or "").strip().lower()
    if not phenotype_text or role_text not in {"positive_regulator", "negative_regulator"}:
        return []

    phenotype_key = phenotype_text.lower()
    if phenotype_key in _DIRECTIONAL_PATHWAY_TERMS:
        return list(_DIRECTIONAL_PATHWAY_TERMS[phenotype_key].get(role_text, []))

    human_role = "positive regulation of" if role_text == "positive_regulator" else "negative regulation of"
    return [f"{human_role} {phenotype_text.lower()}"]


def infer_expected_target_role(phenotype_context: Dict[str, Any]) -> Dict[str, Any]:
    context = dict(phenotype_context or {})
    phenotype = str(context.get("phenotype") or "").strip() or None
    observed_change = _canonicalize_observed_change(
        context.get("observed_change") or context.get("direction")
    )
    perturbation = _canonicalize_perturbation(context.get("miRNA_perturbation"))
    raw_phrase = str(context.get("raw_phrase") or "").strip()

    inference: Dict[str, Any] = {
        "enabled": bool(phenotype or raw_phrase),
        "assumption": "miRNAs usually repress target gene expression",
        "expected_target_effect_on_phenotype": "unknown",
        "reasoning": "",
    }
    if not inference["enabled"]:
        return inference

    if perturbation == "overexpression":
        if observed_change in {"promoted", "increased"}:
            inference["expected_target_effect_on_phenotype"] = "negative_regulator"
            inference["reasoning"] = (
                f"The user reported miRNA overexpression increased {phenotype or 'the phenotype'}. "
                "Since miRNAs usually repress target genes, the most direct target interpretation is "
                f"repression of genes that normally suppress {phenotype or 'that phenotype'}."
            )
        elif observed_change in {"suppressed", "decreased"}:
            inference["expected_target_effect_on_phenotype"] = "positive_regulator"
            inference["reasoning"] = (
                f"The user reported miRNA overexpression decreased {phenotype or 'the phenotype'}. "
                "Since miRNAs usually repress target genes, the most direct target interpretation is "
                f"repression of genes that normally promote {phenotype or 'that phenotype'}."
            )
        else:
            inference["reasoning"] = (
                "miRNA overexpression was mentioned, but the phenotype change was not directional enough "
                "to infer whether direct targets are positive or negative regulators."
            )
        return inference

    if perturbation in {"knockdown", "inhibition"}:
        inference["reasoning"] = (
            "The user described miRNA knockdown or inhibition. Because that reduces repression of target genes, "
            "direct directional target-role inference is treated as ambiguous here unless stronger context is provided."
        )
        return inference

    inference["reasoning"] = (
        "The user did not provide a confident miRNA perturbation direction, so pathway filtering should stay "
        "at the general phenotype level without assuming positive- or negative-regulator targets."
    )
    return inference


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
        {
            "phenotype": None,
            "observed_change": None,
            "miRNA_perturbation": None,
            "raw_phrase": None,
            "direction": None,
        },
    )
    qs["phenotype_context"].setdefault("phenotype", None)
    qs["phenotype_context"].setdefault("observed_change", None)
    qs["phenotype_context"].setdefault("miRNA_perturbation", None)
    qs["phenotype_context"].setdefault("direction", None)
    qs["phenotype_context"].setdefault("raw_phrase", None)
    qs["phenotype_context"] = _enrich_phenotype_context(question, qs["phenotype_context"])
    qs["target_role_inference"] = infer_expected_target_role(qs["phenotype_context"])
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

    observed_change = qs["phenotype_context"].get("observed_change")
    if observed_change not in {"promoted", "suppressed", "increased", "decreased", "associated", None}:
        qs["phenotype_context"]["observed_change"] = None

    perturbation = qs["phenotype_context"].get("miRNA_perturbation")
    if perturbation not in {"overexpression", "knockdown", "inhibition", "unknown", None}:
        qs["phenotype_context"]["miRNA_perturbation"] = None

    qs["target_role_inference"] = infer_expected_target_role(qs["phenotype_context"])

    phenotype = qs["phenotype_context"].get("phenotype")
    directional_terms = build_directional_query_terms(
        phenotype,
        (qs.get("target_role_inference") or {}).get("expected_target_effect_on_phenotype"),
    )
    existing_directional_terms = [
        str(term).strip()
        for term in (qs["pathway_selection_request"].get("directional_query_terms") or [])
        if str(term).strip()
    ]
    if directional_terms:
        qs["pathway_selection_request"]["directional_query_terms"] = list(
            dict.fromkeys(directional_terms + existing_directional_terms)
        )
    else:
        qs["pathway_selection_request"]["directional_query_terms"] = existing_directional_terms

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
