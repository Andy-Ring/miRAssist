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
    "expected_target_expression_change": "increased" | "decreased" | "unknown",
    "expected_target_effect_on_phenotype": "positive_regulator" | "negative_regulator" | "unknown",
    "reasoning": string,
    "confidence": "moderate" | "low"
  },
  "pathway_selection_request": {
    "enabled": boolean,
    "query_terms": [string],
    "directional_query_terms": [string],
    "strict": boolean,
    "strict_directional": boolean
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
- Under the usual repression assumption, increased miRNA implies decreased target expression, while decreased miRNA (including inhibitor, antagomir, knockdown, knockout, or depletion) implies increased target expression.
- Combine target-expression direction with phenotype direction: decreased miRNA + increased phenotype => positive regulators; decreased miRNA + decreased phenotype => negative regulators; increased miRNA + increased phenotype => negative regulators; increased miRNA + decreased phenotype => positive regulators.
- If either direction is unknown or contradictory, keep both inferred target fields unknown.
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
    (re.compile(r"\b(decrease(?:d|s)?|reduc(?:e|ed|es|tion)|inhibit(?:ed|s)?)\b", re.IGNORECASE), "decreased"),
    (re.compile(r"\b(unchanged|no\s+(?:change|effect)|not\s+(?:changed|altered))\b", re.IGNORECASE), "unchanged"),
    (re.compile(r"\b(unknown|unclear|not\s+(?:known|determined))\b", re.IGNORECASE), "unknown"),
    (re.compile(r"\bassociated with\b", re.IGNORECASE), "associated"),
]
_PERTURBATION_PATTERNS = [
    (
        re.compile(
            r"\b(increas(?:e|ed|ing)|elevat(?:e|ed|ion)|overexpress(?:ed|ion)?|upregulat(?:e|ed|ion)|ectopic expression|forced expression|mimic(?: transfection| treatment)?|miRNA mimic)\b",
            re.IGNORECASE,
        ),
        "overexpression",
    ),
    (
        re.compile(
            r"\b(decreas(?:e|ed|ing)|reduc(?:e|ed|tion)|downregulat(?:e|ed|ion)|"
            r"knock(?:ed)?[- ]?down|knockout|knock(?:ed)?[- ]?out|deplet(?:e|ed|ion))\b",
            re.IGNORECASE,
        ),
        "knockdown",
    ),
    (
        re.compile(
            r"\b(inhibit(?:ed|ion)?|suppress(?:ed|ion)?|inhibitor(?: treatment)?|antagomir(?: treatment)?|anti-?mir)\b",
            re.IGNORECASE,
        ),
        "inhibition",
    ),
]
_PHENOTYPE_PATTERNS = [
    (re.compile(r"\b(apoptosis|apoptotic process|apoptotic)\b", re.IGNORECASE), "apoptosis"),
    (re.compile(r"\b(proliferation|cell proliferation|cell cycle)\b", re.IGNORECASE), "proliferation"),
    (re.compile(r"\b(migration)\b", re.IGNORECASE), "migration"),
    (re.compile(r"\b(invasion|invasive)\b", re.IGNORECASE), "invasion"),
    (re.compile(r"\b(emt|epithelial[- ]mesenchymal transition)\b", re.IGNORECASE), "EMT"),
    (re.compile(r"\b(energy metabolism|metabolism|metabolic reprogramming)\b", re.IGNORECASE), "energy metabolism"),
]
_PHENOTYPE_CONCEPT_PATTERNS = [
    (re.compile(r"\bcell\s+invasion\b|\binvasive\b|\binvasion\b", re.IGNORECASE), "cell invasion"),
    (re.compile(r"\bcell\s+migration\b|\bmigration\b", re.IGNORECASE), "cell migration"),
    (re.compile(r"\bcell\s+motility\b|\bmotility\b", re.IGNORECASE), "cell motility"),
    (re.compile(r"\bcell\s+proliferation\b|\bproliferation\b", re.IGNORECASE), "cell proliferation"),
    (re.compile(r"\bcell\s+survival\b|\bsurvival\b", re.IGNORECASE), "cell survival"),
    (re.compile(r"\bangiogenesis\b", re.IGNORECASE), "angiogenesis"),
    (re.compile(r"\bmetastasis\b|\bmetastatic\b", re.IGNORECASE), "metastasis"),
    (re.compile(r"\bapoptosis\b|\bapoptotic\b", re.IGNORECASE), "apoptosis"),
    (re.compile(r"\bemt\b|\bepithelial[- ]mesenchymal transition\b", re.IGNORECASE), "EMT"),
    (re.compile(r"\benergy metabolism\b|\bmetabolic reprogramming\b|\bmetabolism\b", re.IGNORECASE), "energy metabolism"),
]
_MIRNA_TEXT_RE = re.compile(r"\b(?:miR(?:NA)?s?|microRNAs?)\b", re.IGNORECASE)
_MIRNA_TOKEN_RE = re.compile(r"\b(?:hsa[-_\s]*)?(?:miR(?:NA)?|microRNA)[-_\s]*\d+[A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*(?:[-_](?:3p|5p))?\b", re.IGNORECASE)
_GENE_TO_MIRNA_PATTERNS = [
    re.compile(r"\bmiR(?:NA)?s?\b.*\b(?:regulat\w*|regult\w*|regulte|target\w*|bind\w*)\b", re.IGNORECASE),
    re.compile(r"\b(?:regulat\w*|regult\w*|regulte|target\w*|bind\w*)\b.*\b(?:by\s+)?miR(?:NA)?s?\b", re.IGNORECASE),
    re.compile(r"\bmiR(?:NA)?\s+regulators?\s+of\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+regulates\b", re.IGNORECASE),
]
_STOPWORD_ENTITY_TOKENS = {
    "A",
    "AN",
    "AND",
    "ARE",
    "AS",
    "BY",
    "CANDIDATE",
    "DO",
    "DOES",
    "FOR",
    "GENE",
    "GENES",
    "IN",
    "IS",
    "MAY",
    "MIR",
    "MIRNA",
    "MIRNAS",
    "OF",
    "PREDICTED",
    "REGULATE",
    "REGULATES",
    "REGULTE",
    "REGULATORS",
    "TARGET",
    "TARGETING",
    "TARGETS",
    "THE",
    "TO",
    "WHAT",
    "WHICH",
}
_DIRECTIONAL_PATHWAY_TERMS = {
    "cell migration": {
        "negative_regulator": [
            "negative regulation of cell migration",
            "negative regulation of migration",
            "negative regulation of cell motility",
            "cell migration suppressor",
            "migration inhibition",
        ],
        "positive_regulator": [
            "positive regulation of cell migration",
            "positive regulation of migration",
            "positive regulation of cell motility",
            "cell migration",
            "cell motility",
        ],
    },
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


def _observed_direction(value: Any) -> str:
    observed = str(value or "").strip().lower()
    if observed in {"promoted", "increased"}:
        return "increased"
    if observed in {"suppressed", "decreased"}:
        return "decreased"
    return "unknown"


def _perturbation_direction(value: Any) -> str:
    perturbation = str(value or "").strip().lower()
    if perturbation == "overexpression":
        return "increased"
    if perturbation in {"knockdown", "inhibition"}:
        return "decreased"
    return "unknown"


def _canonicalize_observed_change(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in _LEGACY_DIRECTION_TO_OBSERVED:
        return _LEGACY_DIRECTION_TO_OBSERVED[text]
    if text in {
        "promoted", "increased", "suppressed", "decreased",
        "unchanged", "unknown", "associated",
    }:
        return text
    matches = {
        canonical
        for pattern, canonical in _OBSERVED_CHANGE_PATTERNS
        if pattern.search(text)
    }
    directions = {
        _observed_direction(canonical)
        for canonical in matches
        if _observed_direction(canonical) in {"increased", "decreased"}
    }
    if len(directions) > 1:
        return "unknown"
    for canonical in (
        "unchanged", "unknown", "promoted", "increased",
        "suppressed", "decreased", "associated",
    ):
        if canonical in matches:
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
    matches = [
        canonical
        for pattern, canonical in _PERTURBATION_PATTERNS
        if pattern.search(text)
    ]
    directions = {_perturbation_direction(canonical) for canonical in matches}
    directions.discard("unknown")
    if len(directions) > 1:
        return "unknown"
    if matches:
        return matches[0]
    return None


def _directions_near_entity(
    text: str,
    entity_pattern: re.Pattern[str],
    signal_patterns: List[tuple[re.Pattern[str], str]],
    *,
    max_distance: int = 60,
) -> List[tuple[str, str]]:
    """Associate direction words with an entity without crossing another named entity."""
    results: List[tuple[str, str]] = []
    for entity_match in entity_pattern.finditer(text):
        for signal_pattern, label in signal_patterns:
            for signal_match in signal_pattern.finditer(text):
                if signal_match.end() <= entity_match.start():
                    between = text[signal_match.end():entity_match.start()]
                    distance = entity_match.start() - signal_match.end()
                elif signal_match.start() >= entity_match.end():
                    between = text[entity_match.end():signal_match.start()]
                    distance = signal_match.start() - entity_match.end()
                else:
                    between = ""
                    distance = 0
                if distance > max_distance:
                    continue
                if _MIRNA_TOKEN_RE.search(between) or any(
                    pattern.search(between) for pattern, _ in _PHENOTYPE_PATTERNS
                ):
                    continue
                results.append((label, signal_match.group(0)))
    return results


def _extract_mirna_perturbation(question: str) -> Optional[str]:
    text = str(question or "")
    if not text:
        return None
    safe_patterns = [
        (
            re.compile(
                r"\b(overexpress(?:ed|ion)?|upregulat(?:e|ed|ion)|elevat(?:e|ed|ion)|"
                r"ectopic expression|forced expression|mimic(?: transfection| treatment)?|miRNA mimic)\b",
                re.IGNORECASE,
            ),
            "overexpression",
        ),
        (
            re.compile(
                r"\b(downregulat(?:e|ed|ion)|knock(?:ed)?[- ]?down|knockout|"
                r"knock(?:ed)?[- ]?out|deplet(?:e|ed|ion))\b",
                re.IGNORECASE,
            ),
            "knockdown",
        ),
        (
            re.compile(
                r"\b(inhibitor(?: treatment)?|antagomir(?: treatment)?|anti-?mir)\b",
                re.IGNORECASE,
            ),
            "inhibition",
        ),
    ]
    associated = _directions_near_entity(
        text, _MIRNA_TOKEN_RE, safe_patterns, max_distance=75
    )
    if not associated and _MIRNA_TEXT_RE.search(text):
        for pattern, label in safe_patterns:
            for match in pattern.finditer(text):
                associated.append((label, match.group(0)))
    mirna_expr = _MIRNA_TOKEN_RE.pattern
    anchored_patterns = [
        (re.compile(mirna_expr + r"(?:\s+levels?)?\s+(?:was|were|is|are)\s+(?:increased|elevated|upregulated)\b", re.IGNORECASE), "overexpression"),
        (re.compile(r"\b(?:increased|elevated|upregulated)\s+(?:levels?\s+of\s+)?" + mirna_expr, re.IGNORECASE), "overexpression"),
        (re.compile(mirna_expr + r"(?:\s+levels?)?\s+(?:was|were|is|are)\s+(?:decreased|reduced|inhibited|suppressed)\b", re.IGNORECASE), "inhibition"),
        (re.compile(r"\b(?:decreased|reduced|inhibited|suppressed)\s+(?:levels?\s+of\s+)?" + mirna_expr, re.IGNORECASE), "inhibition"),
        (re.compile(mirna_expr + r"\s+(?:increased|elevated|upregulated)\s*(?:(?:and|while|but)\b|[,;])", re.IGNORECASE), "overexpression"),
        (re.compile(mirna_expr + r"\s+(?:decreased|reduced|inhibited|suppressed)\s*(?:(?:and|while|but)\b|[,;])", re.IGNORECASE), "inhibition"),
    ]
    for pattern, label in anchored_patterns:
        match = pattern.search(text)
        if match:
            associated.append((label, match.group(0)))
    phenotype_starts = [
        match.start()
        for pattern, _ in _PHENOTYPE_PATTERNS
        for match in pattern.finditer(text)
    ]
    for mirna_match in _MIRNA_TOKEN_RE.finditer(text):
        stop = min(
            [position for position in phenotype_starts if position > mirna_match.end()]
            or [min(len(text), mirna_match.end() + 90)]
        )
        segment = text[mirna_match.end():stop]
        if re.search(r"\b(?:was|were|is|are|levels?)\b", segment, re.IGNORECASE):
            if re.search(r"\b(?:increased|elevated|upregulated)\b", segment, re.IGNORECASE):
                associated.append(("overexpression", segment))
            if re.search(r"\b(?:decreased|reduced|inhibited|suppressed)\b", segment, re.IGNORECASE):
                associated.append(("inhibition", segment))
    if _MIRNA_TOKEN_RE.search(text):
        for pattern, label in safe_patterns:
            for match in pattern.finditer(text):
                nearby = text[max(0, match.start() - 35):match.end() + 35]
                if re.search(r"\b(?:it|mimic|inhibitor|antagomir)\b", nearby, re.IGNORECASE):
                    associated.append((label, match.group(0)))
    labels = [label for label, _ in associated]
    directions = {_perturbation_direction(label) for label in labels}
    directions.discard("unknown")
    if len(directions) > 1:
        return "unknown"
    if not labels:
        return None
    if "inhibition" in labels:
        return "inhibition"
    if "knockdown" in labels:
        return "knockdown"
    return "overexpression"


def _extract_observed_change(question: str, phenotype: Any) -> Optional[str]:
    text = str(question or "")
    phenotype_text = str(phenotype or "").strip()
    if not text or not phenotype_text:
        return None
    phenotype_pattern = re.compile(
        r"\b" + r"[\s_-]+".join(re.escape(token) for token in phenotype_text.split()) + r"\b",
        re.IGNORECASE,
    )
    candidates: List[tuple[str, int, str]] = []
    for phenotype_match in phenotype_pattern.finditer(text):
        for signal_pattern, label in _OBSERVED_CHANGE_PATTERNS:
            for signal_match in signal_pattern.finditer(text):
                if signal_match.end() <= phenotype_match.start():
                    between = text[signal_match.end():phenotype_match.start()]
                    distance = phenotype_match.start() - signal_match.end()
                    side = "before"
                elif signal_match.start() >= phenotype_match.end():
                    between = text[phenotype_match.end():signal_match.start()]
                    distance = signal_match.start() - phenotype_match.end()
                    side = "after"
                else:
                    between = ""
                    distance = 0
                    side = "overlap"
                if distance > 55 or _MIRNA_TOKEN_RE.search(between):
                    continue
                prior_mirnas = [
                    match for match in _MIRNA_TOKEN_RE.finditer(text[:signal_match.start()])
                ]
                if prior_mirnas:
                    prior_mirna = prior_mirnas[-1]
                    mirna_gap = text[prior_mirna.end():signal_match.start()]
                    if (
                        len(mirna_gap) <= 25
                        and not (
                            prior_mirna.end() <= phenotype_match.start() <= signal_match.start()
                        )
                        and re.search(
                        r"\b(?:was|were|is|are|levels?)\b",
                        mirna_gap,
                        re.IGNORECASE,
                        )
                    ):
                        continue
                candidates.append((label, distance, side))
    if not candidates:
        return None
    directional = [item for item in candidates if _observed_direction(item[0]) != "unknown"]
    directions = {_observed_direction(item[0]) for item in directional}
    if len(directions) > 1:
        increased_sides = {
            side for label, _, side in directional
            if _observed_direction(label) == "increased"
        }
        decreased_sides = {
            side for label, _, side in directional
            if _observed_direction(label) == "decreased"
        }
        if increased_sides & decreased_sides:
            return "unknown"
        nearest = min(directional, key=lambda item: item[1])
        return _observed_direction(nearest[0])
    values = [value for value, _, _ in candidates]
    if any(value == "unchanged" for value in values):
        return "unchanged"
    if any(value == "unknown" for value in values):
        return "unknown"
    if directions == {"increased"}:
        return "increased"
    if directions == {"decreased"}:
        return "decreased"
    return "associated" if "associated" in values else None

def _infer_phenotype_name(text: str) -> Optional[str]:
    for pattern, phenotype in _PHENOTYPE_PATTERNS:
        if pattern.search(text):
            return phenotype
    return None


def extract_phenotype_keywords(question: str, phenotype: Any = None) -> List[str]:
    """Return ordered, distinct biological concepts mentioned in a question."""
    concepts: List[str] = []
    for pattern, concept in _PHENOTYPE_CONCEPT_PATTERNS:
        if pattern.search(str(question or "")) and concept not in concepts:
            concepts.append(concept)
    canonical = str(phenotype or "").strip()
    if canonical:
        canonical = {"migration": "cell migration", "invasion": "cell invasion", "proliferation": "cell proliferation"}.get(canonical.lower(), canonical)
        if canonical not in concepts:
            concepts.insert(0, canonical)
    return concepts

def _looks_like_mirna_query_text(question: str) -> bool:
    return bool(_MIRNA_TOKEN_RE.search(str(question or "")))


def _looks_like_gene_to_mirnas_question(question: str) -> bool:
    text = str(question or "")
    if not _MIRNA_TEXT_RE.search(text):
        return False
    if _looks_like_mirna_query_text(text):
        return False
    return any(pattern.search(text) for pattern in _GENE_TO_MIRNA_PATTERNS) or bool(_extract_likely_gene_symbol(text))


def _extract_likely_gene_symbol(question: str) -> Optional[str]:
    text = str(question or "")
    clean = re.sub(r"[^\w\s-]", " ", text)
    tokens = [token.strip("-_") for token in clean.split() if token.strip("-_")]
    candidates: List[str] = []
    for token in tokens:
        normalized = re.sub(r"[^A-Za-z0-9]", "", token).upper()
        if not normalized or normalized in _STOPWORD_ENTITY_TOKENS:
            continue
        if normalized.startswith("MIR") or normalized.startswith("MICRORNA"):
            continue
        if 2 <= len(normalized) <= 15 and any(ch.isalpha() for ch in normalized):
            candidates.append(normalized)
    return candidates[-1] if candidates else None


def _apply_direction_text_overrides(qs: Dict[str, Any], question: str) -> None:
    if not _looks_like_gene_to_mirnas_question(question):
        return

    inferred_gene = _extract_likely_gene_symbol(question)
    if not inferred_gene:
        return

    qs["mode"] = "gene_to_mirnas"
    qs["gene"] = str(qs.get("gene") or inferred_gene).strip().upper()
    qs["mirna"] = None
    qs.setdefault("debug_warnings", [])
    warning = (
        "Planner direction corrected to gene_to_mirnas from question phrasing; "
        f"using gene {qs['gene']}."
    )
    if warning not in qs["debug_warnings"]:
        qs["debug_warnings"].append(warning)


def _enrich_phenotype_context(question: str, context: Dict[str, Any]) -> Dict[str, Any]:
    question_text = str(question or "").strip()
    context = dict(context or {})

    phenotype = str(context.get("phenotype") or "").strip() or None
    supplied_observed = context.get("observed_change") or context.get("direction")
    supplied_perturbation = context.get("miRNA_perturbation")
    observed_change = _canonicalize_observed_change(supplied_observed)
    perturbation = _canonicalize_perturbation(supplied_perturbation)
    raw_phrase = str(context.get("raw_phrase") or "").strip() or None

    if phenotype is None:
        phenotype = _infer_phenotype_name(question_text)
    if not supplied_observed:
        observed_change = _extract_observed_change(question_text, phenotype)
    if not supplied_perturbation:
        perturbation = _extract_mirna_perturbation(question_text)

    if raw_phrase is None and (phenotype or observed_change or perturbation):
        raw_phrase = question_text or None

    if perturbation is None and (phenotype or observed_change):
        perturbation = "unknown"
    if observed_change is None and phenotype:
        observed_change = "unknown"

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

    canonical_term = (
        "positive regulation of " if role_text == "positive_regulator"
        else "negative regulation of "
    ) + phenotype_text.lower()
    phenotype_key = phenotype_text.lower()
    if phenotype_key in _DIRECTIONAL_PATHWAY_TERMS:
        terms = [canonical_term] + list(
            _DIRECTIONAL_PATHWAY_TERMS[phenotype_key].get(role_text, [])
        )
        return list(dict.fromkeys(terms))
    return [canonical_term]


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
        "assumption": (
            "Assumption based on typical miRNA-mediated repression: "
            "miRNAs usually repress their direct targets."
        ),
        "expected_target_expression_change": "unknown",
        "expected_target_effect_on_phenotype": "unknown",
        "reasoning": "",
        "confidence": "low",
    }
    if not inference["enabled"]:
        return inference

    mirna_change = _perturbation_direction(perturbation)
    phenotype_change = _observed_direction(observed_change)
    if mirna_change == "increased":
        inference["expected_target_expression_change"] = "decreased"
    elif mirna_change == "decreased":
        inference["expected_target_expression_change"] = "increased"

    if mirna_change == "unknown" or phenotype_change == "unknown":
        inference["reasoning"] = (
            "Either the miRNA perturbation or the observed phenotype direction is unknown, "
            "unchanged, or contradictory. Target-role direction is therefore unknown, and "
            "general phenotype pathways should be used without inventing directionality."
        )
        return inference

    target_change = inference["expected_target_expression_change"]
    expected_role = (
        "positive_regulator" if target_change == phenotype_change
        else "negative_regulator"
    )
    inference["expected_target_effect_on_phenotype"] = expected_role
    inference["confidence"] = "moderate"
    role_words = "promote" if expected_role == "positive_regulator" else "suppress"
    phenotype_label = phenotype or "the phenotype"
    inference["reasoning"] = (
        f"The miRNA was {mirna_change}, so under the typical repression assumption its "
        f"direct targets are expected to be {target_change}. The phenotype was "
        f"{phenotype_change}; therefore candidate targets that normally {role_words} "
        f"{phenotype_label} are directionally consistent."
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


def _new_queryspec(question: str) -> Dict[str, Any]:
    return {
        "original_question": question,
        "mode": "mirna_to_targets",
        "mirna": None,
        "gene": None,
        "cancer": {"name": None, "tcga": None},
        "phenotype_context": {
            "phenotype": None,
            "observed_change": None,
            "miRNA_perturbation": None,
            "raw_phrase": None,
            "direction": None,
        },
        "target_role_inference": {
            "enabled": False,
            "assumption": (
                "Assumption based on typical miRNA-mediated repression: "
                "miRNAs usually repress their direct targets."
            ),
            "expected_target_expression_change": "unknown",
            "expected_target_effect_on_phenotype": "unknown",
            "reasoning": "",
            "confidence": "low",
        },
        "pathway_selection_request": {
            "enabled": False,
            "query_terms": [],
            "directional_query_terms": [],
            "strict": False,
            "strict_directional": False,
        },
        "phenotype_keywords": [],
        "pathway_keywords": [],
        "pathway_filter": {"enabled": False, "mode": "filter", "min_gene_sets": 1},
        "novel": False,
        "k": get_default_k(),
        "result_count": None,
        "filters": {
            "min_support": 1,
            "require_binding_evidence": False,
            "require_expression": False,
        },
        "needs_clarification": [],
    }


def _build_rule_based_queryspec(question: str) -> Optional[Dict[str, Any]]:
    """
    Handle direct entity lookup prompts without an LLM planner call.

    Posit Connect has shown a native segfault immediately after the planner LLM
    call completes. These common query shapes do not need generative planning:
    the database retrieval layer can answer them once the entity/direction is
    known.
    """
    text = str(question or "").strip()
    if not text:
        return None

    mirna_match = _MIRNA_TOKEN_RE.search(text)
    if mirna_match:
        qs = _new_queryspec(text)
        qs["mode"] = "mirna_to_targets"
        qs["mirna"] = mirna_match.group(0)
        return qs

    if _looks_like_gene_to_mirnas_question(text):
        gene = _extract_likely_gene_symbol(text)
        if gene:
            qs = _new_queryspec(text)
            qs["mode"] = "gene_to_mirnas"
            qs["gene"] = gene
            return qs

    return None


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
    extracted_keywords = extract_phenotype_keywords(question, qs["phenotype_context"].get("phenotype"))
    existing_keywords = [str(term).strip() for term in (qs.get("phenotype_keywords") or []) if str(term).strip()]
    qs["phenotype_keywords"] = list(dict.fromkeys(extracted_keywords + existing_keywords))
    existing_pathway_keywords = [str(term).strip() for term in (qs.get("pathway_keywords") or []) if str(term).strip()]
    qs["pathway_keywords"] = list(dict.fromkeys(existing_pathway_keywords))
    direct_terms = qs["phenotype_keywords"] + qs["pathway_keywords"] + [str(term).strip() for term in (qs.get("pathway_selection_request", {}).get("query_terms") or []) if str(term).strip()]
    qs.setdefault(
        "pathway_selection_request",
        {"enabled": False, "query_terms": [], "directional_query_terms": [], "strict": False, "strict_directional": False},
    )
    qs["pathway_selection_request"].setdefault("enabled", False)
    qs["pathway_selection_request"].setdefault("query_terms", [])
    qs["pathway_selection_request"].setdefault("directional_query_terms", [])
    qs["pathway_selection_request"].setdefault("strict", False)
    qs["pathway_selection_request"].setdefault("strict_directional", False)
    if re.search(
        r"\b(?:strict(?:ly)? directional|directionally consistent only|only directionally consistent|only (?:positive|negative) regulators?)\b",
        question,
        re.IGNORECASE,
    ):
        qs["pathway_selection_request"]["strict_directional"] = True

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

    _apply_direction_text_overrides(qs, question)

    direction = qs["phenotype_context"].get("direction")
    if direction not in {"promotes", "suppresses", "increases", "decreases", "associated", None}:
        qs["phenotype_context"]["direction"] = None

    observed_change = qs["phenotype_context"].get("observed_change")
    if observed_change not in {
        "promoted", "suppressed", "increased", "decreased",
        "unchanged", "unknown", "associated", None,
    }:
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
    expected_role = (qs.get("target_role_inference") or {}).get("expected_target_effect_on_phenotype")
    if directional_terms and expected_role in {"positive_regulator", "negative_regulator"}:
        qs["pathway_selection_request"]["directional_query_terms"] = directional_terms
    else:
        qs["pathway_selection_request"]["directional_query_terms"] = []

    pathway_enabled = bool(
        qs["pathway_selection_request"].get("enabled")
        or qs["pathway_filter"].get("enabled")
        or qs.get("phenotype_keywords")
        or qs.get("pathway_keywords")
        or qs["phenotype_context"].get("phenotype")
    )
    qs["pathway_selection_request"]["enabled"] = pathway_enabled
    qs["pathway_selection_request"]["query_terms"] = list(dict.fromkeys([str(term).strip() for term in direct_terms if str(term).strip()]))
    qs["pathway_selection_request"]["strict"] = pathway_enabled
    qs["pathway_filter"]["enabled"] = pathway_enabled
    qs["pathway_filter"]["mode"] = "filter"

    _normalize_cancer_context(qs)
    _normalize_optional_clarifications(qs)

    return qs


def run_planner(question: str) -> Dict[str, Any]:
    """
    Main planner entrypoint for the direct Streamlit app.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("Question is empty.")

    rule_based_qs = _build_rule_based_queryspec(question)
    if rule_based_qs is not None:
        print("[miRAssist] planner using deterministic entity parser", flush=True)
        return _validate_and_fill(rule_based_qs, question)

    from backend.llm_backend import chat

    print("[miRAssist] planner starting LLM call", flush=True)
    response = chat(
        system=PLANNER_SYSTEM_PROMPT,
        user=f"User question:\n{question}\n\nReturn JSON QuerySpec only.",
        model=get_planner_model(),
        max_new_tokens=700,
        temperature=get_planner_temperature(),
        top_p=1.0,
    )
    print(f"[miRAssist] planner received LLM text: {len(response)} chars", flush=True)

    qs = _json_from_text(response)
    print("[miRAssist] planner parsed QuerySpec JSON", flush=True)
    validated = _validate_and_fill(qs, question)
    print("[miRAssist] planner validated QuerySpec", flush=True)
    return validated
