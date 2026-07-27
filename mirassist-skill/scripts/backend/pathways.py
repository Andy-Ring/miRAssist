from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from backend.config import ROOT_DIR
from backend.planner import build_directional_query_terms


PATHWAYS_DIR = ROOT_DIR / "data" / "processed" / "pathways"
DEFAULT_PATHWAYS_PATH = PATHWAYS_DIR / "pathways.parquet"
DEFAULT_GENE_TO_PATHWAYS_PATH = PATHWAYS_DIR / "gene_to_pathways.parquet"
DEFAULT_PATHWAYS_CSV_GZ_PATH = PATHWAYS_DIR / "pathways.csv.gz"
DEFAULT_PATHWAYS_CSV_PATH = PATHWAYS_DIR / "pathways.csv"
DEFAULT_GENE_TO_PATHWAYS_CSV_GZ_PATH = PATHWAYS_DIR / "gene_to_pathways.csv.gz"
DEFAULT_GENE_TO_PATHWAYS_CSV_PATH = PATHWAYS_DIR / "gene_to_pathways.csv"

logger = logging.getLogger(__name__)

_PATHWAYS_CACHE: Optional[pd.DataFrame] = None
_PATHWAYS_SOURCE: Optional[str] = None
_GENE_TO_PATHWAYS_CACHE: Optional[pd.DataFrame] = None
_GENE_TO_PATHWAYS_SOURCE: Optional[str] = None

_WORD_RE = re.compile(r"[^a-z0-9]+")
_DEFAULT_MIN_RELEVANCE_SCORE = 0.72
_COLLECTION_PRIORITY = {
    "GO Biological Process": 4,
    "Reactome": 3,
    "WikiPathways": 2,
    "Other curated": 2,
    "Hallmark": 1,
    "Unknown": 0,
}
_COLLECTION_SCORE_BONUS = {
    "GO Biological Process": 0.04,
    "Reactome": 0.03,
    "WikiPathways": 0.02,
    "Other curated": 0.01,
    "Hallmark": 0.0,
    "Unknown": 0.0,
}
_MATCH_PRIORITY = {
    "exact_normalized_name_match": 4,
    "exact_description_phrase_match": 3,
    "controlled_synonym": 2,
    "controlled_ontology_relation": 2,
    "broader_semantic_association": 1,
}

# Each tuple is (term, relationship to the source concept, strength). These
# expansions are deliberately small and auditable. Gene overlap is never used
# to infer phenotype relevance.
_CONTROLLED_PHENOTYPE_TERMS: Dict[str, List[tuple[str, str, float]]] = {
    "cell migration": [
        ("cell migration", "direct", 1.00),
        ("regulation of cell migration", "ontology", 0.94),
        ("cell motility", "synonym", 0.93),
        ("chemotaxis", "ontology", 0.90),
        ("focal adhesion", "ontology", 0.88),
        ("extracellular matrix organization", "ontology", 0.88),
        ("integrin signaling", "ontology", 0.88),
        ("regulation of actin cytoskeleton", "ontology", 0.88),
        ("epithelial mesenchymal transition", "ontology", 0.86),
    ],
    "migration": [
        ("cell migration", "synonym", 0.98),
        ("regulation of cell migration", "ontology", 0.94),
        ("cell motility", "synonym", 0.93),
        ("chemotaxis", "ontology", 0.90),
        ("focal adhesion", "ontology", 0.88),
        ("extracellular matrix organization", "ontology", 0.88),
        ("integrin signaling", "ontology", 0.88),
        ("regulation of actin cytoskeleton", "ontology", 0.88),
        ("epithelial mesenchymal transition", "ontology", 0.86),
    ],
    "cell invasion": [
        ("cell invasion", "direct", 1.00),
        ("cell migration", "ontology", 0.93),
        ("cell motility", "ontology", 0.91),
        ("focal adhesion", "ontology", 0.88),
        ("extracellular matrix organization", "ontology", 0.88),
        ("integrin signaling", "ontology", 0.86),
        ("regulation of actin cytoskeleton", "ontology", 0.86),
        ("epithelial mesenchymal transition", "ontology", 0.90),
    ],
    "invasion": [
        ("cell invasion", "synonym", 0.98),
        ("cell migration", "ontology", 0.93),
        ("cell motility", "ontology", 0.91),
        ("focal adhesion", "ontology", 0.88),
        ("extracellular matrix organization", "ontology", 0.88),
        ("integrin signaling", "ontology", 0.86),
        ("regulation of actin cytoskeleton", "ontology", 0.86),
        ("epithelial mesenchymal transition", "ontology", 0.90),
    ],
    "energy metabolism": [
        ("energy metabolism", "direct", 1.00),
        ("metabolic reprogramming", "ontology", 0.84),
        ("glycolysis", "ontology", 0.90),
        ("oxidative phosphorylation", "ontology", 0.90),
        ("cellular respiration", "ontology", 0.88),
        ("mitochondrial function", "ontology", 0.84),
        ("atp metabolism", "ontology", 0.86),
    ],
    "apoptosis": [
        ("apoptosis", "direct", 1.00),
        ("apoptotic process", "synonym", 0.96),
    ],
    "cell proliferation": [
        ("cell proliferation", "direct", 1.00),
        ("cell cycle", "ontology", 0.82),
        ("growth factor signaling", "ontology", 0.78),
        ("survival signaling", "ontology", 0.76),
    ],
    "proliferation": [
        ("cell proliferation", "synonym", 0.98),
        ("cell cycle", "ontology", 0.82),
        ("growth factor signaling", "ontology", 0.78),
        ("survival signaling", "ontology", 0.76),
    ],
    "cell survival": [
        ("cell survival", "direct", 1.00),
        ("survival signaling", "ontology", 0.86),
        ("anti apoptotic", "ontology", 0.82),
        ("pi3k akt signaling", "ontology", 0.80),
    ],
    "survival": [
        ("cell survival", "synonym", 0.98),
        ("survival signaling", "ontology", 0.86),
        ("anti apoptotic", "ontology", 0.82),
        ("pi3k akt signaling", "ontology", 0.80),
    ],
    "angiogenesis": [
        ("angiogenesis", "direct", 1.00),
        ("vascular endothelial growth factor", "ontology", 0.90),
        ("vegf signaling", "ontology", 0.88),
        ("blood vessel development", "ontology", 0.88),
    ],
    "metastasis": [
        ("metastasis", "direct", 1.00),
        ("cell migration", "ontology", 0.88),
        ("cell invasion", "ontology", 0.90),
        ("epithelial mesenchymal transition", "ontology", 0.88),
        ("focal adhesion", "ontology", 0.82),
        ("extracellular matrix organization", "ontology", 0.84),
    ],
}
_PHENOTYPE_PRIMARY_EXCLUSIONS = {
    "cell migration": ("adipogenesis", "xenobiotic metabolism"),
    "migration": ("adipogenesis", "xenobiotic metabolism"),
}
_SEMANTIC_STOPWORDS = {
    "a", "an", "and", "by", "for", "in", "of", "pathway", "process", "signaling",
    "the", "to",
}


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = _WORD_RE.sub(" ", text)
    return " ".join(text.split())


def _clean_scalar(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _normalize_pathway_name(value: Any) -> str:
    normalized = _normalize_text(value)
    normalized = re.sub(
        r"^(?:hallmark|reactome|wikipathways|wp|go biological process)\s+",
        "",
        normalized,
    )
    normalized = re.sub(r"\s+(?:go\s+\d+|wp\s*\d+)$", "", normalized)
    return normalized


def _infer_collection(row: pd.Series, pathway_name: str, pathway_id: str) -> str:
    metadata = " ".join(
        _clean_scalar(row.get(col))
        for col in row.index
        if str(col).strip().lower() in {
            "collection", "collection_name", "category", "source", "database", "library"
        }
    )
    normalized = _normalize_text(f"{metadata} {pathway_name} {pathway_id}")
    if re.search(r"\bgo\b", normalized) and (
        "biological process" in normalized or re.search(r"\bgo\s+\d+", normalized)
    ):
        return "GO Biological Process"
    if "reactome" in normalized:
        return "Reactome"
    if "wikipathway" in normalized or re.search(r"\bwp\s*\d+\b", normalized):
        return "WikiPathways"
    if "hallmark" in normalized:
        return "Hallmark"
    if any(token in normalized for token in ("curated", "kegg", "biocarta", "pid")):
        return "Other curated"
    return "Unknown"


def _split_aliases(value: Any) -> List[str]:
    text = _clean_scalar(value)
    if not text:
        return []
    text = text.strip("[]")
    aliases = re.split(r"[|;,]", text)
    return [
        alias.strip().strip("'\"")
        for alias in aliases
        if alias.strip().strip("'\"")
    ]


def _parquet_fallback_enabled() -> bool:
    return (os.getenv("MIRASSIST_ENABLE_PARQUET_FALLBACK", "") or "").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _resolve_lookup_path(env_name: str, csv_gz_path: Path, csv_path: Path, parquet_path: Path) -> Path:
    configured = os.getenv(env_name)
    if configured:
        path = Path(configured).expanduser().resolve()
        if path.suffix.lower() == ".parquet" and not _parquet_fallback_enabled():
            raise RuntimeError(
                f"{env_name} points to Parquet, but Parquet fallback is disabled. "
                "Set MIRASSIST_ENABLE_PARQUET_FALLBACK=1 to enable it explicitly."
            )
        return path
    for path in (csv_gz_path, csv_path):
        if path.exists():
            return path.resolve()
    if _parquet_fallback_enabled() and parquet_path.exists():
        return parquet_path.resolve()
    raise FileNotFoundError(
        f"No pathway lookup table found. Tried {csv_gz_path}, {csv_path}"
        + (f", {parquet_path} (Parquet fallback enabled)" if _parquet_fallback_enabled() else "")
    )


def _resolve_pathways_path() -> Path:
    return _resolve_lookup_path(
        "MIRASSIST_PATHWAYS_PATH",
        DEFAULT_PATHWAYS_CSV_GZ_PATH,
        DEFAULT_PATHWAYS_CSV_PATH,
        DEFAULT_PATHWAYS_PATH,
    )


def _resolve_gene_to_pathways_path() -> Path:
    return _resolve_lookup_path(
        "MIRASSIST_GENE_TO_PATHWAYS_PATH",
        DEFAULT_GENE_TO_PATHWAYS_CSV_GZ_PATH,
        DEFAULT_GENE_TO_PATHWAYS_CSV_PATH,
        DEFAULT_GENE_TO_PATHWAYS_PATH,
    )


def _load_lookup_table(path: Path, table_name: str) -> pd.DataFrame:
    file_type = "csv.gz" if path.name.lower().endswith(".csv.gz") else path.suffix.lower().lstrip(".")
    if file_type == "parquet":
        if not _parquet_fallback_enabled():
            raise RuntimeError(f"{table_name} resolved to Parquet while Parquet fallback is disabled.")
        df = pd.read_parquet(path)
    elif file_type in {"csv", "csv.gz"}:
        # Gene symbols such as NA are data, not missing-value sentinels.
        df = pd.read_csv(
            path,
            compression="gzip" if file_type == "csv.gz" else None,
            keep_default_na=False,
        )
    else:
        raise ValueError(f"Unsupported {table_name} file type: {path}")
    logger.info(
        "Loaded %s: selected pathway file=%s file_type=%s rows=%d columns=%d status=success",
        table_name, path, file_type, len(df), len(df.columns)
    )
    return df


def load_pathways(force_reload: bool = False) -> pd.DataFrame:
    global _PATHWAYS_CACHE, _PATHWAYS_SOURCE

    path = _resolve_pathways_path()
    source = str(path)
    if not force_reload and _PATHWAYS_CACHE is not None and _PATHWAYS_SOURCE == source:
        return _PATHWAYS_CACHE

    df = _load_lookup_table(path, "pathways")
    _PATHWAYS_CACHE = df
    _PATHWAYS_SOURCE = source
    return df


def load_gene_to_pathways(force_reload: bool = False) -> pd.DataFrame:
    global _GENE_TO_PATHWAYS_CACHE, _GENE_TO_PATHWAYS_SOURCE

    path = _resolve_gene_to_pathways_path()
    source = str(path)
    if (
        not force_reload
        and _GENE_TO_PATHWAYS_CACHE is not None
        and _GENE_TO_PATHWAYS_SOURCE == source
    ):
        return _GENE_TO_PATHWAYS_CACHE

    df = _load_lookup_table(path, "gene_to_pathways")
    _GENE_TO_PATHWAYS_CACHE = df
    _GENE_TO_PATHWAYS_SOURCE = source
    return df


def _pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower_to_actual = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        actual = lower_to_actual.get(candidate.lower())
        if actual is not None:
            return str(actual)
    return None


def _build_pathway_query_candidates(queryspec: Dict[str, Any]) -> List[Dict[str, Any]]:
    pathway_request = queryspec.get("pathway_selection_request") or {}
    phenotype_context = queryspec.get("phenotype_context") or {}
    target_role_inference = queryspec.get("target_role_inference") or {}
    phenotype = _clean_scalar(phenotype_context.get("phenotype"))
    expected_target_role = _clean_scalar(
        target_role_inference.get("expected_target_effect_on_phenotype")
    ).lower()

    explicit_terms = {
        _normalize_text(term)
        for term in (queryspec.get("pathway_keywords") or [])
        if _normalize_text(term)
    }
    directional_terms = {
        _normalize_text(term)
        for term in (
            list(pathway_request.get("directional_query_terms") or [])
            + build_directional_query_terms(phenotype, expected_target_role)
        )
        if _normalize_text(term)
    }

    raw_terms: List[str] = []
    for source in (
        pathway_request.get("query_terms") or [],
        pathway_request.get("directional_query_terms") or [],
        queryspec.get("phenotype_keywords") or [],
        queryspec.get("pathway_keywords") or [],
        [phenotype] if phenotype else [],
    ):
        raw_terms.extend(_clean_scalar(term) for term in source if _clean_scalar(term))

    candidates: List[Dict[str, Any]] = []
    concept_inputs = list(queryspec.get("phenotype_keywords") or [])
    if phenotype:
        concept_inputs.append(phenotype)
    recognized_concepts = {
        _normalize_text(concept)
        for concept in concept_inputs
        if _normalize_text(concept) in _CONTROLLED_PHENOTYPE_TERMS
    }

    for term in raw_terms:
        normalized = _normalize_text(term)
        relation = "ontology" if normalized in directional_terms else "direct"
        source_concept = phenotype or term
        if normalized in recognized_concepts:
            source_concept = term
        candidates.append(
            {
                "term": term,
                "normalized": normalized,
                "relation": relation,
                "strength": 0.96 if relation == "ontology" else 1.0,
                "source_concept": source_concept,
                "explicit": normalized in explicit_terms or not recognized_concepts,
            }
        )

    for concept in concept_inputs:
        concept_key = _normalize_text(concept)
        for term, relation, strength in _CONTROLLED_PHENOTYPE_TERMS.get(concept_key, []):
            candidates.append(
                {
                    "term": term,
                    "normalized": _normalize_text(term),
                    "relation": relation,
                    "strength": strength,
                    "source_concept": _clean_scalar(concept),
                    "explicit": False,
                }
            )

    observed_change = _clean_scalar(phenotype_context.get("observed_change")).lower()
    if _normalize_text(phenotype) == "apoptosis" and observed_change == "associated":
        candidates.append(
            {
                "term": "apoptotic process",
                "normalized": "apoptotic process",
                "relation": "synonym",
                "strength": 0.96,
                "source_concept": phenotype,
                "explicit": False,
            }
        )

    best_by_term: Dict[str, Dict[str, Any]] = {}
    ordered: List[str] = []
    for candidate in candidates:
        normalized = candidate["normalized"]
        if not normalized:
            continue
        if normalized not in best_by_term:
            ordered.append(normalized)
            best_by_term[normalized] = candidate
            continue
        current = best_by_term[normalized]
        if (
            bool(candidate["explicit"]),
            float(candidate["strength"]),
            candidate["relation"] == "direct",
        ) > (
            bool(current["explicit"]),
            float(current["strength"]),
            current["relation"] == "direct",
        ):
            best_by_term[normalized] = candidate
    return [best_by_term[normalized] for normalized in ordered]


def _build_pathway_query_terms(queryspec: Dict[str, Any]) -> List[str]:
    return [candidate["term"] for candidate in _build_pathway_query_candidates(queryspec)]


def _pathway_context_enabled(queryspec: Dict[str, Any]) -> bool:
    pathway_request = queryspec.get("pathway_selection_request") or {}
    pathway_filter = queryspec.get("pathway_filter") or {}
    return bool(
        pathway_request.get("enabled")
        or pathway_filter.get("enabled")
        or (queryspec.get("phenotype_keywords") or [])
        or (queryspec.get("pathway_keywords") or [])
    )



def _minimum_relevance_score(queryspec: Dict[str, Any]) -> float:
    configured = (queryspec.get("pathway_filter") or {}).get(
        "min_relevance_score", _DEFAULT_MIN_RELEVANCE_SCORE
    )
    try:
        score = float(configured)
    except (TypeError, ValueError):
        score = _DEFAULT_MIN_RELEVANCE_SCORE
    return max(0.0, min(1.0, score))


def _semantic_overlap_score(term: str, pathway_text: str) -> float:
    term_tokens = {
        token for token in term.split() if token not in _SEMANTIC_STOPWORDS
    }
    pathway_tokens = {
        token for token in pathway_text.split() if token not in _SEMANTIC_STOPWORDS
    }
    overlap = term_tokens & pathway_tokens
    if len(overlap) < 2:
        return 0.0
    coverage = len(overlap) / max(1, len(term_tokens))
    precision = len(overlap) / max(1, len(pathway_tokens))
    if coverage < 0.66:
        return 0.0
    return min(0.71, 0.56 + 0.10 * coverage + 0.05 * precision)


def _is_controlled_name_phrase(term: str, pathway_name: str) -> bool:
    if not term or term not in pathway_name:
        return False
    remainder = pathway_name.replace(term, " ", 1)
    extra_tokens = set(remainder.split())
    allowed_modifiers = {
        "assembly", "cell", "disassembly", "negative", "of", "organization",
        "positive", "regulation",
    }
    return len(extra_tokens) <= 3 and extra_tokens.issubset(allowed_modifiers)


def _score_pathway_row(
    row: pd.Series,
    *,
    pathway_id_col: Optional[str],
    pathway_name_col: str,
    description_cols: List[str],
    alias_cols: List[str],
    candidates: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    pathway_name = _clean_scalar(row.get(pathway_name_col))
    pathway_id = _clean_scalar(row.get(pathway_id_col)) if pathway_id_col else pathway_name
    pathway_id = pathway_id or pathway_name
    full_name = _normalize_text(pathway_name)
    core_name = _normalize_pathway_name(pathway_name)
    normalized_id = _normalize_text(pathway_id)
    descriptions = [
        _clean_scalar(row.get(col)) for col in description_cols if _clean_scalar(row.get(col))
    ]
    description = _normalize_text(" | ".join(descriptions))
    aliases = [
        alias
        for col in alias_cols
        for alias in _split_aliases(row.get(col))
    ]
    normalized_aliases = [_normalize_text(alias) for alias in aliases]
    collection = _infer_collection(row, pathway_name, pathway_id)
    collection_bonus = _COLLECTION_SCORE_BONUS[collection]

    best: Optional[Dict[str, Any]] = None
    matched_terms: List[str] = []
    for candidate in candidates:
        term = candidate["normalized"]
        if not term:
            continue
        relation = candidate["relation"]
        strength = float(candidate["strength"])
        match_type = ""
        base_score = 0.0
        matched_field = ""

        exact_name = term in {full_name, core_name, normalized_id}
        exact_alias = term in normalized_aliases
        name_phrase = bool(term and (term in full_name or term in core_name))
        controlled_name_phrase = _is_controlled_name_phrase(term, core_name)
        description_phrase = bool(term and term in description)

        if relation == "direct" and (exact_name or exact_alias):
            match_type = "exact_normalized_name_match"
            base_score = 1.0 if exact_name else 0.98
            matched_field = "name or identifier" if exact_name else "alias"
        elif relation == "direct" and description_phrase:
            match_type = "exact_description_phrase_match"
            base_score = 0.93
            matched_field = "description"
        elif relation in {"synonym", "ontology"} and (
            exact_name or exact_alias or controlled_name_phrase or description_phrase
        ):
            match_type = (
                "controlled_synonym" if relation == "synonym"
                else "controlled_ontology_relation"
            )
            if exact_name or exact_alias:
                base_score = strength
                matched_field = "name or controlled alias"
            elif controlled_name_phrase:
                base_score = strength * 0.94
                matched_field = "name phrase or controlled alias"
            else:
                base_score = strength * 0.90
                matched_field = "description"
        elif relation == "direct" and name_phrase:
            match_type = "broader_semantic_association"
            base_score = 0.66
            matched_field = "broader name phrase"
        else:
            semantic_score = _semantic_overlap_score(
                term, " ".join([core_name, description] + normalized_aliases)
            )
            if semantic_score:
                match_type = "broader_semantic_association"
                base_score = semantic_score * strength
                matched_field = "token-level semantic overlap"

        if not match_type:
            continue
        matched_terms.append(candidate["term"])
        relevance_score = min(1.0, base_score + collection_bonus)
        penalty_reasons: List[str] = []
        source_key = _normalize_text(candidate["source_concept"])
        exclusions = _PHENOTYPE_PRIMARY_EXCLUSIONS.get(source_key, ())
        if exclusions and any(exclusion in core_name for exclusion in exclusions):
            explicitly_named = bool(candidate["explicit"] and (exact_name or exact_alias))
            if not explicitly_named:
                relevance_score -= 0.75
                penalty_reasons.append(
                    "primary pathway definition is unrelated to the requested phenotype"
                )
        if match_type == "broader_semantic_association" and matched_field == "token-level semantic overlap":
            relevance_score -= 0.08
            penalty_reasons.append("weak token-only association")
        relevance_score = max(0.0, relevance_score)

        rationale = (
            f"{match_type.replace('_', ' ')} for '{candidate['term']}' in the "
            f"pathway {matched_field}; {collection} priority applied"
        )
        if penalty_reasons:
            rationale += "; penalized because " + " and ".join(penalty_reasons)
        result = {
            "pathway_id": pathway_id,
            "pathway_name": pathway_name,
            "match_type": match_type,
            "matched_term": candidate["term"],
            "source_concept": candidate["source_concept"],
            "relevance_score": round(relevance_score, 3),
            "collection": collection,
            "rationale": rationale,
            "_match_priority": _MATCH_PRIORITY[match_type],
            "_collection_priority": _COLLECTION_PRIORITY[collection],
        }
        if best is None or (
            result["_match_priority"],
            result["relevance_score"],
            result["_collection_priority"],
        ) > (
            best["_match_priority"],
            best["relevance_score"],
            best["_collection_priority"],
        ):
            best = result

    if best is None:
        return None
    best["matched_terms"] = list(dict.fromkeys(matched_terms))
    return best


def compact_pathway_selection(selection: Dict[str, Any], include_internal: bool = False) -> Dict[str, Any]:
    compact: Dict[str, Any] = {
        "enabled": bool(selection.get("enabled")),
        "mode": selection.get("mode", "none"),
        "phenotype": selection.get("phenotype"),
        "direction": selection.get("direction"),
        "observed_change": selection.get("observed_change"),
        "miRNA_perturbation": selection.get("miRNA_perturbation"),
        "expected_target_effect_on_phenotype": selection.get("expected_target_effect_on_phenotype"),
        "query_terms": list(selection.get("query_terms") or []),
        "minimum_relevance_score": float(
            selection.get("minimum_relevance_score", _DEFAULT_MIN_RELEVANCE_SCORE)
        ),
        "available_collections": list(selection.get("available_collections") or []),
        "selected_pathways": list(selection.get("selected_pathways") or []),
        "n_selected_pathways": int(selection.get("n_selected_pathways") or 0),
        "n_selected_genes": int(selection.get("n_selected_genes") or 0),
        "selected_gene_examples": list(selection.get("selected_gene_examples") or []),
        "warnings": list(selection.get("warnings") or []),
    }
    if include_internal:
        if selection.get("selected_genes") is not None:
            compact["selected_genes"] = list(selection.get("selected_genes") or [])
        if selection.get("selected_gene_pathways") is not None:
            compact["selected_gene_pathways"] = dict(selection.get("selected_gene_pathways") or {})
        if selection.get("_selected_gene_set") is not None:
            compact["_selected_gene_set"] = set(selection.get("_selected_gene_set") or set())
        if selection.get("_selected_gene_pathways") is not None:
            compact["_selected_gene_pathways"] = dict(selection.get("_selected_gene_pathways") or {})
    return compact


def resolve_pathway_selection(queryspec: Dict[str, Any]) -> Dict[str, Any]:
    phenotype_context = queryspec.get("phenotype_context") or {}
    target_role_inference = queryspec.get("target_role_inference") or {}
    enabled = _pathway_context_enabled(queryspec)

    selection: Dict[str, Any] = {
        "enabled": enabled,
        "mode": "filter" if enabled else "none",
        "phenotype": phenotype_context.get("phenotype"),
        "direction": phenotype_context.get("direction"),
        "observed_change": phenotype_context.get("observed_change"),
        "miRNA_perturbation": phenotype_context.get("miRNA_perturbation"),
        "expected_target_effect_on_phenotype": target_role_inference.get(
            "expected_target_effect_on_phenotype"
        ),
        "query_terms": [],
        "minimum_relevance_score": _minimum_relevance_score(queryspec),
        "available_collections": [],
        "selected_pathways": [],
        "n_selected_pathways": 0,
        "n_selected_genes": 0,
        "selected_gene_examples": [],
        "warnings": [],
        "_selected_gene_set": set(),
        "_selected_gene_pathways": {},
    }

    if not enabled:
        return selection

    query_candidates = _build_pathway_query_candidates(queryspec)
    query_terms = [candidate["term"] for candidate in query_candidates]
    selection["query_terms"] = query_terms

    try:
        pathways_df = load_pathways()
        gene_to_pathways_df = load_gene_to_pathways()
    except Exception as exc:
        selection["warnings"].append(
            f"Pathway filtering requested but pathway files could not be loaded: {exc}"
        )
        return selection

    pathway_id_col = _pick_column(pathways_df, ["pathway_id", "id", "term_id", "geneset_id"])
    pathway_name_col = _pick_column(pathways_df, ["pathway_name", "name", "term_name", "geneset_name"])
    description_cols = [
        str(col) for col in pathways_df.columns
        if str(col).strip().lower() in {"description", "pathway_desc", "definition", "summary"}
    ]
    alias_cols = [
        str(col) for col in pathways_df.columns
        if str(col).strip().lower() in {"alias", "aliases", "synonym", "synonyms"}
    ]

    if pathway_name_col is None:
        selection["warnings"].append("Could not find a pathway name column in pathways data.")
        return selection

    available_collections = {
        _infer_collection(
            row,
            _clean_scalar(row.get(pathway_name_col)),
            _clean_scalar(row.get(pathway_id_col)) if pathway_id_col else "",
        )
        for _, row in pathways_df.iterrows()
    }
    available_collections.discard("Unknown")
    selection["available_collections"] = sorted(
        available_collections,
        key=lambda collection: (-_COLLECTION_PRIORITY[collection], collection),
    )
    if available_collections == {"Hallmark"}:
        selection["warnings"].append(
            "The pathway database contains only Hallmark sets; specific phenotype filtering "
            "requires GO Biological Process and curated pathway collections."
        )

    minimum_score = selection["minimum_relevance_score"]
    scored_matches: List[Dict[str, Any]] = []
    for _, row in pathways_df.iterrows():
        match = _score_pathway_row(
            row,
            pathway_id_col=pathway_id_col,
            pathway_name_col=pathway_name_col,
            description_cols=description_cols,
            alias_cols=alias_cols,
            candidates=query_candidates,
        )
        if match is not None and match["relevance_score"] >= minimum_score:
            scored_matches.append(match)

    scored_matches.sort(
        key=lambda item: (
            -item["_match_priority"],
            -item["relevance_score"],
            -item["_collection_priority"],
            item["pathway_name"].lower(),
            item["pathway_id"].lower(),
        )
    )

    selected_pathways: List[Dict[str, Any]] = []
    seen_pairs = set()
    for item in scored_matches:
        key = (item["pathway_id"], item["pathway_name"])
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        selected_pathways.append(
            {
                key: value
                for key, value in item.items()
                if not key.startswith("_")
            }
        )

    selection["selected_pathways"] = selected_pathways
    selection["n_selected_pathways"] = len(selected_pathways)

    if not selected_pathways:
        selection["warnings"].append(
            "No pathways met the minimum biological relevance threshold for the requested "
            "phenotype/pathway terms."
        )
        return selection

    gene_pathway_id_col = _pick_column(gene_to_pathways_df, ["pathway_id", "id", "term_id", "geneset_id"])
    gene_pathway_name_col = _pick_column(gene_to_pathways_df, ["pathway_name", "name", "term_name", "geneset_name"])
    gene_symbol_col = _pick_column(gene_to_pathways_df, ["gene_symbol", "gene", "symbol"])

    if gene_symbol_col is None:
        selection["warnings"].append("Could not find a gene symbol column in gene_to_pathways data.")
        return selection

    pathway_ids = {item["pathway_id"] for item in selected_pathways}
    pathway_names = {item["pathway_name"] for item in selected_pathways}

    if gene_pathway_id_col is not None:
        match_mask = gene_to_pathways_df[gene_pathway_id_col].astype(str).isin(pathway_ids)
    elif gene_pathway_name_col is not None:
        match_mask = gene_to_pathways_df[gene_pathway_name_col].astype(str).isin(pathway_names)
    else:
        selection["warnings"].append("Could not find pathway identifier columns in gene_to_pathways data.")
        return selection

    matched_gene_rows = gene_to_pathways_df.loc[match_mask].copy()
    if matched_gene_rows.empty:
        selection["warnings"].append("Matching pathways were found, but no genes were linked to them in gene_to_pathways.")
        return selection

    gene_pathway_map: Dict[str, List[str]] = {}
    for _, row in matched_gene_rows.iterrows():
        gene_symbol = str(row.get(gene_symbol_col) or "").strip().upper()
        if not gene_symbol:
            continue
        pathway_name = ""
        if gene_pathway_name_col is not None:
            pathway_name = str(row.get(gene_pathway_name_col) or "").strip()
        elif gene_pathway_id_col is not None:
            pathway_name = str(row.get(gene_pathway_id_col) or "").strip()
        if not pathway_name:
            pathway_name = "UNKNOWN_PATHWAY"
        gene_pathway_map.setdefault(gene_symbol, [])
        if pathway_name not in gene_pathway_map[gene_symbol]:
            gene_pathway_map[gene_symbol].append(pathway_name)

    selected_genes = sorted(gene_pathway_map.keys())
    selection["n_selected_genes"] = len(selected_genes)
    selection["selected_gene_examples"] = selected_genes[:20]
    selection["_selected_gene_set"] = set(selected_genes)
    selection["_selected_gene_pathways"] = gene_pathway_map
    return selection
