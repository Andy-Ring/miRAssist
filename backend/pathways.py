from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from backend.config import ROOT_DIR


PATHWAYS_DIR = ROOT_DIR / "data" / "processed" / "pathways"
DEFAULT_PATHWAYS_PATH = PATHWAYS_DIR / "pathways.parquet"
DEFAULT_GENE_TO_PATHWAYS_PATH = PATHWAYS_DIR / "gene_to_pathways.parquet"

_PATHWAYS_CACHE: Optional[pd.DataFrame] = None
_PATHWAYS_SOURCE: Optional[str] = None
_GENE_TO_PATHWAYS_CACHE: Optional[pd.DataFrame] = None
_GENE_TO_PATHWAYS_SOURCE: Optional[str] = None

_WORD_RE = re.compile(r"[^a-z0-9]+")


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = _WORD_RE.sub(" ", text)
    return " ".join(text.split())


def _resolve_pathways_path() -> Path:
    return Path(
        os.getenv("MIRASSIST_PATHWAYS_PATH", str(DEFAULT_PATHWAYS_PATH))
    ).expanduser().resolve()


def _resolve_gene_to_pathways_path() -> Path:
    return Path(
        os.getenv("MIRASSIST_GENE_TO_PATHWAYS_PATH", str(DEFAULT_GENE_TO_PATHWAYS_PATH))
    ).expanduser().resolve()


def load_pathways(force_reload: bool = False) -> pd.DataFrame:
    global _PATHWAYS_CACHE, _PATHWAYS_SOURCE

    path = _resolve_pathways_path()
    source = str(path)
    if not force_reload and _PATHWAYS_CACHE is not None and _PATHWAYS_SOURCE == source:
        return _PATHWAYS_CACHE

    df = pd.read_parquet(path)
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

    df = pd.read_parquet(path)
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


def _build_pathway_query_terms(queryspec: Dict[str, Any]) -> List[str]:
    pathway_request = queryspec.get("pathway_selection_request") or {}
    phenotype_context = queryspec.get("phenotype_context") or {}

    terms: List[str] = []
    for source in (
        pathway_request.get("query_terms") or [],
        pathway_request.get("directional_query_terms") or [],
        queryspec.get("phenotype_keywords") or [],
        queryspec.get("pathway_keywords") or [],
    ):
        for term in source:
            text = str(term or "").strip()
            if text:
                terms.append(text)

    phenotype = str(phenotype_context.get("phenotype") or "").strip()
    direction = str(phenotype_context.get("direction") or "").strip().lower()
    raw_phrase = str(phenotype_context.get("raw_phrase") or "").strip()

    if raw_phrase:
        terms.append(raw_phrase)
    if phenotype:
        terms.append(phenotype)

    if phenotype.lower() == "apoptosis":
        if direction in {"promotes", "increases"}:
            terms.extend(
                [
                    "positive regulation of apoptosis",
                    "positive regulation of apoptotic process",
                    "activation of apoptotic signaling pathway",
                    "apoptotic process",
                ]
            )
        elif direction in {"suppresses", "decreases"}:
            terms.extend(
                [
                    "negative regulation of apoptosis",
                    "negative regulation of apoptotic process",
                    "anti apoptotic",
                    "regulation of apoptotic signaling pathway",
                ]
            )

    seen = set()
    ordered_terms: List[str] = []
    for term in terms:
        normalized = _normalize_text(term)
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered_terms.append(term.strip())
    return ordered_terms


def _pathway_context_enabled(queryspec: Dict[str, Any]) -> bool:
    pathway_request = queryspec.get("pathway_selection_request") or {}
    pathway_filter = queryspec.get("pathway_filter") or {}
    return bool(
        pathway_request.get("enabled")
        or pathway_filter.get("enabled")
        or (queryspec.get("phenotype_keywords") or [])
        or (queryspec.get("pathway_keywords") or [])
    )


def resolve_pathway_selection(queryspec: Dict[str, Any]) -> Dict[str, Any]:
    phenotype_context = queryspec.get("phenotype_context") or {}
    enabled = _pathway_context_enabled(queryspec)

    selection: Dict[str, Any] = {
        "enabled": enabled,
        "mode": "filter" if enabled else "none",
        "phenotype": phenotype_context.get("phenotype"),
        "direction": phenotype_context.get("direction"),
        "query_terms": [],
        "selected_pathways": [],
        "n_selected_pathways": 0,
        "selected_genes": [],
        "n_selected_genes": 0,
        "warnings": [],
    }

    if not enabled:
        return selection

    query_terms = _build_pathway_query_terms(queryspec)
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
    description_col = _pick_column(pathways_df, ["description"])
    source_col = _pick_column(pathways_df, ["source"])
    category_col = _pick_column(pathways_df, ["category"])

    if pathway_name_col is None:
        selection["warnings"].append("Could not find a pathway name column in pathways data.")
        return selection

    scored_matches: List[Dict[str, Any]] = []
    for _, row in pathways_df.iterrows():
        searchable_parts = [
            row.get(pathway_name_col),
            row.get(description_col) if description_col else None,
            row.get(source_col) if source_col else None,
            row.get(category_col) if category_col else None,
            row.get(pathway_id_col) if pathway_id_col else None,
        ]
        searchable_text = " | ".join(str(part or "") for part in searchable_parts)
        searchable_normalized = _normalize_text(searchable_text)

        matched_terms = []
        score = 0
        for term in query_terms:
            term_normalized = _normalize_text(term)
            if term_normalized and term_normalized in searchable_normalized:
                matched_terms.append(term)
                score += max(1, len(term_normalized.split()))

        if matched_terms:
            scored_matches.append(
                {
                    "pathway_id": str(row.get(pathway_id_col) or row.get(pathway_name_col)),
                    "pathway_name": str(row.get(pathway_name_col) or ""),
                    "matched_terms": matched_terms,
                    "_score": score,
                }
            )

    scored_matches.sort(
        key=lambda item: (-item["_score"], item["pathway_name"].lower(), item["pathway_id"].lower())
    )

    selected_pathways = []
    seen_pairs = set()
    for item in scored_matches:
        key = (item["pathway_id"], item["pathway_name"])
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        selected_pathways.append(
            {
                "pathway_id": item["pathway_id"],
                "pathway_name": item["pathway_name"],
                "matched_terms": item["matched_terms"],
            }
        )

    selection["selected_pathways"] = selected_pathways
    selection["n_selected_pathways"] = len(selected_pathways)

    if not selected_pathways:
        selection["warnings"].append("No matching pathways were found for the requested phenotype/pathway terms.")
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

    selection["selected_genes"] = sorted(gene_pathway_map.keys())
    selection["n_selected_genes"] = len(selection["selected_genes"])
    selection["selected_gene_pathways"] = gene_pathway_map
    return selection
