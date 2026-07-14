"""
Memory-safe evidence fetch from a local parquet snapshot.

Reads only the rows for the queried miRNA (or gene) from the cached evidence
snapshot using pyarrow predicate pushdown, so the full table is never loaded
into memory. This mirrors backend.retrieval._fetch_postgres_candidate_pool but
against the GitHub-hosted parquet snapshot instead of Supabase.

All scoring/ranking still happens in pandas inside retrieve_candidates; this
module only produces the bounded candidate pool. Percentile columns are expected
to be precomputed in the snapshot (as they are in the Supabase table), so no
full-table feature-percentile annotation is required.
"""
from __future__ import annotations

from functools import reduce
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pds

from backend.config import get_db_candidate_limit, get_default_mirna_arm
from backend.evidence_bootstrap import ensure_evidence_parquet

_DATASET_CACHE: Dict[str, "pds.Dataset"] = {}


def _dataset(path: str) -> "pds.Dataset":
    if path not in _DATASET_CACHE:
        if path.split("?", 1)[0].lower().endswith(".csv"):
            from pyarrow.csv import ParseOptions

            fmt = pds.CsvFileFormat(parse_options=ParseOptions(newlines_in_values=True))
        else:
            fmt = "parquet"
        _DATASET_CACHE[path] = pds.dataset(path, format=fmt)
    return _DATASET_CACHE[path]


def _isin_lower(col: str, values: List[str]):
    lowered = [str(v).lower() for v in values if str(v or "").strip()]
    return pc.is_in(pc.utf8_lower(pc.field(col)), value_set=pa.array(lowered, pa.string()))


def _eq_upper(col: str, value: str):
    return pc.equal(pc.utf8_upper(pc.field(col)), pa.scalar(str(value).upper()))


def _starts_with_lower(col: str, prefix: str):
    return pc.starts_with(pc.utf8_lower(pc.field(col)), pa.scalar(str(prefix).lower()))


def _base_filters(cfg: "Any", available: set) -> Optional["pc.Expression"]:
    exprs: List["pc.Expression"] = []
    if cfg.novel and cfg.use_mirtarbase_evidence:
        for col in ("mirtarbase_pos", "label_mirtarbase"):
            if col in available:
                exprs.append(pc.is_null(pc.field(col)) | pc.equal(pc.field(col), 0))
    if cfg.require_binding_evidence:
        bind = [c for c in ("support_targetscan", "support_encori", "support_mirdb") if c in available]
        if bind:
            exprs.append(reduce(lambda a, b: a | b, [pc.equal(pc.field(c), 1) for c in bind]))
    if cfg.require_expression and cfg.tcga:
        pair_col = f"{str(cfg.tcga).upper()}_pair_expressed"
        if pair_col in available:
            exprs.append(pc.equal(pc.field(pair_col), 1))
    pathway_selection = cfg.pathway_selection or {}
    if bool(pathway_selection.get("enabled")) and cfg.pathway_gene_set:
        gene_norm_col = "gene_symbol_norm" if "gene_symbol_norm" in available else (
            "gene_symbol_normalized" if "gene_symbol_normalized" in available else None
        )
        if gene_norm_col:
            genes = sorted({str(g).upper() for g in cfg.pathway_gene_set if str(g or "").strip()})
            if genes:
                exprs.append(pc.is_in(pc.utf8_upper(pc.field(gene_norm_col)), value_set=pa.array(genes, pa.string())))
    if not exprs:
        return None
    return reduce(lambda a, b: a & b, exprs)


def fetch_parquet_candidate_pool(query_token: str, cfg: "Any") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    from backend.retrieval import (
        _direction_from_token,
        _first_available_column,
        _mirna_prefix_fallback_patterns,
        _normalize_gene_symbol,
        _normalize_mirna_query,
        _normalize_token,
        _select_production_evidence_columns,
        expand_mirna_query_variants,
    )

    path = ensure_evidence_parquet()
    dataset = _dataset(path)
    available = {str(c) for c in dataset.schema.names}
    selected = _select_production_evidence_columns(available, cfg.tcga)

    direction = _direction_from_token(_normalize_token(query_token))
    mirna_norm_col = _first_available_column(available, ("mirna_name_norm", "mirna_name_normalized"))
    gene_norm_col = _first_available_column(available, ("gene_symbol_norm", "gene_symbol_normalized"))

    required = {"mirna_name", "gene_symbol", "support_count"}
    if direction == "mirna_to_targets" and mirna_norm_col is None:
        required.add("mirna_name_norm")
    if direction == "gene_to_mirnas" and gene_norm_col is None:
        required.add("gene_symbol_norm")
    missing = sorted(c for c in required if c not in selected)
    if missing:
        raise RuntimeError(
            "Evidence snapshot is missing required columns: " + ", ".join(missing)
        )

    base = _base_filters(cfg, available)
    default_arm = get_default_mirna_arm()
    expansion = (
        expand_mirna_query_variants(query_token, default_arm=default_arm)
        if direction == "mirna_to_targets"
        else {}
    )
    primary = list(expansion.get("primary_variants") or [])
    fallback = list(expansion.get("fallback_variants") or [])
    prefixes = _mirna_prefix_fallback_patterns(query_token) if direction == "mirna_to_targets" else []

    diagnostics: Dict[str, Any] = {
        "evidence_backend": "snapshot",
        "evidence_source": path,
        "supabase_table_name": None,
        "snapshot_path": path,
        "db_candidate_limit": int(get_db_candidate_limit()),
        "query_direction": direction,
        "raw_mirna_query": str(query_token or "") if direction == "mirna_to_targets" else None,
        "normalized_mirna_query": expansion.get("normalized_input") if direction == "mirna_to_targets" else None,
        "query_mirna_normalized": _normalize_mirna_query(query_token)[0] if direction == "mirna_to_targets" else None,
        "primary_mirna_variants": list(primary),
        "fallback_mirna_variants": list(fallback),
        "variants_used": [],
        "arm_interpretation_note": expansion.get("note") if direction == "mirna_to_targets" else None,
    }

    def _run(entity_expr) -> pd.DataFrame:
        expr = entity_expr if base is None else (entity_expr & base)
        table = dataset.to_table(columns=selected, filter=expr)
        limit = int(get_db_candidate_limit())
        if table.num_rows > limit:
            table = table.slice(0, limit)
        return table.to_pandas()

    if direction == "mirna_to_targets":
        df = _run(_isin_lower(str(mirna_norm_col), primary)) if primary else pd.DataFrame()
        if not df.empty:
            diagnostics["variants_used"] = list(primary)
        if df.empty and fallback:
            df = _run(_isin_lower(str(mirna_norm_col), fallback))
            if not df.empty:
                diagnostics["variants_used"] = list(fallback)
        if df.empty and prefixes:
            stripped = [p[:-1] if p.endswith("%") else p for p in prefixes]
            expr = reduce(lambda a, b: a | b, [_starts_with_lower(str(mirna_norm_col), p) for p in stripped])
            df = _run(expr)
            if not df.empty:
                diagnostics["variants_used"] = list(prefixes)
    else:
        df = _run(_eq_upper(str(gene_norm_col), _normalize_gene_symbol(query_token)))

    if not df.empty and df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    diagnostics["n_rows_fetched_from_db"] = int(len(df))
    diagnostics["sql_selected_column_count"] = int(len(selected))
    diagnostics["sql_returned_column_count"] = int(len(df.columns))
    return df, diagnostics
