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

from backend.runtime_diagnostics import trace
from backend.config import get_db_candidate_limit, get_default_mirna_arm
from backend.evidence_bootstrap import ensure_evidence_parquet

_DATASET_CACHE: Dict[str, Any] = {}
_PANDAS_SNAPSHOT_CACHE: Dict[str, pd.DataFrame] = {}


def _is_csv_snapshot(path: str) -> bool:
    lower = str(path).split("?", 1)[0].lower()
    return lower.endswith(".csv") or lower.endswith(".csv.gz")


def _snapshot_reader() -> str:
    import os

    return (os.getenv("MIRASSIST_SNAPSHOT_READER", "pandas") or "pandas").strip().lower()


def _dataset(path: str) -> Any:
    import pyarrow.dataset as pds

    trace(f"before pyarrow dataset open path={path}")
    if path not in _DATASET_CACHE:
        if _is_csv_snapshot(path):
            from pyarrow.csv import ParseOptions

            fmt = pds.CsvFileFormat(parse_options=ParseOptions(newlines_in_values=True))
        else:
            fmt = "parquet"
        _DATASET_CACHE[path] = pds.dataset(path, format=fmt)
        trace(f"after pyarrow dataset open path={path} schema_cols={len(_DATASET_CACHE[path].schema.names)}")
    return _DATASET_CACHE[path]


def _isin_lower(col: str, values: List[str]) -> Any:
    import pyarrow as pa
    import pyarrow.compute as pc

    lowered = [str(v).lower() for v in values if str(v or "").strip()]
    return pc.is_in(pc.utf8_lower(pc.field(col)), value_set=pa.array(lowered, pa.string()))


def _eq_upper(col: str, value: str) -> Any:
    import pyarrow as pa
    import pyarrow.compute as pc

    return pc.equal(pc.utf8_upper(pc.field(col)), pa.scalar(str(value).upper()))


def _starts_with_lower(col: str, prefix: str) -> Any:
    import pyarrow as pa
    import pyarrow.compute as pc

    return pc.starts_with(pc.utf8_lower(pc.field(col)), pa.scalar(str(prefix).lower()))


def _base_filters(cfg: "Any", available: set) -> Optional[Any]:
    import pyarrow as pa
    import pyarrow.compute as pc
    from backend.retrieval import MIRTARBASE_KNOWN_POSITIVE_COLUMNS

    exprs: List[Any] = []
    if cfg.novel and cfg.use_mirtarbase_evidence:
        for col in MIRTARBASE_KNOWN_POSITIVE_COLUMNS:
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


def _read_snapshot_columns(path: str) -> List[str]:
    if _is_csv_snapshot(path):
        print("[miRAssist] snapshot reading CSV header", flush=True)
        return list(pd.read_csv(path, nrows=0).columns)

    trace("before pyarrow parquet metadata read")
    print("[miRAssist] snapshot reading parquet metadata", flush=True)
    import pyarrow.parquet as pq

    columns = [str(name) for name in pq.ParquetFile(path).schema_arrow.names]
    print(f"[miRAssist] snapshot parquet metadata read: columns={len(columns)}", flush=True)
    trace(f"after pyarrow parquet metadata read columns={len(columns)}")
    return columns


def _read_snapshot_frame(path: str, columns: List[str]) -> pd.DataFrame:
    cache_key = f"{path}||{'|'.join(columns)}"
    if cache_key in _PANDAS_SNAPSHOT_CACHE:
        return _PANDAS_SNAPSHOT_CACHE[cache_key].copy()

    if _is_csv_snapshot(path):
        print(f"[miRAssist] snapshot pandas read_csv starting: columns={len(columns)}", flush=True)
        df = pd.read_csv(path, usecols=columns)
        print(f"[miRAssist] snapshot pandas read_csv complete: rows={len(df)}", flush=True)
    else:
        trace(f"before pandas read_parquet columns={len(columns)}")
        print(f"[miRAssist] snapshot pandas read_parquet starting: columns={len(columns)}", flush=True)
        df = pd.read_parquet(path, columns=columns)
        print(f"[miRAssist] snapshot pandas read_parquet complete: rows={len(df)}", flush=True)
        trace(f"after pandas read_parquet rows={len(df)} cols={len(df.columns)}")
    _PANDAS_SNAPSHOT_CACHE[cache_key] = df
    return df.copy()


def _filter_snapshot_frame(
    df: pd.DataFrame,
    cfg: "Any",
    available: set,
    pathway_gene_set: set,
) -> pd.Series:
    from backend.retrieval import MIRTARBASE_KNOWN_POSITIVE_COLUMNS

    mask = pd.Series(True, index=df.index)
    if cfg.novel and cfg.use_mirtarbase_evidence:
        for col in MIRTARBASE_KNOWN_POSITIVE_COLUMNS:
            if col in df.columns:
                values = pd.to_numeric(df[col], errors="coerce")
                mask &= values.isna() | values.eq(0)
    if cfg.require_binding_evidence:
        bind = [c for c in ("support_targetscan", "support_encori", "support_mirdb") if c in df.columns]
        if bind:
            bind_mask = pd.Series(False, index=df.index)
            for col in bind:
                bind_mask |= pd.to_numeric(df[col], errors="coerce").fillna(0).eq(1)
            mask &= bind_mask
    if cfg.require_expression and cfg.tcga:
        pair_col = f"{str(cfg.tcga).upper()}_pair_expressed"
        if pair_col in df.columns:
            mask &= pd.to_numeric(df[pair_col], errors="coerce").fillna(0).eq(1)
    pathway_selection = cfg.pathway_selection or {}
    if bool(pathway_selection.get("enabled")) and pathway_gene_set:
        gene_norm_col = "gene_symbol_norm" if "gene_symbol_norm" in available else (
            "gene_symbol_normalized" if "gene_symbol_normalized" in available else None
        )
        if gene_norm_col and gene_norm_col in df.columns:
            mask &= df[gene_norm_col].astype(str).str.upper().isin(pathway_gene_set)
    return mask


def _fetch_pandas_candidate_pool(
    *,
    path: str,
    query_token: str,
    cfg: "Any",
    available: set,
    selected: List[str],
    direction: str,
    mirna_norm_col: Optional[str],
    gene_norm_col: Optional[str],
    expansion: Dict[str, Any],
    primary: List[str],
    fallback: List[str],
    prefixes: List[str],
    normalized_gene: str,
    diagnostics: Dict[str, Any],
) -> pd.DataFrame:
    print("[miRAssist] snapshot reader using pandas", flush=True)
    df_all = _read_snapshot_frame(path, selected)
    base_mask = _filter_snapshot_frame(
        df_all,
        cfg,
        available,
        {str(g).upper() for g in cfg.pathway_gene_set if str(g or "").strip()},
    )
    limit = int(get_db_candidate_limit())

    def _limit(df: pd.DataFrame) -> pd.DataFrame:
        return df.head(limit).copy()

    if direction == "mirna_to_targets":
        col = str(mirna_norm_col)
        lowered = df_all[col].astype(str).str.lower()
        if primary:
            primary_values = {str(v).lower() for v in primary if str(v or "").strip()}
            df = _limit(df_all[base_mask & lowered.isin(primary_values)])
            if not df.empty:
                diagnostics["variants_used"] = list(primary)
                return df
        if fallback:
            fallback_values = {str(v).lower() for v in fallback if str(v or "").strip()}
            df = _limit(df_all[base_mask & lowered.isin(fallback_values)])
            if not df.empty:
                diagnostics["variants_used"] = list(fallback)
                return df
        if prefixes:
            stripped = [p[:-1] if p.endswith("%") else p for p in prefixes]
            prefix_mask = pd.Series(False, index=df_all.index)
            for prefix in stripped:
                prefix_mask |= lowered.str.startswith(str(prefix).lower(), na=False)
            df = _limit(df_all[base_mask & prefix_mask])
            if not df.empty:
                diagnostics["variants_used"] = list(prefixes)
                return df
        return df_all.head(0).copy()

    gene_mask = df_all[str(gene_norm_col)].astype(str).str.upper().eq(normalized_gene)
    return _limit(df_all[base_mask & gene_mask])


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

    print("[miRAssist] snapshot fetch starting", flush=True)
    path = ensure_evidence_parquet()
    print(f"[miRAssist] snapshot path ready: {path}", flush=True)

    reader = _snapshot_reader()
    if reader == "pyarrow_dataset":
        print("[miRAssist] snapshot reader using pyarrow_dataset", flush=True)
        dataset = _dataset(path)
        available = {str(c) for c in dataset.schema.names}
    else:
        print("[miRAssist] snapshot reading schema without pyarrow.dataset", flush=True)
        dataset = None
        available = set(_read_snapshot_columns(path))

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

    if reader != "pyarrow_dataset":
        df = _fetch_pandas_candidate_pool(
            path=path,
            query_token=query_token,
            cfg=cfg,
            available=available,
            selected=selected,
            direction=direction,
            mirna_norm_col=mirna_norm_col,
            gene_norm_col=gene_norm_col,
            expansion=expansion,
            primary=primary,
            fallback=fallback,
            prefixes=prefixes,
            normalized_gene=_normalize_gene_symbol(query_token),
            diagnostics=diagnostics,
        )
        if not df.empty and df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()].copy()
        diagnostics["n_rows_fetched_from_db"] = int(len(df))
        diagnostics["sql_selected_column_count"] = int(len(selected))
        diagnostics["sql_returned_column_count"] = int(len(df.columns))
        print(f"[miRAssist] snapshot pandas fetch complete: rows={len(df)}", flush=True)
        return df, diagnostics

    base = _base_filters(cfg, available)

    def _run(entity_expr: Any) -> pd.DataFrame:
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
