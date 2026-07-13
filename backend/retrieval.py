from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import os
import re

import numpy as np
import pandas as pd

from backend.config import (
    get_db_candidate_limit,
    get_default_mirna_arm,
    get_default_k,
    get_evidence_backend,
    get_evidence_table,
    get_learned_score_column,
    get_use_learned_score,
    get_use_structure_in_score,
    resolve_evidence_path,
    use_mirtarbase_evidence,
)


# =============================================================================
# Evidence loading (Colab-friendly)
# =============================================================================

_EVIDENCE_CACHE: Optional[pd.DataFrame] = None
_EVIDENCE_SOURCE: Optional[str] = None
_POSTGRES_COLUMN_CACHE: Dict[str, List[str]] = {}

PRODUCTION_EVIDENCE_COLUMNS: Tuple[str, ...] = (
    "mirna_name",
    "gene_symbol",
    "mirna_name_norm",
    "gene_symbol_norm",
    "mirna_name_normalized",
    "gene_symbol_normalized",
    "transcript_id",
    "mirassist_xgboost_score",
    "best_backend_model_score",
    "learned_score",
    "model_score",
    "learned_score_xgb_raw_v1",
    "learned_score_xgb_raw_nomissing_v1",
    "learned_score_model_version",
    "learned_score_feature_set",
    "learned_score_updated_at",
    "retrieval_score",
    "support_count",
    "support_targetscan",
    "support_mirdb",
    "support_encori",
    "support_rnahybrid",
    "mirtarbase_pos",
    "label_mirtarbase",
    "mirdb_best_score",
    "mirdb_mean_score",
    "ts_best_contextpp",
    "ts_best_percentile",
    "ts_context_strength",
    "clip_exp_sum",
    "clip_exp_max",
    "n_clip_sites",
    "has_seed_features",
    "best_seed_rank",
    "best_seed_class",
    "n_total_sites",
    "n_sites_6mer",
    "n_sites_7mer_a1",
    "n_sites_7mer_m8",
    "n_sites_8mer",
    "site_density_per_kb",
    "has_rnahybrid",
    "n_rnahybrid_sites",
    "best_mfe",
    "mfe_strength",
    "mean_top3_mfe",
    "mean_top3_mfe_strength",
    "best_local_au",
    "best_local_au_by_mfe",
    "BRCA_spearman_rho",
    "COAD_spearman_rho",
    "PRAD_spearman_rho",
    "BRCA_support_tcga",
    "COAD_support_tcga",
    "PRAD_support_tcga",
    "BRCA_anticorrelated",
    "COAD_anticorrelated",
    "PRAD_anticorrelated",
    "BRCA_repression_evidence",
    "COAD_repression_evidence",
    "PRAD_repression_evidence",
    "BRCA_pair_expressed",
    "COAD_pair_expressed",
    "PRAD_pair_expressed",
    "gene_pathway_hits",
    "overall_evidence_support_percentile",
    "evidence_family_count",
    "evidence_family_summary_json",
    "sequence_complementarity_available",
    "sequence_complementarity_support_percentile",
    "sequence_complementarity_evidence_count",
    "thermodynamic_stability_available",
    "thermodynamic_stability_support_percentile",
    "thermodynamic_stability_evidence_count",
    "sequence_conservation_available",
    "sequence_conservation_support_percentile",
    "sequence_conservation_evidence_count",
    "target_site_accessibility_available",
    "target_site_accessibility_support_percentile",
    "target_site_accessibility_evidence_count",
    "functional_binding_available",
    "functional_binding_support_percentile",
    "functional_binding_evidence_count",
    "functional_repression_available",
    "functional_repression_support_percentile",
    "functional_repression_evidence_count",
    "seed_match_type",
    "seed_pairing_score",
    "seed_pairing_score_percentile",
    "n_seed_sites",
    "n_seed_sites_percentile",
    "best_seed_site_type",
    "has_seed_evidence",
    "rnahybrid_mfe",
    "rnahybrid_mfe_percentile",
    "rnahybrid_mfe_best_site",
    "rnahybrid_mfe_best_site_percentile",
    "rnahybrid_seed_mfe",
    "rnahybrid_seed_mfe_percentile",
    "rnahybrid_strength",
    "rnahybrid_strength_percentile",
    "has_rnahybrid_evidence",
    "targetscan_context_score",
    "targetscan_context_score_support_percentile",
    "targetscan_context_score_percentile",
    "targetscan_aggregate_context_score",
    "targetscan_aggregate_context_score_percentile",
    "targetscan_conserved_site",
    "targetscan_pct",
    "targetscan_pct_percentile",
    "targetscan_branch_length_score",
    "targetscan_branch_length_score_percentile",
    "has_targetscan_evidence",
    "rnaplfold_best_seed_unpaired_prob",
    "rnaplfold_best_seed_unpaired_prob_percentile",
    "rnaplfold_mean_seed_unpaired_prob",
    "rnaplfold_mean_seed_unpaired_prob_percentile",
    "rnaplfold_best_site_unpaired_prob",
    "rnaplfold_best_site_unpaired_prob_percentile",
    "rnaplfold_mean_site_unpaired_prob",
    "rnaplfold_mean_site_unpaired_prob_percentile",
    "rnaplfold_best_flank_unpaired_prob",
    "rnaplfold_best_flank_unpaired_prob_percentile",
    "rnaplfold_mean_flank_unpaired_prob",
    "rnaplfold_mean_flank_unpaired_prob_percentile",
    "rnaplfold_n_sites_scored",
    "rnaplfold_n_sites_scored_percentile",
    "rnaplfold_n_accessible_sites",
    "rnaplfold_n_accessible_sites_percentile",
    "has_rnaplfold_evidence",
    "clip_any_support",
    "clip_max_score",
    "clip_max_score_percentile",
    "clip_n_experiments",
    "clip_n_experiments_percentile",
    "clip_n_cell_lines",
    "clip_n_cell_lines_percentile",
    "encori_clip_score",
    "encori_clip_score_percentile",
    "has_clip_evidence",
    "tcga_any_anticorrelated",
    "tcga_n_supported_contexts",
    "tcga_n_supported_contexts_percentile",
    "tcga_best_repression_evidence",
    "tcga_best_repression_evidence_percentile",
    "tcga_mean_spearman_rho",
    "tcga_mean_spearman_rho_percentile",
    "has_tcga_evidence",
)


def _load_evidence_parquet(
    evidence_path: str | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    resolved_path = str(resolve_evidence_path(evidence_path))
    if columns:
        return pd.read_parquet(resolved_path, columns=columns)
    return pd.read_parquet(resolved_path)


def _load_evidence_postgres(
    table_name: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    from backend.db import get_database_engine, quote_identifier

    engine = get_database_engine()
    if engine is None:
        raise RuntimeError("DATABASE_URL is not configured for postgres evidence loading.")

    quoted_table = quote_identifier(table_name)
    if columns:
        quoted_columns = ", ".join(quote_identifier(col) for col in columns)
    else:
        quoted_columns = "*"

    query = f"SELECT {quoted_columns} FROM {quoted_table}"
    try:
        return pd.read_sql_query(query, engine)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load postgres evidence table '{table_name}'. "
            "Check EVIDENCE_TABLE, DATABASE_URL, and that the table/schema exists."
        ) from exc


def load_evidence(
    evidence_path: str | None = None,
    columns: list[str] | None = None,
    force_reload: bool = False,
) -> pd.DataFrame:
    """
    Load the evidence parquet once and cache it in-process.

    Resolution order:
      1) explicit argument
      2) env MIASSIST_EVIDENCE
      3) env MIASSIST_BASE + /data/processed/evidence_pairs_tcga.parquet
      4) fallback: data/processed/evidence_pairs_tcga.parquet (relative)

    Notes:
      - Drops duplicate column labels if present.
      - Designed for single-process direct app usage.
    """
    global _EVIDENCE_CACHE, _EVIDENCE_SOURCE

    backend_name = get_evidence_backend()
    source = ""
    if backend_name == "postgres":
        source = f"postgres:{get_evidence_table()}"
    else:
        source = str(resolve_evidence_path(evidence_path))

    if (
        not force_reload
        and _EVIDENCE_CACHE is not None
        and _EVIDENCE_SOURCE == source
    ):
        return _EVIDENCE_CACHE

    if backend_name == "postgres":
        df = _load_evidence_postgres(get_evidence_table(), columns=columns)
    else:
        df = _load_evidence_parquet(evidence_path, columns=columns)

    # Safety: remove duplicate column labels (can happen after merges)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    _EVIDENCE_CACHE = df
    _EVIDENCE_SOURCE = source
    return df


def _split_table_name(table_name: str) -> Tuple[str, str]:
    raw = str(table_name or "").strip()
    if "." in raw:
        schema_name, bare_name = raw.split(".", 1)
        return schema_name.strip() or "public", bare_name.strip()
    return "public", raw


def _get_postgres_table_columns(table_name: str) -> List[str]:
    from sqlalchemy import text

    from backend.db import get_database_engine

    engine = get_database_engine()
    if engine is None:
        raise RuntimeError("DATABASE_URL is not configured for postgres evidence loading.")

    schema_name, bare_name = _split_table_name(table_name)
    cache_key = f"{engine.url}|{schema_name}.{bare_name}"
    if cache_key in _POSTGRES_COLUMN_CACHE:
        return list(_POSTGRES_COLUMN_CACHE[cache_key])

    with engine.begin() as conn:
        rows = conn.execute(
            text(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = :schema_name
                  AND table_name = :table_name
                ORDER BY ordinal_position
                """
            ),
            {"schema_name": schema_name, "table_name": bare_name},
        ).fetchall()

    columns = [str(row[0]) for row in rows]
    _POSTGRES_COLUMN_CACHE[cache_key] = columns
    return list(columns)


def _dynamic_tcga_columns(tcga: Optional[str]) -> List[str]:
    if not tcga:
        return []
    tcga = str(tcga).upper()
    return [
        f"{tcga}_spearman_rho",
        f"{tcga}_spearman_p",
        f"{tcga}_support_tcga",
        f"{tcga}_anticorrelated",
        f"{tcga}_repression_evidence",
        f"{tcga}_pair_expressed",
    ]


def _select_production_evidence_columns(available_columns: Iterable[str], tcga: Optional[str]) -> List[str]:
    available = {str(col) for col in available_columns}
    selected = [col for col in PRODUCTION_EVIDENCE_COLUMNS if col in available]
    for col in _dynamic_tcga_columns(tcga):
        if col in available and col not in selected:
            selected.append(col)
    return selected


def _first_available_column(available_columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    available = {str(col) for col in available_columns}
    for candidate in candidates:
        if str(candidate) in available:
            return str(candidate)
    return None


def _sql_in_clause(column_sql: str, param_prefix: str, values: List[str], params: Dict[str, Any]) -> str:
    placeholders: List[str] = []
    for idx, value in enumerate(values):
        param_name = f"{param_prefix}_{idx}"
        placeholders.append(f":{param_name}")
        params[param_name] = str(value).strip().lower()
    if not placeholders:
        return "1 = 0"
    return f"LOWER({column_sql}) IN ({', '.join(placeholders)})"


def _postgres_rank_tiebreak_columns(available: set[str]) -> List[str]:
    order_columns: List[str] = []
    if "support_count" in available:
        order_columns.append('"support_count" DESC NULLS LAST')
    if "mirdb_best_score" in available:
        order_columns.append('"mirdb_best_score" DESC NULLS LAST')
    if "ts_context_strength" in available:
        order_columns.append('"ts_context_strength" DESC NULLS LAST')
    elif "ts_best_contextpp" in available:
        order_columns.append('"ts_best_contextpp" ASC NULLS LAST')
    if "clip_exp_sum" in available:
        order_columns.append('"clip_exp_sum" DESC NULLS LAST')
    if "best_mfe" in available:
        order_columns.append('"best_mfe" ASC NULLS LAST')
    return order_columns


def build_postgres_candidate_query(
    query_token: str,
    cfg: "RetrievalConfig",
    available_columns: Iterable[str],
    *,
    mirna_variants: Optional[List[str]] = None,
    mirna_prefix_patterns: Optional[List[str]] = None,
) -> Tuple[str, Dict[str, Any], List[str], Dict[str, Any]]:
    from backend.db import quote_identifier

    direction = _direction_from_token(_normalize_token(query_token))
    available = {str(col) for col in available_columns}
    selected_columns = _select_production_evidence_columns(available, cfg.tcga)
    required_columns = {"mirna_name", "gene_symbol", "support_count"}
    mirna_norm_col = _first_available_column(available, ("mirna_name_norm", "mirna_name_normalized"))
    gene_norm_col = _first_available_column(available, ("gene_symbol_norm", "gene_symbol_normalized"))
    if direction == "mirna_to_targets":
        if mirna_norm_col is None:
            required_columns.add("mirna_name_norm")
    else:
        if gene_norm_col is None:
            required_columns.add("gene_symbol_norm")
    missing_required = sorted(col for col in required_columns if col not in selected_columns)
    if missing_required:
        raise RuntimeError(
            "Postgres evidence retrieval requires columns missing from the evidence table: "
            + ", ".join(missing_required)
        )

    params: Dict[str, Any] = {"candidate_limit": int(get_db_candidate_limit())}
    where_clauses: List[str] = []

    if cfg.novel and cfg.use_mirtarbase_evidence and "mirtarbase_pos" in available:
        where_clauses.append("COALESCE(" + quote_identifier("mirtarbase_pos") + ", 0) = 0")

    if cfg.require_binding_evidence:
        binding_cols = [col for col in ("support_targetscan", "support_encori", "support_mirdb") if col in available]
        if binding_cols:
            binding_bits = [f"COALESCE({quote_identifier(col)}, 0) = 1" for col in binding_cols]
            where_clauses.append("(" + " OR ".join(binding_bits) + ")")

    if cfg.require_expression and cfg.tcga:
        pair_expr_col = f"{str(cfg.tcga).upper()}_pair_expressed"
        if pair_expr_col in available:
            where_clauses.append("COALESCE(" + quote_identifier(pair_expr_col) + ", 0) = 1")

    if direction == "mirna_to_targets":
        mirna_bits: List[str] = []
        if mirna_variants:
            mirna_bits.append(
                _sql_in_clause(quote_identifier(str(mirna_norm_col)), "mirna_norm", mirna_variants, params)
            )
        if mirna_prefix_patterns:
            for idx, pattern in enumerate(mirna_prefix_patterns):
                param_name = f"mirna_prefix_{idx}"
                params[param_name] = str(pattern).strip().lower()
                mirna_bits.append(f"LOWER({quote_identifier(str(mirna_norm_col))}) LIKE :{param_name}")
        where_clauses.append("(" + " OR ".join(mirna_bits) + ")" if mirna_bits else "1 = 0")
    else:
        params["gene_norm"] = _normalize_gene_symbol(query_token)
        where_clauses.append(f"{quote_identifier(str(gene_norm_col))} = :gene_norm")

    pathway_selection = cfg.pathway_selection or {}
    if bool(pathway_selection.get("enabled")) and cfg.pathway_gene_set and gene_norm_col is not None:
        pathway_genes = sorted({_normalize_gene_symbol(gene) for gene in cfg.pathway_gene_set if str(gene or "").strip()})
        where_clauses.append(
            _sql_in_clause(quote_identifier(str(gene_norm_col)), "pathway_gene", pathway_genes, params)
        )

    learned_score_column = get_learned_score_column()
    selected_db_score_column = _resolve_score_column_from_available(available, learned_score_column)
    order_columns: List[str] = []
    if selected_db_score_column and selected_db_score_column != "retrieval_score":
        order_columns.append(f"{quote_identifier(selected_db_score_column)} DESC NULLS LAST")
        order_columns.extend(_postgres_rank_tiebreak_columns(available))
        if "retrieval_score" in available:
            order_columns.append('"retrieval_score" DESC NULLS LAST')
    else:
        if "retrieval_score" in available:
            order_columns.append('"retrieval_score" DESC NULLS LAST')
        order_columns.extend(_postgres_rank_tiebreak_columns(available))
    if not order_columns:
        order_columns.append(f"{quote_identifier('gene_symbol')} ASC")

    quoted_table = quote_identifier(get_evidence_table())
    select_sql = ", ".join(quote_identifier(col) for col in selected_columns)
    where_sql = " AND ".join(where_clauses) if where_clauses else "1 = 1"
    order_sql = ", ".join(order_columns)
    query = (
        f"SELECT {select_sql} "
        f"FROM {quoted_table} "
        f"WHERE {where_sql} "
        f"ORDER BY {order_sql} "
        f"LIMIT :candidate_limit"
    )
    diagnostics = {
        "evidence_backend": "postgres",
        "supabase_table_name": get_evidence_table(),
        "db_candidate_limit": int(get_db_candidate_limit()),
        "learned_score_column": selected_db_score_column or learned_score_column,
        "sort_column_used": selected_db_score_column or "retrieval_score",
        "query_direction": direction,
        "sql_mirna_norm_column": mirna_norm_col,
        "sql_gene_norm_column": gene_norm_col,
        "sql_selected_columns": list(selected_columns),
        "sql_order_columns": list(order_columns),
        "primary_mirna_variants": list(mirna_variants or []),
        "mirna_prefix_patterns_used": list(mirna_prefix_patterns or []),
    }
    return query, params, selected_columns, diagnostics


def _fetch_postgres_candidate_pool(
    query_token: str,
    cfg: "RetrievalConfig",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    from sqlalchemy import text

    from backend.db import get_database_engine

    engine = get_database_engine()
    if engine is None:
        raise RuntimeError("DATABASE_URL is not configured for postgres evidence retrieval.")

    table_name = get_evidence_table()
    available_columns = _get_postgres_table_columns(table_name)
    direction = _direction_from_token(_normalize_token(query_token))
    default_arm = get_default_mirna_arm()
    expansion = expand_mirna_query_variants(query_token, default_arm=default_arm) if direction == "mirna_to_targets" else {}
    primary_variants = list(expansion.get("primary_variants") or []) if direction == "mirna_to_targets" else []
    fallback_variants = list(expansion.get("fallback_variants") or []) if direction == "mirna_to_targets" else []
    prefix_patterns = _mirna_prefix_fallback_patterns(query_token) if direction == "mirna_to_targets" else []

    diagnostics: Dict[str, Any] = {
        "evidence_backend": "postgres",
        "raw_mirna_query": str(query_token or "") if direction == "mirna_to_targets" else None,
        "normalized_mirna_query": expansion.get("normalized_input") if direction == "mirna_to_targets" else None,
        "query_mirna_normalized": _normalize_mirna_query(query_token)[0] if direction == "mirna_to_targets" else None,
        "explicit_mirna_arm": bool(expansion.get("explicit_arm")) if direction == "mirna_to_targets" else False,
        "default_mirna_arm": default_arm if direction == "mirna_to_targets" else None,
        "primary_mirna_variants": list(primary_variants),
        "fallback_mirna_variants": list(fallback_variants),
        "exact_mirna_variants_used": list(primary_variants),
        "searched_mirna_variants": list(primary_variants),
        "variants_used": [],
        "mature_arm_expansion_attempted": bool(direction == "mirna_to_targets" and not expansion.get("explicit_arm")),
        "mature_arm_expansion_used": False,
        "mirna_prefix_fallback_attempted": False,
        "mirna_prefix_fallback_used": False,
        "n_rows_primary": 0,
        "n_rows_fallback": 0,
        "arm_interpretation_note": expansion.get("note") if direction == "mirna_to_targets" else None,
    }

    def _run_query(mirna_variants: Optional[List[str]], mirna_like_patterns: Optional[List[str]]) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
        query, params, selected_cols, diag = build_postgres_candidate_query(
            query_token,
            cfg,
            available_columns,
            mirna_variants=mirna_variants,
            mirna_prefix_patterns=mirna_like_patterns,
        )
        try:
            frame = pd.read_sql_query(text(query), engine, params=params)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch bounded postgres evidence candidates from '{table_name}'."
            ) from exc
        return frame, diag, selected_cols

    if direction == "mirna_to_targets":
        df, query_diagnostics, selected_columns = _run_query(primary_variants, None)
        diagnostics.update(query_diagnostics)
        diagnostics["n_rows_primary"] = int(len(df))
        if not df.empty:
            diagnostics["variants_used"] = list(primary_variants)
            diagnostics["mature_arm_expansion_used"] = not bool(expansion.get("explicit_arm"))
        if df.empty and fallback_variants:
            df, query_diagnostics, selected_columns = _run_query(fallback_variants, None)
            diagnostics.update(query_diagnostics)
            diagnostics["n_rows_fallback"] = int(len(df))
            diagnostics["searched_mirna_variants"] = list(fallback_variants)
            if not df.empty:
                diagnostics["variants_used"] = list(fallback_variants)
        if df.empty and prefix_patterns:
            diagnostics["mirna_prefix_fallback_attempted"] = True
            df, query_diagnostics, selected_columns = _run_query(None, prefix_patterns)
            diagnostics.update(query_diagnostics)
            diagnostics["mirna_prefix_fallback_used"] = not df.empty
            if not df.empty:
                diagnostics["variants_used"] = list(prefix_patterns)
    else:
        df, query_diagnostics, selected_columns = _run_query(None, None)
        diagnostics.update(query_diagnostics)

    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    diagnostics["n_rows_fetched_from_db"] = int(len(df))
    diagnostics["sql_selected_column_count"] = int(len(selected_columns))
    diagnostics["sql_returned_column_count"] = int(len(df.columns))
    return df, diagnostics


# =============================================================================
# Retrieval config
# =============================================================================

@dataclass
class RetrievalConfig:
    # Core
    k_shortlist: int = 200
    min_support: int = 1
    novel: bool = False

    # Context
    tcga: Optional[str] = None  # e.g., "BRCA"
    keywords: Optional[List[str]] = None  # generic soft keywords (optional)

    # Phenotype/pathway conditioning
    phenotype_keywords: Optional[List[str]] = None
    pathway_keywords: Optional[List[str]] = None
    pathway_filter: Optional[Dict[str, Any]] = None
    pathway_selection: Optional[Dict[str, Any]] = None
    pathway_gene_set: Optional[set[str]] = None
    pathway_gene_map: Optional[Dict[str, List[str]]] = None

    # Optional soft gates (off by default)
    require_binding_evidence: bool = False
    require_expression: bool = False

    # Collapse duplicate (miRNA,gene) rows before scoring
    collapse_duplicates: bool = True
    use_mirtarbase_evidence: bool = True
    allow_min_support_relaxation: bool = False
    user_requested_strict_support: bool = False


# =============================================================================
# Helpers
# =============================================================================

def _normalize_token(x: str) -> str:
    return str(x).strip()


def _ensure_cols(ev: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in ev.columns]
    if missing:
        raise ValueError(f"Evidence table missing columns: {missing[:25]}")


def _bool_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.zeros(len(df), dtype=int), index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)


def _safe_float_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.full(len(df), default, dtype=float), index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def _safe_int_col(df: pd.DataFrame, col: str, default: int = 0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.full(len(df), default, dtype=int), index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(int)


def _candidate_score_columns(configured_score_column: str) -> List[str]:
    candidates: List[str] = ["mirassist_xgboost_score"]
    configured = str(configured_score_column or "").strip()
    if configured:
        candidates.append(configured)
    candidates.extend(
        [
            "best_backend_model_score",
            "learned_score",
            "model_score",
            "overall_evidence_support_percentile",
            "retrieval_score",
        ]
    )
    out: List[str] = []
    for candidate in candidates:
        if candidate and candidate not in out:
            out.append(candidate)
    return out


def _resolve_score_column(df: pd.DataFrame, configured_score_column: str) -> Optional[str]:
    for candidate in _candidate_score_columns(configured_score_column):
        if candidate in df.columns:
            return candidate
    return None


def _resolve_score_column_from_available(available_columns: Iterable[str], configured_score_column: str) -> Optional[str]:
    available = {str(col) for col in available_columns}
    for candidate in _candidate_score_columns(configured_score_column):
        if candidate in available:
            return candidate
    return None


def apply_learned_score_ranking(
    df: pd.DataFrame,
    learned_score_column: str,
    enabled: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    ranked = df.copy()
    retrieval_score = _safe_float_col(ranked, "retrieval_score", default=0.0)
    support_tiebreak = _safe_float_col(ranked, "support_count", default=0.0)
    mirdb_tiebreak = _safe_float_col(ranked, "mirdb_best_score", default=0.0)
    clip_tiebreak = _safe_float_col(ranked, "clip_exp_sum", default=0.0)
    mfe_tiebreak = np.clip(-_safe_float_col(ranked, "best_mfe", default=0.0), 0.0, 100.0)
    if "ts_context_strength" in ranked.columns:
        ts_tiebreak = _safe_float_col(ranked, "ts_context_strength", default=0.0)
    elif "retrieval_ts_contrib" in ranked.columns:
        ts_tiebreak = _safe_float_col(ranked, "retrieval_ts_contrib", default=0.0)
    else:
        ts_tiebreak = np.clip(-_safe_float_col(ranked, "ts_best_contextpp", default=0.0), 0.0, 2.0)

    diagnostics: Dict[str, Any] = {
        "learned_score_enabled": False,
        "retrieval_ranking_mode": "manual",
        "learned_score_column": learned_score_column,
        "score_column_used": "retrieval_score",
        "learned_score_present_count": 0,
        "learned_score_missing_count": int(len(ranked)),
    }

    ranked = ranked.assign(
        retrieval_rank_score=retrieval_score.astype(float),
        learned_score_used=retrieval_score.astype(float),
        _learned_score_missing=pd.Series(np.ones(len(ranked), dtype=int), index=ranked.index),
        score_column_used=pd.Series(["retrieval_score"] * len(ranked), index=ranked.index, dtype="object"),
        _retrieval_support_tiebreak=support_tiebreak.astype(float),
        _retrieval_mirdb_tiebreak=mirdb_tiebreak.astype(float),
        _retrieval_ts_tiebreak=ts_tiebreak.astype(float),
        _retrieval_clip_tiebreak=clip_tiebreak.astype(float),
        _retrieval_mfe_tiebreak=mfe_tiebreak.astype(float),
    )

    if not enabled:
        ranked = ranked.sort_values(
            [
                "retrieval_rank_score",
                "_retrieval_support_tiebreak",
                "_retrieval_mirdb_tiebreak",
                "_retrieval_ts_tiebreak",
                "_retrieval_clip_tiebreak",
                "_retrieval_mfe_tiebreak",
                "retrieval_score",
            ],
            ascending=[False, False, False, False, False, False, False],
        )
        return ranked, diagnostics

    selected_score_column = _resolve_score_column(ranked, learned_score_column)
    diagnostics["learned_score_column"] = selected_score_column or learned_score_column
    diagnostics["score_column_used"] = selected_score_column or "retrieval_score"

    if selected_score_column is None:
        diagnostics["warnings"] = [
            "No preferred learned-score columns were present in the evidence rows; using manual retrieval_score ranking."
        ]
        ranked = ranked.sort_values(
            [
                "retrieval_rank_score",
                "_retrieval_support_tiebreak",
                "_retrieval_mirdb_tiebreak",
                "_retrieval_ts_tiebreak",
                "_retrieval_clip_tiebreak",
                "_retrieval_mfe_tiebreak",
                "retrieval_score",
            ],
            ascending=[False, False, False, False, False, False, False],
        )
        return ranked, diagnostics

    learned_score_values = pd.to_numeric(ranked[selected_score_column], errors="coerce")
    learned_score_missing = learned_score_values.isna()
    score_column_used = pd.Series(
        np.where(learned_score_missing.to_numpy(), "retrieval_score", selected_score_column),
        index=ranked.index,
        dtype="object",
    )
    ranked = ranked.assign(
        retrieval_rank_score=learned_score_values.where(~learned_score_missing, retrieval_score).astype(float),
        learned_score_used=learned_score_values.where(~learned_score_missing, retrieval_score).astype(float),
        _learned_score_missing=learned_score_missing.astype(int),
        score_column_used=score_column_used,
    )
    ranked = ranked.sort_values(
        [
            "_learned_score_missing",
            "retrieval_rank_score",
            "_retrieval_support_tiebreak",
            "_retrieval_mirdb_tiebreak",
            "_retrieval_ts_tiebreak",
            "_retrieval_clip_tiebreak",
            "_retrieval_mfe_tiebreak",
            "retrieval_score",
        ],
        ascending=[True, False, False, False, False, False, False, False],
    )
    diagnostics.update(
        {
            "learned_score_enabled": bool(selected_score_column != "retrieval_score"),
            "retrieval_ranking_mode": f"learned:{selected_score_column}",
            "learned_score_present_count": int((~learned_score_missing).sum()),
            "learned_score_missing_count": int(learned_score_missing.sum()),
        }
    )
    return ranked, diagnostics


# =============================================================================
# miRNA normalization + matching (robust exact matching via normalization)
# =============================================================================

_ARM_RE = re.compile(r"(?i)(?:^|[-_])(3p|5p)$")
_SPECIES_PREFIX_RE = re.compile(r"(?i)^(hsa|mmu|rno|dme|cel|ath)[-_]")
_MICRORNA_PREFIX_RE = re.compile(r"(?i)^microrna[\s_-]*")
_MIRNA_WORD_PREFIX_RE = re.compile(r"(?i)^mirna[\s_-]*")
_MIR_PREFIX_RE = re.compile(r"(?i)^(mir|miR|let|Let)")


def _normalize_gene_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _strip_species_prefix(s: str) -> str:
    return _SPECIES_PREFIX_RE.sub("", s.strip())


def normalize_mirna_name(raw: str) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""

    s = (
        s.replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
        .replace("_", "-")
    )
    s = re.sub(r"\s+", "", s)
    s = s.lower()
    species_prefix = _species_prefix_from_raw_mirna(s)
    if species_prefix:
        s = _strip_species_prefix(s)

    s = _MICRORNA_PREFIX_RE.sub("mir-", s)
    s = _MIRNA_WORD_PREFIX_RE.sub("mir-", s)
    s = re.sub(r"^(mir|let)(?=[0-9a-z])", r"\1-", s)
    s = re.sub(r"^mir(?=[0-9a-z])", "mir-", s)
    s = re.sub(r"^(mir|let)(?=[0-9a-z])", r"\1-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    if species_prefix:
        s = f"{species_prefix}-{s}"
    return s


def mirna_has_explicit_arm(norm: str) -> bool:
    return bool(re.search(r"(?i)-(3p|5p)$", str(norm or "").strip()))


def _normalize_mirna_query(user_mirna: str) -> Tuple[str, Optional[str]]:
    """
    Returns (base, arm) where base is normalized like:
      "mir-21", "mir-17-5", "let-7a", etc. (lowercase, hyphen-delimited)
    arm is "3p"/"5p" if explicitly provided by user, else None.
    """
    normalized = normalize_mirna_name(user_mirna)
    if not normalized:
        return "", None

    s = _strip_species_prefix(normalized).lower()
    arm = None
    m = _ARM_RE.search(s)
    if m:
        arm = m.group(1).lower()
        s = _ARM_RE.sub("", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s, arm


def _species_prefix_from_raw_mirna(value: str) -> Optional[str]:
    s = str(value or "").strip().replace("_", "-")
    m = _SPECIES_PREFIX_RE.match(s)
    if not m:
        return None
    return m.group(1).lower()


def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _display_mirna_name(norm: str) -> str:
    text = str(norm or "").strip()
    if not text:
        return ""
    if text.startswith("hsa-mir-"):
        return "hsa-miR-" + text[len("hsa-mir-") :]
    if text.startswith("mir-"):
        return "miR-" + text[len("mir-") :]
    return text


def expand_mirna_query_variants(raw_mirna: str, default_arm: str = "5p") -> Dict[str, Any]:
    normalized_input = normalize_mirna_name(raw_mirna)
    species_prefix = _species_prefix_from_raw_mirna(raw_mirna)
    species_prefixes = _dedupe_preserve_order([species_prefix, "hsa"])
    base, arm = _normalize_mirna_query(raw_mirna)
    explicit_arm = arm in {"3p", "5p"}
    default_arm = str(default_arm or "5p").strip().lower()
    if default_arm not in {"5p", "3p", "both"}:
        default_arm = "5p"

    if not base:
        return {
            "normalized_input": normalized_input,
            "explicit_arm": False,
            "primary_variants": [],
            "fallback_variants": [],
            "note": None,
        }

    def _with_species(forms: List[str]) -> List[str]:
        prefixed = [f"{prefix}-{form}" for prefix in species_prefixes for form in forms]
        return _dedupe_preserve_order(forms + prefixed)

    if explicit_arm:
        primary_forms = [f"{base}-{arm}"]
        note = None
    elif default_arm == "both":
        primary_forms = [f"{base}-5p", f"{base}-3p"]
        note = (
            f"No mature arm was specified, so miRAssist searched both {_display_mirna_name(f'{base}-5p')} and {_display_mirna_name(f'{base}-3p')} mature arms."
        )
    else:
        primary_forms = [f"{base}-{default_arm}"]
        other_arm = "3p" if default_arm == "5p" else "5p"
        note = (
            f"No mature arm was specified, so miRAssist interpreted this as {_display_mirna_name(f'{base}-{default_arm}')} by default. "
            f"Search {_display_mirna_name(f'{base}-{other_arm}')} explicitly to retrieve {other_arm}-arm targets."
        )

    fallback_forms = [] if explicit_arm else [base]
    return {
        "normalized_input": normalized_input,
        "explicit_arm": explicit_arm,
        "primary_variants": _with_species(primary_forms),
        "fallback_variants": _with_species(fallback_forms),
        "note": note,
    }


def _mirna_prefix_fallback_patterns(raw_mirna: str) -> List[str]:
    base, arm = _normalize_mirna_query(raw_mirna)
    if not base or arm in {"3p", "5p"}:
        return []

    species_prefix = _species_prefix_from_raw_mirna(raw_mirna)
    species_prefixes = _dedupe_preserve_order([species_prefix, "hsa"])
    prefixed = [f"{prefix}-{base}-%" for prefix in species_prefixes]
    return _dedupe_preserve_order([f"{base}-%"] + prefixed)


def _normalize_mirna_table_value(v: str) -> Tuple[str, Optional[str]]:
    """
    Normalize a table miRNA string into (base, arm) similar to query normalization.
    Returns ("", None) if unusable.
    """
    if v is None:
        return "", None
    normalized = normalize_mirna_name(v)
    if not normalized:
        return "", None
    s = _strip_species_prefix(normalized).lower()

    arm = None
    m = _ARM_RE.search(s)
    if m:
        arm = m.group(1).lower()
        s = _ARM_RE.sub("", s)

    # Some tables contain "mir" alone, ignore that
    if s in ("mir", "let"):
        return "", None

    return s, arm


def _match_mirna_names_for_table(user_mirna: str, mirna_series: pd.Series) -> Dict[str, List[str]]:
    expansion = expand_mirna_query_variants(user_mirna, default_arm=get_default_mirna_arm())
    vals = mirna_series.dropna().astype(str)
    if vals.empty:
        return {"primary": [], "fallback": [], "prefix": []}

    primary_variants = {str(value or "").strip().lower() for value in (expansion.get("primary_variants") or []) if str(value or "").strip()}
    fallback_variants = {str(value or "").strip().lower() for value in (expansion.get("fallback_variants") or []) if str(value or "").strip()}
    prefix_patterns = [str(pattern or "").strip().lower() for pattern in _mirna_prefix_fallback_patterns(user_mirna) if str(pattern or "").strip()]

    primary_hits: List[str] = []
    fallback_hits: List[str] = []
    prefix_hits: List[str] = []
    for raw in vals.unique().tolist():
        normalized_name = normalize_mirna_name(raw)
        if not normalized_name:
            continue
        lowered = normalized_name.lower()
        if lowered in primary_variants:
            primary_hits.append(raw)
        elif lowered in fallback_variants:
            fallback_hits.append(raw)
        elif prefix_patterns and any(lowered.startswith(pattern[:-1]) for pattern in prefix_patterns):
            prefix_hits.append(raw)

    return {
        "primary": _dedupe_preserve_order(primary_hits),
        "fallback": _dedupe_preserve_order(fallback_hits),
        "prefix": _dedupe_preserve_order(prefix_hits),
    }


def resolve_mirna_names_for_table(user_mirna: str, mirna_series: pd.Series) -> List[str]:
    """
    Map user query -> EXACT values present in the table, but matched via normalization.
    If no arm specified: prefer the configured default mature arm, then legacy base rows.
    """
    matches = _match_mirna_names_for_table(user_mirna, mirna_series)
    if matches["primary"]:
        return matches["primary"]
    if matches["fallback"]:
        return matches["fallback"]
    return matches["prefix"]


def _filter_mirna_rows(df: pd.DataFrame, raw_mirna: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    default_arm = get_default_mirna_arm()
    expansion = expand_mirna_query_variants(raw_mirna, default_arm=default_arm)
    diagnostics: Dict[str, Any] = {
        "raw_mirna_query": str(raw_mirna or ""),
        "normalized_mirna_query": expansion.get("normalized_input"),
        "query_mirna_normalized": _normalize_mirna_query(raw_mirna)[0],
        "explicit_mirna_arm": bool(expansion.get("explicit_arm")),
        "default_mirna_arm": default_arm,
        "primary_mirna_variants": list(expansion.get("primary_variants") or []),
        "fallback_mirna_variants": list(expansion.get("fallback_variants") or []),
        "exact_mirna_variants_used": list(expansion.get("primary_variants") or []),
        "variants_used": [],
        "searched_mirna_variants": list(expansion.get("primary_variants") or []),
        "mature_arm_expansion_attempted": not bool(expansion.get("explicit_arm")),
        "mature_arm_expansion_used": False,
        "mirna_prefix_fallback_attempted": False,
        "mirna_prefix_fallback_used": False,
        "n_rows_primary": 0,
        "n_rows_fallback": 0,
        "arm_interpretation_note": expansion.get("note"),
    }
    if df.empty:
        return df, diagnostics

    primary_variants = list(expansion.get("primary_variants") or [])
    fallback_variants = list(expansion.get("fallback_variants") or [])
    prefix_patterns = _mirna_prefix_fallback_patterns(raw_mirna)

    normalized_mirna_col = _first_available_column(df.columns, ("mirna_name_norm", "mirna_name_normalized"))
    if normalized_mirna_col is not None:
        norm_series = df[normalized_mirna_col].astype(str).str.strip().str.lower()
        primary_mask = norm_series.isin([value.lower() for value in primary_variants])
        filtered = df.loc[primary_mask].copy()
        diagnostics["n_rows_primary"] = int(len(filtered))
        if not filtered.empty:
            diagnostics["variants_used"] = list(primary_variants)
            diagnostics["mature_arm_expansion_used"] = not bool(expansion.get("explicit_arm"))

        if filtered.empty and fallback_variants:
            fallback_mask = norm_series.isin([value.lower() for value in fallback_variants])
            filtered = df.loc[fallback_mask].copy()
            diagnostics["n_rows_fallback"] = int(len(filtered))
            if not filtered.empty:
                diagnostics["variants_used"] = list(fallback_variants)
                diagnostics["searched_mirna_variants"] = list(fallback_variants)

        if filtered.empty and prefix_patterns:
            diagnostics["mirna_prefix_fallback_attempted"] = True
            prefix_mask = pd.Series(False, index=df.index)
            for pattern in prefix_patterns:
                prefix_mask |= norm_series.str.startswith(pattern[:-1])
            filtered = df.loc[prefix_mask].copy()
            diagnostics["mirna_prefix_fallback_used"] = not filtered.empty
            if not filtered.empty:
                diagnostics["variants_used"] = list(prefix_patterns)

        matched = filtered["mirna_name"].dropna().astype(str).unique().tolist() if "mirna_name" in filtered.columns else []
        diagnostics["matched_mirna_names"] = matched[:20]
        diagnostics["normalized_mirna_column_used"] = normalized_mirna_col
        return filtered, diagnostics

    matches = _match_mirna_names_for_table(raw_mirna, df["mirna_name"])
    filtered = df.head(0).copy()

    primary_allowed = matches["primary"]
    fallback_allowed = matches["fallback"]
    prefix_allowed = matches["prefix"]
    if primary_allowed:
        allowed_l = {value.lower() for value in primary_allowed}
        filtered = df[df["mirna_name"].astype(str).str.lower().isin(allowed_l)].copy()
        diagnostics["n_rows_primary"] = int(len(filtered))
        diagnostics["variants_used"] = list(primary_variants)
        diagnostics["matched_mirna_names"] = primary_allowed[:20]
        diagnostics["mature_arm_expansion_used"] = not bool(expansion.get("explicit_arm"))
        return filtered, diagnostics

    if fallback_allowed:
        allowed_l = {value.lower() for value in fallback_allowed}
        filtered = df[df["mirna_name"].astype(str).str.lower().isin(allowed_l)].copy()
        diagnostics["n_rows_fallback"] = int(len(filtered))
        diagnostics["variants_used"] = list(fallback_variants)
        diagnostics["searched_mirna_variants"] = list(fallback_variants)
        diagnostics["matched_mirna_names"] = fallback_allowed[:20]
        return filtered, diagnostics

    if prefix_allowed:
        diagnostics["mirna_prefix_fallback_attempted"] = True
        allowed_l = {value.lower() for value in prefix_allowed}
        filtered = df[df["mirna_name"].astype(str).str.lower().isin(allowed_l)].copy()
        diagnostics["mirna_prefix_fallback_used"] = not filtered.empty
        diagnostics["variants_used"] = list(prefix_patterns)
        diagnostics["matched_mirna_names"] = prefix_allowed[:20]
        return filtered, diagnostics

    if prefix_patterns:
        diagnostics["mirna_prefix_fallback_attempted"] = True
    diagnostics["matched_mirna_names"] = []
    return filtered, diagnostics


def _apply_min_support_with_relaxation(
    df: pd.DataFrame,
    cfg: "RetrievalConfig",
    diagnostics: Dict[str, Any],
) -> pd.DataFrame:
    initial_min_support = int(cfg.min_support)
    diagnostics["initial_min_support"] = initial_min_support
    diagnostics["effective_min_support"] = initial_min_support
    diagnostics["relaxed_min_support_reason"] = None
    diagnostics["n_rows_before_min_support"] = int(len(df))

    if df.empty:
        diagnostics["n_rows_after_min_support"] = 0
        return df

    support_count = _safe_int_col(df, "support_count", default=0)

    def _filter_for(threshold: int) -> pd.DataFrame:
        if threshold <= 0:
            return df.copy()
        return df.loc[support_count >= threshold].copy()

    filtered = _filter_for(initial_min_support)
    diagnostics["n_rows_after_min_support"] = int(len(filtered))
    diagnostics["n_after_min_support"] = int(len(filtered))
    if not filtered.empty or not cfg.allow_min_support_relaxation or cfg.user_requested_strict_support:
        return filtered

    if initial_min_support >= 2:
        relaxed_one = _filter_for(1)
        if not relaxed_one.empty:
            diagnostics["effective_min_support"] = 1
            diagnostics["relaxed_min_support_reason"] = (
                "No candidates passed the initial min_support filter for a broad miRNA query, so the threshold was relaxed to 1 while keeping learned-score ranking enabled."
            )
            diagnostics["n_rows_after_min_support"] = int(len(relaxed_one))
            diagnostics["n_after_min_support"] = int(len(relaxed_one))
            diagnostics.setdefault("warnings", []).append(
                f"No candidates passed min_support >= {initial_min_support}; retrieval was relaxed to min_support >= 1 for a broad miRNA query."
            )
            return relaxed_one

        relaxed_zero = _filter_for(0)
        if not relaxed_zero.empty:
            diagnostics["effective_min_support"] = 0
            diagnostics["relaxed_min_support_reason"] = (
                "No candidates passed min_support >= 2 or >= 1, so retrieval fell back to min_support 0 and ranked candidates by learned score."
            )
            diagnostics["n_rows_after_min_support"] = int(len(relaxed_zero))
            diagnostics["n_after_min_support"] = int(len(relaxed_zero))
            diagnostics.setdefault("warnings", []).append(
                f"No candidates passed min_support >= {initial_min_support}; retrieval was relaxed to min_support >= 0 and ranked by learned score."
            )
            return relaxed_zero

    return filtered


# =============================================================================
# Direction inference
# =============================================================================

def _is_mirna_token(token: str) -> bool:
    t = token.lower()
    return ("mir" in t) or t.startswith(("hsa-", "mmu-", "rno-"))


def _direction_from_token(token: str) -> str:
    return "mirna_to_targets" if _is_mirna_token(token) else "gene_to_mirnas"


# =============================================================================
# Duplicate collapse (miRNA,gene) -> single row
# =============================================================================

def _first_nonnull_value(series: pd.Series):
    """
    Return first 'meaningful' value from a groupby series.
    Handles list/array cells without raising ambiguous truth errors.
    """
    for v in series:
        if v is None:
            continue
        try:
            na = pd.isna(v)
            if isinstance(na, (bool, np.bool_)) and na:
                continue
        except Exception:
            pass

        if isinstance(v, (list, tuple, set)):
            return v if len(v) > 0 else None
        if isinstance(v, np.ndarray):
            return v if v.size > 0 else None
        return v
    return None


_SEED_CLASS_STRENGTHS = {
    "8mer": 0.55,
    "7mer-m8": 0.40,
    "7mer_a1": 0.35,
    "7mer-a1": 0.35,
    "6mer": 0.20,
}


def _seed_class_strength_from_text(value: Any) -> float:
    text = str(value or "").strip().lower().replace("_", "-")
    if not text:
        return 0.0
    for key, score in _SEED_CLASS_STRENGTHS.items():
        if key in text:
            return score
    return 0.0


def _best_seed_class_value(series: pd.Series):
    best_value = None
    best_score = -1.0
    for value in series:
        score = _seed_class_strength_from_text(value)
        if score > best_score:
            best_score = score
            best_value = value
    return best_value if best_value is not None else _first_nonnull_value(series)


def _seed_class_contrib_series(df: pd.DataFrame) -> pd.Series:
    if "best_seed_class" not in df.columns:
        return pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    return df["best_seed_class"].map(_seed_class_strength_from_text).fillna(0.0).astype(float)


def _compute_seed_contrib(df: pd.DataFrame) -> pd.Series:
    has_seed = 0.20 * _bool_col(df, "has_seed_features").astype(float)
    seed_class = _seed_class_contrib_series(df)
    total_sites = 0.25 * np.clip(_safe_float_col(df, "n_total_sites", default=0.0), 0.0, 6.0) / 6.0
    site_specific = (
        0.04 * np.clip(_safe_float_col(df, "n_sites_6mer", default=0.0), 0.0, 4.0)
        + 0.06 * np.clip(_safe_float_col(df, "n_sites_7mer_a1", default=0.0), 0.0, 4.0)
        + 0.07 * np.clip(_safe_float_col(df, "n_sites_7mer_m8", default=0.0), 0.0, 4.0)
        + 0.09 * np.clip(_safe_float_col(df, "n_sites_8mer", default=0.0), 0.0, 4.0)
    )
    site_specific = np.clip(site_specific, 0.0, 0.35)
    site_density = 0.20 * np.clip(_safe_float_col(df, "site_density_per_kb", default=0.0), 0.0, 3.0) / 3.0
    return (has_seed + seed_class + total_sites + site_specific + site_density).astype(float)


def _compute_rnahybrid_contrib(df: pd.DataFrame) -> pd.Series:
    has_rnahybrid = 0.15 * _bool_col(df, "has_rnahybrid").astype(float)
    n_sites = 0.35 * np.clip(_safe_float_col(df, "n_rnahybrid_sites", default=0.0), 0.0, 5.0) / 5.0
    mfe_strength = 0.35 * np.clip(_safe_float_col(df, "mfe_strength", default=0.0), 0.0, 5.0) / 5.0
    mean_top3_strength = 0.15 * np.clip(_safe_float_col(df, "mean_top3_mfe_strength", default=0.0), 0.0, 5.0) / 5.0
    best_mfe = _safe_float_col(df, "best_mfe", default=0.0)
    best_mfe_contrib = 0.20 * np.clip((-best_mfe - 10.0) / 20.0, 0.0, 1.0)
    low_mfe_sites = (
        0.10 * np.clip(_safe_float_col(df, "n_sites_mfe_lt_-20", default=0.0), 0.0, 4.0) / 4.0
        + 0.12 * np.clip(_safe_float_col(df, "n_sites_mfe_lt_-25", default=0.0), 0.0, 4.0) / 4.0
    )
    return (has_rnahybrid + n_sites + mfe_strength + mean_top3_strength + best_mfe_contrib + low_mfe_sites).astype(float)


def _compute_local_au_contrib(df: pd.DataFrame) -> pd.Series:
    best_local_au = 0.20 * np.clip(_safe_float_col(df, "best_local_au", default=0.0), 0.0, 1.0)
    best_local_au_by_mfe = 0.20 * np.clip(_safe_float_col(df, "best_local_au_by_mfe", default=0.0), 0.0, 1.0)
    return (best_local_au + best_local_au_by_mfe).astype(float)


def _compute_retrieval_components(
    df: pd.DataFrame,
    tcga: Optional[str],
    pathway_gene_set: set[str],
    pathway_gene_map: Dict[str, List[str]],
) -> Dict[str, pd.Series]:
    support = _safe_float_col(df, "support_count", default=0.0)

    ts_ctx = _safe_float_col(df, "ts_best_contextpp", default=0.0)
    ts_contrib = np.clip(-ts_ctx, 0, 2.0)

    clip_sum = _safe_float_col(df, "clip_exp_sum", default=0.0)
    clip_contrib = np.log1p(clip_sum) / 5.0

    mirdb_best = _safe_float_col(df, "mirdb_best_score", default=0.0)
    mirdb_contrib = mirdb_best / 100.0

    seed_contrib = _compute_seed_contrib(df)
    rnahybrid_contrib = _compute_rnahybrid_contrib(df)
    local_au_contrib = _compute_local_au_contrib(df)
    structure_contrib = (seed_contrib + rnahybrid_contrib + local_au_contrib).astype(float)

    tcga_rho_strength = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    tcga_support_flag = pd.Series(np.zeros(len(df), dtype=int), index=df.index)
    tcga_repression_flag = pd.Series(np.zeros(len(df), dtype=int), index=df.index)
    tcga_p = pd.Series(np.full(len(df), np.nan, dtype=float), index=df.index)

    if tcga:
        tcga = str(tcga).upper()
        rho_col = f"{tcga}_spearman_rho"
        p_col = f"{tcga}_spearman_p"
        rep_col = f"{tcga}_repression_evidence"

        if rho_col in df.columns:
            rho = _safe_float_col(df, rho_col, default=0.0)
            tcga_rho_strength = np.clip(-rho, 0, 1.0)

        if p_col in df.columns:
            tcga_p = _safe_float_col(df, p_col, default=np.nan)

        tcga_support_flag = _derive_tcga_support_flag(df, tcga)

        if rep_col in df.columns:
            tcga_repression_flag = _bool_col(df, rep_col)

    tcga_contrib = (
        (1.0 * tcga_rho_strength)
        + (0.8 * tcga_support_flag.astype(float))
        + (0.3 * tcga_repression_flag.astype(float))
    )

    pathway_bonus = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    pathway_match_count = df["gene_symbol"].astype(str).map(_normalize_gene_symbol).map(
        lambda gene: len(pathway_gene_map.get(gene, []))
    )
    pathway_selected_names = df["gene_symbol"].astype(str).map(_normalize_gene_symbol).map(
        lambda gene: pathway_gene_map.get(gene, [])
    )
    pathway_selected_gene = df["gene_symbol"].astype(str).map(_normalize_gene_symbol).isin(pathway_gene_set).astype(int)

    return {
        "support": support.astype(float),
        "ts_contrib": ts_contrib.astype(float),
        "clip_contrib": clip_contrib.astype(float),
        "mirdb_contrib": mirdb_contrib.astype(float),
        "seed_contrib": seed_contrib.astype(float),
        "rnahybrid_contrib": rnahybrid_contrib.astype(float),
        "local_au_contrib": local_au_contrib.astype(float),
        "structure_contrib": structure_contrib.astype(float),
        "tcga_contrib": tcga_contrib.astype(float),
        "tcga_rho_strength": tcga_rho_strength.astype(float),
        "tcga_support_flag": tcga_support_flag.astype(int),
        "tcga_repression_flag": tcga_repression_flag.astype(int),
        "tcga_p": tcga_p.astype(float),
        "pathway_bonus": pathway_bonus.astype(float),
        "pathway_selected_gene": pathway_selected_gene.astype(int),
        "pathway_match_count": pathway_match_count.astype(int),
        "pathway_selected_names": pathway_selected_names,
    }


def _collapse_pair_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse duplicates so that each (mirna_name, gene_symbol) appears once.
    """
    if df.empty:
        return df

    keys = ["mirna_name", "gene_symbol"]
    if not set(keys).issubset(df.columns):
        return df

    agg: Dict[str, Any] = {}

    # Normalized identifiers are needed for app/debug consistency after collapse.
    for c in ["mirna_name_norm", "gene_symbol_norm", "mirna_name_normalized", "gene_symbol_normalized"]:
        if c in df.columns:
            agg[c] = _first_nonnull_value

    # Common evidence fields
    for c in ["support_encori", "support_targetscan", "support_mirdb", "mirtarbase_pos", "label_mirtarbase"]:
        if c in df.columns:
            agg[c] = "max"
    if "support_count" in df.columns:
        agg["support_count"] = "max"

    for c in ["evidence_family_count", "overall_evidence_support_percentile"]:
        if c in df.columns:
            agg[c] = "max"
    if "evidence_family_summary_json" in df.columns:
        agg["evidence_family_summary_json"] = _first_nonnull_value

    for family in [
        "sequence_complementarity",
        "thermodynamic_stability",
        "sequence_conservation",
        "target_site_accessibility",
        "functional_binding",
        "functional_repression",
    ]:
        available_col = f"{family}_available"
        percentile_col = f"{family}_support_percentile"
        count_col = f"{family}_evidence_count"
        if available_col in df.columns:
            agg[available_col] = "max"
        if percentile_col in df.columns:
            agg[percentile_col] = "max"
        if count_col in df.columns:
            agg[count_col] = "max"

    # Support tcga (new evidence line)
    for col in df.columns:
        if col.endswith("_support_tcga") or col == "support_tcga_any":
            agg[col] = "max"

    # TargetScan
    for c in ["ts_best_contextpp", "targetscan_context_score", "targetscan_aggregate_context_score"]:
        if c in df.columns:
            agg[c] = "min"
    for c in [
        "ts_best_percentile",
        "targetscan_context_score_support_percentile",
        "targetscan_context_score_percentile",
        "targetscan_aggregate_context_score_percentile",
        "targetscan_pct",
        "targetscan_pct_percentile",
        "targetscan_branch_length_score",
        "targetscan_branch_length_score_percentile",
    ]:
        if c in df.columns:
            agg[c] = "max"
    for c in ["ts_n_sites", "targetscan_conserved_site", "has_targetscan_evidence"]:
        if c in df.columns:
            agg[c] = "max"
    if "ts_best_site" in df.columns:
        agg["ts_best_site"] = "min"

    # Seed/site and structure-aware raw fields
    for c in [
        "has_seed_features",
        "has_seed_evidence",
        "has_rnahybrid",
        "has_rnahybrid_evidence",
    ]:
        if c in df.columns:
            agg[c] = "max"
    for c in ["best_seed_class", "seed_match_type", "best_seed_site_type"]:
        if c in df.columns:
            agg[c] = _best_seed_class_value
    for c in [
        "n_total_sites",
        "n_seed_sites",
        "n_seed_sites_percentile",
        "n_sites_6mer",
        "n_sites_7mer_a1",
        "n_sites_7mer_m8",
        "n_sites_8mer",
        "site_density_per_kb",
        "n_rnahybrid_sites",
        "mfe_strength",
        "mean_top3_mfe_strength",
        "n_sites_mfe_lt_-20",
        "n_sites_mfe_lt_-25",
        "best_local_au",
        "best_local_au_by_mfe",
        "seed_pairing_score",
        "seed_pairing_score_percentile",
        "rnahybrid_strength",
        "rnahybrid_strength_percentile",
    ]:
        if c in df.columns:
            agg[c] = "max"
    for c in [
        "best_mfe",
        "mean_top3_mfe",
        "rnahybrid_mfe",
        "rnahybrid_mfe_best_site",
        "rnahybrid_seed_mfe",
    ]:
        if c in df.columns:
            agg[c] = "min"
    for c in [
        "rnahybrid_mfe_percentile",
        "rnahybrid_mfe_best_site_percentile",
        "rnahybrid_seed_mfe_percentile",
    ]:
        if c in df.columns:
            agg[c] = "max"

    # ENCORI
    for c in [
        "clip_exp_sum",
        "clip_exp_max",
        "n_clip_sites",
        "clip_any_support",
        "clip_max_score",
        "clip_max_score_percentile",
        "clip_n_experiments",
        "clip_n_experiments_percentile",
        "clip_n_cell_lines",
        "clip_n_cell_lines_percentile",
        "encori_clip_score",
        "encori_clip_score_percentile",
        "has_clip_evidence",
    ]:
        if c in df.columns:
            agg[c] = "max"

    # RNAplfold/accessibility
    for c in [
        "rnaplfold_best_seed_unpaired_prob",
        "rnaplfold_best_seed_unpaired_prob_percentile",
        "rnaplfold_mean_seed_unpaired_prob",
        "rnaplfold_mean_seed_unpaired_prob_percentile",
        "rnaplfold_best_site_unpaired_prob",
        "rnaplfold_best_site_unpaired_prob_percentile",
        "rnaplfold_mean_site_unpaired_prob",
        "rnaplfold_mean_site_unpaired_prob_percentile",
        "rnaplfold_best_flank_unpaired_prob",
        "rnaplfold_best_flank_unpaired_prob_percentile",
        "rnaplfold_mean_flank_unpaired_prob",
        "rnaplfold_mean_flank_unpaired_prob_percentile",
        "rnaplfold_n_sites_scored",
        "rnaplfold_n_sites_scored_percentile",
        "rnaplfold_n_accessible_sites",
        "rnaplfold_n_accessible_sites_percentile",
        "has_rnaplfold_evidence",
    ]:
        if c in df.columns:
            agg[c] = "max"

    # miRDB
    for c in ["mirdb_best_score", "mirdb_mean_score", "mirdb_n_transcripts"]:
        if c in df.columns:
            agg[c] = "max"

    # Pathway hits
    if "gene_pathway_hits" in df.columns:
        agg["gene_pathway_hits"] = "max"

    for c in [
        "mirassist_xgboost_score",
        "best_backend_model_score",
        "learned_score",
        "model_score",
        "learned_score_xgb_raw_v1",
        "learned_score_xgb_raw_nomissing_v1",
    ]:
        if c in df.columns:
            agg[c] = "max"
    for c in ["learned_score_model_version", "learned_score_feature_set", "learned_score_updated_at"]:
        if c in df.columns:
            agg[c] = _first_nonnull_value

    # TCGA columns (raw correlations + booleans)
    for col in df.columns:
        if col.endswith("_spearman_rho"):
            agg[col] = "min"  # most negative = strongest repression signal
        elif col.endswith("_spearman_p"):
            agg[col] = "min"
        elif col.endswith("_anticorrelated"):
            agg[col] = "max"  # boolean evidence
        elif col.endswith("_repression_evidence"):
            agg[col] = "max"  # boolean evidence
        elif col.endswith("_pair_expressed"):
            agg[col] = "max"
        elif col.endswith("_gene_expressed") or col.endswith("_mirna_expressed"):
            agg[col] = "max"
        elif col.endswith("_gene_expr_median") or col.endswith("_mirna_expr_median"):
            agg[col] = "max"
        elif col.endswith("_gene_present_frac") or col.endswith("_mirna_present_frac"):
            agg[col] = "max"
        elif col.endswith("_mrna_n_samples") or col.endswith("_mirna_n_samples"):
            agg[col] = "max"

    # List-like columns
    for c in [
        "cellline_tissue_set",
        "mirtarbase_pmids",
        "mirtarbase_experiments",
        "ts_gene_id_base",
        "entrez_ids",
    ]:
        if c in df.columns:
            agg[c] = _first_nonnull_value

    out = df.groupby(keys, as_index=False).agg(agg)
    return out


# =============================================================================
# TCGA derivation helpers
# =============================================================================

def _derive_tcga_support_flag(df: pd.DataFrame, tcga: str) -> pd.Series:
    """
    Returns int 0/1 aligned to df.index indicating TCGA anti-correlation support.

    Priority:
      1) {TCGA}_support_tcga (explicit evidence line)
      2) {TCGA}_anticorrelated (raw boolean)
      3) derive from rho/p: (rho < 0) & (p <= 0.05) if both exist
      4) else zeros
    """
    tcga = str(tcga).upper()

    support_col = f"{tcga}_support_tcga"
    if support_col in df.columns:
        return _bool_col(df, support_col)

    antic_col = f"{tcga}_anticorrelated"
    if antic_col in df.columns:
        return _bool_col(df, antic_col)

    rho_col = f"{tcga}_spearman_rho"
    p_col = f"{tcga}_spearman_p"
    if (rho_col in df.columns) and (p_col in df.columns):
        rho = _safe_float_col(df, rho_col, default=0.0)
        p = _safe_float_col(df, p_col, default=1.0)
        return ((rho < 0) & (p <= 0.05)).astype(int)

    return pd.Series(np.zeros(len(df), dtype=int), index=df.index)


# =============================================================================
# Main retrieval
# =============================================================================

def retrieve_candidates(
    ev: pd.DataFrame,
    query_token: str,
    cfg: RetrievalConfig,
) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    """
    Returns (shortlist_df, direction)
    """
    query_token_raw = str(query_token or "")
    query_token = _normalize_token(query_token_raw)
    direction = _direction_from_token(query_token)
    diagnostics: Dict[str, Any] = {
        "query_token": query_token_raw,
        "direction": direction,
        "query_mirna_raw": query_token_raw if direction == "mirna_to_targets" else None,
        "query_mirna_normalized": None,
        "query_gene_normalized": _normalize_gene_symbol(query_token) if direction == "gene_to_mirnas" else None,
        "n_evidence_rows_total": int(len(ev)),
        "n_after_novel_filter": int(len(ev)),
        "n_after_min_support": 0,
        "matched_mirna_names": [],
        "exact_mirna_variants_used": [],
        "mature_arm_expansion_attempted": False,
        "mature_arm_expansion_used": False,
        "mirna_prefix_fallback_attempted": False,
        "mirna_prefix_fallback_used": False,
        "n_after_query_filter": 0,
        "pathway_filter_enabled": bool((cfg.pathway_selection or {}).get("enabled")),
        "n_selected_pathway_genes": 0,
        "n_candidate_genes_before_pathway_filter": 0,
        "n_candidate_genes_in_selected_pathways": 0,
        "n_candidate_genes_removed_by_pathway_filter": 0,
        "example_candidate_genes_before_filter": [],
        "example_selected_pathway_genes": [],
        "example_remaining_genes_after_filter": [],
        "pathway_gene_normalization": "upper",
        "n_after_pathway_filter": 0,
        "n_after_collapse_duplicates": 0,
        "n_final_shortlist": 0,
        "retrieval_structure_in_score": bool(get_use_structure_in_score()),
        "retrieval_ranking_mode": "manual",
        "learned_score_column": get_learned_score_column(),
        "initial_min_support": int(cfg.min_support),
        "effective_min_support": int(cfg.min_support),
        "relaxed_min_support_reason": None,
        "n_rows_before_min_support": 0,
        "n_rows_after_min_support": 0,
        "warnings": [],
    }
    if direction == "mirna_to_targets":
        diagnostics["query_mirna_normalized"] = _normalize_mirna_query(query_token)[0]

    _ensure_cols(ev, ["mirna_name", "gene_symbol", "support_count"])
    df = ev.copy()

    matched_tokens: List[str] = []
    if direction == "mirna_to_targets":
        df, mirna_match_diagnostics = _filter_mirna_rows(df, query_token)
        diagnostics.update(mirna_match_diagnostics)
        matched_tokens = list(diagnostics.get("matched_mirna_names") or [])
        if df.empty:
            attempted = (
                diagnostics.get("primary_mirna_variants")
                or diagnostics.get("fallback_mirna_variants")
                or [diagnostics.get("query_mirna_normalized")]
            )
            diagnostics["warnings"].append(
                "No miRNA names in the evidence table matched the query after normalization. "
                f"Searched normalized forms: {', '.join([str(v) for v in attempted if v])}."
            )
            return df.head(0), direction, diagnostics
    else:
        gene_norm = _normalize_gene_symbol(query_token)
        df = df[df["gene_symbol"].astype(str).map(_normalize_gene_symbol) == gene_norm].copy()
        matched_tokens = [query_token]
    diagnostics["n_after_query_filter"] = int(len(df))

    if df.empty:
        if direction == "mirna_to_targets":
            diagnostics["warnings"].append("The miRNA query matched no rows after table filtering.")
        else:
            diagnostics["warnings"].append("The gene query matched no rows after table filtering.")
        return df.head(0), direction, diagnostics

    # --- Novel mode and optional soft gates after query narrowing ---
    if cfg.novel and cfg.use_mirtarbase_evidence and "mirtarbase_pos" in df.columns:
        df = df.loc[_bool_col(df, "mirtarbase_pos") == 0].copy()
    diagnostics["n_after_novel_filter"] = int(len(df))

    if cfg.require_binding_evidence:
        binding_mask = (
            (_bool_col(df, "support_targetscan") == 1)
            | (_bool_col(df, "support_encori") == 1)
            | (_bool_col(df, "support_mirdb") == 1)
        )
        df = df.loc[binding_mask].copy()

    if cfg.require_expression and cfg.tcga:
        pair_expr = f"{str(cfg.tcga).upper()}_pair_expressed"
        if pair_expr in df.columns:
            df = df.loc[_bool_col(df, pair_expr) == 1].copy()

    if df.empty:
        diagnostics["warnings"].append("No evidence rows remained after novelty/binding/expression filters.")
        return df.head(0), direction, diagnostics

    df = _apply_min_support_with_relaxation(df, cfg, diagnostics)
    if df.empty:
        searched_form = diagnostics.get("query_mirna_normalized") if direction == "mirna_to_targets" else diagnostics.get("query_gene_normalized")
        matched_names = diagnostics.get("matched_mirna_names") or []
        detail = ""
        if matched_names:
            detail = " Matching rows were found for " + "/".join([str(name) for name in matched_names[:4]]) + ", but relaxed filtering may be needed."
        explanation = (
            f"No candidates passed min_support >= {diagnostics.get('initial_min_support', cfg.min_support)} for {searched_form or query_token}. "
            f"Rows found before min_support: {diagnostics.get('n_rows_before_min_support', 0)}.{detail}"
        )
        diagnostics["no_candidates_explanation"] = explanation
        diagnostics["warnings"].append(explanation)
        return df.head(0), direction, diagnostics

    # --- Collapse duplicates so each (miRNA,gene) appears once ---
    if cfg.collapse_duplicates:
        df = _collapse_pair_rows(df)
        if df.empty:
            diagnostics["warnings"].append("Rows were found, but none remained after duplicate collapse.")
            return df.head(0), direction, diagnostics
    diagnostics["n_after_collapse_duplicates"] = int(len(df))

    if direction == "mirna_to_targets":
        arm_note = diagnostics.get("arm_interpretation_note")
        if arm_note:
            diagnostics.setdefault("user_notes", []).append(str(arm_note))

    # --- Pathway filtering (strict, filter-only) ---
    pathway_selection = cfg.pathway_selection or {}
    pathway_enabled = bool(pathway_selection.get("enabled"))
    pathway_gene_map = {
        _normalize_gene_symbol(gene): list(names or [])
        for gene, names in (cfg.pathway_gene_map or {}).items()
    }
    pathway_gene_set = {_normalize_gene_symbol(gene) for gene in (cfg.pathway_gene_set or set())}
    diagnostics["n_selected_pathway_genes"] = int(len(pathway_gene_set))
    diagnostics["example_selected_pathway_genes"] = sorted(pathway_gene_set)[:20]

    if pathway_enabled:
        if not pathway_gene_set:
            diagnostics["warnings"].append("Pathway filtering was enabled, but no genes were resolved from the selected pathways.")
            return df.head(0), direction, diagnostics

        candidate_genes_before = sorted({_normalize_gene_symbol(gene) for gene in df["gene_symbol"].tolist() if str(gene or "").strip()})
        remaining_genes = [gene for gene in candidate_genes_before if gene in pathway_gene_set]
        diagnostics["n_candidate_genes_before_pathway_filter"] = int(len(candidate_genes_before))
        diagnostics["n_candidate_genes_in_selected_pathways"] = int(len(remaining_genes))
        diagnostics["n_candidate_genes_removed_by_pathway_filter"] = int(len(candidate_genes_before) - len(remaining_genes))
        diagnostics["example_candidate_genes_before_filter"] = candidate_genes_before[:20]

        df = df[df["gene_symbol"].astype(str).map(_normalize_gene_symbol).isin(pathway_gene_set)].copy()
        diagnostics["example_remaining_genes_after_filter"] = sorted(
            {_normalize_gene_symbol(gene) for gene in df["gene_symbol"].tolist() if str(gene or "").strip()}
        )[:20]
        diagnostics["n_after_pathway_filter"] = int(len(df))
        if df.empty:
            diagnostics["warnings"].append("Pathway filtering removed all candidates.")
            return df.head(0), direction, diagnostics
    else:
        diagnostics["n_after_pathway_filter"] = int(len(df))
        diagnostics["example_remaining_genes_after_filter"] = sorted(
            {_normalize_gene_symbol(gene) for gene in df["gene_symbol"].tolist() if str(gene or "").strip()}
        )[:20]

    components = _compute_retrieval_components(
        df,
        cfg.tcga,
        pathway_gene_set,
        pathway_gene_map,
    )
    structure_in_score = get_use_structure_in_score()

    # --- Final score ---
    score = (
        1.0 * components["support"]
        + 1.0 * components["ts_contrib"]
        + 0.7 * components["clip_contrib"]
        + 0.7 * components["mirdb_contrib"]
        + 0.9 * components["tcga_contrib"]
        + 0.6 * components["pathway_bonus"]
        + (0.7 * components["structure_contrib"] if structure_in_score else 0.0)
    )

    df = df.assign(
        retrieval_score=score,
        retrieval_support=components["support"],
        retrieval_ts_contrib=components["ts_contrib"],
        retrieval_clip_contrib=components["clip_contrib"],
        retrieval_mirdb_contrib=components["mirdb_contrib"],
        retrieval_seed_contrib=components["seed_contrib"],
        retrieval_rnahybrid_contrib=components["rnahybrid_contrib"],
        retrieval_local_au_contrib=components["local_au_contrib"],
        retrieval_structure_contrib=components["structure_contrib"],
        retrieval_structure_in_score=int(structure_in_score),
        retrieval_tcga_contrib=components["tcga_contrib"],
        retrieval_tcga_rho_strength=components["tcga_rho_strength"],
        retrieval_tcga_support_flag=components["tcga_support_flag"],
        retrieval_tcga_repression_flag=components["tcga_repression_flag"],
        retrieval_tcga_p=components["tcga_p"],
        retrieval_pathway_bonus=components["pathway_bonus"],
        matched_query_tokens=";".join(matched_tokens),
        pathway_selected_gene=components["pathway_selected_gene"],
        pathway_match_count=components["pathway_match_count"],
        pathway_selected_names=components["pathway_selected_names"],
    )

    df, ranking_info = apply_learned_score_ranking(
        df,
        learned_score_column=get_learned_score_column(),
        enabled=bool(get_use_learned_score() or _resolve_score_column(df, get_learned_score_column())),
    )
    diagnostics.update({k: v for k, v in ranking_info.items() if k != "warnings"})
    for warning in ranking_info.get("warnings", []):
        diagnostics["warnings"].append(warning)

    df = df.head(int(cfg.k_shortlist)).reset_index(drop=True)
    df = df.drop(
        columns=[
            col
            for col in [
                "_retrieval_support_tiebreak",
                "_retrieval_mirdb_tiebreak",
                "_retrieval_ts_tiebreak",
                "_retrieval_clip_tiebreak",
                "_retrieval_mfe_tiebreak",
            ]
            if col in df.columns
        ],
        errors="ignore",
    )
    diagnostics["n_final_shortlist"] = int(len(df))

    return df, direction, diagnostics


def retrieve_from_queryspec(
    ev: Optional[pd.DataFrame],
    queryspec: Dict[str, Any],
    pathway_selection: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    """
    QuerySpec wrapper used by the direct app:
      queryspec -> RetrievalConfig -> retrieve_candidates
    """
    token = queryspec.get("mirna") or queryspec.get("gene") or queryspec.get("query_token")
    if not token:
        raise ValueError("QuerySpec missing 'mirna'/'gene'/'query_token'.")

    # tcga can be nested (new schema) or top-level (old schema)
    tcga = None
    if isinstance(queryspec.get("cancer"), dict):
        tcga = queryspec["cancer"].get("tcga")
    else:
        tcga = queryspec.get("tcga")

    filters = queryspec.get("filters") or {}
    original_question = str(queryspec.get("original_question") or queryspec.get("question") or "")
    original_question_l = original_question.lower()
    user_requested_strict_support = any(
        phrase in original_question_l
        for phrase in [
            "high confidence",
            "high-confidence",
            "multiple evidence",
            "multiple lines of evidence",
            "at least 2",
            "minimum support 2",
            "min support 2",
        ]
    )
    pathway_filter_cfg = queryspec.get("pathway_filter") or {}
    simple_broad_mirna_query = bool(
        (queryspec.get("mode") == "mirna_to_targets")
        and not tcga
        and not bool(pathway_filter_cfg.get("enabled"))
        and not bool(filters.get("require_binding_evidence", False))
        and not bool(filters.get("require_expression", False))
    )
    allow_min_support_relaxation = bool(
        simple_broad_mirna_query
        and get_use_learned_score()
        and int(filters.get("min_support", 1)) <= 2
        and not user_requested_strict_support
    )
    pathway_selection = pathway_selection or queryspec.get("pathway_selection") or {}
    pathway_gene_map = (
        pathway_selection.get("_selected_gene_pathways")
        or pathway_selection.get("selected_gene_pathways")
        or {}
    )
    pathway_gene_set = (
        pathway_selection.get("_selected_gene_set")
        or set(pathway_selection.get("selected_genes") or [])
    )

    cfg = RetrievalConfig(
        k_shortlist=int(queryspec.get("k", get_default_k())),
        min_support=int(filters.get("min_support", 1)),
        novel=bool(queryspec.get("novel", False)),
        tcga=(str(tcga).upper() if tcga else None),
        phenotype_keywords=queryspec.get("phenotype_keywords") or [],
        pathway_keywords=queryspec.get("pathway_keywords") or [],
        pathway_filter=queryspec.get("pathway_filter") or None,
        pathway_selection=pathway_selection,
        pathway_gene_set={_normalize_gene_symbol(gene) for gene in pathway_gene_set},
        pathway_gene_map={
            _normalize_gene_symbol(gene): list(names or [])
            for gene, names in pathway_gene_map.items()
        },
        require_binding_evidence=bool(filters.get("require_binding_evidence", False)),
        require_expression=bool(filters.get("require_expression", False)),
        collapse_duplicates=True,
        use_mirtarbase_evidence=use_mirtarbase_evidence(),
        allow_min_support_relaxation=allow_min_support_relaxation,
        user_requested_strict_support=user_requested_strict_support,
    )

    fetch_diagnostics: Dict[str, Any] = {"evidence_backend": get_evidence_backend()}
    if get_evidence_backend() == "postgres":
        ev, fetch_diagnostics = _fetch_postgres_candidate_pool(str(token), cfg)
    elif ev is None:
        ev = load_evidence()

    shortlist_df, direction, diagnostics = retrieve_candidates(ev, str(token), cfg)
    diagnostics.update(
        {
            "evidence_backend": fetch_diagnostics.get("evidence_backend", diagnostics.get("evidence_backend")),
            "db_candidate_limit": fetch_diagnostics.get("db_candidate_limit"),
            "n_rows_fetched_from_db": fetch_diagnostics.get("n_rows_fetched_from_db"),
            "sql_selected_column_count": fetch_diagnostics.get("sql_selected_column_count"),
            "sql_returned_column_count": fetch_diagnostics.get("sql_returned_column_count"),
            "supabase_table_name": fetch_diagnostics.get("supabase_table_name") or get_evidence_table(),
            "sort_column_used": diagnostics.get("score_column_used")
            or fetch_diagnostics.get("sort_column_used")
            or fetch_diagnostics.get("learned_score_column"),
            "learned_score_column": fetch_diagnostics.get("learned_score_column", diagnostics.get("learned_score_column")),
        }
    )
    if fetch_diagnostics.get("sql_mirna_norm_column"):
        diagnostics["sql_mirna_norm_column"] = fetch_diagnostics["sql_mirna_norm_column"]
    if fetch_diagnostics.get("query_direction"):
        diagnostics["query_direction"] = fetch_diagnostics["query_direction"]
    if fetch_diagnostics.get("sql_order_columns"):
        diagnostics["sql_order_columns"] = fetch_diagnostics["sql_order_columns"]
    return shortlist_df, direction, diagnostics
