from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import os
import re

import numpy as np
import pandas as pd

from backend.config import (
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
      - Designed for single-process Colab/uvicorn usage.
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


def _normalize_mirna_query(user_mirna: str) -> Tuple[str, Optional[str]]:
    """
    Returns (base, arm) where base is normalized like:
      "mir-21", "mir-17-5", "let-7a", etc. (lowercase, hyphen-delimited)
    arm is "3p"/"5p" if explicitly provided by user, else None.
    """
    s = (user_mirna or "").strip()
    if not s:
        return "", None

    # normalize weird hyphens/underscores/spaces
    s = s.replace("_", "-").replace("-", "-")
    s = re.sub(r"\s+", "", s)

    # strip "microRNA" textual prefix if present
    s = _MICRORNA_PREFIX_RE.sub("mir-", s)
    s = _MIRNA_WORD_PREFIX_RE.sub("mir-", s)

    # strip species prefix (hsa-/mmu- etc)
    s = _strip_species_prefix(s)

    s = s.lower()

    arm = None
    m = _ARM_RE.search(s)
    if m:
        arm = m.group(1).lower()
        s = _ARM_RE.sub("", s)

    # ensure mir- / let- delimiter
    s = re.sub(r"^(mir|let)(?=[0-9a-z])", r"\1-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s, arm


def _normalize_mirna_table_value(v: str) -> Tuple[str, Optional[str]]:
    """
    Normalize a table miRNA string into (base, arm) similar to query normalization.
    Returns ("", None) if unusable.
    """
    if v is None:
        return "", None
    s = str(v).strip()
    if not s:
        return "", None
    s = s.replace("_", "-").replace("-", "-")
    s = re.sub(r"\s+", "", s)
    s = _MICRORNA_PREFIX_RE.sub("mir-", s)
    s = _MIRNA_WORD_PREFIX_RE.sub("mir-", s)
    s = _strip_species_prefix(s)
    s = s.lower()

    arm = None
    m = _ARM_RE.search(s)
    if m:
        arm = m.group(1).lower()
        s = _ARM_RE.sub("", s)

    # "mir21" -> "mir-21", "let7a" -> "let-7a"
    s = re.sub(r"^(mir|let)(?=[0-9a-z])", r"\1-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")

    # Some tables contain "mir" alone, ignore that
    if s in ("mir", "let"):
        return "", None

    return s, arm


def resolve_mirna_names_for_table(user_mirna: str, mirna_series: pd.Series) -> List[str]:
    """
    Map user query -> EXACT values present in the table, but matched via normalization.
    If no arm specified: prefer 5p, then base-only, then 3p.
    """
    q_base, q_arm = _normalize_mirna_query(user_mirna)
    if not q_base:
        return []

    vals = mirna_series.dropna().astype(str)
    if vals.empty:
        return []

    # Build normalization map once for unique names
    uniq = vals.unique().tolist()
    norm_map: Dict[Tuple[str, Optional[str]], List[str]] = {}
    for raw in uniq:
        b, a = _normalize_mirna_table_value(raw)
        if not b:
            continue
        norm_map.setdefault((b, a), []).append(raw)

    def hits_for(base: str, arm: Optional[str]) -> List[str]:
        out = []
        if (base, arm) in norm_map:
            out.extend(norm_map[(base, arm)])
        return out

    # if explicit arm, try that first, then base-only
    if q_arm in ("3p", "5p"):
        h = hits_for(q_base, q_arm)
        if h:
            return sorted(set(h))
        h = hits_for(q_base, None)
        return sorted(set(h))

    # no arm: match available arm-specific and base-only entries for this family
    combined: List[str] = []
    for arm_try in ("5p", None, "3p"):
        combined.extend(hits_for(q_base, arm_try))
    return sorted(set(combined))


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

    # Common evidence fields
    for c in ["support_encori", "support_targetscan", "support_mirdb", "mirtarbase_pos", "label_mirtarbase"]:
        if c in df.columns:
            agg[c] = "max"
    if "support_count" in df.columns:
        agg["support_count"] = "max"

    # Support tcga (new evidence line)
    for col in df.columns:
        if col.endswith("_support_tcga") or col == "support_tcga_any":
            agg[col] = "max"

    # TargetScan
    if "ts_best_contextpp" in df.columns:
        agg["ts_best_contextpp"] = "min"
    if "ts_best_percentile" in df.columns:
        agg["ts_best_percentile"] = "max"
    if "ts_n_sites" in df.columns:
        agg["ts_n_sites"] = "max"
    if "ts_best_site" in df.columns:
        agg["ts_best_site"] = "min"

    # Seed/site and structure-aware raw fields
    for c in [
        "has_seed_features",
        "has_rnahybrid",
    ]:
        if c in df.columns:
            agg[c] = "max"
    if "best_seed_class" in df.columns:
        agg["best_seed_class"] = _best_seed_class_value
    for c in [
        "n_total_sites",
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
    ]:
        if c in df.columns:
            agg[c] = "max"
    for c in ["best_mfe", "mean_top3_mfe"]:
        if c in df.columns:
            agg[c] = "min"

    # ENCORI
    for c in ["clip_exp_sum", "clip_exp_max", "n_clip_sites"]:
        if c in df.columns:
            agg[c] = "max"

    # miRDB
    for c in ["mirdb_best_score", "mirdb_mean_score", "mirdb_n_transcripts"]:
        if c in df.columns:
            agg[c] = "max"

    # Pathway hits
    if "gene_pathway_hits" in df.columns:
        agg["gene_pathway_hits"] = "max"

    for c in ["learned_score_xgb_raw_v1", "learned_score_xgb_raw_nomissing_v1"]:
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
        "n_after_min_support": int(len(ev)),
        "matched_mirna_names": [],
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
        "warnings": [],
    }
    if direction == "mirna_to_targets":
        diagnostics["query_mirna_normalized"] = _normalize_mirna_query(query_token)[0]

    _ensure_cols(ev, ["mirna_name", "gene_symbol", "support_count"])
    df = ev

    # Build a single mask aligned to df.index
    mask = pd.Series(True, index=df.index)

    # --- Novel mode (only hard exclude by design) ---
    if cfg.novel and cfg.use_mirtarbase_evidence and "mirtarbase_pos" in df.columns:
        mask &= (_bool_col(df, "mirtarbase_pos") == 0)
    diagnostics["n_after_novel_filter"] = int(mask.sum())

    # --- Minimal pre-filter ---
    if cfg.min_support > 0:
        sc = _safe_int_col(df, "support_count", default=0)
        mask &= (sc >= int(cfg.min_support))
    diagnostics["n_after_min_support"] = int(mask.sum())

    # --- Optional soft gates ---
    if cfg.require_binding_evidence:
        mask &= (
            (_bool_col(df, "support_targetscan") == 1)
            | (_bool_col(df, "support_encori") == 1)
            | (_bool_col(df, "support_mirdb") == 1)
        )

    if cfg.require_expression and cfg.tcga:
        pair_expr = f"{str(cfg.tcga).upper()}_pair_expressed"
        if pair_expr in df.columns:
            mask &= (_bool_col(df, pair_expr) == 1)

    df = df.loc[mask].copy()
    if df.empty:
        diagnostics["warnings"].append("No evidence rows remained after novelty/support/expression filters.")
        return df.head(0), direction, diagnostics

    # --- Restrict by direction ---
    matched_tokens: List[str] = []
    if direction == "mirna_to_targets":
        allowed = resolve_mirna_names_for_table(query_token, df["mirna_name"])
        matched_tokens = allowed
        diagnostics["matched_mirna_names"] = allowed[:20]
        if not allowed:
            diagnostics["warnings"].append("No miRNA names in the evidence table matched the query after normalization.")
            return df.head(0), direction, diagnostics
        allowed_l = {a.lower() for a in allowed}
        df = df[df["mirna_name"].astype(str).str.lower().isin(allowed_l)].copy()
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

    # --- Collapse duplicates so each (miRNA,gene) appears once ---
    if cfg.collapse_duplicates:
        df = _collapse_pair_rows(df)
        if df.empty:
            diagnostics["warnings"].append("Rows were found, but none remained after duplicate collapse.")
            return df.head(0), direction, diagnostics
    diagnostics["n_after_collapse_duplicates"] = int(len(df))

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

    learned_score_column = get_learned_score_column()
    learned_score_enabled = bool(get_use_learned_score() and learned_score_column in df.columns)
    if learned_score_enabled:
        learned_score_values = pd.to_numeric(df[learned_score_column], errors="coerce")
        learned_score_missing = learned_score_values.isna().astype(int)
        learned_score_rank = learned_score_values.where(~learned_score_values.isna(), df["retrieval_score"])
        mirdb_tiebreak = _safe_float_col(df, "mirdb_best_score", default=0.0)
        ts_tiebreak = df["retrieval_ts_contrib"] if "retrieval_ts_contrib" in df.columns else np.clip(
            -_safe_float_col(df, "ts_best_contextpp", default=0.0),
            0.0,
            2.0,
        )
        df = df.assign(
            _learned_score_rank=learned_score_rank.astype(float),
            _learned_score_missing=learned_score_missing.astype(int),
            _learned_score_tiebreak_mirdb=mirdb_tiebreak.astype(float),
            _learned_score_tiebreak_ts=ts_tiebreak.astype(float),
        )
        df = df.sort_values(
            [
                "_learned_score_missing",
                "_learned_score_rank",
                "retrieval_score",
                "support_count",
                "_learned_score_tiebreak_mirdb",
                "_learned_score_tiebreak_ts",
            ],
            ascending=[True, False, False, False, False, False],
        )
        diagnostics["retrieval_ranking_mode"] = f"learned:{learned_score_column}"
        diagnostics["learned_score_enabled"] = True
    else:
        df = df.sort_values("retrieval_score", ascending=False)
        diagnostics["learned_score_enabled"] = False

    df = df.head(int(cfg.k_shortlist)).reset_index(drop=True)
    df = df.drop(
        columns=[
            col
            for col in [
                "_learned_score_rank",
                "_learned_score_missing",
                "_learned_score_tiebreak_mirdb",
                "_learned_score_tiebreak_ts",
            ]
            if col in df.columns
        ],
        errors="ignore",
    )
    diagnostics["n_final_shortlist"] = int(len(df))

    return df, direction, diagnostics


def retrieve_from_queryspec(
    ev: pd.DataFrame,
    queryspec: Dict[str, Any],
    pathway_selection: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    """
    Wrapper expected by backend/app.py:
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
    )

    return retrieve_candidates(ev, str(token), cfg)
