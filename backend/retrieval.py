from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import os
import re

import numpy as np
import pandas as pd

from backend.config import get_evidence_backend, get_evidence_table, resolve_evidence_path
from backend.db import get_database_engine, quote_identifier


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
_MICRORNA_PREFIX_RE = re.compile(r"(?i)^microrna[\s_-]*")  # includes non-ascii hyphen
_MIR_PREFIX_RE = re.compile(r"(?i)^(mir|miR|let|Let)")


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
    s = _MICRORNA_PREFIX_RE.sub("", s)

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
    s = _MICRORNA_PREFIX_RE.sub("", s)
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

    # no arm: prefer 5p, then base-only, then 3p
    for arm_try in ("5p", None, "3p"):
        h = hits_for(q_base, arm_try)
        if h:
            return sorted(set(h))

    return []


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
    for c in ["support_encori", "support_targetscan", "support_mirdb", "mirtarbase_pos"]:
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
) -> Tuple[pd.DataFrame, str]:
    """
    Returns (shortlist_df, direction)
    """
    query_token = _normalize_token(query_token)
    direction = _direction_from_token(query_token)

    _ensure_cols(ev, ["mirna_name", "gene_symbol", "support_count"])
    df = ev

    # Build a single mask aligned to df.index
    mask = pd.Series(True, index=df.index)

    # --- Novel mode (only hard exclude by design) ---
    if cfg.novel and "mirtarbase_pos" in df.columns:
        mask &= (_bool_col(df, "mirtarbase_pos") == 0)

    # --- Minimal pre-filter ---
    if cfg.min_support > 0:
        sc = _safe_int_col(df, "support_count", default=0)
        mask &= (sc >= int(cfg.min_support))

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
        return df.head(0), direction

    # --- Restrict by direction ---
    matched_tokens: List[str] = []
    if direction == "mirna_to_targets":
        allowed = resolve_mirna_names_for_table(query_token, df["mirna_name"])
        matched_tokens = allowed
        if not allowed:
            return df.head(0), direction
        allowed_l = {a.lower() for a in allowed}
        df = df[df["mirna_name"].astype(str).str.lower().isin(allowed_l)].copy()
    else:
        df = df[df["gene_symbol"].astype(str).str.upper() == query_token.upper()].copy()
        matched_tokens = [query_token]

    if df.empty:
        return df.head(0), direction

    # --- Collapse duplicates so each (miRNA,gene) appears once ---
    if cfg.collapse_duplicates:
        df = _collapse_pair_rows(df)
        if df.empty:
            return df.head(0), direction

    # --- Base scoring components ---
    support = _safe_float_col(df, "support_count", default=0.0)

    ts_ctx = _safe_float_col(df, "ts_best_contextpp", default=0.0)
    ts_contrib = np.clip(-ts_ctx, 0, 2.0)  # more negative -> higher contrib

    clip_sum = _safe_float_col(df, "clip_exp_sum", default=0.0)
    clip_contrib = np.log1p(clip_sum) / 5.0

    mirdb_best = _safe_float_col(df, "mirdb_best_score", default=0.0)
    mirdb_contrib = (mirdb_best / 100.0)

    # --- TCGA: treat anti-correlation as its own evidence line ---
    tcga_rho_strength = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    tcga_support_flag = pd.Series(np.zeros(len(df), dtype=int), index=df.index)
    tcga_repression_flag = pd.Series(np.zeros(len(df), dtype=int), index=df.index)
    tcga_p = pd.Series(np.full(len(df), np.nan, dtype=float), index=df.index)

    if cfg.tcga:
        tcga = str(cfg.tcga).upper()
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

    # Combine TCGA contributions:
    # - rho strength (continuous)
    # - tcga_support_flag (binary evidence line)
    # - repression evidence (optional extra flag)
    tcga_contrib = (
        (1.0 * tcga_rho_strength)
        + (0.8 * tcga_support_flag.astype(float))
        + (0.3 * tcga_repression_flag.astype(float))
    )

    # --- Pathway filtering (strict, filter-only) ---
    pathway_bonus = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    pathway_selection = cfg.pathway_selection or {}
    pathway_enabled = bool(pathway_selection.get("enabled"))
    pathway_gene_map = cfg.pathway_gene_map or {}
    pathway_gene_set = {str(gene).upper() for gene in (cfg.pathway_gene_set or set())}

    if pathway_enabled:
        if not pathway_gene_set:
            return df.head(0), direction

        df = df[df["gene_symbol"].astype(str).str.upper().isin(pathway_gene_set)].copy()
        if df.empty:
            return df.head(0), direction

        support = _safe_float_col(df, "support_count", default=0.0)
        ts_ctx = _safe_float_col(df, "ts_best_contextpp", default=0.0)
        ts_contrib = np.clip(-ts_ctx, 0, 2.0)
        clip_sum = _safe_float_col(df, "clip_exp_sum", default=0.0)
        clip_contrib = np.log1p(clip_sum) / 5.0
        mirdb_best = _safe_float_col(df, "mirdb_best_score", default=0.0)
        mirdb_contrib = (mirdb_best / 100.0)

        tcga_rho_strength = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
        tcga_support_flag = pd.Series(np.zeros(len(df), dtype=int), index=df.index)
        tcga_repression_flag = pd.Series(np.zeros(len(df), dtype=int), index=df.index)
        tcga_p = pd.Series(np.full(len(df), np.nan, dtype=float), index=df.index)

        if cfg.tcga:
            tcga = str(cfg.tcga).upper()
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

    # --- Final score ---
    score = (
        1.0 * support
        + 1.0 * ts_contrib
        + 0.7 * clip_contrib
        + 0.7 * mirdb_contrib
        + 0.9 * tcga_contrib
        + 0.6 * pathway_bonus
    )

    df = df.assign(
        retrieval_score=score,
        retrieval_support=support,
        retrieval_ts_contrib=ts_contrib,
        retrieval_clip_contrib=clip_contrib,
        retrieval_mirdb_contrib=mirdb_contrib,
        retrieval_tcga_contrib=tcga_contrib,
        retrieval_tcga_rho_strength=tcga_rho_strength,
        retrieval_tcga_support_flag=tcga_support_flag,
        retrieval_tcga_repression_flag=tcga_repression_flag,
        retrieval_tcga_p=tcga_p,
        retrieval_pathway_bonus=pathway_bonus,
        matched_query_tokens=";".join(matched_tokens),
        pathway_selected_gene=df["gene_symbol"].astype(str).str.upper().isin(pathway_gene_set).astype(int),
        pathway_match_count=df["gene_symbol"].astype(str).str.upper().map(
            lambda gene: len(pathway_gene_map.get(gene, []))
        ),
        pathway_selected_names=df["gene_symbol"].astype(str).str.upper().map(
            lambda gene: pathway_gene_map.get(gene, [])
        ),
    )

    df = df.sort_values("retrieval_score", ascending=False)
    df = df.head(int(cfg.k_shortlist)).reset_index(drop=True)

    return df, direction


def retrieve_from_queryspec(
    ev: pd.DataFrame,
    queryspec: Dict[str, Any],
    pathway_selection: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, str]:
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
    pathway_gene_map = pathway_selection.get("selected_gene_pathways") or {}
    pathway_gene_set = set(pathway_selection.get("selected_genes") or [])

    cfg = RetrievalConfig(
        k_shortlist=int(queryspec.get("k", 200)),
        min_support=int(filters.get("min_support", 1)),
        novel=bool(queryspec.get("novel", False)),
        tcga=(str(tcga).upper() if tcga else None),
        phenotype_keywords=queryspec.get("phenotype_keywords") or [],
        pathway_keywords=queryspec.get("pathway_keywords") or [],
        pathway_filter=queryspec.get("pathway_filter") or None,
        pathway_selection=pathway_selection,
        pathway_gene_set={str(gene).upper() for gene in pathway_gene_set},
        pathway_gene_map={
            str(gene).upper(): list(names or [])
            for gene, names in pathway_gene_map.items()
        },
        require_binding_evidence=bool(filters.get("require_binding_evidence", False)),
        require_expression=bool(filters.get("require_expression", False)),
        collapse_duplicates=True,
    )

    return retrieve_candidates(ev, str(token), cfg)
