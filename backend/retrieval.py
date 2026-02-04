from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Any

import re
import numpy as np
import pandas as pd


@dataclass
class RetrievalConfig:
    # Core
    k_shortlist: int = 200
    min_support: int = 1
    novel: bool = False

    # Context
    tcga: Optional[str] = None  # e.g., "BRCA"
    keywords: Optional[List[str]] = None  # generic soft keywords (optional)

    # New: phenotype + pathway conditioning
    phenotype_keywords: Optional[List[str]] = None
    pathway_keywords: Optional[List[str]] = None
    pathway_filter: Optional[Dict[str, Any]] = None  # {"enabled":bool,"mode":"boost|filter","min_gene_sets":int}

    # Optional "soft gates" (keep as False by default)
    require_binding_evidence: bool = False
    require_expression: bool = False


# ----------------------------
# Helpers: safety + columns
# ----------------------------
def _normalize_token(x: str) -> str:
    return str(x).strip()


def _ensure_cols(ev: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in ev.columns]
    if missing:
        raise ValueError(f"Evidence table missing columns: {missing[:15]}")


def _bool_col(ev: pd.DataFrame, col: str) -> pd.Series:
    if col not in ev.columns:
        return pd.Series(np.zeros(len(ev), dtype=int), index=ev.index)
    return ev[col].fillna(0).astype(int)


def _safe_float_col(ev: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in ev.columns:
        return pd.Series(np.full(len(ev), default, dtype=float), index=ev.index)
    return pd.to_numeric(ev[col], errors="coerce").fillna(default).astype(float)


# ----------------------------
# miRNA matching (NO substring!)
# ----------------------------
_ARM_RE = re.compile(r"(?i)(?:^|[-_])(3p|5p)$")
_SPECIES_PREFIX_RE = re.compile(r"(?i)^(hsa|mmu|rno|dme|cel|ath)[-_]")


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

    s = s.replace("_", "-")
    s = re.sub(r"\s+", "", s)

    s = _strip_species_prefix(s)
    s = s.lower()

    # capture explicit arm at end
    arm = None
    m = _ARM_RE.search(s)
    if m:
        arm = m.group(1).lower()
        s = _ARM_RE.sub("", s)

    # Ensure mir/let have hyphen if missing (mir21 -> mir-21)
    s = re.sub(r"^(mir|let)(?=[0-9a-z])", r"\1-", s)

    # collapse double hyphens
    s = re.sub(r"-{2,}", "-", s).strip("-")

    return s, arm


def resolve_mirna_names_for_table(user_mirna: str, mirna_series: pd.Series) -> List[str]:
    """
    Map user query (miR-21, hsa-miR-21, miR-21-3p) -> exact table tokens to match.
    Defaults:
      - if no arm specified: prefer 5p
    Safety:
      - EXACT match only (prevents miR-21 matching miR-214)
    Fallback if preferred token not in table:
      1) base-only exact
      2) other arm exact (3p)
    """
    base, arm = _normalize_mirna_query(user_mirna)
    if not base:
        return []

    # Table normalization for matching
    vals = mirna_series.dropna().astype(str)
    vals_lower = vals.str.lower()

    # Table uses hsa- prefix typically, but allow missing prefix if any.
    prefixes = ["hsa-", ""]  # keep minimal and deterministic

    def cand(arm_: Optional[str]) -> List[str]:
        out: List[str] = []
        for pref in prefixes:
            if arm_:
                out.append(f"{pref}{base}-{arm_}")
            else:
                out.append(f"{pref}{base}")
        return out

    # If arm specified: try exact arm then base-only
    if arm in ("3p", "5p"):
        primary = cand(arm)
        hits = vals[vals_lower.isin([x.lower() for x in primary])].unique().tolist()
        if hits:
            return hits

        base_only = cand(None)
        hits = vals[vals_lower.isin([x.lower() for x in base_only])].unique().tolist()
        return hits

    # No arm: default to 5p
    primary = cand("5p")
    hits = vals[vals_lower.isin([x.lower() for x in primary])].unique().tolist()
    if hits:
        return hits

    # fallback: base-only exact
    base_only = cand(None)
    hits = vals[vals_lower.isin([x.lower() for x in base_only])].unique().tolist()
    if hits:
        return hits

    # fallback: try 3p exact
    fallback = cand("3p")
    hits = vals[vals_lower.isin([x.lower() for x in fallback])].unique().tolist()
    return hits


# ----------------------------
# Direction inference
# ----------------------------
def _is_mirna_token(token: str) -> bool:
    t = token.lower()
    return ("mir" in t) or t.startswith("hsa-") or t.startswith("mmu-") or t.startswith("rno-")


def _direction_from_token(token: str) -> str:
    return "mirna_to_targets" if _is_mirna_token(token) else "gene_to_mirnas"


# ----------------------------
# Main retrieval
# ----------------------------
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

    # Start from full table; apply filters with masks on same frame.
    df = ev

    # --- Novel mode is the only hard exclude by design ---
    if cfg.novel and "mirtarbase_pos" in df.columns:
        df = df[_bool_col(df, "mirtarbase_pos") == 0]

    # --- Minimal pre-filter only (soft) ---
    if cfg.min_support > 0:
        sc = pd.to_numeric(df["support_count"], errors="coerce").fillna(0).astype(int)
        df = df[sc >= cfg.min_support]

    # --- Optional soft gates (off by default) ---
    if cfg.require_binding_evidence:
        df = df[
            (_bool_col(df, "support_targetscan") == 1)
            | (_bool_col(df, "support_encori") == 1)
            | (_bool_col(df, "support_mirdb") == 1)
        ]

    if cfg.require_expression and cfg.tcga:
        pair_expr = f"{cfg.tcga}_pair_expressed"
        if pair_expr in df.columns:
            df = df[_bool_col(df, pair_expr) == 1]

    # --- Restrict to relevant row set by direction ---
    matched_tokens: List[str] = []
    if direction == "mirna_to_targets":
        allowed = resolve_mirna_names_for_table(query_token, df["mirna_name"])
        matched_tokens = allowed
        if not allowed:
            return df.head(0), direction

        df = df[df["mirna_name"].astype(str).str.lower().isin([a.lower() for a in allowed])]
    else:
        df = df[df["gene_symbol"].astype(str).str.upper() == query_token.upper()]
        matched_tokens = [query_token]

    if len(df) == 0:
        return df.head(0), direction

    # --- Scoring ---
    support = pd.to_numeric(df["support_count"], errors="coerce").fillna(0).astype(float)

    ts_ctx = _safe_float_col(df, "ts_best_contextpp", default=0.0)
    ts_contrib = np.clip(-ts_ctx, 0, 2.0)

    clip_sum = _safe_float_col(df, "clip_exp_sum", default=0.0)
    clip_contrib = np.log1p(clip_sum) / 5.0

    mirdb_best = _safe_float_col(df, "mirdb_best_score", default=0.0)
    mirdb_contrib = (mirdb_best / 100.0)

    tcga_contrib = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    if cfg.tcga:
        rho_col = f"{cfg.tcga}_spearman_rho"
        if rho_col in df.columns:
            rho = _safe_float_col(df, rho_col, default=0.0)
            tcga_contrib = np.clip(-rho, 0, 1.0)
        rep_col = f"{cfg.tcga}_repression_evidence"
        if rep_col in df.columns:
            tcga_contrib = tcga_contrib + 0.2 * _bool_col(df, rep_col)

    pathway_bonus = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    pf = cfg.pathway_filter or {}
    enabled = bool(pf.get("enabled", False))
    mode = str(pf.get("mode", "boost")).lower()
    min_gene_sets = int(pf.get("min_gene_sets", 1))

    if enabled and (cfg.pathway_keywords or cfg.phenotype_keywords):
        if "gene_pathway_hits" in df.columns:
            hits_i = pd.to_numeric(df["gene_pathway_hits"], errors="coerce").fillna(0).astype(int)

            if mode == "filter":
                df = df[hits_i >= min_gene_sets].copy()
                if len(df) == 0:
                    return df.head(0), direction

                support = pd.to_numeric(df["support_count"], errors="coerce").fillna(0).astype(float)
                ts_ctx = _safe_float_col(df, "ts_best_contextpp", default=0.0)
                ts_contrib = np.clip(-ts_ctx, 0, 2.0)
                clip_sum = _safe_float_col(df, "clip_exp_sum", default=0.0)
                clip_contrib = np.log1p(clip_sum) / 5.0
                mirdb_best = _safe_float_col(df, "mirdb_best_score", default=0.0)
                mirdb_contrib = (mirdb_best / 100.0)

                tcga_contrib = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
                if cfg.tcga:
                    rho_col = f"{cfg.tcga}_spearman_rho"
                    if rho_col in df.columns:
                        rho = _safe_float_col(df, rho_col, default=0.0)
                        tcga_contrib = np.clip(-rho, 0, 1.0)
                    rep_col = f"{cfg.tcga}_repression_evidence"
                    if rep_col in df.columns:
                        tcga_contrib = tcga_contrib + 0.2 * _bool_col(df, rep_col)

            hits_f = pd.to_numeric(df["gene_pathway_hits"], errors="coerce").fillna(0).astype(float)
            pathway_bonus = np.clip(hits_f / 5.0, 0, 1.0)

    score = (
        1.0 * support
        + 1.0 * ts_contrib
        + 0.7 * clip_contrib
        + 0.7 * mirdb_contrib
        + 0.8 * tcga_contrib
        + 0.6 * pathway_bonus
    )

    df = df.assign(
        retrieval_score=score,
        retrieval_support=support,
        retrieval_ts_contrib=ts_contrib,
        retrieval_clip_contrib=clip_contrib,
        retrieval_mirdb_contrib=mirdb_contrib,
        retrieval_tcga_contrib=tcga_contrib,
        retrieval_pathway_bonus=pathway_bonus,
        matched_query_tokens=";".join(matched_tokens),
    )

    df = df.sort_values("retrieval_score", ascending=False)
    df = df.head(int(cfg.k_shortlist)).reset_index(drop=True)

    return df, direction


def retrieve_from_queryspec(ev: pd.DataFrame, queryspec: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
    """
    Convenience wrapper for the FastAPI orchestrator:
      queryspec -> RetrievalConfig -> retrieve_candidates(...)
    """
    token = queryspec.get("mirna") or queryspec.get("gene") or queryspec.get("query_token")
    if not token:
        raise ValueError("QuerySpec missing 'mirna'/'gene'/'query_token'.")

    tcga = None
    if isinstance(queryspec.get("cancer"), dict):
        tcga = queryspec["cancer"].get("tcga")
    else:
        tcga = queryspec.get("tcga")

    filters = queryspec.get("filters") or {}
    cfg = RetrievalConfig(
        k_shortlist=int(queryspec.get("k", 50)),
        min_support=int(filters.get("min_support", 1)),
        novel=bool(queryspec.get("novel", False)),
        tcga=tcga,
        phenotype_keywords=queryspec.get("phenotype_keywords") or [],
        pathway_keywords=queryspec.get("pathway_keywords") or [],
        pathway_filter=queryspec.get("pathway_filter") or None,
        require_binding_evidence=bool(filters.get("require_binding_evidence", False)),
        require_expression=bool(filters.get("require_expression", False)),
    )

    return retrieve_candidates(ev, str(token), cfg)
