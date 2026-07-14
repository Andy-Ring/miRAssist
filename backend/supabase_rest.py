"""
Supabase REST (PostgREST) evidence fetch for the miRAssist Claude skill.

This mirrors the interface of retrieval._fetch_postgres_candidate_pool, but reads
the bounded candidate pool over Supabase's auto-generated REST API using the
*publishable* anon key instead of a direct Postgres connection. The anon key is
safe to commit when the evidence table is protected by a read-only Row-Level
Security (RLS) policy (SELECT only). See SKILL.md for the one-time setup.

All scoring, ranking, duplicate-collapse and pathway logic still happens in
pandas inside retrieval.retrieve_candidates - this module only fetches rows.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from backend.config import (
    get_db_candidate_limit,
    get_default_mirna_arm,
    get_evidence_table_bare,
    get_learned_score_column,
    get_supabase_anon_key,
    get_supabase_url,
)

_REST_COLUMN_CACHE: Dict[str, List[str]] = {}
_HTTP_TIMEOUT = 60


def _headers() -> Dict[str, str]:
    key = get_supabase_anon_key() or ""
    headers = {"apikey": key, "Accept": "application/json"}
    # Legacy anon keys are JWTs (start with "eyJ") and are also passed as a Bearer
    # token. Newer publishable keys ("sb_publishable_...") are NOT JWTs; sending
    # them as a Bearer token makes PostgREST reject them, so use apikey only.
    if key.startswith("eyJ"):
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _rest_base() -> str:
    url = (get_supabase_url() or "").rstrip("/")
    if not url:
        raise RuntimeError(
            "Supabase URL is not configured. Set 'supabase_url' in skill_settings.json "
            "or the MIRASSIST_SUPABASE_URL environment variable."
        )
    return f"{url}/rest/v1"


def get_rest_table_columns(table: str) -> List[str]:
    if table in _REST_COLUMN_CACHE:
        return list(_REST_COLUMN_CACHE[table])
    resp = requests.get(
        f"{_rest_base()}/{table}",
        headers=_headers(),
        params={"select": "*", "limit": 1},
        timeout=_HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    rows = resp.json()
    columns = list(rows[0].keys()) if rows else []
    _REST_COLUMN_CACHE[table] = columns
    return list(columns)


def _quote_in_values(values: List[str]) -> str:
    # PostgREST in.(...) list; quote values to be safe with hyphens/spaces.
    escaped = [str(v).replace('"', '\\"') for v in values]
    return "in.(" + ",".join(f'"{v}"' for v in escaped) + ")"


def _rest_order(available: set, score_column: Optional[str]) -> str:
    parts: List[str] = []
    if score_column and score_column != "retrieval_score" and score_column in available:
        parts.append(f"{score_column}.desc.nullslast")
    if "support_count" in available:
        parts.append("support_count.desc.nullslast")
    if "mirdb_best_score" in available:
        parts.append("mirdb_best_score.desc.nullslast")
    if "ts_context_strength" in available:
        parts.append("ts_context_strength.desc.nullslast")
    elif "ts_best_contextpp" in available:
        parts.append("ts_best_contextpp.asc.nullslast")
    if "clip_exp_sum" in available:
        parts.append("clip_exp_sum.desc.nullslast")
    if "best_mfe" in available:
        parts.append("best_mfe.asc.nullslast")
    if "retrieval_score" in available:
        parts.append("retrieval_score.desc.nullslast")
    if not parts:
        parts.append("gene_symbol.asc")
    return ",".join(parts)


def _build_query_params(
    query_token: str,
    cfg: "Any",
    available_columns: List[str],
    *,
    mirna_variants: Optional[List[str]] = None,
    mirna_prefix_patterns: Optional[List[str]] = None,
) -> Tuple[List[Tuple[str, str]], List[str], Dict[str, Any]]:
    # Imported here to avoid a circular import at module load time.
    from backend.retrieval import (
        _direction_from_token,
        _first_available_column,
        MIRTARBASE_KNOWN_POSITIVE_COLUMNS,
        _normalize_gene_symbol,
        _normalize_token,
        _resolve_score_column_from_available,
        _select_production_evidence_columns,
    )

    direction = _direction_from_token(_normalize_token(query_token))
    available = {str(c) for c in available_columns}
    selected_columns = _select_production_evidence_columns(available, cfg.tcga)

    mirna_norm_col = _first_available_column(available, ("mirna_name_norm", "mirna_name_normalized"))
    gene_norm_col = _first_available_column(available, ("gene_symbol_norm", "gene_symbol_normalized"))

    required = {"mirna_name", "gene_symbol", "support_count"}
    if direction == "mirna_to_targets" and mirna_norm_col is None:
        required.add("mirna_name_norm")
    if direction == "gene_to_mirnas" and gene_norm_col is None:
        required.add("gene_symbol_norm")
    missing = sorted(c for c in required if c not in selected_columns)
    if missing:
        raise RuntimeError(
            "Supabase evidence table is missing required columns: " + ", ".join(missing)
        )

    params: List[Tuple[str, str]] = []

    # entity / direction filter
    if direction == "mirna_to_targets":
        col = str(mirna_norm_col)
        if mirna_variants:
            params.append((col, _quote_in_values([v.lower() for v in mirna_variants])))
        elif mirna_prefix_patterns:
            like_bits = [f"{col}.like.{p.lower()}" for p in mirna_prefix_patterns]
            params.append(("or", "(" + ",".join(like_bits) + ")"))
        else:
            params.append((col, "in.()"))  # matches nothing
    else:
        params.append((str(gene_norm_col), f"eq.{_normalize_gene_symbol(query_token)}"))

    # novelty / gates
    if cfg.novel and cfg.use_mirtarbase_evidence:
        for col in MIRTARBASE_KNOWN_POSITIVE_COLUMNS:
            if col in available:
                params.append(("or", f"({col}.is.null,{col}.eq.0)"))
    if cfg.require_binding_evidence:
        bind = [c for c in ("support_targetscan", "support_encori", "support_mirdb") if c in available]
        if bind:
            params.append(("or", "(" + ",".join(f"{c}.eq.1" for c in bind) + ")"))
    if cfg.require_expression and cfg.tcga:
        pair_col = f"{str(cfg.tcga).upper()}_pair_expressed"
        if pair_col in available:
            params.append((pair_col, "eq.1"))

    # grounded pathway gene restriction
    pathway_selection = cfg.pathway_selection or {}
    if bool(pathway_selection.get("enabled")) and cfg.pathway_gene_set and gene_norm_col is not None:
        genes = sorted({_normalize_gene_symbol(g) for g in cfg.pathway_gene_set if str(g or "").strip()})
        if genes:
            params.append((str(gene_norm_col), _quote_in_values(genes)))

    score_col = _resolve_score_column_from_available(available, get_learned_score_column())
    params.append(("select", ",".join(selected_columns)))
    params.append(("order", _rest_order(available, score_col)))
    params.append(("limit", str(int(get_db_candidate_limit()))))

    diagnostics = {
        "evidence_backend": "rest",
        "supabase_table_name": get_evidence_table_bare(),
        "db_candidate_limit": int(get_db_candidate_limit()),
        "learned_score_column": score_col or get_learned_score_column(),
        "sort_column_used": score_col or "retrieval_score",
        "query_direction": direction,
        "sql_mirna_norm_column": mirna_norm_col,
        "sql_gene_norm_column": gene_norm_col,
    }
    return params, selected_columns, diagnostics


def fetch_rest_candidate_pool(query_token: str, cfg: "Any") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    from backend.retrieval import (
        _direction_from_token,
        _mirna_prefix_fallback_patterns,
        _normalize_mirna_query,
        _normalize_token,
        expand_mirna_query_variants,
    )

    table = get_evidence_table_bare()
    available_columns = get_rest_table_columns(table)
    direction = _direction_from_token(_normalize_token(query_token))
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
        "evidence_backend": "rest",
        "raw_mirna_query": str(query_token or "") if direction == "mirna_to_targets" else None,
        "normalized_mirna_query": expansion.get("normalized_input") if direction == "mirna_to_targets" else None,
        "query_mirna_normalized": _normalize_mirna_query(query_token)[0] if direction == "mirna_to_targets" else None,
        "explicit_mirna_arm": bool(expansion.get("explicit_arm")) if direction == "mirna_to_targets" else False,
        "primary_mirna_variants": list(primary),
        "fallback_mirna_variants": list(fallback),
        "variants_used": [],
        "arm_interpretation_note": expansion.get("note") if direction == "mirna_to_targets" else None,
    }

    def _get(mirna_variants: Optional[List[str]], like_patterns: Optional[List[str]]):
        params, selected, diag = _build_query_params(
            query_token, cfg, available_columns,
            mirna_variants=mirna_variants, mirna_prefix_patterns=like_patterns,
        )
        resp = requests.get(f"{_rest_base()}/{table}", headers=_headers(), params=params, timeout=_HTTP_TIMEOUT)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Supabase REST request failed ({resp.status_code}) for table '{table}': {resp.text[:300]}. "
                "Check the anon key, the table name, and that a read-only RLS SELECT policy is enabled."
            )
        return pd.DataFrame(resp.json()), diag, selected

    if direction == "mirna_to_targets":
        df, qdiag, selected = _get(primary, None)
        diagnostics.update(qdiag)
        if not df.empty:
            diagnostics["variants_used"] = list(primary)
        if df.empty and fallback:
            df, qdiag, selected = _get(fallback, None)
            diagnostics.update(qdiag)
            if not df.empty:
                diagnostics["variants_used"] = list(fallback)
        if df.empty and prefixes:
            df, qdiag, selected = _get(None, prefixes)
            diagnostics.update(qdiag)
            if not df.empty:
                diagnostics["variants_used"] = list(prefixes)
    else:
        df, qdiag, selected = _get(None, None)
        diagnostics.update(qdiag)

    if not df.empty and df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    diagnostics["n_rows_fetched_from_db"] = int(len(df))
    diagnostics["sql_selected_column_count"] = int(len(selected))
    diagnostics["sql_returned_column_count"] = int(len(df.columns))
    return df, diagnostics
