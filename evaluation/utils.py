from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from backend.retrieval import (
    _normalize_gene_symbol,
    _normalize_mirna_query,
    load_evidence,
    retrieve_from_queryspec,
)


DEFAULT_LABEL_COLUMNS = ("mirtarbase_pos", "label_mirtarbase")
LABEL_COLUMN_CANDIDATES = (
    "mirtarbase_pos",
    "label_mirtarbase",
    "mirtarbase_pos_label",
    "label_mirtarbase_label",
)


@dataclass
class EvalRunResult:
    query_id: str
    queryspec: Dict[str, Any]
    retrieval_diagnostics: Dict[str, Any]
    shortlist: pd.DataFrame
    answer: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: str | Path, payload: Any) -> None:
    safe_payload = sanitize_json_payload(payload)
    Path(path).write_text(
        json.dumps(safe_payload, ensure_ascii=False, allow_nan=False, indent=2),
        encoding="utf-8",
    )


def sanitize_json_payload(payload: Any) -> Any:
    if payload is None:
        return None
    if isinstance(payload, (str, bool, int)):
        return payload
    if isinstance(payload, float):
        return payload if math.isfinite(payload) else None
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, dict):
        return {str(key): sanitize_json_payload(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple, set)):
        return [sanitize_json_payload(value) for value in payload]
    if isinstance(payload, pd.DataFrame):
        return sanitize_json_payload(payload.to_dict(orient="records"))
    if isinstance(payload, pd.Series):
        return sanitize_json_payload(payload.to_dict() if payload.index.is_unique else payload.tolist())
    if payload is pd.NA or payload is pd.NaT:
        return None
    if isinstance(payload, pd.Timestamp):
        return payload.isoformat()
    if isinstance(payload, np.ndarray):
        return sanitize_json_payload(payload.tolist())
    if isinstance(payload, (np.integer, np.floating, np.bool_)):
        return sanitize_json_payload(payload.item())
    return str(payload)


def normalize_mirna_name(value: Any) -> str:
    base, arm = _normalize_mirna_query(str(value or ""))
    if not base:
        return ""
    return f"{base}-{arm}" if arm else base


def normalize_gene_symbol(value: Any) -> str:
    return _normalize_gene_symbol(value)


def get_numeric_series(df: pd.DataFrame, col: str, default: float = 0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _resolve_normalized_name_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    mirna_col = None
    gene_col = None
    for candidate in ("mirna_name_normalized", "mirna_name_norm"):
        if candidate in df.columns:
            mirna_col = candidate
            break
    for candidate in ("gene_symbol_normalized", "gene_symbol_norm"):
        if candidate in df.columns:
            gene_col = candidate
            break
    return mirna_col, gene_col


def _resolve_label_columns(df: pd.DataFrame) -> List[str]:
    return [column for column in LABEL_COLUMN_CANDIDATES if column in df.columns]


def add_eval_row_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "eval_row_id" not in out.columns:
        out.insert(0, "eval_row_id", np.arange(len(out), dtype=int))
    return out


def _looks_like_mirtarbase_auxiliary(column_name: str) -> bool:
    text = str(column_name or "").strip().lower()
    if "mirtarbase" in text or "mir_tar_base" in text:
        return True
    if "mirtar" in text and any(token in text for token in ("pmid", "experiment", "validated", "functional")):
        return True
    return False


def find_mirtarbase_like_columns(columns: Iterable[str]) -> List[str]:
    matches: List[str] = []
    for column in columns:
        if _looks_like_mirtarbase_auxiliary(str(column)):
            matches.append(str(column))
    return matches


def recompute_blinded_support_count(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.zeros(len(df), dtype=int), index=df.index)

    def _bool(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(np.zeros(len(df), dtype=int), index=df.index)
        return pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    def _float(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(np.full(len(df), np.nan, dtype=float), index=df.index)
        return pd.to_numeric(df[col], errors="coerce")

    category_masks: List[pd.Series] = []
    for col in ("support_targetscan", "support_mirdb", "support_encori", "support_rnahybrid"):
        if col in df.columns:
            category_masks.append(_bool(col) > 0)

    seed_mask = pd.Series(False, index=df.index)
    if "has_seed_features" in df.columns:
        seed_mask |= _bool("has_seed_features") > 0
    if "n_total_sites" in df.columns:
        seed_mask |= _float("n_total_sites").fillna(0) > 0
    for col in ("n_sites_6mer", "n_sites_7mer_a1", "n_sites_7mer_m8", "n_sites_8mer"):
        if col in df.columns:
            seed_mask |= _float(col).fillna(0) > 0
    if "best_seed_class" in df.columns:
        seed_mask |= df["best_seed_class"].fillna("").astype(str).str.strip() != ""
    category_masks.append(seed_mask)

    structure_mask = pd.Series(False, index=df.index)
    for col in ("has_rnahybrid", "n_rnahybrid_sites", "n_sites_mfe_lt_-20", "n_sites_mfe_lt_-25"):
        if col in df.columns:
            structure_mask |= _float(col).fillna(0) > 0
    for col in ("best_mfe", "mfe_strength", "mean_top3_mfe", "mean_top3_mfe_strength"):
        if col in df.columns:
            structure_mask |= _float(col).notna()
    category_masks.append(structure_mask)

    tcga_mask = pd.Series(False, index=df.index)
    for col in df.columns:
        lower = str(col).lower()
        if lower.endswith("_support_tcga") or lower.endswith("_anticorrelated") or lower.endswith("_repression_evidence"):
            tcga_mask |= _bool(col) > 0
        elif lower.endswith("_spearman_rho"):
            tcga_mask |= _float(col) < 0
    category_masks.append(tcga_mask)

    for mask in category_masks:
        out += mask.fillna(False).astype(int)

    return out.astype(int)


def make_blinded_evidence(
    evidence_df: pd.DataFrame,
    label_cols: Sequence[str] = DEFAULT_LABEL_COLUMNS,
    keep_neutral_label_cols: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    original = add_eval_row_id(evidence_df)
    labels_to_keep = [col for col in label_cols if col in original.columns]

    heldout_cols = ["eval_row_id", "mirna_name", "gene_symbol"]
    heldout_cols.extend(col for col in ("transcript_id", "ts_gene_id_base", "entrez_ids") if col in original.columns)
    heldout_cols.extend(labels_to_keep)
    heldout = original.loc[:, list(dict.fromkeys(heldout_cols))].copy()
    heldout["mirna_name_normalized"] = heldout["mirna_name"].map(normalize_mirna_name)
    heldout["gene_symbol_normalized"] = heldout["gene_symbol"].map(normalize_gene_symbol)

    blinded = original.copy()
    related_columns = find_mirtarbase_like_columns(blinded.columns)
    neutralized_columns: List[str] = []
    dropped_columns: List[str] = []
    allowed_neutralized = {col for col in ("mirtarbase_pos", "label_mirtarbase") if col in blinded.columns}

    for col in related_columns:
        if keep_neutral_label_cols and col in allowed_neutralized:
            blinded[col] = 0
            neutralized_columns.append(col)
        elif col in blinded.columns:
            blinded = blinded.drop(columns=[col])
            dropped_columns.append(col)

    blinded["support_count"] = recompute_blinded_support_count(blinded)
    blinded["mirna_name_normalized"] = blinded["mirna_name"].map(normalize_mirna_name)
    blinded["gene_symbol_normalized"] = blinded["gene_symbol"].map(normalize_gene_symbol)

    remaining_related = [
        col for col in find_mirtarbase_like_columns(blinded.columns) if col not in neutralized_columns
    ]
    if remaining_related:
        raise ValueError(f"Blinded evidence still contains miRTarBase-related columns: {remaining_related}")

    for col in neutralized_columns:
        if col in blinded.columns and int(pd.to_numeric(blinded[col], errors="coerce").fillna(0).sum()) != 0:
            raise ValueError(f"Neutralized column {col} still contains non-zero values after blinding.")

    audit = {
        "input_shape": [int(original.shape[0]), int(original.shape[1])],
        "blinded_shape": [int(blinded.shape[0]), int(blinded.shape[1])],
        "heldout_shape": [int(heldout.shape[0]), int(heldout.shape[1])],
        "dropped_columns": dropped_columns,
        "neutralized_columns": neutralized_columns,
        "support_count_column": "support_count",
        "support_count_recomputed": True,
        "label_columns_preserved": labels_to_keep,
        "n_positive_rows": int(
            pd.to_numeric(heldout.get("mirtarbase_pos", 0), errors="coerce").fillna(0).astype(int).sum()
        ),
    }
    positives = audit["n_positive_rows"]
    audit["positive_fraction"] = float(positives / len(heldout)) if len(heldout) else 0.0
    return blinded, heldout, audit


def assert_eval_mode() -> None:
    if (os.getenv("MIRASSIST_EVAL_MODE", "0") or "0").strip() != "1":
        raise RuntimeError("MIRASSIST_EVAL_MODE must be set to 1 for evaluation runs.")


def assert_no_mirtarbase_leakage(df: pd.DataFrame) -> None:
    if "mirtarbase_pos" in df.columns:
        positives = pd.to_numeric(df["mirtarbase_pos"], errors="coerce").fillna(0)
        if int(positives.sum()) > 0:
            raise RuntimeError("Blinded evidence still contains positive mirtarbase_pos values.")
    if "label_mirtarbase" in df.columns:
        positives = pd.to_numeric(df["label_mirtarbase"], errors="coerce").fillna(0)
        if int(positives.sum()) > 0:
            raise RuntimeError("Blinded evidence still contains positive label_mirtarbase values.")


def build_eval_queryspec(
    *,
    query_id: str,
    mode: str,
    mirna: Optional[str],
    gene: Optional[str],
    k: int,
    min_support: int,
    novel: bool,
    use_pathway_filter: bool = False,
    cancer_name: Optional[str] = None,
    tcga: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "query_id": query_id,
        "original_question": f"What genes are regulated by {mirna}?" if mode == "mirna_to_targets" else f"What miRNAs regulate {gene}?",
        "mode": mode,
        "mirna": mirna,
        "gene": gene,
        "cancer": {"name": cancer_name, "tcga": tcga},
        "phenotype_context": {
            "phenotype": None,
            "observed_change": None,
            "miRNA_perturbation": None,
            "raw_phrase": None,
            "direction": None,
        },
        "target_role_inference": {
            "enabled": False,
            "assumption": "miRNAs usually repress target gene expression",
            "expected_target_effect_on_phenotype": "unknown",
            "reasoning": "",
        },
        "pathway_selection_request": {
            "enabled": bool(use_pathway_filter),
            "query_terms": [],
            "directional_query_terms": [],
            "strict": bool(use_pathway_filter),
        },
        "phenotype_keywords": [],
        "pathway_keywords": [],
        "pathway_filter": {
            "enabled": bool(use_pathway_filter),
            "mode": "filter",
            "min_gene_sets": 1,
        },
        "novel": bool(novel),
        "k": int(k),
        "result_count": None,
        "filters": {
            "min_support": int(min_support),
            "require_binding_evidence": False,
            "require_expression": False,
        },
        "needs_clarification": [],
        "optional_clarifications": [],
    }


def run_eval_query(
    *,
    queryspec: Dict[str, Any],
    evidence_path: str | Path,
    disable_synthesis: bool = True,
) -> EvalRunResult:
    assert_eval_mode()
    evidence_df = load_evidence(str(evidence_path), force_reload=True)
    assert_no_mirtarbase_leakage(evidence_df)

    shortlist_df, direction, diagnostics = retrieve_from_queryspec(evidence_df, queryspec)
    diagnostics = dict(diagnostics)
    diagnostics["direction"] = direction

    answer = None
    if disable_synthesis:
        answer = {
            "summary": f"Deterministic evaluation run completed with {len(shortlist_df)} ranked candidates.",
            "raw_text": "",
            "suggested_experiments": [],
        }
    else:
        from backend.cards import cards_from_dataframe
        from backend.feature_stats import annotate_feature_percentiles
        from backend.prompting import build_prompt_bundle
        from backend.synthesizer import run_synthesizer

        annotated = annotate_feature_percentiles(shortlist_df, evidence_df)
        cards = cards_from_dataframe(annotated, tcga=((queryspec.get("cancer") or {}).get("tcga")))
        bundle = build_prompt_bundle(
            queryspec=queryspec,
            shortlist=annotated,
            cards=cards,
            direction=direction,
            retrieval_diagnostics=diagnostics,
        )
        answer = run_synthesizer(bundle)

    metadata = {
        "eval_mode": True,
        "disable_synthesis": bool(disable_synthesis),
        "evidence_path": str(Path(evidence_path).resolve()),
        "cwd": os.getcwd(),
    }
    return EvalRunResult(
        query_id=str(queryspec.get("query_id") or ""),
        queryspec=queryspec,
        retrieval_diagnostics=diagnostics,
        shortlist=shortlist_df,
        answer=answer,
        metadata=metadata,
    )


def aggregate_labels_for_join(labels_df: pd.DataFrame) -> pd.DataFrame:
    labels = labels_df.copy()
    mirna_norm_col, gene_norm_col = _resolve_normalized_name_columns(labels)
    if mirna_norm_col is None:
        labels["mirna_name_normalized"] = labels["mirna_name"].map(normalize_mirna_name)
        mirna_norm_col = "mirna_name_normalized"
    elif mirna_norm_col != "mirna_name_normalized":
        labels["mirna_name_normalized"] = labels[mirna_norm_col]
        mirna_norm_col = "mirna_name_normalized"
    if gene_norm_col is None:
        labels["gene_symbol_normalized"] = labels["gene_symbol"].map(normalize_gene_symbol)
        gene_norm_col = "gene_symbol_normalized"
    elif gene_norm_col != "gene_symbol_normalized":
        labels["gene_symbol_normalized"] = labels[gene_norm_col]
        gene_norm_col = "gene_symbol_normalized"

    label_columns_used = _resolve_label_columns(labels)
    if not label_columns_used:
        labels["is_positive"] = 0
        grouped = labels.groupby(["mirna_name_normalized", "gene_symbol_normalized"], as_index=False)[
            ["is_positive"]
        ].max()
        grouped["label_columns_used"] = ""
        grouped.attrs["label_columns_used"] = []
        grouped.attrs["label_warning"] = "No held-out label columns were present."
        return grouped

    grouped = labels.groupby(["mirna_name_normalized", "gene_symbol_normalized"], as_index=False)[
        label_columns_used
    ].max()
    positive_mask = pd.Series(False, index=grouped.index)
    for column in label_columns_used:
        positive_mask |= get_numeric_series(grouped, column, default=0) > 0
    grouped["is_positive"] = positive_mask.astype(int)
    grouped.attrs["label_columns_used"] = label_columns_used
    grouped.attrs["label_warning"] = ""
    return grouped


def collect_rankings_from_json(
    json_dir: str | Path,
    labels_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    label_lookup = aggregate_labels_for_join(labels_df)
    label_columns_used = list(label_lookup.attrs.get("label_columns_used") or [])
    label_warning = str(label_lookup.attrs.get("label_warning") or "")
    ranking_rows: List[Dict[str, Any]] = []
    query_rows: List[Dict[str, Any]] = []
    json_files_read = 0

    for path in sorted(Path(json_dir).glob("*.json")):
        json_files_read += 1
        payload = json.loads(path.read_text(encoding="utf-8"))
        query_id = str(payload.get("query_id") or path.stem)
        queryspec = payload.get("queryspec") or {}
        shortlist = payload.get("shortlist") or []
        diagnostics = payload.get("retrieval_diagnostics") or {}
        error_text = payload.get("error")

        if shortlist:
            for idx, record in enumerate(shortlist, start=1):
                row = dict(record)
                row["query_id"] = query_id
                row["rank"] = idx
                row["mirna"] = row.get("mirna_name") or queryspec.get("mirna")
                row["mirna_name_normalized"] = normalize_mirna_name(row.get("mirna"))
                row["gene_symbol_normalized"] = normalize_gene_symbol(row.get("gene_symbol"))
                ranking_rows.append(row)
        else:
            error_text = error_text or "; ".join(diagnostics.get("warnings") or [])

        query_rows.append(
            {
                "query_id": query_id,
                "mirna": queryspec.get("mirna"),
                "gene": queryspec.get("gene"),
                "n_ranked": int(len(shortlist)),
                "errors": error_text or "",
            }
        )

    rankings_df = pd.DataFrame(ranking_rows)
    if rankings_df.empty:
        rankings_df = pd.DataFrame(
            columns=[
                "query_id",
                "mirna",
                "gene_symbol",
                "rank",
                "retrieval_score",
                "retrieval_support",
                "retrieval_ts_contrib",
                "retrieval_clip_contrib",
                "retrieval_mirdb_contrib",
                "retrieval_seed_contrib",
                "retrieval_rnahybrid_contrib",
                "retrieval_local_au_contrib",
                "retrieval_structure_contrib",
                "retrieval_structure_in_score",
                "retrieval_tcga_contrib",
                "retrieval_pathway_bonus",
                "mirna_name_normalized",
                "gene_symbol_normalized",
                "mirtarbase_pos",
                "label_mirtarbase",
                "is_positive",
            ]
        )
        missing_label_join_rows = 0
    else:
        rankings_df = rankings_df.merge(
            label_lookup,
            how="left",
            on=["mirna_name_normalized", "gene_symbol_normalized"],
        )
        missing_label_join_rows = int(get_numeric_series(rankings_df, "is_positive", default=np.nan).isna().sum())
        if not label_columns_used:
            rankings_df["is_positive"] = 0
        else:
            positive_mask = pd.Series(False, index=rankings_df.index)
            for column in label_columns_used:
                positive_mask |= get_numeric_series(rankings_df, column, default=0) > 0
            rankings_df["is_positive"] = positive_mask.astype(int)
        for preferred in ("mirtarbase_pos", "label_mirtarbase"):
            if preferred not in rankings_df.columns:
                matching = [col for col in label_columns_used if preferred in col]
                if matching:
                    rankings_df[preferred] = get_numeric_series(rankings_df, matching[0], default=0).astype(int)
                else:
                    rankings_df[preferred] = 0
            else:
                rankings_df[preferred] = get_numeric_series(rankings_df, preferred, default=0).astype(int)

    query_summary = pd.DataFrame(query_rows)
    if query_summary.empty:
        collection_summary = {
            "json_files_read": json_files_read,
            "ranked_rows_collected": int(len(rankings_df)),
            "rows_with_heldout_positive_label": int(get_numeric_series(rankings_df, "is_positive", default=0).sum()),
            "label_columns_used": label_columns_used,
            "missing_label_join_rows": int(missing_label_join_rows),
            "warnings": [label_warning] if label_warning else [],
        }
        return rankings_df, query_summary, collection_summary

    label_counts = (
        labels_df.assign(
            mirna_name_normalized=(
                labels_df[_resolve_normalized_name_columns(labels_df)[0]]
                if _resolve_normalized_name_columns(labels_df)[0] is not None
                else labels_df["mirna_name"].map(normalize_mirna_name)
            ),
            gene_symbol_normalized=(
                labels_df[_resolve_normalized_name_columns(labels_df)[1]]
                if _resolve_normalized_name_columns(labels_df)[1] is not None
                else labels_df["gene_symbol"].map(normalize_gene_symbol)
            ),
        )
    )
    if label_columns_used:
        positive_series = pd.Series(False, index=label_counts.index)
        for column in label_columns_used:
            positive_series |= get_numeric_series(label_counts, column, default=0) > 0
        label_counts = label_counts.assign(positive=positive_series.astype(int))
    else:
        label_counts = label_counts.assign(positive=0)
    label_counts = (
        label_counts.groupby("mirna_name_normalized", as_index=False)["positive"]
        .sum()
        .rename(columns={"positive": "n_positives_total"})
    )
    query_summary["mirna_name_normalized"] = query_summary["mirna"].map(normalize_mirna_name)
    query_summary = query_summary.merge(label_counts, how="left", on="mirna_name_normalized")
    query_summary["n_positives_total"] = pd.to_numeric(
        query_summary["n_positives_total"], errors="coerce"
    ).fillna(0).astype(int)

    if not rankings_df.empty:
        positive_ranks = rankings_df[rankings_df["is_positive"] == 1]
        best_ranks = (
            positive_ranks.groupby("query_id", as_index=False)["rank"].min().rename(columns={"rank": "best_positive_rank"})
            if not positive_ranks.empty
            else pd.DataFrame(columns=["query_id", "best_positive_rank"])
        )
        query_summary = query_summary.merge(best_ranks, how="left", on="query_id")
        query_summary["best_positive_rank"] = pd.to_numeric(
            query_summary["best_positive_rank"], errors="coerce"
        )
        retrieved_counts = (
            positive_ranks.groupby("query_id", as_index=False).size().rename(columns={"size": "n_positives_retrieved"})
            if not positive_ranks.empty
            else pd.DataFrame(columns=["query_id", "n_positives_retrieved"])
        )
        query_summary = query_summary.merge(retrieved_counts, how="left", on="query_id")
        query_summary["n_positives_retrieved"] = pd.to_numeric(
            query_summary["n_positives_retrieved"], errors="coerce"
        ).fillna(0).astype(int)
        for k in (10, 25, 50, 100):
            counts = (
                positive_ranks[positive_ranks["rank"] <= k]
                .groupby("query_id", as_index=False)
                .size()
                .rename(columns={"size": f"top{k}_positive_count"})
            )
            query_summary = query_summary.merge(counts, how="left", on="query_id")
            query_summary[f"top{k}_positive_count"] = pd.to_numeric(
                query_summary[f"top{k}_positive_count"], errors="coerce"
            ).fillna(0).astype(int)
    else:
        query_summary["best_positive_rank"] = np.nan
        query_summary["n_positives_retrieved"] = 0
        for k in (10, 25, 50, 100):
            query_summary[f"top{k}_positive_count"] = 0

    query_summary["label_columns_used"] = ", ".join(label_columns_used)
    query_summary["missing_label_join_rows"] = int(missing_label_join_rows)
    collection_summary = {
        "json_files_read": json_files_read,
        "ranked_rows_collected": int(len(rankings_df)),
        "rows_with_heldout_positive_label": int(get_numeric_series(rankings_df, "is_positive", default=0).sum()),
        "label_columns_used": label_columns_used,
        "missing_label_join_rows": int(missing_label_join_rows),
        "warnings": [label_warning] if label_warning else [],
    }
    return rankings_df, query_summary, collection_summary


def roc_auc_score_manual(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    y = np.asarray(labels, dtype=float)
    s = np.asarray(scores, dtype=float)
    if len(y) == 0 or np.unique(y).size < 2:
        return None
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    pos = y == 1
    n_pos = float(pos.sum())
    n_neg = float((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum = ranks[pos].sum()
    auc = (rank_sum - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def average_precision_manual(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    if len(y) == 0 or y.sum() == 0:
        return None
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    precision = tp / np.arange(1, len(y_sorted) + 1)
    ap = (precision * (y_sorted == 1)).sum() / max(1, int(y.sum()))
    return float(ap)
