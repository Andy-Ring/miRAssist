from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.utils import average_precision_manual, json_dump, roc_auc_score_manual  # noqa: E402


DEFAULT_KS = (1, 5, 10, 25, 50, 100)
STRUCTURE_COMPONENT_CANDIDATES = (
    "retrieval_structure_contrib",
    "retrieval_seed_contrib",
    "retrieval_rnahybrid_contrib",
    "retrieval_local_au_contrib",
    "retrieval_mfe_contrib",
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rankings", required=True)
    ap.add_argument("--query-summary", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ks", default="1,5,10,25,50,100")
    ap.add_argument("--score-modes", default=None)
    ap.add_argument("--include-random-baseline", default=None)
    return ap.parse_args()


def parse_ks(text: str) -> List[int]:
    values = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise ValueError("At least one K value is required.")
    return sorted(set(values))


def require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def _safe_float_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _safe_text_series(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=str)
    return df[col].fillna(default).astype(str)


def _compute_metrics_from_vector(is_positive: np.ndarray, ks: Sequence[int], total_positives: int | None) -> Dict[str, float]:
    n_ranked = int(len(is_positive))
    positive_positions = np.flatnonzero(is_positive > 0) + 1
    positives_retrieved = int(is_positive.sum())
    denominator = int(total_positives) if total_positives is not None and int(total_positives) > 0 else positives_retrieved

    metrics: Dict[str, float] = {
        "n_ranked": float(n_ranked),
        "n_positives_retrieved": float(positives_retrieved),
        "n_positives_total_effective": float(denominator),
        "best_positive_rank": float(positive_positions.min()) if positive_positions.size else np.nan,
        "reciprocal_rank": (1.0 / float(positive_positions.min())) if positive_positions.size else 0.0,
    }
    for k in ks:
        positives_in_topk = int(is_positive[: min(k, n_ranked)].sum()) if n_ranked > 0 else 0
        metrics[f"positive_count_at_{k}"] = float(positives_in_topk)
        metrics[f"recall_at_{k}"] = positives_in_topk / float(denominator) if denominator > 0 else np.nan
        metrics[f"precision_at_{k}"] = positives_in_topk / float(max(1, min(k, n_ranked))) if n_ranked > 0 else 0.0
    return metrics


def _build_structure_component(df: pd.DataFrame) -> pd.Series | None:
    if "retrieval_structure_contrib" in df.columns:
        return _safe_float_series(df, "retrieval_structure_contrib", default=0.0)
    available = [col for col in STRUCTURE_COMPONENT_CANDIDATES if col in df.columns and col != "retrieval_structure_contrib"]
    if not available:
        return None
    total = pd.Series(0.0, index=df.index, dtype=float)
    for col in available:
        total += _safe_float_series(df, col, default=0.0)
    return total


def _available_score_modes(df: pd.DataFrame) -> tuple[Dict[str, pd.Series], Dict[str, Any]]:
    warnings: Dict[str, Any] = {
        "missing_columns": {},
        "skipped_modes": [],
        "notes": [],
    }
    modes: Dict[str, pd.Series] = {}

    def add_mode(name: str, required_cols: Sequence[str], builder) -> None:
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            warnings["missing_columns"][name] = missing
            warnings["skipped_modes"].append(name)
            return
        modes[name] = builder()

    add_mode("full", ["retrieval_score"], lambda: _safe_float_series(df, "retrieval_score", default=0.0))
    add_mode("support_only", ["retrieval_support"], lambda: _safe_float_series(df, "retrieval_support", default=0.0))
    add_mode("targetscan_only", ["retrieval_ts_contrib"], lambda: _safe_float_series(df, "retrieval_ts_contrib", default=0.0))
    add_mode("clip_only", ["retrieval_clip_contrib"], lambda: _safe_float_series(df, "retrieval_clip_contrib", default=0.0))
    add_mode("mirdb_only", ["retrieval_mirdb_contrib"], lambda: _safe_float_series(df, "retrieval_mirdb_contrib", default=0.0))
    add_mode("tcga_only", ["retrieval_tcga_contrib"], lambda: _safe_float_series(df, "retrieval_tcga_contrib", default=0.0))
    add_mode("seed_only", ["retrieval_seed_contrib"], lambda: _safe_float_series(df, "retrieval_seed_contrib", default=0.0))
    add_mode(
        "rnahybrid_only",
        ["retrieval_rnahybrid_contrib"],
        lambda: _safe_float_series(df, "retrieval_rnahybrid_contrib", default=0.0),
    )
    add_mode(
        "local_au_only",
        ["retrieval_local_au_contrib"],
        lambda: _safe_float_series(df, "retrieval_local_au_contrib", default=0.0),
    )
    add_mode(
        "mirdb_targetscan_only",
        ["retrieval_mirdb_contrib", "retrieval_ts_contrib"],
        lambda: _safe_float_series(df, "retrieval_mirdb_contrib", 0.0) + _safe_float_series(df, "retrieval_ts_contrib", 0.0),
    )

    add_mode(
        "no_targetscan",
        ["retrieval_score", "retrieval_ts_contrib"],
        lambda: _safe_float_series(df, "retrieval_score", 0.0) - _safe_float_series(df, "retrieval_ts_contrib", 0.0),
    )
    add_mode(
        "no_clip",
        ["retrieval_score", "retrieval_clip_contrib"],
        lambda: _safe_float_series(df, "retrieval_score", 0.0) - _safe_float_series(df, "retrieval_clip_contrib", 0.0),
    )
    add_mode(
        "no_mirdb",
        ["retrieval_score", "retrieval_mirdb_contrib"],
        lambda: _safe_float_series(df, "retrieval_score", 0.0) - _safe_float_series(df, "retrieval_mirdb_contrib", 0.0),
    )
    add_mode(
        "no_tcga",
        ["retrieval_score", "retrieval_tcga_contrib"],
        lambda: _safe_float_series(df, "retrieval_score", 0.0) - _safe_float_series(df, "retrieval_tcga_contrib", 0.0),
    )
    add_mode(
        "no_pathway",
        ["retrieval_score", "retrieval_pathway_bonus"],
        lambda: _safe_float_series(df, "retrieval_score", 0.0) - _safe_float_series(df, "retrieval_pathway_bonus", 0.0),
    )
    add_mode(
        "non_tcga_non_pathway",
        ["retrieval_score", "retrieval_tcga_contrib", "retrieval_pathway_bonus"],
        lambda: (
            _safe_float_series(df, "retrieval_score", 0.0)
            - _safe_float_series(df, "retrieval_tcga_contrib", 0.0)
            - _safe_float_series(df, "retrieval_pathway_bonus", 0.0)
        ),
    )
    add_mode(
        "no_mirdb_no_targetscan",
        ["retrieval_score", "retrieval_mirdb_contrib", "retrieval_ts_contrib"],
        lambda: (
            _safe_float_series(df, "retrieval_score", 0.0)
            - _safe_float_series(df, "retrieval_mirdb_contrib", 0.0)
            - _safe_float_series(df, "retrieval_ts_contrib", 0.0)
        ),
    )

    structure_total = _build_structure_component(df)
    if structure_total is None:
        warnings["notes"].append(
            "Structure-specific ablation could not be computed because structure contribution columns were not present in rankings_long. "
            "To evaluate structure-aware contribution, add structure component scores during ranking and rerun evaluation."
        )
        warnings["skipped_modes"].extend(
            ["structure_only", "seed_only", "rnahybrid_only", "local_au_only", "mirdb_targetscan_structure", "no_structure"]
        )
    else:
        modes["structure_only"] = structure_total
        missing_mts = [col for col in ("retrieval_mirdb_contrib", "retrieval_ts_contrib") if col not in df.columns]
        if missing_mts:
            warnings["missing_columns"]["mirdb_targetscan_structure"] = missing_mts
            warnings["skipped_modes"].append("mirdb_targetscan_structure")
        else:
            modes["mirdb_targetscan_structure"] = (
                _safe_float_series(df, "retrieval_mirdb_contrib", 0.0)
                + _safe_float_series(df, "retrieval_ts_contrib", 0.0)
                + structure_total
            )
        if "retrieval_score" in df.columns:
            modes["no_structure"] = _safe_float_series(df, "retrieval_score", 0.0) - structure_total
        else:
            warnings["missing_columns"]["no_structure"] = ["retrieval_score"]
            warnings["skipped_modes"].append("no_structure")
        if "retrieval_structure_in_score" in df.columns:
            structure_in_score = _safe_float_series(df, "retrieval_structure_in_score", default=0.0)
            if float(structure_in_score.max()) <= 0.0:
                warnings["notes"].append(
                    "retrieval_structure_contrib was exported diagnostically, but structure features were not included in retrieval_score. "
                    "Interpret no_structure and other leave-one-out comparisons relative to retrieval_score cautiously unless you rerun with MIRASSIST_USE_STRUCTURE_IN_SCORE=1."
                )

    warnings["notes"].append("NaN ablation scores were filled with 0 before reranking.")
    return modes, warnings


def rerank_within_query(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    work = df.copy()
    work["_tie_original_rank"] = _safe_float_series(work, "rank", default=np.inf)
    work["_tie_gene"] = _safe_text_series(work, "gene_symbol", default="")
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce").fillna(0.0)
    work = work.sort_values(
        ["query_id", score_col, "_tie_original_rank", "_tie_gene"],
        ascending=[True, False, True, True],
    ).copy()
    work["ablation_rank"] = work.groupby("query_id").cumcount() + 1
    return work.drop(columns=["_tie_original_rank", "_tie_gene"])


def save_bar_plot(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str, path: Path) -> None:
    plot_df = df.copy()
    plt.figure(figsize=(8, 4))
    plt.bar(plot_df[x].astype(str), plot_df[y])
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    ks = parse_ks(args.ks)
    rankings_df = pd.read_parquet(Path(args.rankings).resolve())
    query_summary_df = pd.read_parquet(Path(args.query_summary).resolve())
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    require_columns(rankings_df, ["query_id", "is_positive"])
    if "rank" not in rankings_df.columns and "retrieval_score" not in rankings_df.columns:
        raise ValueError("rankings_long must contain either 'rank' or 'retrieval_score'.")

    score_modes, warnings_payload = _available_score_modes(rankings_df)
    if args.score_modes:
        requested = [item.strip() for item in str(args.score_modes).split(",") if item.strip()]
        score_modes = {name: series for name, series in score_modes.items() if name in requested}
        missing_requested = [name for name in requested if name not in score_modes]
        if missing_requested:
            warnings_payload["notes"].append(f"Requested score modes not available and skipped: {missing_requested}")
    if not score_modes:
        warnings_payload["notes"].append("No score modes were available to evaluate.")

    summary_lookup = {}
    if not query_summary_df.empty and "query_id" in query_summary_df.columns:
        summary_lookup = query_summary_df.set_index("query_id").to_dict(orient="index")

    ablation_rankings_frames: List[pd.DataFrame] = []
    ablation_query_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    recall_rows: List[Dict[str, Any]] = []
    precision_rows: List[Dict[str, Any]] = []

    for mode_name, score_series in score_modes.items():
        mode_df = rankings_df.copy()
        mode_df["ablation_score"] = pd.to_numeric(score_series, errors="coerce").fillna(0.0)
        reranked = rerank_within_query(mode_df, "ablation_score")
        reranked["score_mode"] = mode_name
        reranked["original_rank"] = pd.to_numeric(reranked.get("rank"), errors="coerce")
        ablation_rankings_frames.append(
            reranked[
                [
                    col
                    for col in [
                        "query_id",
                        "mirna",
                        "gene_symbol",
                        "score_mode",
                        "ablation_score",
                        "ablation_rank",
                        "original_rank",
                        "is_positive",
                    ]
                    if col in reranked.columns
                ]
            ].copy()
        )

        per_query_rows: List[Dict[str, Any]] = []
        for query_id, group in reranked.groupby("query_id", sort=True):
            ordered = group.sort_values("ablation_rank").reset_index(drop=True)
            is_positive = pd.to_numeric(ordered["is_positive"], errors="coerce").fillna(0).astype(int).to_numpy()
            summary_row = summary_lookup.get(query_id, {})
            total_positives = _safe_int(summary_row.get("n_positives_total"), default=int(is_positive.sum()))
            metrics = _compute_metrics_from_vector(is_positive, ks, total_positives)
            row = {
                "score_mode": mode_name,
                "query_id": query_id,
                "mirna": summary_row.get("mirna", ordered["mirna"].iloc[0] if "mirna" in ordered.columns and len(ordered) else ""),
                "n_ranked": int(metrics["n_ranked"]),
                "n_positives_total": int(total_positives),
                "n_positives_retrieved": int(metrics["n_positives_retrieved"]),
                "best_positive_rank": metrics["best_positive_rank"],
                "reciprocal_rank": metrics["reciprocal_rank"],
            }
            for k in ks:
                row[f"positive_count_at_{k}"] = int(metrics[f"positive_count_at_{k}"])
                row[f"recall_at_{k}"] = metrics[f"recall_at_{k}"]
                row[f"precision_at_{k}"] = metrics[f"precision_at_{k}"]
            per_query_rows.append(row)
        per_query_df = pd.DataFrame(per_query_rows)
        ablation_query_rows.extend(per_query_rows)

        for k in ks:
            recall_rows.append(
                {
                    "score_mode": mode_name,
                    "k": k,
                    "recall_at_k": float(np.nanmean(per_query_df[f"recall_at_{k}"])) if not per_query_df.empty else np.nan,
                }
            )
            precision_rows.append(
                {
                    "score_mode": mode_name,
                    "k": k,
                    "precision_at_k": float(np.nanmean(per_query_df[f"precision_at_{k}"])) if not per_query_df.empty else np.nan,
                }
            )

        best_rank_series = pd.to_numeric(per_query_df["best_positive_rank"], errors="coerce")
        labels = pd.to_numeric(reranked["is_positive"], errors="coerce").fillna(0).astype(int).tolist()
        scores = pd.to_numeric(reranked["ablation_score"], errors="coerce").fillna(0.0).tolist()
        auroc = roc_auc_score_manual(labels, scores) if len(set(labels)) > 1 else None
        auprc = average_precision_manual(labels, scores) if any(labels) else None
        summary_rows.append(
            {
                "score_mode": mode_name,
                "n_queries": int(len(per_query_df)),
                "n_ranked_interactions": int(len(reranked)),
                "positives_retrieved": int(pd.to_numeric(per_query_df["n_positives_retrieved"], errors="coerce").fillna(0).sum())
                if not per_query_df.empty
                else 0,
                "mrr": float(np.nanmean(per_query_df["reciprocal_rank"])) if not per_query_df.empty else np.nan,
                "mean_best_positive_rank": float(best_rank_series.dropna().mean()) if best_rank_series.notna().any() else np.nan,
                "median_best_positive_rank": float(best_rank_series.dropna().median()) if best_rank_series.notna().any() else np.nan,
                "auroc": auroc,
                "auprc": auprc,
            }
        )

    ablation_query_df = pd.DataFrame(ablation_query_rows)
    summary_df = pd.DataFrame(summary_rows)
    recall_df = pd.DataFrame(recall_rows)
    precision_df = pd.DataFrame(precision_rows)
    ablation_rankings_df = pd.concat(ablation_rankings_frames, ignore_index=True) if ablation_rankings_frames else pd.DataFrame()

    summary_df.to_csv(outdir / "ablation_metrics_summary.csv", index=False)
    recall_df.to_csv(outdir / "ablation_recall_at_k.csv", index=False)
    precision_df.to_csv(outdir / "ablation_precision_at_k.csv", index=False)
    ablation_query_df.to_parquet(outdir / "ablation_query_metrics.parquet", index=False)
    ablation_query_df.to_csv(outdir / "ablation_query_metrics.csv", index=False)
    ablation_rankings_df.to_parquet(outdir / "ablation_rankings_long.parquet", index=False)
    json_dump(outdir / "ablation_warnings.json", warnings_payload)

    if not summary_df.empty:
        save_bar_plot(summary_df, "score_mode", "mrr", "Ablation Comparison: MRR", "MRR", outdir / "ablation_mrr.png")
        if "auprc" in summary_df.columns and summary_df["auprc"].notna().any():
            save_bar_plot(summary_df, "score_mode", "auprc", "Ablation Comparison: AUPRC", "AUPRC", outdir / "ablation_auprc.png")
        save_bar_plot(
            summary_df,
            "score_mode",
            "median_best_positive_rank",
            "Ablation Comparison: Median Best Positive Rank",
            "Median best positive rank",
            outdir / "ablation_median_best_positive_rank.png",
        )
    if not recall_df.empty:
        recall_subset = recall_df[recall_df["k"].isin([k for k in (10, 25, 50) if k in ks])].copy()
        if not recall_subset.empty:
            pivot = recall_subset.pivot(index="score_mode", columns="k", values="recall_at_k").fillna(0.0)
            ax = pivot.plot(kind="bar", figsize=(8, 4))
            ax.set_title("Ablation Comparison: Recall@10/25/50")
            ax.set_ylabel("Recall")
            ax.set_xlabel("Score mode")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(outdir / "ablation_recall_at_10_25_50.png", dpi=160)
            plt.close()
    if not precision_df.empty:
        precision_subset = precision_df[precision_df["k"].isin([k for k in (10, 25, 50) if k in ks])].copy()
        if not precision_subset.empty:
            pivot = precision_subset.pivot(index="score_mode", columns="k", values="precision_at_k").fillna(0.0)
            ax = pivot.plot(kind="bar", figsize=(8, 4))
            ax.set_title("Ablation Comparison: Precision@10/25/50")
            ax.set_ylabel("Precision")
            ax.set_xlabel("Score mode")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(outdir / "ablation_precision_at_10_25_50.png", dpi=160)
            plt.close()

    if args.include_random_baseline:
        warnings_payload["notes"].append(
            f"Random baseline outputs were provided at {args.include_random_baseline} but are not consumed directly by this script."
        )
        json_dump(outdir / "ablation_warnings.json", warnings_payload)


if __name__ == "__main__":
    main()
