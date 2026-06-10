from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.utils import json_dump  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rankings", required=True, help="Path to rankings_long.parquet")
    ap.add_argument("--query-summary", required=True, help="Path to query_summary.parquet")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-permutations", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--ks", default="1,5,10,25,50,100")
    return ap.parse_args()


def parse_ks(text: str) -> List[int]:
    values = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
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


def _compute_metrics_from_vector(is_positive: np.ndarray, ks: List[int], total_positives: int | None) -> Dict[str, float]:
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
        metrics[f"precision_at_{k}"] = (
            positives_in_topk / float(max(1, min(k, n_ranked))) if n_ranked > 0 else 0.0
        )
        metrics[f"recall_at_{k}"] = (
            positives_in_topk / float(denominator) if denominator > 0 else np.nan
        )
    return metrics


def summarize_random_distribution(values: np.ndarray) -> Dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": np.nan, "sd": np.nan, "p025": np.nan, "p975": np.nan}
    return {
        "mean": float(np.mean(finite)),
        "sd": float(np.std(finite, ddof=0)),
        "p025": float(np.percentile(finite, 2.5)),
        "p975": float(np.percentile(finite, 97.5)),
    }


def empirical_p_value(random_values: np.ndarray, observed_value: float, higher_is_better: bool) -> float | None:
    finite = random_values[np.isfinite(random_values)]
    if finite.size == 0 or not np.isfinite(observed_value):
        return None
    if higher_is_better:
        return float(np.mean(finite >= observed_value))
    return float(np.mean(finite <= observed_value))


def save_observed_vs_random_plot(
    df: pd.DataFrame,
    observed_col: str,
    random_mean_col: str,
    lower_col: str,
    upper_col: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(df["k"], df[observed_col], marker="o", label="Observed")
    plt.plot(df["k"], df[random_mean_col], marker="o", label="Random mean")
    plt.fill_between(df["k"], df[lower_col], df[upper_col], alpha=0.2, label="Random 95% interval")
    plt.xlabel("K")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    ks = parse_ks(args.ks)
    rankings_df = pd.read_parquet(Path(args.rankings).resolve())
    query_summary_df = pd.read_parquet(Path(args.query_summary).resolve())

    require_columns(rankings_df, ["query_id", "rank", "is_positive"])

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))

    summary_lookup = {}
    if not query_summary_df.empty and "query_id" in query_summary_df.columns:
        summary_lookup = query_summary_df.set_index("query_id").to_dict(orient="index")

    observed_query_rows: List[Dict[str, Any]] = []
    random_query_rows: List[Dict[str, Any]] = []

    aggregate_random_rr = np.zeros(int(args.n_permutations), dtype=float)
    aggregate_random_best_rank_lists: List[List[float]] = [[] for _ in range(int(args.n_permutations))]
    aggregate_random_recall = {k: np.full(int(args.n_permutations), np.nan, dtype=float) for k in ks}
    aggregate_random_precision = {k: np.zeros(int(args.n_permutations), dtype=float) for k in ks}

    observed_rr_values: List[float] = []
    observed_best_rank_values: List[float] = []
    observed_recall_values = {k: [] for k in ks}
    observed_precision_values = {k: [] for k in ks}

    grouped = rankings_df.sort_values(["query_id", "rank"]).groupby("query_id", sort=True)
    n_queries = 0
    for query_id, group in grouped:
        n_queries += 1
        ordered = group.sort_values("rank").reset_index(drop=True)
        is_positive = pd.to_numeric(ordered["is_positive"], errors="coerce").fillna(0).astype(int).to_numpy()
        summary_row = summary_lookup.get(query_id, {})
        total_positives = _safe_int(summary_row.get("n_positives_total"), default=int(is_positive.sum()))
        observed_metrics = _compute_metrics_from_vector(is_positive, ks, total_positives)
        observed_row: Dict[str, Any] = {
            "query_id": query_id,
            "mirna": summary_row.get("mirna", ordered.get("mirna", pd.Series([""])).iloc[0] if "mirna" in ordered.columns else ""),
            "n_ranked": int(observed_metrics["n_ranked"]),
            "n_positives_total": int(total_positives),
            "n_positives_retrieved": int(observed_metrics["n_positives_retrieved"]),
            "best_positive_rank": observed_metrics["best_positive_rank"],
            "reciprocal_rank": observed_metrics["reciprocal_rank"],
        }

        random_metric_arrays: Dict[str, np.ndarray] = {}
        for metric_name in ["best_positive_rank", "reciprocal_rank"]:
            fill_value = np.nan if metric_name == "best_positive_rank" else 0.0
            random_metric_arrays[metric_name] = np.full(int(args.n_permutations), fill_value, dtype=float)
        for k in ks:
            random_metric_arrays[f"positive_count_at_{k}"] = np.zeros(int(args.n_permutations), dtype=float)
            random_metric_arrays[f"recall_at_{k}"] = np.full(int(args.n_permutations), np.nan, dtype=float)
            random_metric_arrays[f"precision_at_{k}"] = np.zeros(int(args.n_permutations), dtype=float)

        for perm_idx in range(int(args.n_permutations)):
            shuffled = np.array(is_positive, copy=True)
            rng.shuffle(shuffled)
            random_metrics = _compute_metrics_from_vector(shuffled, ks, total_positives)
            for key, value in random_metrics.items():
                if key in random_metric_arrays:
                    random_metric_arrays[key][perm_idx] = value

        for k in ks:
            observed_row[f"positive_count_at_{k}"] = observed_metrics[f"positive_count_at_{k}"]
            observed_row[f"recall_at_{k}"] = observed_metrics[f"recall_at_{k}"]
            observed_row[f"precision_at_{k}"] = observed_metrics[f"precision_at_{k}"]
            observed_recall_values[k].append(observed_metrics[f"recall_at_{k}"])
            observed_precision_values[k].append(observed_metrics[f"precision_at_{k}"])
        observed_rr_values.append(observed_metrics["reciprocal_rank"])
        observed_best_rank_values.append(observed_metrics["best_positive_rank"])
        observed_query_rows.append(observed_row)

        random_row: Dict[str, Any] = {
            "query_id": query_id,
            "mirna": observed_row["mirna"],
            "n_ranked": observed_row["n_ranked"],
            "n_positives_total": observed_row["n_positives_total"],
            "random_reciprocal_rank_mean": float(np.nanmean(random_metric_arrays["reciprocal_rank"])),
            "random_reciprocal_rank_sd": float(np.nanstd(random_metric_arrays["reciprocal_rank"])),
            "random_best_positive_rank_mean": float(np.nanmean(random_metric_arrays["best_positive_rank"]))
            if np.isfinite(random_metric_arrays["best_positive_rank"]).any()
            else np.nan,
            "random_best_positive_rank_sd": float(np.nanstd(random_metric_arrays["best_positive_rank"]))
            if np.isfinite(random_metric_arrays["best_positive_rank"]).any()
            else np.nan,
        }
        for k in ks:
            random_row[f"random_positive_count_at_{k}_mean"] = float(np.nanmean(random_metric_arrays[f"positive_count_at_{k}"]))
            random_row[f"random_recall_at_{k}_mean"] = float(np.nanmean(random_metric_arrays[f"recall_at_{k}"]))
            random_row[f"random_precision_at_{k}_mean"] = float(np.nanmean(random_metric_arrays[f"precision_at_{k}"]))
        random_query_rows.append(random_row)

        aggregate_random_rr += np.nan_to_num(random_metric_arrays["reciprocal_rank"], nan=0.0)
        if np.isfinite(observed_metrics["best_positive_rank"]):
            observed_best_rank_values[-1] = observed_metrics["best_positive_rank"]
        for perm_idx in range(int(args.n_permutations)):
            if np.isfinite(random_metric_arrays["best_positive_rank"][perm_idx]):
                aggregate_random_best_rank_lists[perm_idx].append(random_metric_arrays["best_positive_rank"][perm_idx])
        for k in ks:
            if query_id not in summary_lookup or _safe_int(summary_row.get("n_positives_total"), default=0) > 0:
                if np.isnan(aggregate_random_recall[k]).all():
                    aggregate_random_recall[k] = np.zeros(int(args.n_permutations), dtype=float)
                aggregate_random_recall[k] = np.nan_to_num(aggregate_random_recall[k], nan=0.0)
                aggregate_random_recall[k] += np.nan_to_num(random_metric_arrays[f"recall_at_{k}"], nan=0.0)
            aggregate_random_precision[k] += np.nan_to_num(random_metric_arrays[f"precision_at_{k}"], nan=0.0)

    observed_query_df = pd.DataFrame(observed_query_rows)
    random_query_df = pd.DataFrame(random_query_rows)

    observed_query_df.to_parquet(outdir / "observed_query_metrics.parquet", index=False)
    random_query_df.to_parquet(outdir / "random_query_metrics.parquet", index=False)

    n_queries_for_rr = max(1, len(observed_query_df))
    random_mrr_distribution = aggregate_random_rr / float(n_queries_for_rr)
    observed_mrr = float(np.nanmean(observed_rr_values)) if observed_rr_values else 0.0
    observed_best_rank_series = pd.to_numeric(
        observed_query_df["best_positive_rank"], errors="coerce"
    ) if not observed_query_df.empty else pd.Series(dtype=float)
    observed_median_best_rank = (
        float(np.nanmedian(observed_best_rank_series))
        if not observed_best_rank_series.empty and observed_best_rank_series.notna().any()
        else np.nan
    )

    recall_rows = []
    precision_rows = []
    for k in ks:
        observed_recall = float(np.nanmean([v for v in observed_recall_values[k] if np.isfinite(v)])) if any(
            np.isfinite(v) for v in observed_recall_values[k]
        ) else np.nan
        valid_recall_query_count = max(1, sum(1 for v in observed_recall_values[k] if np.isfinite(v)))
        random_recall_distribution = aggregate_random_recall[k] / float(valid_recall_query_count)
        recall_summary = summarize_random_distribution(random_recall_distribution)
        recall_rows.append(
            {
                "k": k,
                "observed_recall_at_k": observed_recall,
                "random_mean_recall_at_k": recall_summary["mean"],
                "random_sd_recall_at_k": recall_summary["sd"],
                "random_p025_recall_at_k": recall_summary["p025"],
                "random_p975_recall_at_k": recall_summary["p975"],
                "fold_enrichment": (
                    observed_recall / recall_summary["mean"]
                    if np.isfinite(observed_recall) and np.isfinite(recall_summary["mean"]) and recall_summary["mean"] != 0
                    else np.nan
                ),
                "empirical_p_value": empirical_p_value(random_recall_distribution, observed_recall, higher_is_better=True),
            }
        )

        observed_precision = float(np.nanmean(observed_precision_values[k])) if observed_precision_values[k] else 0.0
        random_precision_distribution = aggregate_random_precision[k] / float(n_queries_for_rr)
        precision_summary = summarize_random_distribution(random_precision_distribution)
        precision_rows.append(
            {
                "k": k,
                "observed_precision_at_k": observed_precision,
                "random_mean_precision_at_k": precision_summary["mean"],
                "random_sd_precision_at_k": precision_summary["sd"],
                "random_p025_precision_at_k": precision_summary["p025"],
                "random_p975_precision_at_k": precision_summary["p975"],
                "fold_enrichment": (
                    observed_precision / precision_summary["mean"]
                    if np.isfinite(precision_summary["mean"]) and precision_summary["mean"] != 0
                    else np.nan
                ),
                "empirical_p_value": empirical_p_value(random_precision_distribution, observed_precision, higher_is_better=True),
            }
        )

    recall_df = pd.DataFrame(recall_rows)
    precision_df = pd.DataFrame(precision_rows)
    recall_df.to_csv(outdir / "observed_vs_random_recall_at_k.csv", index=False)
    precision_df.to_csv(outdir / "observed_vs_random_precision_at_k.csv", index=False)

    random_best_rank_distribution = np.array(
        [float(np.median(values)) if values else np.nan for values in aggregate_random_best_rank_lists],
        dtype=float,
    )
    mrr_summary = summarize_random_distribution(random_mrr_distribution)
    best_rank_summary = summarize_random_distribution(random_best_rank_distribution)
    observed_vs_random_mrr_df = pd.DataFrame(
        [
            {
                "metric": "mrr",
                "observed": observed_mrr,
                "random_mean": mrr_summary["mean"],
                "random_sd": mrr_summary["sd"],
                "random_p025": mrr_summary["p025"],
                "random_p975": mrr_summary["p975"],
                "fold_enrichment": (
                    observed_mrr / mrr_summary["mean"]
                    if np.isfinite(mrr_summary["mean"]) and mrr_summary["mean"] != 0
                    else np.nan
                ),
                "empirical_p_value": empirical_p_value(random_mrr_distribution, observed_mrr, higher_is_better=True),
            },
            {
                "metric": "median_best_positive_rank",
                "observed": observed_median_best_rank,
                "random_mean": best_rank_summary["mean"],
                "random_sd": best_rank_summary["sd"],
                "random_p025": best_rank_summary["p025"],
                "random_p975": best_rank_summary["p975"],
                "fold_enrichment": np.nan,
                "empirical_p_value": empirical_p_value(
                    random_best_rank_distribution,
                    observed_median_best_rank,
                    higher_is_better=False,
                ),
            },
        ]
    )
    observed_vs_random_mrr_df.to_csv(outdir / "observed_vs_random_mrr.csv", index=False)

    summary_payload = {
        "n_queries": int(len(observed_query_df)),
        "n_ranked_rows": int(len(rankings_df)),
        "n_permutations": int(args.n_permutations),
        "seed": int(args.seed),
        "ks": ks,
        "observed_mrr": observed_mrr,
        "random_mrr_mean": mrr_summary["mean"],
        "random_mrr_sd": mrr_summary["sd"],
        "random_mrr_p025": mrr_summary["p025"],
        "random_mrr_p975": mrr_summary["p975"],
        "observed_median_best_positive_rank": (None if not np.isfinite(observed_median_best_rank) else observed_median_best_rank),
        "random_median_best_positive_rank_mean": best_rank_summary["mean"],
        "random_median_best_positive_rank_sd": best_rank_summary["sd"],
    }
    json_dump(outdir / "random_baseline_summary.json", summary_payload)

    if not recall_df.empty:
        save_observed_vs_random_plot(
            recall_df,
            observed_col="observed_recall_at_k",
            random_mean_col="random_mean_recall_at_k",
            lower_col="random_p025_recall_at_k",
            upper_col="random_p975_recall_at_k",
            ylabel="Recall",
            title="Observed vs Random Recall@K",
            path=outdir / "observed_vs_random_recall_at_k.png",
        )
    if not precision_df.empty:
        save_observed_vs_random_plot(
            precision_df,
            observed_col="observed_precision_at_k",
            random_mean_col="random_mean_precision_at_k",
            lower_col="random_p025_precision_at_k",
            upper_col="random_p975_precision_at_k",
            ylabel="Precision",
            title="Observed vs Random Precision@K",
            path=outdir / "observed_vs_random_precision_at_k.png",
        )
    plt.figure(figsize=(5, 4))
    mrr_plot_df = observed_vs_random_mrr_df[observed_vs_random_mrr_df["metric"] == "mrr"].copy()
    if not mrr_plot_df.empty:
        row = mrr_plot_df.iloc[0]
        plt.bar(["Observed", "Random mean"], [row["observed"], row["random_mean"]], color=["#1f77b4", "#ff7f0e"])
        plt.errorbar(
            [1],
            [row["random_mean"]],
            yerr=[[row["random_mean"] - row["random_p025"]], [row["random_p975"] - row["random_mean"]]],
            fmt="none",
            ecolor="black",
            capsize=4,
        )
        plt.ylabel("MRR")
        plt.title("Observed vs Random MRR")
        plt.tight_layout()
        plt.savefig(outdir / "observed_vs_random_mrr.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
