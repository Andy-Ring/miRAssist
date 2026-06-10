from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.utils import average_precision_manual, json_dump, roc_auc_score_manual  # noqa: E402


TOP_KS = (1, 5, 10, 25, 50, 100)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rankings-long", required=True)
    ap.add_argument("--query-summary", required=True)
    ap.add_argument("--outdir", required=True)
    return ap.parse_args()


def compute_recall_precision(rankings_df: pd.DataFrame, query_summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    recall_rows = []
    precision_rows = []
    total_positives = max(1, int(query_summary_df["n_positives_total"].sum()))
    n_queries = max(1, int(len(query_summary_df)))

    for k in TOP_KS:
        topk = rankings_df[rankings_df["rank"] <= k]
        positives_topk = int(topk["is_positive"].sum()) if not topk.empty else 0
        evaluated_slots = int(
            query_summary_df["n_ranked"].clip(upper=k).sum()
        ) if "n_ranked" in query_summary_df.columns else (k * n_queries)
        evaluated_slots = max(1, evaluated_slots)
        recall_rows.append(
            {
                "k": k,
                "positives_in_top_k": positives_topk,
                "total_positives": int(query_summary_df["n_positives_total"].sum()),
                "recall_at_k": positives_topk / total_positives,
            }
        )
        precision_rows.append(
            {
                "k": k,
                "positives_in_top_k": positives_topk,
                "evaluated_slots": evaluated_slots,
                "precision_at_k": positives_topk / float(evaluated_slots),
            }
        )
    return pd.DataFrame(recall_rows), pd.DataFrame(precision_rows)


def main() -> None:
    args = parse_args()
    rankings_df = pd.read_parquet(Path(args.rankings_long).resolve())
    query_summary_df = pd.read_parquet(Path(args.query_summary).resolve())

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    recall_df, precision_df = compute_recall_precision(rankings_df, query_summary_df)
    recall_df.to_csv(outdir / "recall_at_k.csv", index=False)
    precision_df.to_csv(outdir / "precision_at_k.csv", index=False)

    best_rank_series = pd.to_numeric(query_summary_df["best_positive_rank"], errors="coerce")
    reciprocal = best_rank_series.dropna().map(lambda value: 1.0 / float(value) if value > 0 else 0.0)
    positives_retrieved = int(query_summary_df["n_positives_retrieved"].sum()) if "n_positives_retrieved" in query_summary_df.columns else 0

    enrichment_rows = []
    background_rate = (
        float(rankings_df["is_positive"].mean())
        if not rankings_df.empty and "is_positive" in rankings_df.columns
        else 0.0
    )
    for k in TOP_KS:
        topk = rankings_df[rankings_df["rank"] <= k]
        observed_rate = float(topk["is_positive"].mean()) if not topk.empty else 0.0
        enrichment_rows.append(
            {
                "k": k,
                "observed_positive_rate": observed_rate,
                "background_positive_rate": background_rate,
                "enrichment": (observed_rate / background_rate) if background_rate > 0 else np.nan,
            }
        )
    enrichment_df = pd.DataFrame(enrichment_rows)
    enrichment_df.to_csv(outdir / "topk_enrichment.csv", index=False)

    metrics_by_query = query_summary_df.copy()
    if "n_positives_total" in metrics_by_query.columns:
        metrics_by_query["retrieval_fraction"] = np.where(
            metrics_by_query["n_positives_total"] > 0,
            metrics_by_query["n_positives_retrieved"] / metrics_by_query["n_positives_total"],
            np.nan,
        )
    metrics_by_query.to_csv(outdir / "metrics_by_query.csv", index=False)
    metrics_by_query.to_parquet(outdir / "metrics_by_query.parquet", index=False)

    auroc = None
    auprc = None
    if not rankings_df.empty and "retrieval_score" in rankings_df.columns:
        auroc = roc_auc_score_manual(rankings_df["is_positive"].tolist(), rankings_df["retrieval_score"].tolist())
        auprc = average_precision_manual(rankings_df["is_positive"].tolist(), rankings_df["retrieval_score"].tolist())

    metrics_summary = {
        "number_of_queries": int(len(query_summary_df)),
        "number_of_ranked_interactions": int(len(rankings_df)),
        "number_of_heldout_positives": int(query_summary_df["n_positives_total"].sum()) if "n_positives_total" in query_summary_df.columns else 0,
        "positives_retrieved": positives_retrieved,
        "mrr": float(reciprocal.mean()) if len(reciprocal) else None,
        "mean_best_positive_rank": float(best_rank_series.dropna().mean()) if best_rank_series.notna().any() else None,
        "median_best_positive_rank": float(best_rank_series.dropna().median()) if best_rank_series.notna().any() else None,
        "auroc": auroc,
        "auprc": auprc,
    }
    json_dump(outdir / "metrics_summary.json", metrics_summary)
    print(json.dumps(metrics_summary, indent=2))


if __name__ == "__main__":
    main()
