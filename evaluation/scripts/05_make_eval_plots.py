from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", required=True)
    ap.add_argument("--tables-dir", required=True)
    ap.add_argument("--outdir", required=True)
    return ap.parse_args()


def save_line_plot(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str, path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(df[x], df[y], marker="o")
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    metrics_dir = Path(args.metrics_dir).resolve()
    tables_dir = Path(args.tables_dir).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    recall_df = pd.read_csv(metrics_dir / "recall_at_k.csv")
    precision_df = pd.read_csv(metrics_dir / "precision_at_k.csv")
    query_summary_df = pd.read_parquet(metrics_dir / "metrics_by_query.parquet")
    rankings_df = pd.read_parquet(tables_dir / "rankings_long.parquet")
    enrichment_df = pd.read_csv(metrics_dir / "topk_enrichment.csv")

    save_line_plot(recall_df, "k", "recall_at_k", "Recall@K", "Recall", outdir / "recall_at_k.png")
    save_line_plot(precision_df, "k", "precision_at_k", "Precision@K", "Precision", outdir / "precision_at_k.png")
    save_line_plot(enrichment_df, "k", "enrichment", "Top-K Positive Enrichment", "Enrichment", outdir / "topK_enrichment.png")

    plt.figure(figsize=(6, 4))
    best_ranks = pd.to_numeric(query_summary_df["best_positive_rank"], errors="coerce").dropna()
    plt.hist(best_ranks, bins=min(30, max(5, len(best_ranks))))
    plt.title("Best Positive Rank Distribution")
    plt.xlabel("Best positive rank")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outdir / "histogram_best_positive_rank.png", dpi=160)
    plt.close()

    if not rankings_df.empty and "retrieval_score" in rankings_df.columns:
        plt.figure(figsize=(6, 4))
        rankings_df = rankings_df.copy()
        rankings_df["label_name"] = rankings_df["is_positive"].map({1: "Positive", 0: "Other"})
        for label_name, subset in rankings_df.groupby("label_name"):
            plt.hist(subset["retrieval_score"], bins=40, alpha=0.6, label=label_name)
        plt.legend()
        plt.title("Retrieval Score Distribution")
        plt.xlabel("retrieval_score")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(outdir / "score_distribution_positive_vs_other.png", dpi=160)
        plt.close()


if __name__ == "__main__":
    main()
