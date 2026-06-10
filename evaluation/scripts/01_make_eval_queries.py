from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.utils import normalize_mirna_name, normalize_gene_symbol  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blinded-evidence", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="mirna_to_targets", choices=["mirna_to_targets"])
    ap.add_argument("--min-positive-count", type=int, default=1)
    ap.add_argument("--max-mirnas", type=int, default=None)
    ap.add_argument("--only-mirnas", default=None, help="Optional text file with one miRNA per line")
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--min-support", type=int, default=1)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    blinded_df = pd.read_parquet(Path(args.blinded_evidence).resolve())
    labels_df = pd.read_parquet(Path(args.labels).resolve())

    if "mirna_name_normalized" not in labels_df.columns:
        labels_df["mirna_name_normalized"] = labels_df["mirna_name"].map(normalize_mirna_name)
    if "gene_symbol_normalized" not in labels_df.columns:
        labels_df["gene_symbol_normalized"] = labels_df["gene_symbol"].map(normalize_gene_symbol)

    labels_df["mirtarbase_pos"] = pd.to_numeric(labels_df.get("mirtarbase_pos", 0), errors="coerce").fillna(0).astype(int)
    positive_df = labels_df[labels_df["mirtarbase_pos"] > 0].copy()

    if args.only_mirnas:
        allowed = {
            line.strip()
            for line in Path(args.only_mirnas).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        positive_df = positive_df[positive_df["mirna_name"].isin(allowed)].copy()

    candidate_counts = (
        blinded_df.assign(gene_symbol_normalized=blinded_df["gene_symbol"].map(normalize_gene_symbol))
        .groupby("mirna_name", as_index=False)["gene_symbol_normalized"]
        .nunique()
        .rename(columns={"gene_symbol_normalized": "n_candidate_rows"})
    )
    positive_counts = (
        positive_df.groupby("mirna_name", as_index=False)
        .agg(label_positive_count=("gene_symbol_normalized", "nunique"))
        .sort_values(["label_positive_count", "mirna_name"], ascending=[False, True])
    )
    positive_counts = positive_counts[positive_counts["label_positive_count"] >= int(args.min_positive_count)].copy()
    if args.max_mirnas is not None:
        positive_counts = positive_counts.head(int(args.max_mirnas)).copy()

    manifest = positive_counts.merge(candidate_counts, how="left", on="mirna_name")
    manifest["n_candidate_rows"] = pd.to_numeric(manifest["n_candidate_rows"], errors="coerce").fillna(0).astype(int)
    manifest["query_id"] = [f"eval_{idx:05d}" for idx in range(len(manifest))]
    manifest["mode"] = args.mode
    manifest["mirna"] = manifest["mirna_name"]
    manifest["gene"] = ""
    manifest["question"] = manifest["mirna"].map(lambda value: f"What genes are regulated by {value}?")
    manifest["k"] = int(args.k)
    manifest["min_support"] = int(args.min_support)
    manifest["novel"] = False
    manifest["use_pathway_filter"] = False
    manifest["cancer_context"] = ""
    manifest["tcga"] = ""
    manifest["mirna_name_normalized"] = manifest["mirna"].map(normalize_mirna_name)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest[
        [
            "query_id",
            "mode",
            "mirna",
            "mirna_name_normalized",
            "gene",
            "question",
            "k",
            "min_support",
            "novel",
            "use_pathway_filter",
            "cancer_context",
            "tcga",
            "label_positive_count",
            "n_candidate_rows",
        ]
    ].to_csv(out_path, index=False)
    print(f"Wrote {len(manifest)} evaluation queries to {out_path}")


if __name__ == "__main__":
    main()
