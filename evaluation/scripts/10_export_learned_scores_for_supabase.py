from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored-evidence", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--score-columns", default="learned_score_xgb_raw_v1,learned_score_xgb_raw_nomissing_v1")
    return ap.parse_args()


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path).resolve()
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported scored-evidence format for {path}. Use .parquet or .csv.")


def main() -> None:
    args = parse_args()
    df = read_table(args.scored_evidence)
    requested_scores = [part.strip() for part in str(args.score_columns).split(",") if part.strip()]

    required_keys = ["mirna_name_norm", "gene_symbol_norm"]
    optional_keys = ["transcript_id"]
    metadata_cols = [
        "learned_score_model_version",
        "learned_score_feature_set",
        "learned_score_updated_at",
    ]

    missing_required = [col for col in required_keys if col not in df.columns]
    if missing_required:
        raise ValueError(f"Scored evidence was missing required key columns: {missing_required}")

    keep_columns = list(required_keys)
    keep_columns.extend([col for col in optional_keys if col in df.columns])
    keep_columns.extend([col for col in requested_scores if col in df.columns])
    keep_columns.extend([col for col in metadata_cols if col in df.columns])
    if len(keep_columns) == len(required_keys) + len([col for col in optional_keys if col in df.columns]) + len([col for col in metadata_cols if col in df.columns]):
        raise ValueError("None of the requested score columns were present in the scored evidence.")

    out_df = df.loc[:, list(dict.fromkeys(keep_columns))].copy()
    out_path = Path(args.out_csv).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote Supabase learned-score export CSV to {out_path}")


if __name__ == "__main__":
    main()
