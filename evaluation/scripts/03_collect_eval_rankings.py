from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.utils import collect_rankings_from_json  # noqa: E402
from evaluation.utils import json_dump  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--outdir", required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    labels_df = pd.read_parquet(Path(args.labels).resolve())
    rankings_df, query_summary_df, collection_summary = collect_rankings_from_json(args.json_dir, labels_df)

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rankings_df.to_parquet(outdir / "rankings_long.parquet", index=False)
    rankings_df.to_csv(outdir / "rankings_long.csv", index=False)
    query_summary_df.to_parquet(outdir / "query_summary.parquet", index=False)
    query_summary_df.to_csv(outdir / "query_summary.csv", index=False)
    json_dump(outdir / "collection_summary.json", collection_summary)

    print(f"Wrote rankings_long and query_summary tables to {outdir}")


if __name__ == "__main__":
    main()
