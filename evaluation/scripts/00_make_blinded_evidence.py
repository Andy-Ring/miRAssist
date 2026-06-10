from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.utils import hash_file, json_dump, make_blinded_evidence  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True, help="Input evidence parquet")
    ap.add_argument("--outdir", required=True, help="Output directory, usually evaluation/data")
    ap.add_argument(
        "--label-cols",
        default="mirtarbase_pos,label_mirtarbase",
        help="Comma-separated held-out label columns",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    evidence_path = Path(args.evidence).resolve()
    label_cols = [col.strip() for col in args.label_cols.split(",") if col.strip()]

    evidence_df = pd.read_parquet(evidence_path)
    blinded_df, heldout_df, audit = make_blinded_evidence(evidence_df, label_cols=label_cols)

    blinded_path = outdir / "evidence_blinded_no_mirtarbase.parquet"
    labels_path = outdir / "heldout_mirtarbase_labels.parquet"
    audit_path = outdir / "blinding_audit.json"

    blinded_df.to_parquet(blinded_path, index=False)
    heldout_df.to_parquet(labels_path, index=False)

    audit["input_path"] = str(evidence_path)
    audit["blinded_output_path"] = str(blinded_path)
    audit["label_output_path"] = str(labels_path)
    audit["blinded_output_sha256"] = hash_file(blinded_path)
    audit["label_output_sha256"] = hash_file(labels_path)
    json_dump(audit_path, audit)

    print(f"Wrote blinded evidence to {blinded_path}")
    print(f"Wrote held-out labels to {labels_path}")
    print(f"Wrote audit report to {audit_path}")


if __name__ == "__main__":
    main()
