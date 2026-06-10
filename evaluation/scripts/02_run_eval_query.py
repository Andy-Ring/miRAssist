from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys
import traceback

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.utils import (  # noqa: E402
    assert_eval_mode,
    build_eval_queryspec,
    json_dump,
    run_eval_query,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--query-id", default=None)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--min-support", type=int, default=None)
    ap.add_argument(
        "--enable-synthesis",
        action="store_true",
        help="Run cards/prompt/synthesis instead of deterministic rank-only mode",
    )
    return ap.parse_args()


def load_manifest_row(path: Path, index: int | None, query_id: str | None) -> pd.Series:
    manifest = pd.read_csv(path)
    if query_id:
        matches = manifest[manifest["query_id"].astype(str) == str(query_id)]
        if matches.empty:
            raise ValueError(f"Query ID {query_id!r} not found in manifest.")
        return matches.iloc[0]
    if index is None:
        raise ValueError("Provide either --index or --query-id.")
    if index < 0 or index >= len(manifest):
        raise IndexError(f"Manifest index {index} is out of range for {len(manifest)} rows.")
    return manifest.iloc[int(index)]


def main() -> None:
    args = parse_args()
    assert_eval_mode()

    manifest_path = Path(args.manifest).resolve()
    row = load_manifest_row(manifest_path, args.index, args.query_id)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    k_value = int(args.k if args.k is not None else row.get("k", 1000))
    min_support_value = int(args.min_support if args.min_support is not None else row.get("min_support", 1))
    disable_synthesis = not args.enable_synthesis

    queryspec = build_eval_queryspec(
        query_id=str(row["query_id"]),
        mode=str(row.get("mode", "mirna_to_targets")),
        mirna=(str(row.get("mirna", "")).strip() or None),
        gene=(str(row.get("gene", "")).strip() or None),
        k=k_value,
        min_support=min_support_value,
        novel=bool(row.get("novel", False)),
        use_pathway_filter=bool(row.get("use_pathway_filter", False)),
        cancer_name=(str(row.get("cancer_context", "")).strip() or None),
        tcga=(str(row.get("tcga", "")).strip() or None),
    )
    queryspec["original_question"] = str(row.get("question") or queryspec["original_question"])

    output_path = outdir / f"{queryspec['query_id']}.json"

    try:
        result = run_eval_query(
            queryspec=queryspec,
            evidence_path=args.evidence,
            disable_synthesis=disable_synthesis,
        )
        payload = {
            "query_id": result.query_id,
            "queryspec": result.queryspec,
            "retrieval_diagnostics": result.retrieval_diagnostics,
            "shortlist": result.shortlist.to_dict(orient="records"),
            "answer": result.answer,
            "environment": {
                "MIRASSIST_EVAL_MODE": os.getenv("MIRASSIST_EVAL_MODE"),
                "MIRASSIST_DISABLE_SYNTHESIS": os.getenv("MIRASSIST_DISABLE_SYNTHESIS"),
                "MIRASSIST_USE_STRUCTURE_IN_SCORE": os.getenv("MIRASSIST_USE_STRUCTURE_IN_SCORE"),
                "EVIDENCE_BACKEND": os.getenv("EVIDENCE_BACKEND"),
                "JOBSTORE_BACKEND": os.getenv("JOBSTORE_BACKEND"),
                "MIRASSIST_EVIDENCE": os.getenv("MIRASSIST_EVIDENCE"),
            },
            "metadata": result.metadata,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        payload = {
            "query_id": str(row.get("query_id", "")),
            "queryspec": queryspec,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

    json_dump(output_path, payload)
    print(f"Wrote evaluation query output to {output_path}")


if __name__ == "__main__":
    main()
