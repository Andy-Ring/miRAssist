#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd

from backend.planner import run_planner
from backend.retrieval import retrieve_from_queryspec
from backend.prompting import build_prompt_bundle
from backend.llm_backend import generate_answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query_id", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--novel", action="store_true")
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--min_support", type=int, default=2)
    ap.add_argument("--require_binding_evidence", action="store_true")
    ap.add_argument("--require_expression", action="store_true")
    ap.add_argument(
        "--pathway_mode",
        default=None,
        help="Override pathway integration: auto|boost|filter (default: planner decides)",
    )
    args = ap.parse_args()

    # Planner -> QuerySpec
    qs = run_planner(args.question)

    # Apply CLI overrides
    qs["novel"] = bool(args.novel)
    qs["k"] = int(args.k)
    qs.setdefault("filters", {})
    qs["filters"]["min_support"] = int(args.min_support)
    qs["filters"]["require_binding_evidence"] = bool(args.require_binding_evidence)
    qs["filters"]["require_expression"] = bool(args.require_expression)

    # Optional CLI override: pathway filter vs boost.
    if args.pathway_mode is not None:
        pm = str(args.pathway_mode).strip().lower()
        if pm == "auto" or pm == "":
            pass
        elif pm in {"boost", "filter"}:
            qs.setdefault("pathway_filter", {})
            qs["pathway_filter"]["enabled"] = True
            qs["pathway_filter"]["mode"] = pm
        else:
            raise ValueError("--pathway_mode must be one of: auto, boost, filter")

    # Evidence
    ev = pd.read_parquet("data/processed/evidence_pairs_tcga.parquet")

    # Retrieval
    shortlist_df, direction = retrieve_from_queryspec(ev, qs)

    # Prompt bundle
    bundle = build_prompt_bundle(
        queryspec=qs,
        shortlist=shortlist_df,
        direction=direction,
    )

    # LLM
    answer = generate_answer(bundle)

    # Emit artifact files (for debugging / batch use)
    out_dir = Path("runs")
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / f"{args.query_id}.query.json").write_text(json.dumps(qs, indent=2))
    (out_dir / f"{args.query_id}.bundle.json").write_text(json.dumps(bundle, indent=2))
    (out_dir / f"{args.query_id}.answer.txt").write_text(str(answer))

    print("[OK] wrote:")
    print("  ", out_dir / f"{args.query_id}.query.json")
    print("  ", out_dir / f"{args.query_id}.bundle.json")
    print("  ", out_dir / f"{args.query_id}.answer.txt")


if __name__ == "__main__":
    main()
