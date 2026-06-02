from __future__ import annotations

import argparse
import traceback
from typing import Any, Dict

from backend.jobstore import write_job
from backend.planner import run_planner
from backend.prompting import build_prompt_bundle
from backend.retrieval import load_evidence, retrieve_from_queryspec
from backend.synthesizer import run_synthesizer


def _apply_query_overrides(
    *,
    queryspec: Dict[str, Any],
    k: int,
    min_support: int,
    novel: bool,
    require_binding_evidence: bool,
    require_expression: bool,
    pathway_mode: str,
) -> Dict[str, Any]:
    qs = dict(queryspec)
    qs["k"] = int(k)
    qs.setdefault("filters", {})
    qs["filters"]["min_support"] = int(min_support)
    qs["novel"] = bool(novel)
    qs["filters"]["require_binding_evidence"] = bool(require_binding_evidence)
    qs["filters"]["require_expression"] = bool(require_expression)

    if pathway_mode != "auto":
        qs.setdefault("pathway_filter", {})
        qs["pathway_filter"]["enabled"] = True
        qs["pathway_filter"]["mode"] = pathway_mode

    return qs


def run_query_job(
    query_id: str,
    question: str,
    k: int = 200,
    min_support: int = 1,
    novel: bool = True,
    require_binding_evidence: bool = False,
    require_expression: bool = False,
    pathway_mode: str = "auto",
) -> Dict[str, Any]:
    try:
        write_job(query_id, {"status": "running", "stage": "planner"})

        qs = run_planner(question)
        qs = _apply_query_overrides(
            queryspec=qs,
            k=k,
            min_support=min_support,
            novel=novel,
            require_binding_evidence=require_binding_evidence,
            require_expression=require_expression,
            pathway_mode=pathway_mode,
        )

        write_job(
            query_id,
            {
                "status": "running",
                "stage": "retrieval",
                "queryspec": qs,
            },
        )

        ev = load_evidence()
        shortlist_df, direction = retrieve_from_queryspec(ev, qs)
        shortlist_records = shortlist_df.to_dict(orient="records")

        bundle = build_prompt_bundle(
            queryspec=qs,
            shortlist=shortlist_df,
            direction=direction,
        )

        write_job(
            query_id,
            {
                "status": "running",
                "stage": "synthesis",
                "queryspec": qs,
                "shortlist": shortlist_records,
                "bundle": bundle,
            },
        )

        answer_obj = run_synthesizer(bundle)

        final_payload = {
            "status": "done",
            "stage": "done",
            "queryspec": qs,
            "shortlist": shortlist_records,
            "bundle": bundle,
            "answer": answer_obj,
        }
        write_job(query_id, final_payload)
        return final_payload
    except Exception as exc:
        error_payload = {
            "status": "error",
            "stage": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_job(query_id, error_payload)
        return error_payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query_id", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--min_support", type=int, default=1)
    ap.add_argument("--novel", action="store_true")
    ap.add_argument("--require_binding_evidence", action="store_true")
    ap.add_argument("--require_expression", action="store_true")
    ap.add_argument(
        "--pathway_mode",
        default="auto",
        choices=["auto", "boost", "filter"],
        help="Override pathway integration mode. 'auto' uses planner defaults.",
    )
    args = ap.parse_args()

    run_query_job(
        query_id=args.query_id,
        question=args.question,
        k=args.k,
        min_support=args.min_support,
        novel=args.novel,
        require_binding_evidence=args.require_binding_evidence,
        require_expression=args.require_expression,
        pathway_mode=args.pathway_mode,
    )


if __name__ == "__main__":
    main()
