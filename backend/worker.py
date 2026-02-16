import argparse
import traceback

from backend.jobstore import write_job
from backend.planner import run_planner
from backend.retrieval import load_evidence, retrieve_from_queryspec
from backend.prompting import build_prompt_bundle
from backend.synthesizer import run_synthesizer


def main():
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

    query_id = args.query_id
    try:
        write_job(query_id, {"status": "running"})

        qs = run_planner(args.question)

        # Apply overrides
        qs["k"] = int(args.k)
        qs.setdefault("filters", {})
        qs["filters"]["min_support"] = int(args.min_support)
        qs["novel"] = bool(args.novel)
        qs["filters"]["require_binding_evidence"] = bool(args.require_binding_evidence)
        qs["filters"]["require_expression"] = bool(args.require_expression)

        # Optional override: pathway integration mode
        if args.pathway_mode != "auto":
            qs.setdefault("pathway_filter", {})
            qs["pathway_filter"]["enabled"] = True
            qs["pathway_filter"]["mode"] = args.pathway_mode

        ev = load_evidence()

        shortlist_df, direction = retrieve_from_queryspec(ev, qs)

        bundle = build_prompt_bundle(
            qs,
            shortlist_df,
            direction=direction,
            max_prompt_tokens=6500,
        )

        answer_obj = run_synthesizer(bundle)

        write_job(
            query_id,
            {
                "status": "done",
                "queryspec": qs,
                "shortlist": shortlist_df.to_dict(orient="records"),
                "bundle": bundle,
                "answer": answer_obj,
            },
        )

    except Exception as e:
        write_job(
            query_id,
            {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
        )


if __name__ == "__main__":
    main()
