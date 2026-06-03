from __future__ import annotations

import argparse
import traceback
from typing import Any, Dict

from backend.feature_stats import annotate_feature_percentiles
from backend.jobstore import write_job
from backend.pathways import resolve_pathway_selection
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
    debug_warnings = list(qs.get("debug_warnings") or [])
    qs["k"] = int(k)
    qs.setdefault("filters", {})
    qs["filters"]["min_support"] = int(min_support)
    qs["novel"] = bool(novel)
    qs["filters"]["require_binding_evidence"] = bool(require_binding_evidence)
    qs["filters"]["require_expression"] = bool(require_expression)

    pathway_request = qs.get("pathway_selection_request") or {}
    pathway_context_exists = bool(
        pathway_request.get("enabled")
        or qs.get("phenotype_keywords")
        or qs.get("pathway_keywords")
        or (qs.get("phenotype_context") or {}).get("phenotype")
    )
    normalized_pathway_mode = str(pathway_mode or "auto").strip().lower()

    qs.setdefault("pathway_filter", {})
    qs["pathway_filter"]["min_gene_sets"] = int(qs["pathway_filter"].get("min_gene_sets", 1) or 1)
    qs["pathway_filter"]["mode"] = "filter"

    if normalized_pathway_mode == "boost":
        debug_warnings.append("Pathway boost mode has been removed; using strict filter mode.")
        normalized_pathway_mode = "filter"

    if normalized_pathway_mode == "filter":
        qs["pathway_filter"]["enabled"] = True
        qs.setdefault("pathway_selection_request", {})
        qs["pathway_selection_request"]["enabled"] = True
        qs["pathway_selection_request"]["strict"] = True
    elif normalized_pathway_mode == "auto":
        qs["pathway_filter"]["enabled"] = bool(pathway_context_exists)
        qs.setdefault("pathway_selection_request", {})
        qs["pathway_selection_request"]["enabled"] = bool(pathway_context_exists)
        qs["pathway_selection_request"]["strict"] = bool(pathway_context_exists)
    else:
        qs["pathway_filter"]["enabled"] = bool(pathway_context_exists)

    qs["pathway_mode_effective"] = "filter" if qs["pathway_filter"]["enabled"] else "none"
    if debug_warnings:
        qs["debug_warnings"] = debug_warnings

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
        pathway_selection = resolve_pathway_selection(qs)
        if qs.get("debug_warnings"):
            pathway_selection.setdefault("warnings", [])
            for warning in qs["debug_warnings"]:
                if warning not in pathway_selection["warnings"]:
                    pathway_selection["warnings"].append(warning)
        qs["pathway_selection"] = pathway_selection

        write_job(
            query_id,
            {
                "status": "running",
                "stage": "retrieval",
                "queryspec": qs,
                "pathway_selection": pathway_selection,
            },
        )

        ev = load_evidence()
        shortlist_df, direction = retrieve_from_queryspec(ev, qs, pathway_selection=pathway_selection)
        shortlist_df = annotate_feature_percentiles(shortlist_df, ev)
        shortlist_records = shortlist_df.to_dict(orient="records")

        bundle = build_prompt_bundle(
            queryspec=qs,
            shortlist=shortlist_df,
            direction=direction,
            meta={"queryspec": qs, "pathway_selection": pathway_selection},
        )

        write_job(
            query_id,
            {
                "status": "running",
                "stage": "synthesis",
                "queryspec": qs,
                "pathway_selection": pathway_selection,
                "shortlist": shortlist_records,
                "bundle": bundle,
            },
        )

        answer_obj = run_synthesizer(bundle)

        final_payload = {
            "status": "done",
            "stage": "done",
            "queryspec": qs,
            "pathway_selection": pathway_selection,
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
