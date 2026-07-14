from __future__ import annotations

import argparse
import json
import traceback
from typing import Any, Dict

from backend.config import get_debug_max_rows, get_default_k, get_disable_synthesis, get_evidence_backend


SYNTHESIS_EVIDENCE_LIMIT = 25


def _limit_debug_records(shortlist_df) -> list[dict]:
    from backend.jobstore import sanitize_json_payload

    if shortlist_df is None or len(shortlist_df) == 0:
        return []
    max_rows = max(1, int(get_debug_max_rows()))
    limited_df = shortlist_df.head(max_rows).copy()
    return sanitize_json_payload(limited_df.to_dict(orient="records"))


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
    ui_k = int(k)
    planner_k = qs.get("k")
    default_k = get_default_k()
    if planner_k in (None, ""):
        qs["k"] = ui_k
    elif ui_k != default_k:
        qs["k"] = ui_k
    else:
        try:
            qs["k"] = int(planner_k)
        except Exception:
            qs["k"] = default_k
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
    k: int = get_default_k(),
    min_support: int = 1,
    novel: bool = True,
    require_binding_evidence: bool = False,
    require_expression: bool = False,
    pathway_mode: str = "auto",
    disable_synthesis: bool | None = None,
    persist_job: bool = True,
) -> Dict[str, Any]:
    from backend.feature_stats import annotate_feature_percentiles
    from backend.jobstore import write_job
    from backend.pathways import compact_pathway_selection, resolve_pathway_selection
    from backend.planner import run_planner
    from backend.retrieval import load_evidence, retrieve_from_queryspec

    def persist(payload: Dict[str, Any]) -> None:
        if persist_job:
            write_job(query_id, payload)

    try:
        persist({"status": "running", "stage": "planner"})
        synthesis_disabled = get_disable_synthesis() if disable_synthesis is None else bool(disable_synthesis)

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
        pathway_selection_internal = resolve_pathway_selection(qs)
        if qs.get("debug_warnings"):
            pathway_selection_internal.setdefault("warnings", [])
            for warning in qs["debug_warnings"]:
                if warning not in pathway_selection_internal["warnings"]:
                    pathway_selection_internal["warnings"].append(warning)
        pathway_selection = compact_pathway_selection(pathway_selection_internal)
        qs["pathway_selection"] = pathway_selection

        persist(
            {
                "status": "running",
                "stage": "retrieval",
                "queryspec": qs,
                "pathway_selection": pathway_selection,
            }
        )

        evidence_backend = get_evidence_backend()
        ev = None
        # Only the local-parquet dev backend loads the full table into memory and
        # computes percentiles here. The github snapshot, postgres, and rest backends
        # fetch a bounded candidate pool and rely on precomputed percentile columns.
        if evidence_backend == "parquet":
            ev = load_evidence()

        shortlist_df, direction, retrieval_diagnostics = retrieve_from_queryspec(
            ev,
            qs,
            pathway_selection=pathway_selection_internal,
        )
        retrieval_diagnostics["evidence_backend"] = evidence_backend
        if evidence_backend == "parquet":
            shortlist_df = annotate_feature_percentiles(shortlist_df, ev)
        else:
            retrieval_diagnostics.setdefault("warnings", []).append(
                "Feature percentile annotation was skipped; using precomputed percentile "
                "columns from the evidence snapshot/database."
            )

        shortlist_records = _limit_debug_records(shortlist_df)
        synth_shortlist_df = shortlist_df.head(SYNTHESIS_EVIDENCE_LIMIT).copy()
        retrieval_diagnostics["debug_max_rows"] = int(get_debug_max_rows())
        retrieval_diagnostics["n_debug_rows_returned"] = int(len(shortlist_records))
        retrieval_diagnostics["n_rows_sent_to_synthesizer"] = int(0 if synthesis_disabled else len(synth_shortlist_df))
        retrieval_diagnostics["disable_synthesis"] = synthesis_disabled
        card_generation_diagnostics: Dict[str, Any] = {
            "disabled": synthesis_disabled,
            "n_shortlist_rows": int(len(synth_shortlist_df)),
            "n_cards_generated": 0,
            "card_errors": [],
        }
        cards = []

        print(
            json.dumps(
                {
                    "query_id": query_id,
                    "evidence_backend": evidence_backend,
                    "db_candidate_limit": retrieval_diagnostics.get("db_candidate_limit"),
                    "n_rows_fetched_from_db": retrieval_diagnostics.get("n_rows_fetched_from_db"),
                    "n_after_filters": retrieval_diagnostics.get("n_after_pathway_filter"),
                    "n_final_shortlist": retrieval_diagnostics.get("n_final_shortlist"),
                    "n_rows_sent_to_synthesizer": retrieval_diagnostics.get("n_rows_sent_to_synthesizer"),
                    "disable_synthesis": synthesis_disabled,
                    "learned_score_used": retrieval_diagnostics.get("learned_score_enabled"),
                    "learned_score_column": retrieval_diagnostics.get("learned_score_column"),
                },
                ensure_ascii=False,
            )
        )

        if len(synth_shortlist_df) == 0:
            diagnostics_bits = list(retrieval_diagnostics.get("warnings") or [])
            if pathway_selection.get("warnings"):
                diagnostics_bits.extend(pathway_selection.get("warnings") or [])
            summary = retrieval_diagnostics.get("no_candidates_explanation") or "No candidates passed the current filters."
            if diagnostics_bits:
                summary += " Diagnostics: " + "; ".join(dict.fromkeys(diagnostics_bits))
            answer_obj = {
                "raw_text": summary,
                "summary": summary,
                "suggested_experiments": [],
            }
        else:
            answer_obj = None

        if synthesis_disabled:
            final_payload = {
                "status": "done",
                "stage": "done",
                "query_id": query_id,
                "queryspec": qs,
                "pathway_selection": pathway_selection,
                "retrieval_diagnostics": retrieval_diagnostics,
                "card_generation_diagnostics": card_generation_diagnostics,
                "shortlist": shortlist_records,
                "answer": answer_obj,
                "disable_synthesis": True,
                "result_mode": "chart_only",
            }
            persist(final_payload)
            return final_payload

        if answer_obj is not None:
            final_payload = {
                "status": "done",
                "stage": "done",
                "query_id": query_id,
                "queryspec": qs,
                "pathway_selection": pathway_selection,
                "retrieval_diagnostics": retrieval_diagnostics,
                "card_generation_diagnostics": card_generation_diagnostics,
                "shortlist": shortlist_records,
                "answer": answer_obj,
                "disable_synthesis": synthesis_disabled,
                "result_mode": "answer_only",
            }
            persist(final_payload)
            return final_payload

        from backend.cards import cards_from_dataframe_with_diagnostics
        from backend.prompting import build_prompt_bundle
        from backend.synthesizer import run_synthesizer

        tcga = ((qs.get("cancer") or {}).get("tcga") or qs.get("tcga"))
        cards, card_generation_diagnostics = cards_from_dataframe_with_diagnostics(synth_shortlist_df, tcga=tcga)
        if len(synth_shortlist_df) > 0 and not cards:
            raise RuntimeError(
                "Shortlist rows were produced, but card generation returned zero evidence cards. "
                f"Diagnostics: {json.dumps(card_generation_diagnostics, ensure_ascii=False)}"
            )
        if not cards:
            diagnostics_bits = list(retrieval_diagnostics.get("warnings") or [])
            if pathway_selection.get("warnings"):
                diagnostics_bits.extend(pathway_selection.get("warnings") or [])
            summary = retrieval_diagnostics.get("no_candidates_explanation") or "No candidates passed the current filters."
            if diagnostics_bits:
                summary += " Diagnostics: " + "; ".join(dict.fromkeys(diagnostics_bits))
            answer_obj = {
                "raw_text": summary,
                "summary": summary,
                "suggested_experiments": [],
            }
        else:
            answer_obj = None

        bundle = build_prompt_bundle(
            queryspec=qs,
            shortlist=synth_shortlist_df,
            cards=cards,
            direction=direction,
            meta={
                "queryspec": qs,
                "pathway_selection": pathway_selection,
                "retrieval_diagnostics": retrieval_diagnostics,
                "card_generation_diagnostics": card_generation_diagnostics,
            },
            retrieval_diagnostics=retrieval_diagnostics,
        )
        prompt_bundle_debug = {
            "candidate_order_sent_to_llm": (bundle.get("meta") or {}).get("candidate_order_sent_to_llm", []),
            "cards_count": (bundle.get("meta") or {}).get("cards_count", len(cards)),
        }

        persist(
            {
                "status": "running",
                "stage": "synthesis",
                "queryspec": qs,
                "pathway_selection": pathway_selection,
                "retrieval_diagnostics": retrieval_diagnostics,
                "card_generation_diagnostics": card_generation_diagnostics,
                "shortlist": shortlist_records,
            }
        )

        if answer_obj is None:
            answer_obj = run_synthesizer(bundle)

        final_payload = {
            "status": "done",
            "stage": "done",
            "query_id": query_id,
            "queryspec": qs,
            "pathway_selection": pathway_selection,
            "retrieval_diagnostics": retrieval_diagnostics,
            "card_generation_diagnostics": card_generation_diagnostics,
            "prompt_bundle_debug": prompt_bundle_debug,
            "shortlist": shortlist_records,
            "answer": answer_obj,
            "disable_synthesis": synthesis_disabled,
            "result_mode": "answer_and_chart",
        }
        persist(final_payload)
        return final_payload
    except Exception as exc:
        error_payload = {
            "status": "error",
            "stage": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        persist(error_payload)
        return error_payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query_id", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--k", type=int, default=get_default_k())
    ap.add_argument("--min_support", type=int, default=1)
    ap.add_argument("--novel", action="store_true")
    ap.add_argument("--require_binding_evidence", action="store_true")
    ap.add_argument("--require_expression", action="store_true")
    ap.add_argument("--disable_synthesis", action="store_true", default=None)
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
        disable_synthesis=args.disable_synthesis,
    )


if __name__ == "__main__":
    main()
