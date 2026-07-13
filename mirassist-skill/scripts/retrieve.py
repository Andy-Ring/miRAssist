#!/usr/bin/env python3
"""
miRAssist headless retrieval CLI.

This is the deterministic scientific core of miRAssist, exposed as a command-line
tool for use inside a Claude skill. It does NOT call any LLM. Claude itself plays
the roles that OpenAI used to play in the Streamlit app:

  * PLANNER    : Claude reads the researcher's natural-language question and turns
                 it into the structured arguments below (or a full --queryspec-json).
  * SYNTHESIZER: Claude reads this script's JSON output and writes the grounded
                 answer, using ONLY the values and labels returned here.

What this script does:
  1. Builds a QuerySpec from CLI flags (or loads one from --queryspec-json).
  2. Enriches / validates it with the existing planner schema logic
     (arm defaulting, target-role inference, directional pathway terms).
  3. Resolves grounded pathway/phenotype gene sets from the bundled pathway data.
  4. Retrieves candidates from the evidence backend (Supabase/Postgres in
     production, or a local parquet for development) and ranks them by the
     precomputed learned XGBoost score, falling back to the manual retrieval
     score where the learned score is missing.
  5. Prints a compact JSON result: ranked candidates + diagnostics.

Evidence backend is selected by environment variables (see SKILL.md):
  EVIDENCE_BACKEND=postgres  + DATABASE_URL + EVIDENCE_TABLE   (production)
  EVIDENCE_BACKEND=parquet   + MIRASSIST_EVIDENCE=/path.parquet (development)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make the bundled backend package importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ---- Fields we surface to Claude for grounded synthesis --------------------
# Keep this list focused: identity, ranking, breadth, and the human-readable
# percentile/label fields the backend computes. Claude must not invent values
# beyond these.
CORE_OUTPUT_COLUMNS: List[str] = [
    "mirna_name",
    "gene_symbol",
    "support_count",
    "learned_score_used",
    "retrieval_rank_score",
    "score_column_used",
    "_learned_score_missing",
    "retrieval_score",
    # evidence breadth / families
    "evidence_family_count",
    "overall_evidence_support_percentile",
    "mirtarbase_pos",
    "support_targetscan",
    "support_mirdb",
    "support_encori",
    "support_rnahybrid",
    # per-family availability + percentile (produced by feature_stats in parquet
    # mode, and present as precomputed columns in the Supabase table)
    "sequence_complementarity_support_percentile",
    "thermodynamic_stability_support_percentile",
    "sequence_conservation_support_percentile",
    "target_site_accessibility_support_percentile",
    "functional_binding_support_percentile",
    "functional_repression_support_percentile",
    # a few widely available raw signals for context
    "mirdb_best_score",
    "ts_context_strength",
    "ts_best_contextpp",
    "clip_exp_sum",
    "best_mfe",
    "best_seed_class",
    "n_total_sites",
    # pathway grounding
    "pathway_selected_gene",
    "pathway_selected_names",
]


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of numpy/pandas scalars to plain JSON types."""
    try:
        import numpy as np
        import pandas as pd
    except Exception:  # pragma: no cover
        np = None
        pd = None

    if value is None:
        return None
    if np is not None:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return None if np.isnan(value) else float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, np.ndarray):
            return [_jsonable(v) for v in value.tolist()]
    if pd is not None:
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def build_queryspec_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Construct a minimal QuerySpec dict from CLI flags.

    Claude is the planner: it decides mode, entity, cancer/TCGA, phenotype
    context, and filters, and passes them here. We then run the partial spec
    through the existing planner validator so arm defaulting, target-role
    inference and directional pathway terms behave exactly like the app.
    """
    if args.queryspec_json:
        raw = json.loads(Path(args.queryspec_json).read_text(encoding="utf-8"))
        question = raw.get("original_question") or args.question or ""
    else:
        raw = {
            "mode": args.mode,
            "mirna": args.mirna,
            "gene": args.gene,
            "cancer": {"name": args.cancer_name, "tcga": args.tcga},
            "phenotype_context": {
                "phenotype": args.phenotype,
                "observed_change": args.observed_change,
                "miRNA_perturbation": args.perturbation,
                "raw_phrase": args.question or None,
                "direction": None,
            },
            "phenotype_keywords": list(args.phenotype_keyword or []),
            "pathway_keywords": list(args.pathway_keyword or []),
            "novel": bool(args.novel),
            "k": args.k,
            "result_count": args.result_count,
            "filters": {
                "min_support": args.min_support,
                "require_binding_evidence": bool(args.require_binding),
                "require_expression": bool(args.require_expression),
            },
        }
        question = args.question or (args.mirna or args.gene or "")

    from backend.planner import _validate_and_fill

    qs = _validate_and_fill(dict(raw), str(question))

    # Explicit CLI flags win over any inference so Claude stays in control.
    if not args.queryspec_json:
        if args.mode:
            qs["mode"] = args.mode
        if args.mirna:
            qs["mirna"] = args.mirna
        if args.gene:
            qs["gene"] = args.gene
        qs["k"] = int(args.k)
        if args.result_count is not None:
            qs["result_count"] = int(args.result_count)
        qs["novel"] = bool(args.novel)
        qs.setdefault("filters", {})
        qs["filters"]["min_support"] = int(args.min_support)
        qs["filters"]["require_binding_evidence"] = bool(args.require_binding)
        qs["filters"]["require_expression"] = bool(args.require_expression)
    return qs


def run(args: argparse.Namespace) -> Dict[str, Any]:
    from backend.config import get_default_result_count, get_evidence_backend
    from backend.pathways import compact_pathway_selection, resolve_pathway_selection
    from backend.retrieval import load_evidence, retrieve_from_queryspec

    qs = build_queryspec_from_args(args)

    # Grounded pathway/phenotype gene-set resolution (deterministic; bundled data).
    pathway_selection_internal = resolve_pathway_selection(qs)
    pathway_selection = compact_pathway_selection(pathway_selection_internal)
    qs["pathway_selection"] = pathway_selection

    evidence_backend = get_evidence_backend()
    ev = None
    if evidence_backend == "parquet":
        ev = load_evidence()

    shortlist_df, direction, diagnostics = retrieve_from_queryspec(
        ev, qs, pathway_selection=pathway_selection_internal
    )

    # Percentile annotation is only done in parquet mode; in rest/postgres mode
    # the percentile columns are already precomputed in the evidence table.
    if evidence_backend == "parquet" and ev is not None and len(shortlist_df) > 0:
        from backend.feature_stats import annotate_feature_percentiles

        shortlist_df = annotate_feature_percentiles(shortlist_df, ev)

    result_count = qs.get("result_count")
    if result_count in (None, ""):
        result_count = get_default_result_count()
    result_count = int(result_count)

    top = shortlist_df.head(result_count)
    candidates: List[Dict[str, Any]] = []
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        record: Dict[str, Any] = {"rank": rank}
        for col in CORE_OUTPUT_COLUMNS:
            if col in row.index:
                record[col] = _jsonable(row[col])
        # dynamic TCGA columns for the requested cancer, if present
        tcga = (qs.get("cancer") or {}).get("tcga")
        if tcga:
            for suffix in ("_spearman_rho", "_anticorrelated", "_repression_evidence", "_pair_expressed"):
                col = f"{str(tcga).upper()}{suffix}"
                if col in row.index:
                    record[col] = _jsonable(row[col])
        candidates.append(record)

    out = {
        "query": {
            "mode": qs.get("mode"),
            "direction": direction,
            "mirna": qs.get("mirna"),
            "gene": qs.get("gene"),
            "cancer": qs.get("cancer"),
            "phenotype_context": qs.get("phenotype_context"),
            "novel": qs.get("novel"),
            "k": qs.get("k"),
            "result_count": result_count,
            "filters": qs.get("filters"),
        },
        "pathway_selection": {
            "enabled": pathway_selection.get("enabled"),
            "phenotype": pathway_selection.get("phenotype"),
            "expected_target_effect_on_phenotype": pathway_selection.get(
                "expected_target_effect_on_phenotype"
            ),
            "selected_pathways": pathway_selection.get("selected_pathways"),
            "n_selected_genes": pathway_selection.get("n_selected_genes"),
            "warnings": pathway_selection.get("warnings"),
        },
        "ranking": {
            "evidence_backend": diagnostics.get("evidence_backend"),
            "ranking_mode": diagnostics.get("retrieval_ranking_mode"),
            "learned_score_column": diagnostics.get("learned_score_column"),
            "score_column_used": diagnostics.get("score_column_used"),
            "learned_score_present_count": diagnostics.get("learned_score_present_count"),
            "learned_score_missing_count": diagnostics.get("learned_score_missing_count"),
            "n_final_shortlist": diagnostics.get("n_final_shortlist"),
        },
        "arm_interpretation_note": (diagnostics.get("user_notes") or [None])[0]
        if diagnostics.get("user_notes")
        else diagnostics.get("arm_interpretation_note"),
        "warnings": diagnostics.get("warnings") or [],
        "no_candidates_explanation": diagnostics.get("no_candidates_explanation"),
        "n_candidates": len(candidates),
        "candidates": candidates,
    }
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="miRAssist headless retrieval (deterministic XGBoost-ranked miRNA-target evidence)."
    )
    p.add_argument("--question", default="", help="Original researcher question (for logging/relaxation heuristics).")
    p.add_argument("--queryspec-json", default=None, help="Path to a full QuerySpec JSON (overrides individual flags).")
    p.add_argument("--mode", choices=["mirna_to_targets", "gene_to_mirnas"], default=None)
    p.add_argument("--mirna", default=None, help="miRNA name, e.g. hsa-miR-21-5p or miR-21.")
    p.add_argument("--gene", default=None, help="Gene symbol, e.g. PTEN (for gene_to_mirnas mode).")
    p.add_argument("--cancer-name", default=None, help="Free-text cancer name, e.g. 'breast cancer'.")
    p.add_argument("--tcga", default=None, help="TCGA code, e.g. BRCA, COAD, PRAD.")
    p.add_argument("--phenotype", default=None, help="e.g. apoptosis, proliferation, EMT, invasion, migration.")
    p.add_argument("--observed-change", default=None,
                   choices=["promoted", "suppressed", "increased", "decreased", "associated"])
    p.add_argument("--perturbation", default=None,
                   choices=["overexpression", "knockdown", "inhibition", "unknown"])
    p.add_argument("--phenotype-keyword", action="append", default=[], help="Repeatable extra phenotype keyword.")
    p.add_argument("--pathway-keyword", action="append", default=[], help="Repeatable extra pathway keyword.")
    p.add_argument("--novel", action="store_true", help="Restrict to novel/unvalidated (non-miRTarBase) targets.")
    p.add_argument("--k", type=int, default=int(os.getenv("MIRASSIST_DEFAULT_K", "10")),
                   help="Candidate pool size passed forward from retrieval.")
    p.add_argument("--result-count", type=int, default=None, help="Number of ranked results to return (default 5).")
    p.add_argument("--min-support", type=int, default=1, help="Minimum evidence support count.")
    p.add_argument("--require-binding", action="store_true", help="Require binding evidence (TargetScan/ENCORI/miRDB).")
    p.add_argument("--require-expression", action="store_true", help="Require paired TCGA expression in the cancer.")
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    try:
        out = run(args)
    except Exception as exc:  # surface a clean, JSON error for the skill
        err = {"status": "error", "error": str(exc), "type": type(exc).__name__}
        print(json.dumps(err, ensure_ascii=False))
        sys.exit(1)
    print(json.dumps(out, ensure_ascii=False, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
