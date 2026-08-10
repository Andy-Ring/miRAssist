#!/usr/bin/env python3
"""Deterministic staging/production smoke tests for Variant A/RF v1."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from backend.cards import cards_from_dataframe
from backend.prompting import build_prompt_bundle
from backend.retrieval import load_evidence, retrieve_from_queryspec
from frontend.evidence_table import build_evidence_shortlist_table, evidence_shortlist_csv_bytes


DEFAULT_TABLE = ROOT / "data/processed/mirassist_evidence_variant_a_rf_v1.parquet"
SOURCE_TABLE = ROOT / "outputs/sequence_defined_candidates/variant_a_rf_v1_scored_release/tables/variant_a_rf_v1_scored_evidence_table.parquet"
MODEL_VERSION = "mirassist_rf_variant_a_v1"
EMPTY_MESSAGE = (
    "No candidates met the current evidence-supported Variant A eligibility criteria. "
    "This does not establish that the miRNA has no biological targets."
)


def queryspec(
    *,
    mirna: str | None = None,
    gene: str | None = None,
    k: int = 10,
    novel: bool = False,
    tcga: str | None = None,
    pathway: bool = False,
) -> dict[str, Any]:
    return {
        "original_question": f"staging query for {mirna or gene}",
        "mode": "mirna_to_targets" if mirna else "gene_to_mirnas",
        "mirna": mirna,
        "gene": gene,
        "cancer": {"name": None, "tcga": tcga},
        "phenotype_context": {},
        "target_role_inference": {},
        "pathway_selection_request": {"enabled": pathway, "strict_directional": False},
        "phenotype_keywords": [],
        "pathway_keywords": [],
        "pathway_filter": {"enabled": pathway, "mode": "filter", "min_gene_sets": 1},
        "novel": novel,
        "k": k,
        "filters": {
            "min_support": 0,
            "require_binding_evidence": False,
            "require_expression": False,
        },
        "needs_clarification": [],
    }


def assert_release_values(shortlist: pd.DataFrame, source_by_id: pd.DataFrame) -> None:
    if shortlist.empty:
        return
    observed = shortlist.set_index("evidence_row_id")
    expected = source_by_id.loc[observed.index]
    if not np.allclose(
        pd.to_numeric(observed["mirassist_model_score"]).to_numpy(),
        pd.to_numeric(expected["mirassist_model_score"]).to_numpy(),
        atol=0,
        rtol=0,
    ):
        raise AssertionError("A returned RF score differs from the frozen release")
    if not pd.to_numeric(observed["mirassist_score_rank_within_mirna"]).astype(int).equals(
        pd.to_numeric(expected["mirassist_score_rank_within_mirna"]).astype(int)
    ):
        raise AssertionError("A returned global rank differs from the frozen release")
    if not observed["mirassist_model_version"].eq(MODEL_VERSION).all():
        raise AssertionError("Returned model version mismatch")
    if observed["mirassist_xgboost_score"].notna().any():
        raise AssertionError("RF score was placed in the legacy XGBoost field")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    table = args.table.resolve()
    if not table.exists():
        raise FileNotFoundError(table)

    os.environ["EVIDENCE_BACKEND"] = "parquet"
    os.environ["MIRASSIST_EVIDENCE"] = str(table)
    os.environ["MIRASSIST_USE_MIRTARBASE_EVIDENCE"] = "1"
    evidence = load_evidence(str(table), force_reload=True)
    frozen = pd.read_parquet(
        SOURCE_TABLE,
        columns=[
            "evidence_row_id",
            "mirassist_model_score",
            "mirassist_model_version",
            "mirassist_score_rank_within_mirna",
        ],
    ).set_index("evidence_row_id")

    cases: list[dict[str, Any]] = []

    def run_case(
        name: str,
        qs: dict[str, Any],
        pathway_selection: dict[str, Any] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        selection = pathway_selection or {"enabled": False, "warnings": []}
        shortlist, direction, diagnostics = retrieve_from_queryspec(evidence, qs, selection)
        assert_release_values(shortlist, frozen)
        if not shortlist.empty:
            scores = pd.to_numeric(shortlist["mirassist_score"])
            if not scores.is_monotonic_decreasing:
                raise AssertionError(f"{name}: canonical scores are not descending")
            if shortlist["mirassist_filtered_rank"].tolist() != list(range(1, len(shortlist) + 1)):
                raise AssertionError(f"{name}: filtered ranks are not contiguous")
        required_debug = {
            "model_version",
            "candidate_universe_version",
            "active_score_source",
            "score_semantics",
            "candidate_count_returned",
            "retrieval_filters",
        }
        if not required_debug.issubset(diagnostics):
            raise AssertionError(f"{name}: debug metadata is incomplete")
        cases.append(
            {
                "case": name,
                "direction": direction,
                "rows": len(shortlist),
                "candidate_ids": shortlist.get("evidence_row_id", pd.Series(dtype=int)).astype(int).tolist(),
                "scores": shortlist.get("mirassist_model_score", pd.Series(dtype=float)).astype(float).tolist(),
                "global_ranks": shortlist.get("mirassist_score_rank_within_mirna", pd.Series(dtype=int)).astype(int).tolist(),
                "status": "passed",
            }
        )
        return shortlist, diagnostics

    high, _ = run_case("high_coverage_mirna", queryspec(mirna="hsa-miR-186-5p", k=10))
    run_case("median_coverage_mirna", queryspec(mirna="hsa-miR-148a-5p", k=10))
    low, _ = run_case("low_coverage_mirna", queryspec(mirna="hsa-miR-887-3p", k=10))
    zero, zero_diag = run_case("zero_candidate_mirna", queryspec(mirna="hsa-miR-24-1-5p", k=10))
    if not zero.empty or zero_diag.get("no_candidates_explanation") != EMPTY_MESSAGE:
        raise AssertionError("Zero-candidate behavior or wording is incorrect")
    if len(low) != 1:
        raise AssertionError("Frozen low-coverage regression case changed")
    larger_than_count, _ = run_case(
        "top_k_greater_than_candidate_count", queryspec(mirna="hsa-miR-887-3p", k=50)
    )
    if len(larger_than_count) != 1:
        raise AssertionError("Top-k greater than candidate count was not handled safely")

    for tcga in ("BRCA", "COAD", "PRAD"):
        context, _ = run_case(
            f"{tcga.lower()}_context_query",
            queryspec(mirna="hsa-miR-186-5p", k=5, tcga=tcga),
        )
        if f"{tcga}_spearman_rho" not in context.columns:
            raise AssertionError(f"{tcga} context evidence was not preserved")

    pathway_gene = str(high.iloc[0]["gene_symbol"])
    pathway_selection = {
        "enabled": True,
        "warnings": [],
        "selected_genes": [pathway_gene],
        "_selected_gene_set": {pathway_gene},
        "_selected_gene_pathways": {pathway_gene: ["STAGING_FIXED_PATHWAY"]},
    }
    pathway_rows, _ = run_case(
        "pathway_query",
        queryspec(mirna="hsa-miR-186-5p", k=10, pathway=True),
        pathway_selection,
    )
    if pathway_rows.empty or not pathway_rows["gene_symbol"].eq(pathway_gene).all():
        raise AssertionError("Strict pathway filtering failed")

    positive_row = evidence.loc[evidence["mirtarbase_known_positive"].fillna(False).astype(bool)].iloc[0]
    positive_mirna = str(positive_row["mirna_name"])
    novel_rows, _ = run_case(
        "novel_mode", queryspec(mirna=positive_mirna, k=25, novel=True)
    )
    if novel_rows["mirtarbase_known_positive"].fillna(False).astype(bool).any():
        raise AssertionError("Novel mode retained a row aligned to the known-positive set")

    run_case("gene_to_mirna_query", queryspec(gene=pathway_gene, k=10))

    cards = cards_from_dataframe(high.head(5), tcga="BRCA")
    bundle = build_prompt_bundle(
        queryspec=queryspec(mirna="hsa-miR-186-5p", k=5, tcga="BRCA"),
        shortlist=high.head(5),
        cards=cards,
        direction="mirna_to_targets",
        retrieval_diagnostics={"model_version": MODEL_VERSION},
    )
    payloads = bundle["meta"]["candidate_order_sent_to_llm"]
    if not payloads or not all(
        payload.get("mirassist_score") is not None
        and payload.get("mirassist_model_version") == MODEL_VERSION
        for payload in payloads
    ):
        raise AssertionError("Synthesis bundle did not receive canonical RF score metadata")
    export = build_evidence_shortlist_table(high.head(5).to_dict(orient="records"), "mirna_to_targets")
    csv_bytes = evidence_shortlist_csv_bytes(export)
    if b"miRAssist score" not in csv_bytes or b"probability" in csv_bytes.lower():
        raise AssertionError("CSV export terminology is incorrect")

    report = {
        "status": "passed",
        "table": str(table),
        "row_count": len(evidence),
        "model_version": MODEL_VERSION,
        "candidate_universe_version": "variant_a",
        "active_score_field": "mirassist_model_score",
        "cases": cases,
        "planner_retrieval": "passed via fixed QuerySpec regression cases",
        "synthesis_bundle": "passed",
        "csv_export": "passed",
        "debug_metadata": "passed",
        "zero_candidate_wording": EMPTY_MESSAGE,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
