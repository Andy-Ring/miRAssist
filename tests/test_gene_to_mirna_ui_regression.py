from __future__ import annotations

from io import BytesIO

import pandas as pd

from backend.planner import _validate_and_fill
from backend.prompting import build_prompt_bundle
from frontend.evidence_table import build_evidence_shortlist_table, evidence_shortlist_csv_bytes


def test_typo_tolerant_planner_corrects_pten_gene_to_mirnas_direction() -> None:
    question = "What miRNAs regulte pten?"
    llm_qs = {
        "mode": "mirna_to_targets",
        "mirna": None,
        "gene": "pten",
        "cancer": {"name": None, "tcga": None},
        "phenotype_context": {},
        "pathway_selection_request": {"enabled": False, "query_terms": [], "directional_query_terms": [], "strict": False},
        "phenotype_keywords": [],
        "pathway_keywords": [],
        "pathway_filter": {"enabled": False, "mode": "filter", "min_gene_sets": 1},
        "novel": False,
        "k": 10,
        "result_count": None,
        "filters": {"min_support": 1, "require_binding_evidence": False, "require_expression": False},
        "needs_clarification": [],
    }

    qs = _validate_and_fill(llm_qs, question)

    assert qs["mode"] == "gene_to_mirnas"
    assert qs["gene"] == "PTEN"
    assert qs["mirna"] is None


def test_gene_to_mirnas_prompt_and_shortlist_table_use_mirna_candidates() -> None:
    shortlist = [
        {
            "rank": 1,
            "mirna_name": "hsa-miR-22-3p",
            "gene_symbol": "PTEN",
            "transcript_id": "NM_000314",
            "mirassist_xgboost_score": 0.91,
            "overall_evidence_support_percentile": 92.0,
            "evidence_family_count": 5,
        },
        {
            "rank": 2,
            "mirna_name": "hsa-miR-107",
            "gene_symbol": "PTEN",
            "transcript_id": "NM_000314",
            "mirassist_xgboost_score": 0.87,
            "overall_evidence_support_percentile": 89.0,
            "evidence_family_count": 5,
        },
    ]
    table = build_evidence_shortlist_table(shortlist, "gene_to_mirnas")

    assert table["candidate"].tolist() == ["hsa-miR-22-3p", "hsa-miR-107"]
    assert {"mirna_name", "gene_symbol", "mirassist_xgboost_score", "evidence_family_count"}.issubset(table.columns)

    csv_roundtrip = pd.read_csv(BytesIO(evidence_shortlist_csv_bytes(table)))
    assert csv_roundtrip["mirna_name"].tolist() == table["mirna_name"].tolist()
    assert csv_roundtrip["gene_symbol"].tolist() == ["PTEN", "PTEN"]

    bundle = build_prompt_bundle(
        queryspec={
            "original_question": "What miRNAs regulte pten?",
            "mode": "gene_to_mirnas",
            "gene": "PTEN",
            "k": 10,
        },
        shortlist=table,
        direction="gene_to_mirnas",
        retrieval_diagnostics={"query_direction": "gene_to_mirnas", "query_gene_normalized": "PTEN"},
    )
    prompt_text = bundle["user_prompt"]

    assert "Resolved task type: gene -> miRNAs" in prompt_text
    assert "candidate miRNA regulators of the queried gene" in prompt_text
    assert "task is miRNA -> targets" not in prompt_text
    assert [item["mirna_name"] for item in bundle["meta"]["candidate_order_sent_to_llm"]] == [
        "hsa-miR-22-3p",
        "hsa-miR-107",
    ]


def test_mirna_to_targets_shortlist_table_uses_gene_candidates() -> None:
    shortlist = [
        {"mirna_name": "hsa-miR-210-5p", "gene_symbol": "APBB3", "mirassist_xgboost_score": 0.52, "evidence_family_count": 5},
        {"mirna_name": "hsa-miR-210-5p", "gene_symbol": "SCN7A", "mirassist_xgboost_score": 0.48, "evidence_family_count": 5},
    ]

    table = build_evidence_shortlist_table(shortlist, "mirna_to_targets")

    assert table["candidate"].tolist() == ["APBB3", "SCN7A"]
