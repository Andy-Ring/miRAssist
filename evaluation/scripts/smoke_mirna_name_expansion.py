from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.retrieval import (
    RetrievalConfig,
    build_postgres_candidate_query,
    expand_mirna_query_variants,
    retrieve_candidates,
)


def _set_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def _base_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "mirna_name": "hsa-miR-210-3p",
                "mirna_name_norm": "hsa-mir-210-3p",
                "gene_symbol": "GENE1",
                "support_count": 1,
                "learned_score_xgb_raw_v1": 0.91,
                "retrieval_score": 0.20,
            },
            {
                "mirna_name": "hsa-miR-210-5p",
                "mirna_name_norm": "hsa-mir-210-5p",
                "gene_symbol": "GENE2",
                "support_count": 1,
                "learned_score_xgb_raw_v1": 0.73,
                "retrieval_score": 0.19,
            },
            {
                "mirna_name": "hsa-miR-21-5p",
                "mirna_name_norm": "hsa-mir-21-5p",
                "gene_symbol": "GENE3",
                "support_count": 3,
                "learned_score_xgb_raw_v1": 0.88,
                "retrieval_score": 0.40,
            },
        ]
    )


def _assert_variant_expansion() -> None:
    variants = expand_mirna_query_variants("miRNA-210")
    assert "mir-210" in variants
    assert "mir-210-3p" in variants
    assert "mir-210-5p" in variants
    assert "hsa-mir-210" in variants
    assert "hsa-mir-210-3p" in variants
    assert "hsa-mir-210-5p" in variants


def _assert_base_query_matches_arms() -> None:
    df = _base_dataframe()
    cfg = RetrievalConfig(
        k_shortlist=10,
        min_support=2,
        novel=False,
        allow_min_support_relaxation=True,
        user_requested_strict_support=False,
    )
    shortlist_df, direction, diagnostics = retrieve_candidates(df, "miRNA-210", cfg)
    assert direction == "mirna_to_targets"
    assert shortlist_df["gene_symbol"].tolist() == ["GENE1", "GENE2"]
    assert diagnostics["effective_min_support"] == 1
    assert diagnostics["mature_arm_expansion_attempted"] is True
    assert diagnostics["n_rows_before_min_support"] == 2
    assert diagnostics["n_rows_after_min_support"] == 2
    assert diagnostics.get("user_notes")


def _assert_exact_arm_query_stays_exact() -> None:
    df = _base_dataframe()
    cfg = RetrievalConfig(
        k_shortlist=10,
        min_support=1,
        novel=False,
        allow_min_support_relaxation=False,
        user_requested_strict_support=False,
    )
    shortlist_df, _, diagnostics = retrieve_candidates(df, "miR-210-3p", cfg)
    assert shortlist_df["gene_symbol"].tolist() == ["GENE1"]
    assert diagnostics["mature_arm_expansion_attempted"] is False


def _assert_bounded_postgres_query_builder() -> None:
    _set_env("MIRASSIST_DB_CANDIDATE_LIMIT", "500")
    available_columns = [
        "mirna_name",
        "mirna_name_norm",
        "gene_symbol",
        "gene_symbol_norm",
        "support_count",
        "learned_score_xgb_raw_v1",
        "retrieval_score",
        "mirdb_best_score",
        "ts_context_strength",
    ]
    cfg = RetrievalConfig(k_shortlist=10, min_support=2, novel=False)
    query, params, _, _ = build_postgres_candidate_query(
        "miRNA-210",
        cfg,
        available_columns,
        mirna_variants=expand_mirna_query_variants("miRNA-210"),
    )
    assert "LIMIT :candidate_limit" in query
    assert params["candidate_limit"] == 500
    assert "mirna_name_norm" in query


def main() -> None:
    _assert_variant_expansion()
    _assert_base_query_matches_arms()
    _assert_exact_arm_query_stays_exact()
    _assert_bounded_postgres_query_builder()
    print("smoke_mirna_name_expansion: ok")


if __name__ == "__main__":
    main()
