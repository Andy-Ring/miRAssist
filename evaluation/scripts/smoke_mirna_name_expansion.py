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
    mirna_has_explicit_arm,
    normalize_mirna_name,
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
                "mirna_name": "mir-210",
                "mirna_name_norm": "mir-210",
                "gene_symbol": "EPHA2",
                "support_count": 1,
                "learned_score_xgb_raw_v1": 0.05,
                "retrieval_score": 0.05,
            },
            {
                "mirna_name": "hsa-miR-210-5p",
                "mirna_name_norm": "hsa-mir-210-5p",
                "gene_symbol": "KCMF1",
                "support_count": 1,
                "learned_score_xgb_raw_v1": 0.93,
                "retrieval_score": 0.20,
            },
            {
                "mirna_name": "hsa-miR-210-3p",
                "mirna_name_norm": "hsa-mir-210-3p",
                "gene_symbol": "ISCU",
                "support_count": 1,
                "learned_score_xgb_raw_v1": 0.89,
                "retrieval_score": 0.19,
            },
        ]
    )


def _legacy_only_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "mirna_name": "mir-210",
                "mirna_name_norm": "mir-210",
                "gene_symbol": "EPHA2",
                "support_count": 1,
                "learned_score_xgb_raw_v1": 0.07,
                "retrieval_score": 0.04,
            }
        ]
    )


def _no_norm_dataframe() -> pd.DataFrame:
    return _base_dataframe().drop(columns=["mirna_name_norm"])


def _cfg(min_support: int = 2) -> RetrievalConfig:
    return RetrievalConfig(
        k_shortlist=10,
        min_support=min_support,
        novel=False,
        allow_min_support_relaxation=True,
        user_requested_strict_support=False,
    )


def _assert_normalization_helpers() -> None:
    assert normalize_mirna_name("miRNA-210") == "mir-210"
    assert normalize_mirna_name("miR-210") == "mir-210"
    assert normalize_mirna_name("hsa-miR-210") == "hsa-mir-210"
    assert normalize_mirna_name("hsa-miR-210-3p") == "hsa-mir-210-3p"
    assert normalize_mirna_name("hsa-miR-210-5p") == "hsa-mir-210-5p"
    assert mirna_has_explicit_arm("mir-210-3p") is True
    assert mirna_has_explicit_arm("mir-210") is False


def _assert_default_5p_expansion() -> None:
    _set_env("MIRASSIST_DEFAULT_MIRNA_ARM", "5p")
    expansion = expand_mirna_query_variants("miRNA-210", default_arm="5p")
    assert expansion["normalized_input"] == "mir-210"
    assert expansion["explicit_arm"] is False
    assert expansion["primary_variants"] == ["mir-210-5p", "hsa-mir-210-5p"]
    assert expansion["fallback_variants"] == ["mir-210", "hsa-mir-210"]


def _assert_default_5p_query_prefers_5p() -> None:
    _set_env("MIRASSIST_DEFAULT_MIRNA_ARM", "5p")
    shortlist_df, direction, diagnostics = retrieve_candidates(_base_dataframe(), "miRNA-210", _cfg())
    assert direction == "mirna_to_targets"
    assert shortlist_df["gene_symbol"].tolist() == ["KCMF1"]
    assert "EPHA2" not in shortlist_df["gene_symbol"].tolist()
    assert diagnostics["explicit_mirna_arm"] is False
    assert diagnostics["default_mirna_arm"] == "5p"
    assert diagnostics["variants_used"] == ["mir-210-5p", "hsa-mir-210-5p"]
    assert diagnostics["n_rows_primary"] == 1
    assert diagnostics["n_rows_fallback"] == 0
    assert diagnostics["effective_min_support"] == 1
    assert "miR-210-3p" not in " ".join(diagnostics.get("matched_mirna_names") or [])
    assert diagnostics.get("arm_interpretation_note")


def _assert_exact_arm_queries_stay_exact() -> None:
    _set_env("MIRASSIST_DEFAULT_MIRNA_ARM", "5p")
    df = _base_dataframe()

    shortlist_5p, _, diagnostics_5p = retrieve_candidates(df, "miR-210-5p", _cfg(min_support=1))
    assert shortlist_5p["gene_symbol"].tolist() == ["KCMF1"]
    assert diagnostics_5p["explicit_mirna_arm"] is True
    assert diagnostics_5p["n_rows_fallback"] == 0

    shortlist_3p, _, diagnostics_3p = retrieve_candidates(df, "miR-210-3p", _cfg(min_support=1))
    assert shortlist_3p["gene_symbol"].tolist() == ["ISCU"]
    assert diagnostics_3p["explicit_mirna_arm"] is True
    assert diagnostics_3p["n_rows_fallback"] == 0


def _assert_default_both_query_returns_both_arms() -> None:
    _set_env("MIRASSIST_DEFAULT_MIRNA_ARM", "both")
    shortlist_df, _, diagnostics = retrieve_candidates(_base_dataframe(), "miRNA-210", _cfg(min_support=1))
    assert shortlist_df["gene_symbol"].tolist() == ["KCMF1", "ISCU"]
    assert diagnostics["variants_used"] == [
        "mir-210-5p",
        "mir-210-3p",
        "hsa-mir-210-5p",
        "hsa-mir-210-3p",
    ]
    assert diagnostics["arm_interpretation_note"] == (
        "No mature arm was specified, so miRAssist searched both miR-210-5p and miR-210-3p mature arms."
    )


def _assert_generic_fallback_only_when_needed() -> None:
    _set_env("MIRASSIST_DEFAULT_MIRNA_ARM", "5p")
    shortlist_df, _, diagnostics = retrieve_candidates(_legacy_only_dataframe(), "miRNA-210", _cfg(min_support=1))
    assert shortlist_df["gene_symbol"].tolist() == ["EPHA2"]
    assert diagnostics["n_rows_primary"] == 0
    assert diagnostics["n_rows_fallback"] == 1
    assert diagnostics["variants_used"] == ["mir-210", "hsa-mir-210"]


def _assert_primary_arm_stays_primary_without_norm_column() -> None:
    _set_env("MIRASSIST_DEFAULT_MIRNA_ARM", "5p")
    shortlist_df, _, diagnostics = retrieve_candidates(_no_norm_dataframe(), "miRNA-210", _cfg(min_support=1))
    assert shortlist_df["gene_symbol"].tolist() == ["KCMF1"]
    assert "EPHA2" not in shortlist_df["gene_symbol"].tolist()
    assert diagnostics["n_rows_primary"] == 1
    assert diagnostics["n_rows_fallback"] == 0
    assert diagnostics["variants_used"] == ["mir-210-5p", "hsa-mir-210-5p"]
    assert diagnostics["matched_mirna_names"] == ["hsa-miR-210-5p"]


def _assert_bounded_postgres_query_builder() -> None:
    _set_env("MIRASSIST_DB_CANDIDATE_LIMIT", "500")
    _set_env("MIRASSIST_DEFAULT_MIRNA_ARM", "5p")
    expansion = expand_mirna_query_variants("miRNA-210", default_arm="5p")
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
    query, params, _, _ = build_postgres_candidate_query(
        "miRNA-210",
        RetrievalConfig(k_shortlist=10, min_support=2, novel=False),
        available_columns,
        mirna_variants=expansion["primary_variants"],
    )
    assert "LIMIT :candidate_limit" in query
    assert params["candidate_limit"] == 500
    assert "mirna_name_norm" in query


def main() -> None:
    _assert_normalization_helpers()
    _assert_default_5p_expansion()
    _assert_default_5p_query_prefers_5p()
    _assert_exact_arm_queries_stay_exact()
    _assert_default_both_query_returns_both_arms()
    _assert_generic_fallback_only_when_needed()
    _assert_primary_arm_stays_primary_without_norm_column()
    _assert_bounded_postgres_query_builder()
    print("smoke_mirna_name_expansion: ok")


if __name__ == "__main__":
    main()
