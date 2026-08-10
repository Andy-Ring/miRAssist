from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.config import get_debug_max_rows, get_learned_score_column, get_use_learned_score
from backend.jobstore import sanitize_json_payload
from backend.retrieval import (
    RetrievalConfig,
    apply_learned_score_ranking,
    build_postgres_candidate_query,
    expand_mirna_query_variants,
)
from backend.worker import _limit_debug_records


def _set_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def _assert_flag_parsing() -> None:
    cases = {
        None: True,
        "1": True,
        "true": True,
        "yes": True,
        "on": True,
        "0": False,
        "false": False,
        "no": False,
        "off": False,
    }
    for raw_value, expected in cases.items():
        _set_env("MIRASSIST_USE_LEARNED_SCORE", raw_value)
        actual = get_use_learned_score()
        assert actual is expected, "Flag parsing mismatch for {!r}: expected {}, got {}".format(raw_value, expected, actual)

    _set_env("MIRASSIST_LEARNED_SCORE_COLUMN", None)
    assert get_learned_score_column() == "mirassist_model_score"
    _set_env("MIRASSIST_LEARNED_SCORE_COLUMN", "custom_score")
    assert get_learned_score_column() == "custom_score"


def _assert_ranking_behavior() -> None:
    df = pd.DataFrame(
        [
            {
                "gene_symbol": "GENE_A",
                "gene_symbol_normalized": "GENE_A",
                "retrieval_score": 0.60,
                "overall_evidence_support_percentile": 80.0,
                "evidence_family_count": 3,
                "mirassist_model_score": 0.40,
                "mirassist_model_version": "mirassist_rf_variant_a_v1",
                "mirassist_xgboost_score": None,
            },
            {
                "gene_symbol": "GENE_B",
                "gene_symbol_normalized": "GENE_B",
                "retrieval_score": 0.80,
                "overall_evidence_support_percentile": 70.0,
                "evidence_family_count": 2,
                "mirassist_model_score": 0.30,
                "mirassist_model_version": "mirassist_rf_variant_a_v1",
                "mirassist_xgboost_score": None,
            },
            {
                "gene_symbol": "GENE_C",
                "gene_symbol_normalized": "GENE_C",
                "retrieval_score": 0.50,
                "overall_evidence_support_percentile": 60.0,
                "evidence_family_count": 1,
                "mirassist_model_score": 0.90,
                "mirassist_model_version": "mirassist_rf_variant_a_v1",
                "mirassist_xgboost_score": None,
            },
        ]
    )

    ranked_df, diagnostics = apply_learned_score_ranking(
        df,
        learned_score_column="mirassist_model_score",
        enabled=True,
    )
    assert ranked_df["gene_symbol"].tolist() == ["GENE_C", "GENE_A", "GENE_B"]
    assert ranked_df["retrieval_rank_score"].round(4).tolist() == [0.9, 0.4, 0.3]
    assert ranked_df["learned_score_used"].round(4).tolist() == [0.9, 0.4, 0.3]
    assert ranked_df["mirassist_score"].round(4).tolist() == [0.9, 0.4, 0.3]
    assert diagnostics["learned_score_enabled"] is True
    assert diagnostics["active_score_source"] == "mirassist_model_score"
    assert diagnostics["model_version"] == "mirassist_rf_variant_a_v1"
    assert diagnostics["learned_score_present_count"] == 3
    assert diagnostics["learned_score_missing_count"] == 0

    manual_df, manual_diagnostics = apply_learned_score_ranking(
        df,
        learned_score_column="mirassist_model_score",
        enabled=False,
    )
    assert manual_df["gene_symbol"].tolist() == ["GENE_B", "GENE_A", "GENE_C"]
    assert manual_diagnostics["retrieval_ranking_mode"] == "manual"

    missing_df, missing_diagnostics = apply_learned_score_ranking(
        df.drop(columns=["mirassist_model_score", "mirassist_model_version", "mirassist_xgboost_score"]),
        learned_score_column="mirassist_model_score",
        enabled=True,
    )
    assert missing_df["gene_symbol"].tolist() == ["GENE_B", "GENE_A", "GENE_C"]
    assert missing_diagnostics["learned_score_enabled"] is False
    assert missing_diagnostics.get("warnings")

    sparse_df = df.head(2).copy()
    sparse_df.loc[sparse_df.index[1], "mirassist_model_score"] = None
    try:
        apply_learned_score_ranking(
            sparse_df,
            learned_score_column="mirassist_model_score",
            enabled=True,
        )
    except ValueError as exc:
        assert "missing" in str(exc).lower()
    else:
        raise AssertionError("Incomplete persisted model-score coverage was not rejected")


def _assert_debug_truncation_and_json_safety() -> None:
    _set_env("MIRASSIST_DEBUG_MAX_ROWS", "2")
    debug_df = pd.DataFrame(
        [
            {
                "gene_symbol": "GENE_A",
                "retrieval_score": 0.9,
                "learned_score_updated_at": pd.Timestamp("2026-06-10T12:00:00Z"),
                "nan_field": float("nan"),
            },
            {
                "gene_symbol": "GENE_B",
                "retrieval_score": 0.8,
                "learned_score_updated_at": pd.Timestamp("2026-06-10T12:05:00Z"),
                "nan_field": np.nan,
            },
            {
                "gene_symbol": "GENE_C",
                "retrieval_score": 0.7,
                "learned_score_updated_at": pd.Timestamp("2026-06-10T12:10:00Z"),
                "nan_field": np.nan,
            },
        ]
    )
    limited_records = _limit_debug_records(debug_df)
    assert int(get_debug_max_rows()) == 2
    assert len(limited_records) == 2
    assert limited_records[0]["gene_symbol"] == "GENE_A"
    assert limited_records[0]["nan_field"] is None
    assert "2026-06-10T12:00:00" in limited_records[0]["learned_score_updated_at"]

    payload = sanitize_json_payload(
        {
            "x": float("nan"),
            "updated_at": pd.Timestamp("2026-06-10T12:15:00Z"),
            "rows": [{"ts_best_contextpp": np.float64(np.nan)}],
        }
    )
    assert payload["x"] is None
    assert payload["rows"][0]["ts_best_contextpp"] is None
    assert "2026-06-10T12:15:00" in payload["updated_at"]


def _assert_postgres_query_builder() -> None:
    _set_env("MIRASSIST_DB_CANDIDATE_LIMIT", "500")
    _set_env("MIRASSIST_USE_LEARNED_SCORE", "1")
    _set_env("MIRASSIST_LEARNED_SCORE_COLUMN", "mirassist_model_score")
    cfg = RetrievalConfig(
        k_shortlist=10,
        min_support=2,
        novel=False,
        tcga="BRCA",
        require_binding_evidence=False,
        require_expression=False,
        pathway_selection={"enabled": False},
        pathway_gene_set=set(),
        pathway_gene_map={},
    )
    available_columns = [
        "evidence_row_id",
        "mirna_name",
        "gene_symbol",
        "mirna_name_normalized",
        "gene_symbol_normalized",
        "transcript_id",
        "support_count",
        "overall_evidence_support_percentile",
        "evidence_family_count",
        "mirassist_model_score",
        "mirassist_model_version",
        "mirassist_score_rank_within_mirna",
        "mirassist_score_percentile_within_mirna",
        "mirassist_xgboost_score",
    ]
    query, params, selected_columns, diagnostics = build_postgres_candidate_query(
        "miR-210",
        cfg,
        available_columns,
        mirna_variants=expand_mirna_query_variants("miR-210")["primary_variants"],
    )
    assert "LIMIT :candidate_limit" in query
    assert 'LOWER("mirna_name_normalized") IN (' in query
    assert params["candidate_limit"] == 500
    assert any(key.startswith("mirna_norm_") for key in params)
    assert "mirassist_model_score" in selected_columns
    assert "mirassist_xgboost_score" in selected_columns
    assert diagnostics["evidence_backend"] == "postgres"
    assert diagnostics["sql_order_columns"] == [
        '"mirassist_model_score" DESC NULLS LAST',
        '"overall_evidence_support_percentile" DESC NULLS LAST',
        '"evidence_family_count" DESC NULLS LAST',
        '"mirassist_score_rank_within_mirna" ASC NULLS LAST',
        '"mirna_name_normalized" ASC NULLS LAST',
        '"gene_symbol_normalized" ASC NULLS LAST',
        '"transcript_id" ASC NULLS LAST',
        '"evidence_row_id" ASC NULLS LAST',
    ]


def main() -> None:
    _assert_flag_parsing()
    _assert_ranking_behavior()
    _assert_debug_truncation_and_json_safety()
    _assert_postgres_query_builder()
    print("smoke_learned_score_retrieval: ok")


if __name__ == "__main__":
    main()
