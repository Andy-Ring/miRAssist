from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from backend.config import (
    PRODUCTION_EVIDENCE_PATH,
    get_evidence_backend,
    get_evidence_table,
    get_learned_score_column,
    resolve_evidence_path,
)
from backend.retrieval import apply_learned_score_ranking
from backend.score_loader import load_compatible_scores
from frontend.evidence_table import build_evidence_shortlist_table, evidence_shortlist_csv_bytes


def _base() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "evidence_row_id": [1, 2],
            "mirna_name": ["hsa-miR-1-3p", "hsa-miR-1-3p"],
            "mirna_name_normalized": ["mir-1-3p", "mir-1-3p"],
            "gene_symbol": ["A", "B"],
            "gene_symbol_normalized": ["A", "B"],
            "transcript_id": ["ENST1", "ENST2"],
            "support_count": [2, 2],
            "overall_evidence_support_percentile": [80.0, 90.0],
            "evidence_family_count": [3, 4],
            "retrieval_score": [9.0, 8.0],
        }
    )


def test_default_production_pointer_is_variant_a_rf_v1() -> None:
    with patch.dict(os.environ, {}, clear=False):
        for key in (
            "EVIDENCE_BACKEND",
            "EVIDENCE_TABLE",
            "MIRASSIST_EVIDENCE",
            "MIRASSIST_EVIDENCE_PATH",
            "MIRASSIST_LEARNED_SCORE_COLUMN",
        ):
            os.environ.pop(key, None)
        assert get_evidence_backend() == "parquet"
        assert resolve_evidence_path() == PRODUCTION_EVIDENCE_PATH.resolve()
        assert get_evidence_table() == "public.mirassist_evidence_variant_a_rf_v1"
        assert get_learned_score_column() == "mirassist_model_score"


def test_model_score_precedes_legacy_and_conflicts_fail() -> None:
    frame = _base().assign(
        mirassist_model_score=[0.8, 0.7],
        mirassist_model_version="mirassist_rf_variant_a_v1",
        mirassist_xgboost_score=np.nan,
    )
    loaded = load_compatible_scores(frame)
    assert loaded.frame["mirassist_score"].tolist() == [0.8, 0.7]
    assert loaded.metadata["score_source_column"] == "mirassist_model_score"

    conflict = frame.copy()
    conflict["mirassist_xgboost_score"] = [0.1, 0.2]
    with pytest.raises(ValueError, match="Conflicting"):
        load_compatible_scores(conflict)


def test_legacy_xgboost_fallback_remains_supported() -> None:
    frame = _base().assign(mirassist_xgboost_score=[0.2, 0.1])
    with pytest.warns(UserWarning, match="legacy"):
        loaded = load_compatible_scores(frame)
    assert loaded.frame["mirassist_score"].tolist() == [0.2, 0.1]
    assert loaded.metadata["score_source_column"] == "mirassist_xgboost_score"


def test_rf_ranking_uses_approved_ties_and_not_labels_or_manual_score() -> None:
    frame = _base().assign(
        mirassist_model_score=[0.5, 0.5],
        mirassist_model_version="mirassist_rf_variant_a_v1",
        mirassist_xgboost_score=np.nan,
        mirtarbase_known_positive=[True, False],
        mirassist_score_rank_within_mirna=[2, 1],
    )
    ranked, diagnostics = apply_learned_score_ranking(
        frame, learned_score_column="mirassist_model_score", enabled=True
    )
    assert ranked["gene_symbol"].tolist() == ["B", "A"]
    assert ranked["mirassist_score"].tolist() == [0.5, 0.5]
    assert diagnostics["active_score_source"] == "mirassist_model_score"
    assert diagnostics["score_semantics"].startswith("raw uncalibrated random-forest")


def test_new_export_uses_user_facing_score_and_never_probability() -> None:
    rows = _base().assign(
        mirassist_model_score=[0.8, 0.7],
        mirassist_score=[0.8, 0.7],
        mirassist_model_version="mirassist_rf_variant_a_v1",
        mirassist_xgboost_score=np.nan,
        mirassist_score_rank_within_mirna=[1, 2],
        mirassist_filtered_rank=[1, 2],
        mirassist_score_percentile_within_mirna=[1.0, 0.0],
    ).to_dict(orient="records")
    export = build_evidence_shortlist_table(rows, "mirna_to_targets")
    assert "miRAssist score" in export.columns
    assert "Model version" in export.columns
    assert "Global rank within miRNA" in export.columns
    assert "Filtered rank" in export.columns
    assert "mirassist_xgboost_score" not in export.columns
    assert b"probability" not in evidence_shortlist_csv_bytes(export).lower()
