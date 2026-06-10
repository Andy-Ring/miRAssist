from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import types

import numpy as np
import pandas as pd


def _install_fake_matplotlib() -> None:
    if "matplotlib.pyplot" in sys.modules:
        return
    fake_pyplot = types.SimpleNamespace(
        figure=lambda *args, **kwargs: None,
        bar=lambda *args, **kwargs: None,
        xticks=lambda *args, **kwargs: None,
        title=lambda *args, **kwargs: None,
        ylabel=lambda *args, **kwargs: None,
        tight_layout=lambda *args, **kwargs: None,
        savefig=lambda *args, **kwargs: None,
        close=lambda *args, **kwargs: None,
    )
    fake_matplotlib = types.ModuleType("matplotlib")
    fake_matplotlib.pyplot = fake_pyplot
    sys.modules["matplotlib"] = fake_matplotlib
    sys.modules["matplotlib.pyplot"] = fake_pyplot


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "evaluation" / "scripts" / "08_train_learned_ranker.py"
    spec = importlib.util.spec_from_file_location("eval_learned_ranker", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    _install_fake_matplotlib()
    learned = _load_module()

    rankings = pd.DataFrame(
        [
            {"query_id": "q1", "mirna": "mir1", "gene_symbol": "A", "is_positive": 1, "rank": 1, "retrieval_score": 0.9, "retrieval_support": 4, "retrieval_ts_contrib": 0.35, "retrieval_clip_contrib": 0.12, "retrieval_mirdb_contrib": 0.25, "retrieval_tcga_contrib": 0.08, "retrieval_seed_contrib": 0.20, "retrieval_rnahybrid_contrib": 0.16, "retrieval_local_au_contrib": 0.09, "retrieval_structure_contrib": 0.45, "support_count": 4, "mirdb_best_score": 88, "ts_best_contextpp": -0.32, "clip_exp_sum": 10, "n_clip_sites": 3, "best_seed_class": "8mer", "best_mfe": -24.0, "best_local_au": 0.4, "best_local_au_by_mfe": 0.3, "BRCA_spearman_rho": -0.31, "mirtarbase_pos": 1},
            {"query_id": "q1", "mirna": "mir1", "gene_symbol": "B", "is_positive": 0, "rank": 2, "retrieval_score": 0.5, "retrieval_support": 2, "retrieval_ts_contrib": 0.10, "retrieval_clip_contrib": 0.03, "retrieval_mirdb_contrib": 0.10, "retrieval_tcga_contrib": 0.02, "retrieval_seed_contrib": 0.08, "retrieval_rnahybrid_contrib": 0.05, "retrieval_local_au_contrib": 0.02, "retrieval_structure_contrib": 0.15, "support_count": 2, "mirdb_best_score": 60, "ts_best_contextpp": -0.10, "clip_exp_sum": 2, "n_clip_sites": 1, "best_seed_class": "6mer", "best_mfe": -15.0, "best_local_au": 0.2, "best_local_au_by_mfe": 0.1, "BRCA_spearman_rho": -0.05, "mirtarbase_pos": 0},
            {"query_id": "q2", "mirna": "mir2", "gene_symbol": "C", "is_positive": 1, "rank": 1, "retrieval_score": 0.85, "retrieval_support": 4, "retrieval_ts_contrib": 0.30, "retrieval_clip_contrib": 0.11, "retrieval_mirdb_contrib": 0.23, "retrieval_tcga_contrib": 0.07, "retrieval_seed_contrib": 0.18, "retrieval_rnahybrid_contrib": 0.15, "retrieval_local_au_contrib": 0.08, "retrieval_structure_contrib": 0.41, "support_count": 4, "mirdb_best_score": 84, "ts_best_contextpp": -0.29, "clip_exp_sum": 9, "n_clip_sites": 2, "best_seed_class": "7mer-m8", "best_mfe": -23.0, "best_local_au": 0.35, "best_local_au_by_mfe": 0.25, "BRCA_spearman_rho": -0.28, "mirtarbase_pos": 1},
            {"query_id": "q2", "mirna": "mir2", "gene_symbol": "D", "is_positive": 0, "rank": 2, "retrieval_score": 0.45, "retrieval_support": 1, "retrieval_ts_contrib": 0.08, "retrieval_clip_contrib": 0.02, "retrieval_mirdb_contrib": 0.08, "retrieval_tcga_contrib": 0.01, "retrieval_seed_contrib": 0.05, "retrieval_rnahybrid_contrib": 0.03, "retrieval_local_au_contrib": 0.01, "retrieval_structure_contrib": 0.09, "support_count": 1, "mirdb_best_score": 52, "ts_best_contextpp": -0.08, "clip_exp_sum": 1, "n_clip_sites": 1, "best_seed_class": "", "best_mfe": -13.0, "best_local_au": 0.15, "best_local_au_by_mfe": 0.05, "BRCA_spearman_rho": -0.01, "mirtarbase_pos": 0},
            {"query_id": "q3", "mirna": "mir3", "gene_symbol": "E", "is_positive": 1, "rank": 1, "retrieval_score": 0.88, "retrieval_support": 5, "retrieval_ts_contrib": 0.33, "retrieval_clip_contrib": 0.13, "retrieval_mirdb_contrib": 0.26, "retrieval_tcga_contrib": 0.09, "retrieval_seed_contrib": 0.22, "retrieval_rnahybrid_contrib": 0.18, "retrieval_local_au_contrib": 0.10, "retrieval_structure_contrib": 0.50, "support_count": 5, "mirdb_best_score": 90, "ts_best_contextpp": -0.34, "clip_exp_sum": 11, "n_clip_sites": 3, "best_seed_class": "8mer", "best_mfe": -25.0, "best_local_au": 0.42, "best_local_au_by_mfe": 0.31, "BRCA_spearman_rho": -0.33, "mirtarbase_pos": 1},
            {"query_id": "q3", "mirna": "mir3", "gene_symbol": "F", "is_positive": 0, "rank": 2, "retrieval_score": 0.40, "retrieval_support": 1, "retrieval_ts_contrib": 0.06, "retrieval_clip_contrib": 0.01, "retrieval_mirdb_contrib": 0.07, "retrieval_tcga_contrib": 0.00, "retrieval_seed_contrib": 0.04, "retrieval_rnahybrid_contrib": 0.02, "retrieval_local_au_contrib": 0.01, "retrieval_structure_contrib": 0.07, "support_count": 1, "mirdb_best_score": 48, "ts_best_contextpp": -0.05, "clip_exp_sum": 1, "n_clip_sites": 1, "best_seed_class": "6mer", "best_mfe": -12.0, "best_local_au": 0.10, "best_local_au_by_mfe": 0.04, "BRCA_spearman_rho": 0.02, "mirtarbase_pos": 0},
            {"query_id": "q4", "mirna": "mir4", "gene_symbol": "G", "is_positive": 1, "rank": 1, "retrieval_score": 0.86, "retrieval_support": 4, "retrieval_ts_contrib": 0.31, "retrieval_clip_contrib": 0.12, "retrieval_mirdb_contrib": 0.24, "retrieval_tcga_contrib": 0.08, "retrieval_seed_contrib": 0.19, "retrieval_rnahybrid_contrib": 0.17, "retrieval_local_au_contrib": 0.08, "retrieval_structure_contrib": 0.44, "support_count": 4, "mirdb_best_score": 86, "ts_best_contextpp": -0.30, "clip_exp_sum": 10, "n_clip_sites": 2, "best_seed_class": "7mer-a1", "best_mfe": -22.0, "best_local_au": 0.36, "best_local_au_by_mfe": 0.26, "BRCA_spearman_rho": -0.29, "mirtarbase_pos": 1},
            {"query_id": "q4", "mirna": "mir4", "gene_symbol": "H", "is_positive": 0, "rank": 2, "retrieval_score": 0.42, "retrieval_support": 1, "retrieval_ts_contrib": 0.07, "retrieval_clip_contrib": 0.01, "retrieval_mirdb_contrib": 0.06, "retrieval_tcga_contrib": 0.01, "retrieval_seed_contrib": 0.04, "retrieval_rnahybrid_contrib": 0.02, "retrieval_local_au_contrib": 0.01, "retrieval_structure_contrib": 0.07, "support_count": 1, "mirdb_best_score": 50, "ts_best_contextpp": -0.06, "clip_exp_sum": 1, "n_clip_sites": 1, "best_seed_class": "", "best_mfe": -11.0, "best_local_au": 0.12, "best_local_au_by_mfe": 0.05, "BRCA_spearman_rho": 0.01, "mirtarbase_pos": 0},
        ]
    )
    query_summary_lookup = {
        query_id: {"mirna": mirna, "n_positives_total": 1}
        for query_id, mirna in [("q1", "mir1"), ("q2", "mir2"), ("q3", "mir3"), ("q4", "mir4")]
    }

    model_df_all, feature_columns_all, prep_notes_all = learned.prepare_learning_frame(
        rankings,
        feature_set="all",
        include_missingness_indicators=True,
    )
    assert "mirtarbase_pos" not in feature_columns_all
    assert "is_positive" not in feature_columns_all
    assert "best_seed_class_rank_encoded" in feature_columns_all
    assert "best_seed_class" not in feature_columns_all
    assert "BRCA_anticorrelation_strength" in feature_columns_all
    assert "retrieval_score" in feature_columns_all
    assert prep_notes_all["dropped_leakage_columns"]

    _, feature_columns_raw, prep_notes_raw = learned.prepare_learning_frame(
        rankings,
        feature_set="raw",
        include_missingness_indicators=False,
    )
    assert "retrieval_score" not in feature_columns_raw
    assert "retrieval_ts_contrib" not in feature_columns_raw
    assert "mirdb_best_score" in feature_columns_raw
    assert "clip_exp_sum" in feature_columns_raw
    assert not prep_notes_raw["missingness_indicators"]

    model_df, feature_columns, prep_notes = learned.prepare_learning_frame(
        rankings,
        feature_set="components",
        include_missingness_indicators=True,
    )
    assert "retrieval_support" in feature_columns
    assert "retrieval_ts_contrib" in feature_columns
    assert "retrieval_structure_contrib" in feature_columns
    assert "retrieval_score" not in feature_columns
    assert "mirdb_best_score" not in feature_columns
    assert any(col.endswith("_is_missing") for col in feature_columns)
    assert prep_notes["feature_set"] == "components"

    train_groups, test_groups = learned.split_groups(model_df["mirna"].tolist(), test_size=0.5, seed=2026)
    train_df = model_df[model_df["mirna"].isin(train_groups)].copy()
    test_df = model_df[model_df["mirna"].isin(test_groups)].copy()
    assert not train_df.empty and not test_df.empty

    try:
        predicted_scores, resolved_name, feature_importance_df, warnings = learned.fit_predict_model(
            model_name="logistic",
            X_train=train_df[feature_columns],
            y_train=train_df["is_positive"].astype(int),
            X_test=test_df[feature_columns],
            balance_mode="class_weight",
            seed=2026,
            use_calibration=False,
        )
    except Exception as exc:
        if "scikit-learn" in str(exc).lower():
            print("smoke_eval_learned_ranker: SKIPPED (scikit-learn unavailable)")
            return
        raise

    assert resolved_name == "logistic"
    assert len(predicted_scores) == len(test_df)
    assert feature_importance_df is not None and not feature_importance_df.empty
    assert isinstance(warnings, list)

    scored_test = test_df.copy()
    scored_test["predicted_score"] = predicted_scores
    query_metrics_df, summary_row, recall_df, precision_df, ranking_df = learned.compute_query_metrics_from_scores(
        scored_test,
        "predicted_score",
        [1, 2],
        query_summary_lookup,
        resolved_name,
        "components",
    )
    assert not query_metrics_df.empty
    assert summary_row["n_queries"] > 0
    assert not recall_df.empty
    assert not precision_df.empty
    assert not ranking_df.empty

    print("smoke_eval_learned_ranker: OK")


if __name__ == "__main__":
    main()
