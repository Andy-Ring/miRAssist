from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import tempfile
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


def _load_module(filename: str, module_name: str):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "evaluation" / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    _install_fake_matplotlib()
    learned = _load_module("08_train_learned_ranker.py", "eval_learned_ranker")
    scorer = _load_module("09_score_evidence_with_learned_model.py", "score_evidence")

    rankings = pd.DataFrame(
        [
            {"query_id": "q1", "mirna": "mir1", "mirna_name": "mir1", "mirna_name_norm": "mir1", "gene_symbol": "A", "gene_symbol_norm": "A", "transcript_id": "tx1", "is_positive": 1, "retrieval_score": 0.8, "support_count": 4, "support_targetscan": 1, "support_mirdb": 1, "support_encori": 1, "BRCA_support_tcga": 1, "pathway_match_count": 2, "pathway_selected_gene": 1, "mirdb_best_score": 85, "ts_best_contextpp": -0.31, "clip_exp_sum": 10, "n_clip_sites": 3, "best_seed_class": "8mer", "best_mfe": -24.0, "best_local_au": 0.4, "best_local_au_by_mfe": 0.3, "BRCA_spearman_rho": -0.30, "mirtarbase_pos": 1},
            {"query_id": "q1", "mirna": "mir1", "mirna_name": "mir1", "mirna_name_norm": "mir1", "gene_symbol": "B", "gene_symbol_norm": "B", "transcript_id": "tx2", "is_positive": 0, "retrieval_score": 0.4, "support_count": 1, "support_targetscan": 1, "support_mirdb": 0, "support_encori": 0, "BRCA_support_tcga": 0, "pathway_match_count": 1, "pathway_selected_gene": 0, "mirdb_best_score": 55, "ts_best_contextpp": -0.08, "clip_exp_sum": 1, "n_clip_sites": 1, "best_seed_class": "6mer", "best_mfe": -15.0, "best_local_au": 0.1, "best_local_au_by_mfe": 0.04, "BRCA_spearman_rho": 0.01, "mirtarbase_pos": 0},
            {"query_id": "q2", "mirna": "mir2", "mirna_name": "mir2", "mirna_name_norm": "mir2", "gene_symbol": "C", "gene_symbol_norm": "C", "transcript_id": "tx3", "is_positive": 1, "retrieval_score": 0.78, "support_count": 4, "support_targetscan": 1, "support_mirdb": 1, "support_encori": 1, "BRCA_support_tcga": 1, "pathway_match_count": 2, "pathway_selected_gene": 1, "mirdb_best_score": 82, "ts_best_contextpp": -0.28, "clip_exp_sum": 9, "n_clip_sites": 2, "best_seed_class": "7mer-m8", "best_mfe": -22.0, "best_local_au": 0.35, "best_local_au_by_mfe": 0.25, "BRCA_spearman_rho": -0.27, "mirtarbase_pos": 1},
            {"query_id": "q2", "mirna": "mir2", "mirna_name": "mir2", "mirna_name_norm": "mir2", "gene_symbol": "D", "gene_symbol_norm": "D", "transcript_id": "tx4", "is_positive": 0, "retrieval_score": 0.35, "support_count": 1, "support_targetscan": 1, "support_mirdb": 0, "support_encori": 0, "BRCA_support_tcga": 0, "pathway_match_count": 0, "pathway_selected_gene": 0, "mirdb_best_score": 50, "ts_best_contextpp": -0.06, "clip_exp_sum": 1, "n_clip_sites": 1, "best_seed_class": "", "best_mfe": -11.0, "best_local_au": 0.12, "best_local_au_by_mfe": 0.05, "BRCA_spearman_rho": 0.02, "mirtarbase_pos": 0},
        ]
    )

    try:
        model_df, feature_columns, prep_notes = learned.prepare_learning_frame(
            rankings,
            feature_set="all",
            include_missingness_indicators=True,
        )
        estimator, model_type, warnings = learned.fit_model_for_artifact(
            model_name="logistic",
            X_train=model_df[feature_columns],
            y_train=model_df["is_positive"].astype(int),
            balance_mode="class_weight",
            seed=2026,
        )
    except Exception as exc:
        if "scikit-learn" in str(exc).lower():
            print("smoke_score_evidence_with_learned_model: SKIPPED (scikit-learn unavailable)")
            return
        raise

    assert model_type == "logistic"
    assert isinstance(warnings, list)
    assert "mirtarbase_pos" not in feature_columns

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        artifact_path = learned.save_model_artifact(
            tmp / "model.joblib",
            estimator=estimator,
            model_type=model_type,
            model_name="smoke_model_v1",
            feature_set="all",
            include_missingness_indicators=True,
            feature_names=feature_columns,
            leakage_excluded_columns=prep_notes.get("dropped_leakage_columns", []),
            selected_feature_columns=feature_columns,
        )
        evidence_path = tmp / "evidence.parquet"
        out_path = tmp / "scored.parquet"
        scoring_evidence = rankings.drop(columns=["support_count", "pathway_match_count", "pathway_selected_gene"]).copy()
        scoring_evidence.to_parquet(evidence_path, index=False)

        artifact = __import__("joblib").load(artifact_path)
        feature_df, prepared_feature_names, prep_notes_scoring = learned.prepare_feature_frame(
            scoring_evidence,
            feature_set=artifact["feature_set"],
            include_missingness_indicators=artifact["include_missingness_indicators"],
            selected_feature_columns=artifact["feature_names"],
        )
        assert "support_count" in prepared_feature_names
        assert "support_count_is_missing" in prepared_feature_names
        assert "pathway_match_count" in prepared_feature_names
        assert "pathway_selected_gene" in prepared_feature_names
        assert feature_df["pathway_match_count"].eq(0.0).all()
        assert feature_df["pathway_selected_gene"].eq(0.0).all()
        assert feature_df["support_count"].notna().all()
        assert feature_df["support_count_is_missing"].eq(0.0).all()
        assert set(["pathway_match_count", "pathway_selected_gene"]).issubset(set(prep_notes_scoring["created_missing_columns"]))
        scores = scorer.predict_in_batches(artifact["model"], feature_df[list(prepared_feature_names)], batch_size=2)
        scored_df = scoring_evidence.copy()
        scored_df["learned_score_smoke"] = scores
        scored_df["learned_score_model_version"] = artifact["model_name"]
        scored_df["learned_score_feature_set"] = artifact["feature_set"]
        scored_df["learned_score_updated_at"] = "2026-01-01T00:00:00+00:00"
        scorer.write_table(scored_df, out_path)

        roundtrip = pd.read_parquet(out_path)
        assert "learned_score_smoke" in roundtrip.columns
        assert roundtrip["learned_score_smoke"].notna().all()
        for key_col in ["mirna_name", "gene_symbol", "mirna_name_norm", "gene_symbol_norm", "transcript_id"]:
            assert key_col in roundtrip.columns
        assert "mirtarbase_pos" not in prepared_feature_names

    print("smoke_score_evidence_with_learned_model: OK")


if __name__ == "__main__":
    main()
