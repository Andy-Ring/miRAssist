from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_utils import (  # noqa: E402
    DEFAULT_K_VALUES,
    DEFAULT_OUTPUT_ROOT,
    detect_leakage_columns,
    evaluate_score_series,
    fit_and_score_model,
    get_feature_columns_for_families,
    get_feature_family_map,
    json_dump,
    load_labeled_clean_evidence,
    load_pickle,
    load_row_aligned_external_scores,
    make_output_dirs,
    prepare_keys,
    read_table,
    resolve_aligned_external_model_paths,
    save_method_metric_bar_figure,
    score_estimator_on_frame,
    select_family_score,
    select_targetscan_primary_score,
    split_train_test_groups,
    summarise_dataset,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Step 3 clean-evidence evaluation: compare final miRAssist model, evidence-family scores, and external published models."
    )
    ap.add_argument("--evidence", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--best-model-metadata", default="evaluation/clean_evidence_eval/models/best_backend_model_metadata.json")
    ap.add_argument("--best-model-pickle", default="evaluation/clean_evidence_eval/models/best_backend_model.pkl")
    ap.add_argument("--backend-predictions", default="evaluation/clean_evidence_eval/results/backend_model_comparison_predictions.parquet")
    ap.add_argument("--external-root", default="evaluation/clean_evidence_eval/external_models")
    ap.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--mirdb-path", default=None)
    ap.add_argument("--miranda-path", default=None)
    ap.add_argument("--rna22-path", default=None)
    ap.add_argument("--diana-path", default=None)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--limit-rows", type=int, default=None)
    return ap.parse_args()


def _load_best_model_metadata(path: str | Path) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Best backend model metadata not found: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def _metrics_row(
    method_name: str,
    method_type: str,
    score_column: str,
    metrics: Dict[str, Any],
    *,
    n_test_rows: int,
    n_test_positives: int,
    n_predicted_rows: Optional[int],
    prediction_coverage: Optional[float],
) -> Dict[str, Any]:
    row = {
        "method_name": method_name,
        "method_type": method_type,
        "score_column": score_column,
        "n_test_rows": int(n_test_rows),
        "n_test_positives": int(n_test_positives),
        "n_predicted_rows": int(n_predicted_rows) if n_predicted_rows is not None else None,
        "prediction_coverage": float(prediction_coverage) if prediction_coverage is not None else None,
    }
    for metric in ("auroc", "pr_auc") + tuple(f"recall_at_{k}" for k in DEFAULT_K_VALUES) + tuple(
        f"precision_at_{k}" for k in DEFAULT_K_VALUES
    ):
        row[metric] = metrics.get(metric)
    return row


def _status_row(
    method_name: str,
    method_type: str,
    score_info: Dict[str, Any],
    *,
    source_path: Optional[str] = None,
) -> Dict[str, Any]:
    notes = score_info.get("notes")
    if isinstance(notes, (list, tuple)):
        notes_text = " | ".join(str(item) for item in notes)
    else:
        notes_text = "" if notes is None else str(notes)
    return {
        "method_name": method_name,
        "method_type": method_type,
        "score_column": score_info.get("score_column"),
        "score_columns": "|".join(str(item) for item in score_info.get("score_columns", [])),
        "higher_is_stronger": score_info.get("higher_is_stronger"),
        "raw_higher_is_stronger": score_info.get("raw_higher_is_stronger"),
        "n_predicted_rows": score_info.get("n_predicted_rows"),
        "prediction_coverage": score_info.get("prediction_coverage"),
        "n_non_missing_raw": score_info.get("n_non_missing_raw"),
        "prediction_flag_column": score_info.get("prediction_flag_column"),
        "source_path": source_path or score_info.get("source_path"),
        "notes": notes_text,
    }


def _prediction_coverage_from_column(df: pd.DataFrame, column_name: str) -> Tuple[int, float]:
    raw = pd.to_numeric(df[column_name], errors="coerce")
    n = int(raw.notna().sum())
    return n, (float(n) / float(len(df)) if len(df) else 0.0)


def _resolve_final_model_feature_columns(metadata: Dict[str, Any], df: pd.DataFrame) -> List[str]:
    metric_block = metadata.get("metrics", {}) if isinstance(metadata.get("metrics"), dict) else {}
    feature_columns = metric_block.get("feature_columns")
    if isinstance(feature_columns, list) and feature_columns:
        return [str(column) for column in feature_columns]
    return get_feature_columns_for_families(df)


def _load_saved_final_predictions(
    path: str | Path,
    *,
    test_df: pd.DataFrame,
    selected_model_name: str,
) -> Optional[pd.DataFrame]:
    resolved = Path(path).resolve()
    if not resolved.exists():
        return None
    prediction_df = read_table(resolved)
    if prediction_df.empty or "model_name" not in prediction_df.columns:
        return None
    filtered = prediction_df[prediction_df["model_name"].astype(str) == selected_model_name].copy()
    if filtered.empty:
        return None
    if "evaluation_role" in filtered.columns:
        filtered = filtered[filtered["evaluation_role"].fillna("").astype(str) == "test"].copy()
    if "evidence_row_id" not in filtered.columns or "evidence_row_id" not in test_df.columns:
        return None
    filtered["evidence_row_id"] = filtered["evidence_row_id"].fillna("").astype(str)
    key_df = prepare_keys(test_df)[["evidence_row_id"]].copy()
    key_df["evidence_row_id"] = key_df["evidence_row_id"].fillna("").astype(str)
    merged = key_df.merge(filtered, how="left", on="evidence_row_id")
    if "score" not in merged.columns:
        return None
    if pd.to_numeric(merged["score"], errors="coerce").isna().any():
        return None
    output = prepare_keys(test_df).copy()
    output["model_name"] = "final_model"
    output["score"] = pd.to_numeric(merged["score"], errors="coerce").fillna(0.0).astype(float)
    output["evaluation_role"] = "test"
    output["predicted_label"] = (output["score"] >= 0.5).astype(int)
    output["evidence_family_mode"] = "all"
    return output


def _evaluate_final_model(
    args: argparse.Namespace,
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metadata: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    selected_model_name = str(metadata.get("model_name", "xgboost"))
    # Do not use saved backend predictions here.
    # They can be stale or row-misaligned relative to the current clean-evidence split.
    # Step 3 must regenerate final-model scores from the selected backend model.
    if args.backend_predictions:
        print(
            "[WARN] Ignoring --backend-predictions for final_model. "
            "Regenerating scores from the selected Step 1 backend model instead."
        )

    best_model_path = Path(args.best_model_pickle).resolve()
    feature_columns = _resolve_final_model_feature_columns(metadata, train_df)
    leakage = detect_leakage_columns(feature_columns)
    if leakage:
        raise RuntimeError("Leakage features were selected for final-model scoring: " + ", ".join(leakage))
    if best_model_path.exists():
        estimator = load_pickle(best_model_path)
        predictions, metrics = score_estimator_on_frame(
            estimator,
            test_df,
            feature_columns=feature_columns,
            model_name="final_model",
            evidence_family_mode="all",
        )
        info = {
            "score_column": "best_backend_model_score",
            "score_columns": list(feature_columns),
            "higher_is_stronger": True,
            "raw_higher_is_stronger": True,
            "n_predicted_rows": int(len(predictions)),
            "prediction_coverage": 1.0 if len(predictions) else 0.0,
            "notes": ["Loaded fitted best-backend-model artifact and scored the Step 1 test split."],
            "source_path": str(best_model_path),
        }
        return predictions, metrics, info

    estimator, predictions, metrics = fit_and_score_model(
        train_df,
        test_df,
        model_name=selected_model_name,
        seed=args.seed,
    )
    info = {
        "score_column": "best_backend_model_score",
        "score_columns": metrics.get("feature_columns", feature_columns),
        "higher_is_stronger": True,
        "raw_higher_is_stronger": True,
        "n_predicted_rows": int(len(predictions)),
        "prediction_coverage": 1.0 if len(predictions) else 0.0,
        "notes": ["Re-fit the selected backend model on the Step 1 training split because no saved artifact was available."],
        "source_path": None,
    }
    return predictions, metrics, info


def _zero_score_info(method_name: str, source_path: Optional[str], note: str) -> Dict[str, Any]:
    return {
        "score_column": "no_prediction_zero_fallback",
        "score_columns": [],
        "higher_is_stronger": True,
        "raw_higher_is_stronger": True,
        "n_predicted_rows": 0,
        "prediction_coverage": 0.0,
        "notes": [note],
        "source_path": source_path,
        "method_name": method_name,
    }


def _build_long_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = ["auroc", "pr_auc"] + [f"recall_at_{k}" for k in DEFAULT_K_VALUES] + [f"precision_at_{k}" for k in DEFAULT_K_VALUES]
    keep_columns = ["method_name", "method_type"]
    long_df = metrics_df.loc[:, keep_columns + metric_columns].melt(
        id_vars=keep_columns,
        value_vars=metric_columns,
        var_name="metric_name",
        value_name="metric_value",
    )
    return long_df


def main() -> None:
    args = parse_args()
    output_dirs = make_output_dirs(args.output_root)
    metadata = _load_best_model_metadata(args.best_model_metadata)

    labeled_df, load_info = load_labeled_clean_evidence(args.evidence, args.labels, limit_rows=args.limit_rows)
    train_df, test_df, split_info = split_train_test_groups(
        labeled_df,
        group_column="query_group",
        test_size=args.test_size,
        seed=args.seed,
    )
    dataset_summary = summarise_dataset(labeled_df, split_info["split_strategy"])

    print(f"Evidence rows loaded: {len(labeled_df)}")
    print(f"Positive rows: {int(labeled_df['is_positive'].sum())}")
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")

    metrics_rows: List[Dict[str, Any]] = []
    status_rows: List[Dict[str, Any]] = []

    final_predictions, final_metrics, final_info = _evaluate_final_model(
        args,
        train_df=train_df,
        test_df=test_df,
        metadata=metadata,
    )
    metrics_rows.append(
        _metrics_row(
            "final_model",
            "final_model",
            str(final_info["score_column"]),
            final_metrics,
            n_test_rows=len(test_df),
            n_test_positives=int(test_df["is_positive"].sum()),
            n_predicted_rows=int(final_info["n_predicted_rows"]),
            prediction_coverage=float(final_info["prediction_coverage"]),
        )
    )
    status_rows.append(_status_row("final_model", "final_model", final_info))

    family_methods: Sequence[str] = list(get_feature_family_map().keys())
    for family_name in family_methods:
        family_score, family_info = select_family_score(test_df, family_name)
        family_leaks = detect_leakage_columns(family_info.get("score_columns", []))
        if family_leaks:
            raise RuntimeError(f"Leakage columns selected for {family_name}: {', '.join(family_leaks)}")
        pred_df, metrics = evaluate_score_series(test_df, family_score, comparator_name=family_name, ks=DEFAULT_K_VALUES)
        n_predicted_rows = max(int(family_info.get("n_non_missing_raw", 0)), 0)
        prediction_coverage = float(n_predicted_rows) / float(len(test_df)) if len(test_df) else 0.0
        family_info["n_predicted_rows"] = n_predicted_rows
        family_info["prediction_coverage"] = prediction_coverage
        metrics_rows.append(
            _metrics_row(
                family_name,
                "evidence_family",
                str(family_info["score_column"]),
                metrics,
                n_test_rows=len(test_df),
                n_test_positives=int(test_df["is_positive"].sum()),
                n_predicted_rows=n_predicted_rows,
                prediction_coverage=prediction_coverage,
            )
        )
        status_rows.append(_status_row(family_name, "evidence_family", family_info))
        print(f"{family_name}: selected {family_info['score_column']}")

    targetscan_score, targetscan_info = select_targetscan_primary_score(test_df)
    targetscan_leaks = detect_leakage_columns(targetscan_info.get("score_columns", []))
    if targetscan_leaks:
        raise RuntimeError("Leakage columns selected for targetscan: " + ", ".join(targetscan_leaks))
    _, targetscan_metrics = evaluate_score_series(test_df, targetscan_score, comparator_name="targetscan", ks=DEFAULT_K_VALUES)
    targetscan_n_predicted, targetscan_coverage = _prediction_coverage_from_column(test_df, str(targetscan_info["score_column"]))
    targetscan_info["n_predicted_rows"] = targetscan_n_predicted
    targetscan_info["prediction_coverage"] = targetscan_coverage
    metrics_rows.append(
        _metrics_row(
            "targetscan",
            "published_model",
            str(targetscan_info["score_column"]),
            targetscan_metrics,
            n_test_rows=len(test_df),
            n_test_positives=int(test_df["is_positive"].sum()),
            n_predicted_rows=targetscan_n_predicted,
            prediction_coverage=targetscan_coverage,
        )
    )
    status_rows.append(_status_row("targetscan", "published_model", targetscan_info))

    external_paths, external_path_status = resolve_aligned_external_model_paths(
        args.external_root,
        explicit_paths={
            "mirdb": args.mirdb_path,
            "miranda": args.miranda_path,
            "rna22": args.rna22_path,
            "diana_microt": args.diana_path,
        },
    )
    for model_name in ("mirdb", "miranda", "rna22", "diana_microt"):
        path = external_paths.get(model_name)
        if path is None:
            score = pd.Series(np.zeros(len(test_df), dtype=float), index=test_df.index)
            info = _zero_score_info(
                model_name,
                external_path_status[model_name]["path"],
                "Aligned external score file was not found; all test rows were assigned score 0.",
            )
        else:
            score, info = load_row_aligned_external_scores(test_df, model_name=model_name, path=path)
        _, metrics = evaluate_score_series(test_df, score, comparator_name=model_name, ks=DEFAULT_K_VALUES)
        metrics_rows.append(
            _metrics_row(
                model_name,
                "published_model",
                str(info["score_column"]),
                metrics,
                n_test_rows=len(test_df),
                n_test_positives=int(test_df["is_positive"].sum()),
                n_predicted_rows=info.get("n_predicted_rows"),
                prediction_coverage=info.get("prediction_coverage"),
            )
        )
        status_rows.append(_status_row(model_name, "published_model", info, source_path=external_path_status[model_name]["path"]))
        print(f"{model_name}: coverage={info.get('prediction_coverage')}")

    metrics_df = pd.DataFrame(metrics_rows)
    status_df = pd.DataFrame(status_rows)
    long_df = _build_long_metrics(metrics_df)

    metrics_path = output_dirs["results"] / "external_model_comparison_metrics.csv"
    status_path = output_dirs["results"] / "external_model_score_status.csv"
    long_path = output_dirs["results"] / "external_model_comparison_long.csv"
    metrics_df.to_csv(metrics_path, index=False)
    status_df.to_csv(status_path, index=False)
    long_df.to_csv(long_path, index=False)

    save_method_metric_bar_figure(
        metrics_df,
        metric_column="recall_at_10",
        output_prefix=output_dirs["figures"] / "external_model_comparison_recall_at_10",
        title="External Model Comparison: Recall at 10",
    )
    save_method_metric_bar_figure(
        metrics_df,
        metric_column="precision_at_10",
        output_prefix=output_dirs["figures"] / "external_model_comparison_precision_at_10",
        title="External Model Comparison: Precision at 10",
    )
    save_method_metric_bar_figure(
        metrics_df,
        metric_column="pr_auc",
        output_prefix=output_dirs["figures"] / "external_model_comparison_pr_auc",
        title="External Model Comparison: PR-AUC",
    )
    save_method_metric_bar_figure(
        metrics_df,
        metric_column="auroc",
        output_prefix=output_dirs["figures"] / "external_model_comparison_auroc",
        title="External Model Comparison: AUROC",
    )

    json_dump(
        output_dirs["logs"] / "external_model_comparison_log.json",
        {
            "load_info": load_info,
            "split_info": split_info,
            "dataset_summary": dataset_summary,
            "best_model_metadata_path": str(Path(args.best_model_metadata).resolve()),
            "best_model_name": metadata.get("model_name"),
            "external_path_status": external_path_status,
            "limit_rows": args.limit_rows,
            "output_files": {
                "metrics": str(metrics_path),
                "status": str(status_path),
                "long": str(long_path),
            },
        },
    )

    print(f"Selected backend model: {metadata.get('model_name')}")
    print(f"Metrics written to: {metrics_path}")
    print(f"Score status written to: {status_path}")
    print(f"Long metrics written to: {long_path}")


if __name__ == "__main__":
    main()
