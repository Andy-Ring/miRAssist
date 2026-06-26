from __future__ import annotations

import argparse
from datetime import datetime, timezone
import traceback
from pathlib import Path
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_utils import (  # noqa: E402
    DEFAULT_K_VALUES,
    DEFAULT_OUTPUT_ROOT,
    fit_and_score_model,
    format_metric_block,
    json_dump,
    load_labeled_clean_evidence,
    make_output_dirs,
    pick_best_model,
    save_bar_figure,
    save_pickle,
    save_pr_curve_figure,
    save_roc_curve_figure,
    save_text,
    split_train_test_groups,
    summarise_dataset,
)


DEFAULT_MODELS = ("logistic", "xgboost", "svm", "mlp", "naive_bayes")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare backend models on the clean miRAssist evidence table.")
    ap.add_argument("--evidence", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=2026)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    output_dirs = make_output_dirs(args.output_root)
    ks = list(DEFAULT_K_VALUES)

    labeled_df, load_info = load_labeled_clean_evidence(args.evidence, args.labels)
    train_df, test_df, split_info = split_train_test_groups(
        labeled_df,
        group_column="query_group",
        test_size=args.test_size,
        seed=args.seed,
    )

    dataset_summary = summarise_dataset(labeled_df, split_info["split_strategy"])
    model_names = [part.strip() for part in str(args.models).split(",") if part.strip()]
    metric_rows = []
    prediction_frames = []
    logs = []
    successful_models = []

    for model_name in model_names:
        try:
            estimator, predictions, metrics = fit_and_score_model(
                train_df,
                test_df,
                model_name=model_name,
                seed=args.seed,
            )
            metrics["status"] = "ok"
            metric_rows.append(metrics)
            prediction_frames.append(predictions)
            successful_models.append((model_name, estimator, metrics))
        except Exception as exc:
            metric_rows.append(
                {
                    "model_name": model_name,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            logs.append(
                {
                    "model_name": model_name,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = output_dirs["results"] / "backend_model_comparison_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    predictions_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    predictions_path = output_dirs["results"] / "backend_model_comparison_predictions.parquet"
    if not predictions_df.empty:
        predictions_df.to_parquet(predictions_path, index=False)
    else:
        predictions_df.to_csv(output_dirs["results"] / "backend_model_comparison_predictions.csv", index=False)

    log_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_summary": dataset_summary,
        "load_info": load_info,
        "split_info": split_info,
        "model_failures": logs,
    }
    json_dump(output_dirs["logs"] / "backend_model_comparison_log.json", log_payload)

    if successful_models:
        best_row = pick_best_model(metrics_df)
        best_name = str(best_row["model_name"])
        best_estimator = next(estimator for model_name, estimator, _ in successful_models if model_name == best_name)
        best_metadata = {
            "model_name": best_name,
            "selected_at": datetime.now(timezone.utc).isoformat(),
            "dataset_summary": dataset_summary,
            "load_info": load_info,
            "split_info": split_info,
            "selection_priority": [f"recall_at_{k}" for k in ks]
            + [f"precision_at_{k}" for k in ks]
            + ["pr_auc", "auroc"],
            "metrics": best_row.to_dict(),
        }
        save_pickle(output_dirs["models"] / "best_backend_model.pkl", best_estimator)
        json_dump(output_dirs["models"] / "best_backend_model_metadata.json", best_metadata)

        summary_lines = [
            "miRAssist Clean Evidence Backend Model Comparison",
            "=" * 48,
            f"Split strategy: {split_info['split_strategy']}",
            f"Rows: {dataset_summary['rows']}",
            f"Positives: {dataset_summary['positives']}",
            f"Positive rate: {dataset_summary['positive_rate']:.6f}",
            "",
            "Model rankings:",
        ]
        ranked = metrics_df[metrics_df["status"] == "ok"].copy()
        ranked = ranked.sort_values(
            [f"recall_at_{k}" for k in ks] + [f"precision_at_{k}" for k in ks] + ["pr_auc", "auroc"],
            ascending=False,
        )
        for _, row in ranked.iterrows():
            summary_lines.append(
                f"- {row['model_name']}: Recall@1={row.get('recall_at_1')}, Recall@3={row.get('recall_at_3')}, "
                f"Recall@5={row.get('recall_at_5')}, Recall@10={row.get('recall_at_10')}, "
                f"Precision@10={row.get('precision_at_10')}, PR-AUC={row.get('pr_auc')}, AUROC={row.get('auroc')}"
            )
        summary_lines.extend(
            [
                "",
                f"Selected best model: {best_name}",
                "Reasoning for selection:",
                "- Selection prioritized top-rank target recovery using Recall at 1, 3, 5, and 10.",
                "- Precision at 1, 3, 5, and 10 was used as a secondary practical ranking criterion.",
                "- PR-AUC and AUROC were retained as global secondary model-wide metrics.",
                "",
                "Best model metrics:",
                format_metric_block(best_row.to_dict(), ks=ks),
            ]
        )
        save_text(output_dirs["results"] / "best_backend_model_summary.txt", "\n".join(summary_lines) + "\n")

        save_bar_figure(
            ranked,
            category_column="model_name",
            value_columns=[f"recall_at_{k}" for k in ks],
            title="Backend Model Comparison",
            ylabel="Recall",
            output_prefix=output_dirs["figures"] / "backend_model_comparison_recall",
        )
        save_bar_figure(
            ranked,
            category_column="model_name",
            value_columns=[f"precision_at_{k}" for k in ks],
            title="Backend Model Comparison",
            ylabel="Precision",
            output_prefix=output_dirs["figures"] / "backend_model_comparison_precision",
        )
        if prediction_frames:
            save_pr_curve_figure(
                prediction_frames,
                title="Backend Model Comparison Precision-Recall Curves",
                output_prefix=output_dirs["figures"] / "backend_model_comparison_pr_curve",
                include_pr_auc_in_legend=True,
            )
            save_roc_curve_figure(
                prediction_frames,
                title="Backend Model Comparison ROC Curves",
                output_prefix=output_dirs["figures"] / "backend_model_comparison_roc_curve",
                include_auroc_in_legend=True,
            )

        print(f"Best backend model: {best_name}")
        print(f"Rows: {dataset_summary['rows']}")
        print(f"Positives: {dataset_summary['positives']}")
        print(f"Positive rate: {dataset_summary['positive_rate']:.6f}")
    else:
        save_text(
            output_dirs["results"] / "best_backend_model_summary.txt",
            "No backend model completed successfully. See logs/backend_model_comparison_log.json for details.\n",
        )
        print("No backend model completed successfully.")


if __name__ == "__main__":
    main()
