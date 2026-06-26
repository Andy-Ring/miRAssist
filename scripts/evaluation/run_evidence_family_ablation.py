from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_utils import (  # noqa: E402
    DEFAULT_K_VALUES,
    DEFAULT_OUTPUT_ROOT,
    fit_and_score_model,
    format_metric_block,
    get_feature_family_map,
    json_dump,
    load_labeled_clean_evidence,
    load_pickle,
    make_output_dirs,
    read_table,
    save_bar_figure,
    save_pr_curve_figure,
    save_roc_curve_figure,
    save_text,
    split_train_test_groups,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run leave-one-evidence-family-out validation on clean evidence.")
    ap.add_argument("--evidence", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--best-model-metadata", default=None)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=2026)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    output_dirs = make_output_dirs(args.output_root)
    metadata_path = (
        Path(args.best_model_metadata).resolve()
        if args.best_model_metadata
        else (output_dirs["models"] / "best_backend_model_metadata.json")
    )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Best-model metadata was not found at {metadata_path}. Run backend model comparison first."
        )
    model_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_name = str(model_metadata["model_name"])
    family_names = list(get_feature_family_map().keys())

    labeled_df, load_info = load_labeled_clean_evidence(args.evidence, args.labels)
    train_df, test_df, split_info = split_train_test_groups(
        labeled_df,
        group_column="query_group",
        test_size=args.test_size,
        seed=args.seed,
    )

    rows = []
    prediction_frames = []
    full_metrics = None
    for ablation_name, exclude_families in [("full_model", [])] + [
        (f"leave_out_{family}", [family]) for family in family_names
    ]:
        _, predictions, metrics = fit_and_score_model(
            train_df,
            test_df,
            model_name=model_name,
            exclude_families=exclude_families,
            seed=args.seed,
        )
        metrics["ablation_name"] = ablation_name
        metrics["excluded_family"] = exclude_families[0] if exclude_families else ""
        metrics["status"] = "ok"
        rows.append(metrics)
        predictions["ablation_name"] = ablation_name
        prediction_frames.append(predictions)
        if ablation_name == "full_model":
            full_metrics = metrics

    metrics_df = pd.DataFrame(rows)
    if full_metrics is None:
        raise RuntimeError("Full-model ablation metrics were not computed.")
    metric_columns = ["auroc", "pr_auc"] + [f"recall_at_{k}" for k in DEFAULT_K_VALUES] + [
        f"precision_at_{k}" for k in DEFAULT_K_VALUES
    ]
    for column in metric_columns:
        full_value = float(full_metrics.get(column)) if full_metrics.get(column) is not None else float("nan")
        metrics_df[f"{column}_absolute_drop"] = full_value - pd.to_numeric(metrics_df[column], errors="coerce")
        metrics_df[f"{column}_percent_drop"] = np.where(
            full_value and pd.notna(full_value),
            100.0 * (full_value - pd.to_numeric(metrics_df[column], errors="coerce")) / full_value,
            np.nan,
        )

    metrics_df.to_csv(output_dirs["results"] / "evidence_family_ablation_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        output_dirs["results"] / "evidence_family_ablation_predictions.parquet",
        index=False,
    )
    json_dump(
        output_dirs["logs"] / "evidence_family_ablation_log.json",
        {
            "best_model_metadata": model_metadata,
            "load_info": load_info,
            "split_info": split_info,
        },
    )

    ranked_ablation = metrics_df[metrics_df["ablation_name"] != "full_model"].copy()
    ranked_ablation = ranked_ablation.sort_values(
        ["recall_at_10_absolute_drop", "recall_at_5_absolute_drop", "pr_auc_absolute_drop"],
        ascending=False,
    )
    summary_lines = [
        "miRAssist Clean Evidence Leave-One-Evidence-Family-Out Validation",
        "=" * 62,
        f"Best backend model: {model_name}",
        f"Split strategy: {split_info['split_strategy']}",
        "",
        "Full model metrics:",
        format_metric_block(full_metrics, ks=DEFAULT_K_VALUES),
        "",
        "Ablation ranking by performance loss:",
    ]
    for _, row in ranked_ablation.iterrows():
        summary_lines.append(
            f"- {row['excluded_family']}: Recall@10 drop={row['recall_at_10_absolute_drop']}, "
            f"Recall@5 drop={row['recall_at_5_absolute_drop']}, PR-AUC drop={row['pr_auc_absolute_drop']}"
        )
    save_text(output_dirs["results"] / "evidence_family_ablation_summary.txt", "\n".join(summary_lines) + "\n")

    save_bar_figure(
        ranked_ablation,
        category_column="excluded_family",
        value_columns=["recall_at_10_absolute_drop", "pr_auc_absolute_drop"],
        title="Leave-One-Evidence-Family-Out Validation",
        ylabel="Performance Drop",
        output_prefix=output_dirs["figures"] / "evidence_family_ablation_drop",
    )
    save_pr_curve_figure(
        prediction_frames,
        title="Leave-One-Evidence-Family-Out Precision-Recall Curves",
        output_prefix=output_dirs["figures"] / "evidence_family_ablation_pr_curve",
        name_column="ablation_name",
        include_pr_auc_in_legend=True,
    )
    save_roc_curve_figure(
        prediction_frames,
        title="Leave-One-Evidence-Family-Out ROC Curves",
        output_prefix=output_dirs["figures"] / "evidence_family_ablation_roc_curve",
        name_column="ablation_name",
        include_auroc_in_legend=True,
    )

    print(f"Best backend model: {model_name}")
    print("Ablation ranking:")
    for _, row in ranked_ablation.iterrows():
        print(
            f"  {row['excluded_family']}: Recall@10 drop={row['recall_at_10_absolute_drop']}, "
            f"PR-AUC drop={row['pr_auc_absolute_drop']}"
        )


if __name__ == "__main__":
    main()
