from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_utils import (  # noqa: E402
    EXTERNAL_SCORE_CANDIDATES,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_K_VALUES,
    evaluate_score_series,
    family_only_score,
    fit_and_score_model,
    format_metric_block,
    get_feature_family_map,
    join_external_score_table,
    json_dump,
    load_labeled_clean_evidence,
    load_mirdb_scores,
    make_output_dirs,
    resolve_external_model_paths,
    resolve_source_feature_path,
    read_table,
    save_bar_figure,
    save_pr_curve_figure,
    save_roc_curve_figure,
    save_text,
    split_train_test_groups,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare the final clean-evidence model to evidence families and published models.")
    ap.add_argument("--evidence", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--best-model-metadata", default=None)
    ap.add_argument("--source-features", default=None)
    ap.add_argument("--external-model-dir", default=str(DEFAULT_OUTPUT_ROOT / "external_models"))
    ap.add_argument("--diana-path", default=None)
    ap.add_argument("--miranda-path", default=None)
    ap.add_argument("--pita-path", default=None)
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

    _, final_predictions, final_metrics = fit_and_score_model(
        train_df,
        test_df,
        model_name=model_name,
        seed=args.seed,
    )
    final_metrics["model_name"] = "full_model"

    family_metric_rows = [final_metrics]
    family_prediction_frames = [final_predictions.assign(model_name="full_model")]
    for family_name in family_names:
        family_predictions, family_metrics = evaluate_score_series(
            test_df,
            family_only_score(test_df, family_name),
            comparator_name=family_name,
            ks=DEFAULT_K_VALUES,
        )
        family_metric_rows.append(family_metrics)
        family_prediction_frames.append(family_predictions)

    family_metrics_df = pd.DataFrame(family_metric_rows)
    family_metrics_df.to_csv(output_dirs["results"] / "final_model_vs_evidence_families_metrics.csv", index=False)

    external_paths, external_status = resolve_external_model_paths(
        args.external_model_dir,
        explicit_paths={
            "diana_microt": args.diana_path,
            "miranda": args.miranda_path,
            "pita": args.pita_path,
        },
    )

    published_rows = []
    published_prediction_frames = []
    status_lines = [
        "External Model Data Status",
        "=" * 24,
        f"Source feature table for miRDB: {resolve_source_feature_path(args.source_features)}",
        "",
    ]

    try:
        mirdb_score, mirdb_info = load_mirdb_scores(test_df, source_feature_path=args.source_features)
        pred_df, metrics = evaluate_score_series(test_df, mirdb_score, comparator_name="mirdb", ks=DEFAULT_K_VALUES)
        metrics["status"] = "ok"
        published_rows.append(metrics)
        published_prediction_frames.append(pred_df)
        status_lines.append(f"miRDB: available via source feature table ({mirdb_info['source_path']})")
    except Exception as exc:
        published_rows.append({"model_name": "mirdb", "status": "missing", "error": str(exc)})
        status_lines.append(f"miRDB: missing ({exc})")

    for model_name_key, resolved_path in external_paths.items():
        label = model_name_key
        if resolved_path is None:
            published_rows.append({"model_name": label, "status": "missing"})
            status = external_status[model_name_key]
            status_lines.extend(
                [
                    f"{label}: missing",
                    f"  looked for: {', '.join(status['looked_at'])}",
                    "  expected a CSV, TSV, or parquet file with miRNA/gene keys and a score column.",
                ]
            )
            continue
        try:
            external_df = read_table(resolved_path)
            score, join_info = join_external_score_table(
                test_df,
                external_df,
                model_name=label,
                score_candidates=tuple(EXTERNAL_SCORE_CANDIDATES[model_name_key]),
            )
            pred_df, metrics = evaluate_score_series(test_df, score, comparator_name=label, ks=DEFAULT_K_VALUES)
            metrics["status"] = "ok"
            published_rows.append(metrics)
            published_prediction_frames.append(pred_df)
            status_lines.append(
                f"{label}: available ({resolved_path}) using score column {join_info['score_column']} and join keys {join_info['join_keys']}"
            )
        except Exception as exc:
            published_rows.append({"model_name": label, "status": "failed", "error": str(exc)})
            status_lines.append(f"{label}: failed to load ({exc})")
            status_lines.append(traceback.format_exc())

    published_metrics_df = pd.DataFrame([final_metrics] + published_rows)
    published_metrics_df.to_csv(output_dirs["results"] / "published_model_comparison_metrics.csv", index=False)
    if published_prediction_frames:
        pd.concat([final_predictions.assign(model_name="full_model")] + published_prediction_frames, ignore_index=True).to_parquet(
            output_dirs["results"] / "published_model_comparison_predictions.parquet",
            index=False,
        )
    save_text(output_dirs["results"] / "external_model_data_status.txt", "\n".join(status_lines) + "\n")
    json_dump(
        output_dirs["logs"] / "published_model_comparison_log.json",
        {
            "model_metadata": model_metadata,
            "load_info": load_info,
            "split_info": split_info,
            "external_status": external_status,
        },
    )

    family_ranked = family_metrics_df.sort_values(
        ["recall_at_10", "recall_at_5", "pr_auc", "auroc"],
        ascending=False,
    )
    published_ranked = published_metrics_df[published_metrics_df.get("status", "ok") == "ok"].sort_values(
        ["recall_at_10", "recall_at_5", "pr_auc", "auroc"],
        ascending=False,
    )

    save_bar_figure(
        family_ranked,
        category_column="model_name",
        value_columns=["recall_at_10", "pr_auc"],
        title="Final Model Versus Individual Evidence Families",
        ylabel="Metric Value",
        output_prefix=output_dirs["figures"] / "final_model_vs_evidence_families",
    )
    if not published_ranked.empty:
        save_bar_figure(
            published_ranked,
            category_column="model_name",
            value_columns=["recall_at_10", "pr_auc"],
            title="Published Model Comparison",
            ylabel="Metric Value",
            output_prefix=output_dirs["figures"] / "published_model_comparison",
        )
    save_pr_curve_figure(
        family_prediction_frames,
        title="Final Model Versus Evidence Families Precision-Recall Curves",
        output_prefix=output_dirs["figures"] / "final_model_vs_evidence_families_pr_curve",
    )
    save_roc_curve_figure(
        family_prediction_frames,
        title="Final Model Versus Evidence Families ROC Curves",
        output_prefix=output_dirs["figures"] / "final_model_vs_evidence_families_roc_curve",
    )

    summary_lines = [
        "miRAssist Clean Evidence Final Comparison",
        "=" * 42,
        f"Best backend model: {model_name}",
        f"Rows: {len(labeled_df)}",
        f"Positives: {int(labeled_df['is_positive'].sum())}",
        f"Positive rate: {float(labeled_df['is_positive'].mean()) if len(labeled_df) else 0.0:.6f}",
        f"Split strategy: {split_info['split_strategy']}",
        "",
        "Final model metrics:",
        format_metric_block(final_metrics, ks=DEFAULT_K_VALUES),
        "",
        "Final model versus evidence-family ranking:",
    ]
    for _, row in family_ranked.iterrows():
        summary_lines.append(
            f"- {row['model_name']}: Recall@10={row.get('recall_at_10')}, PR-AUC={row.get('pr_auc')}, AUROC={row.get('auroc')}"
        )
    summary_lines.append("")
    summary_lines.append("Published-model comparison ranking:")
    for _, row in published_ranked.iterrows():
        summary_lines.append(
            f"- {row['model_name']}: Recall@10={row.get('recall_at_10')}, PR-AUC={row.get('pr_auc')}, AUROC={row.get('auroc')}"
        )
    save_text(output_dirs["results"] / "final_model_comparison_summary.txt", "\n".join(summary_lines) + "\n")

    print(f"Best backend model: {model_name}")
    print("Final model versus evidence-family ranking:")
    for _, row in family_ranked.iterrows():
        print(f"  {row['model_name']}: Recall@10={row.get('recall_at_10')}, PR-AUC={row.get('pr_auc')}")
    print("Published-model comparison ranking:")
    for _, row in published_ranked.iterrows():
        print(f"  {row['model_name']}: Recall@10={row.get('recall_at_10')}, PR-AUC={row.get('pr_auc')}")


if __name__ == "__main__":
    main()
