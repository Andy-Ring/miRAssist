from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_utils import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    clean_plot_label,
    get_feature_columns_for_families,
    load_labeled_clean_evidence,
    load_pickle,
    prepare_keys,
    read_table,
    require_matplotlib,
    roc_auc_score_manual,
    average_precision_manual,
    score_estimator_on_frame,
    split_train_test_groups,
)


METHOD_SPECS: Dict[str, Dict[str, Any]] = {
    "miRAssist": {
        "method_type": "final_model",
        "score_column": "best_backend_model_score",
        "color": "#0b5d8f",
    },
    "miRDB": {
        "method_type": "published_model",
        "relative_path": Path("mirdb/parsed/mirdb_scores_aligned_to_evidence.csv.gz"),
        "score_column": "mirdb_score",
        "higher_is_better": True,
        "color": "#3f7d20",
    },
    "TargetScan": {
        "method_type": "published_model",
        "score_column": "targetscan_context_score",
        "higher_is_better": False,
        "color": "#a06300",
    },
    "DIANA-MicroT": {
        "method_type": "published_model",
        "relative_path": Path("diana_microt/parsed/diana_microt_scores_aligned_to_evidence.csv.gz"),
        "score_column": "diana_microt_score",
        "higher_is_better": True,
        "color": "#8a3ffc",
    },
    "miRanda": {
        "method_type": "published_model",
        "relative_path": Path("miranda/parsed/miranda_scores_aligned_to_evidence.csv.gz"),
        "score_column": "miranda_best_score",
        "higher_is_better": True,
        "color": "#c73e1d",
    },
    "RNA22": {
        "method_type": "published_model",
        "relative_path": Path("rna22/parsed/rna22_scores_aligned_to_evidence.csv.gz"),
        "score_column": "rna22_best_energy_strength",
        "higher_is_better": True,
        "color": "#00897b",
    },
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate ROC, PR, and Recall@5 plots for the final miRAssist model versus published external models."
    )
    ap.add_argument("--evidence", default="data/processed/mirassist_clean_evidence.parquet")
    ap.add_argument("--labels", default="evaluation/data/heldout_mirtarbase_labels.parquet")
    ap.add_argument("--best-model-metadata", default="evaluation/clean_evidence_eval/models/best_backend_model_metadata.json")
    ap.add_argument("--best-model-pickle", default="evaluation/clean_evidence_eval/models/best_backend_model.pkl")
    ap.add_argument("--external-root", default="evaluation/clean_evidence_eval/external_models")
    ap.add_argument("--step3-metrics", default="evaluation/clean_evidence_eval/results/external_model_comparison_metrics.csv")
    ap.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--test-size", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    return ap.parse_args()


def _load_json(path: str | Path) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Required JSON file was not found: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def _resolve_split_params(args: argparse.Namespace, metadata: Dict[str, Any]) -> Tuple[float, int]:
    metadata_split = metadata.get("split_info", {}) if isinstance(metadata.get("split_info"), dict) else {}
    test_size = float(args.test_size if args.test_size is not None else metadata_split.get("test_size", 0.2))
    seed = int(args.seed if args.seed is not None else metadata.get("seed", 2026))
    return test_size, seed


def _feature_columns_from_metadata(metadata: Dict[str, Any], df: pd.DataFrame) -> List[str]:
    metric_block = metadata.get("metrics", {}) if isinstance(metadata.get("metrics"), dict) else {}
    feature_columns = metric_block.get("feature_columns")
    if isinstance(feature_columns, list) and feature_columns:
        return [str(column) for column in feature_columns]
    if isinstance(feature_columns, str) and feature_columns:
        return [part.strip() for part in feature_columns.split(",") if part.strip()]
    return get_feature_columns_for_families(df)


def _clean_score_values(values: pd.Series | np.ndarray | Sequence[float]) -> np.ndarray:
    series = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return series.to_numpy(dtype=float)


def _join_row_aligned_scores(
    universe_df: pd.DataFrame,
    external_path: str | Path,
    *,
    score_column: str,
    higher_is_better: bool,
) -> Tuple[pd.Series, int, float]:
    external_df = read_table(external_path)
    if "evidence_row_id" not in universe_df.columns:
        raise RuntimeError("The Step 3 test universe does not contain evidence_row_id.")
    if "evidence_row_id" not in external_df.columns:
        raise RuntimeError(f"External file {external_path} does not contain evidence_row_id.")
    if score_column not in external_df.columns:
        raise RuntimeError(f"External file {external_path} does not contain required score column {score_column}.")

    universe_keys = universe_df[["evidence_row_id"]].copy()
    universe_keys["evidence_row_id"] = universe_keys["evidence_row_id"].fillna("").astype(str)
    external = external_df[["evidence_row_id", score_column]].copy()
    external["evidence_row_id"] = external["evidence_row_id"].fillna("").astype(str)
    external = external.drop_duplicates(subset=["evidence_row_id"])

    merged = universe_keys.merge(external, how="left", on="evidence_row_id")
    raw_score = pd.to_numeric(merged[score_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    coverage_n = int(raw_score.notna().sum())
    score = raw_score.astype(float)
    if not higher_is_better:
        score = -1.0 * score
    return score.fillna(0.0), coverage_n, (float(coverage_n) / float(len(universe_df)) if len(universe_df) else 0.0)


def _score_final_model(test_df: pd.DataFrame, metadata: Dict[str, Any], best_model_pickle: str | Path) -> pd.DataFrame:
    pickle_path = Path(best_model_pickle).resolve()
    if not pickle_path.exists():
        raise FileNotFoundError(f"Best backend model pickle was not found: {pickle_path}")
    estimator = load_pickle(pickle_path)
    feature_columns = _feature_columns_from_metadata(metadata, test_df)
    predictions, _ = score_estimator_on_frame(
        estimator,
        test_df,
        feature_columns=feature_columns,
        model_name="miRAssist",
        evidence_family_mode="all",
    )
    predictions["score"] = _clean_score_values(predictions["score"])
    return predictions


def _attach_score_frame(test_df: pd.DataFrame, method_name: str, score: pd.Series) -> pd.DataFrame:
    frame = prepare_keys(test_df).copy()
    frame["method_name"] = method_name
    frame["score"] = _clean_score_values(score)
    return frame


def _compute_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    positives = max(int(y_true.sum()), 1)
    negatives = max(int((1 - y_true).sum()), 1)
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    tpr = np.concatenate(([0.0], tp / positives))
    fpr = np.concatenate(([0.0], fp / negatives))
    return fpr, tpr


def _compute_pr_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    positives = max(int(y_true.sum()), 1)
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    precision = np.concatenate(([precision[0] if len(precision) else 1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return recall, precision


def _compute_recall_at_5_micro(frame: pd.DataFrame) -> float:
    ranked = frame.sort_values(["query_group", "score"], ascending=[True, False]).copy()
    ranked["rank"] = ranked.groupby("query_group").cumcount() + 1
    top5_positives = int(ranked.loc[ranked["rank"] <= 5, "is_positive"].sum())
    total_positives = int(ranked["is_positive"].sum())
    return float(top5_positives) / float(total_positives) if total_positives > 0 else 0.0


def _sanity_check_final_direction(frame: pd.DataFrame) -> None:
    y_true = frame["is_positive"].astype(int).to_numpy()
    score = pd.to_numeric(frame["score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    auroc = float(roc_auc_score_manual(y_true.tolist(), score.tolist()))
    flipped_auroc = float(roc_auc_score_manual(y_true.tolist(), (-1.0 * score).tolist()))
    if auroc < 0.5 and flipped_auroc > 0.7:
        raise RuntimeError(
            f"Final miRAssist score direction appears inverted: AUROC={auroc:.4f}, flipped_AUROC={flipped_auroc:.4f}."
        )


def _plot_roc(frames: Sequence[pd.DataFrame], output_prefix: Path) -> None:
    plt = require_matplotlib()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for frame in frames:
        y_true = frame["is_positive"].astype(int).to_numpy()
        y_score = pd.to_numeric(frame["score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        fpr, tpr = _compute_roc_curve(y_true, y_score)
        auroc = float(roc_auc_score_manual(y_true.tolist(), y_score.tolist()))
        method_name = str(frame["method_name"].iloc[0])
        ax.plot(fpr, tpr, linewidth=2, label=f"{clean_plot_label(method_name)} (AUROC={auroc:.3f})", color=METHOD_SPECS[method_name]["color"])
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("External Models ROC Curve")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_pr(frames: Sequence[pd.DataFrame], prevalence: float, output_prefix: Path) -> None:
    plt = require_matplotlib()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for frame in frames:
        y_true = frame["is_positive"].astype(int).to_numpy()
        y_score = pd.to_numeric(frame["score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        recall, precision = _compute_pr_curve(y_true, y_score)
        pr_auc = float(average_precision_manual(y_true.tolist(), y_score.tolist()))
        method_name = str(frame["method_name"].iloc[0])
        ax.plot(
            recall,
            precision,
            linewidth=2,
            label=f"{clean_plot_label(method_name)} (PR-AUC={pr_auc:.3f})",
            color=METHOD_SPECS[method_name]["color"],
        )
    ax.axhline(prevalence, linestyle="--", color="gray", linewidth=1, label=f"Prevalence ({prevalence:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("External Models Precision-Recall Curve")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_recall_at_5(summary_df: pd.DataFrame, output_prefix: Path) -> None:
    plt = require_matplotlib()
    chart_df = summary_df.sort_values("recall_at_5", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.5, 6))
    colors = [METHOD_SPECS[str(name)]["color"] for name in chart_df["method_name"]]
    ax.bar(np.arange(len(chart_df)), chart_df["recall_at_5"].astype(float), color=colors)
    ax.set_xticks(np.arange(len(chart_df)))
    ax.set_xticklabels([clean_plot_label(name) for name in chart_df["method_name"]], rotation=20, ha="right")
    ax.set_ylabel("Recall at 5")
    ax.set_title("External Models Recall at 5")
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    figures_dir = output_root / "figures"
    results_dir = output_root / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    metadata = _load_json(args.best_model_metadata)
    test_size, seed = _resolve_split_params(args, metadata)
    labeled_df, _ = load_labeled_clean_evidence(args.evidence, args.labels)
    _, test_df, _ = split_train_test_groups(
        labeled_df,
        group_column="query_group",
        test_size=test_size,
        seed=seed,
    )
    test_df = prepare_keys(test_df)

    print(f"Final test rows: {len(test_df)}")
    print(f"Final positives: {int(test_df['is_positive'].sum())}")

    frames: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, Any]] = []

    final_frame = _score_final_model(test_df, metadata, args.best_model_pickle)
    final_frame["method_name"] = "miRAssist"
    _sanity_check_final_direction(final_frame)
    final_auroc = float(roc_auc_score_manual(final_frame["is_positive"].astype(int).tolist(), final_frame["score"].astype(float).tolist()))
    final_pr_auc = float(average_precision_manual(final_frame["is_positive"].astype(int).tolist(), final_frame["score"].astype(float).tolist()))
    final_recall_at_5 = _compute_recall_at_5_micro(final_frame)
    frames.append(final_frame)
    summary_rows.append(
        {
            "method_name": "miRAssist",
            "method_type": "final_model",
            "score_column": "best_backend_model_score",
            "n_test_rows": int(len(final_frame)),
            "n_test_positives": int(final_frame["is_positive"].sum()),
            "n_predicted_rows": int(len(final_frame)),
            "prediction_coverage": 1.0 if len(final_frame) else 0.0,
            "auroc": final_auroc,
            "pr_auc": final_pr_auc,
            "recall_at_5": final_recall_at_5,
        }
    )

    external_root = Path(args.external_root).resolve()
    for method_name, spec in METHOD_SPECS.items():
        if method_name == "miRAssist":
            continue
        if method_name == "TargetScan":
            raw_score = pd.to_numeric(test_df[spec["score_column"]], errors="coerce") if spec["score_column"] in test_df.columns else pd.Series(np.nan, index=test_df.index)
            coverage_n = int(raw_score.notna().sum())
            score = (-1.0 * raw_score.astype(float)).fillna(0.0)
            coverage = float(coverage_n) / float(len(test_df)) if len(test_df) else 0.0
        else:
            external_path = (external_root / spec["relative_path"]).resolve()
            score, coverage_n, coverage = _join_row_aligned_scores(
                test_df,
                external_path,
                score_column=spec["score_column"],
                higher_is_better=bool(spec["higher_is_better"]),
            )
        frame = _attach_score_frame(test_df, method_name, score)
        frames.append(frame)
        auroc = float(roc_auc_score_manual(frame["is_positive"].astype(int).tolist(), frame["score"].astype(float).tolist()))
        pr_auc = float(average_precision_manual(frame["is_positive"].astype(int).tolist(), frame["score"].astype(float).tolist()))
        recall_at_5 = _compute_recall_at_5_micro(frame)
        summary_rows.append(
            {
                "method_name": method_name,
                "method_type": spec["method_type"],
                "score_column": spec["score_column"],
                "n_test_rows": int(len(frame)),
                "n_test_positives": int(frame["is_positive"].sum()),
                "n_predicted_rows": int(coverage_n),
                "prediction_coverage": coverage,
                "auroc": auroc,
                "pr_auc": pr_auc,
                "recall_at_5": recall_at_5,
            }
        )
        print(f"{method_name}: coverage={coverage:.3f} | AUROC={auroc:.3f} | PR-AUC={pr_auc:.3f} | Recall@5={recall_at_5:.3f}")

    summary_df = pd.DataFrame(summary_rows)

    step3_metrics_path = Path(args.step3_metrics).resolve()
    if step3_metrics_path.exists():
        step3_metrics_df = pd.read_csv(step3_metrics_path)
        step3_final = step3_metrics_df[step3_metrics_df["method_name"].astype(str) == "final_model"]
        if not step3_final.empty:
            step3_row = step3_final.iloc[0]
            print(
                "Step 3 final-model metric comparison: "
                f"AUROC current={final_auroc:.3f} vs step3={step3_row.get('auroc')}; "
                f"PR-AUC current={final_pr_auc:.3f} vs step3={step3_row.get('pr_auc')}; "
                f"Recall@10 step3={step3_row.get('recall_at_10')}"
            )

    prevalence = float(test_df["is_positive"].mean()) if len(test_df) else 0.0
    roc_prefix = figures_dir / "external_models_roc_curve"
    pr_prefix = figures_dir / "external_models_precision_recall_curve"
    recall5_prefix = figures_dir / "external_models_recall_at_5"
    summary_path = results_dir / "external_model_curves_summary.csv"

    _plot_roc(frames, roc_prefix)
    _plot_pr(frames, prevalence, pr_prefix)
    _plot_recall_at_5(summary_df, recall5_prefix)
    summary_df.to_csv(summary_path, index=False)

    print(f"ROC figure: {roc_prefix.with_suffix('.png')}")
    print(f"PR figure: {pr_prefix.with_suffix('.png')}")
    print(f"Recall@5 figure: {recall5_prefix.with_suffix('.png')}")
    print(f"Summary table: {summary_path}")


if __name__ == "__main__":
    main()
