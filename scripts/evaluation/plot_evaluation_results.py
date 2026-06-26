from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_utils import save_bar_figure, save_method_metric_bar_figure  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Regenerate publication-ready clean-evidence evaluation figures.")
    ap.add_argument("--output-root", default="evaluation/clean_evidence_eval")
    return ap.parse_args()


def _maybe_plot(path: Path, category_column: str, value_columns: list[str], title: str, ylabel: str, output_prefix: Path) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    save_bar_figure(
        df,
        category_column=category_column,
        value_columns=value_columns,
        title=title,
        ylabel=ylabel,
        output_prefix=output_prefix,
    )


def main() -> None:
    args = parse_args()
    root = Path(args.output_root).resolve()
    results = root / "results"
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    _maybe_plot(
        results / "backend_model_comparison_metrics.csv",
        "model_name",
        ["recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10"],
        "Backend Model Comparison",
        "Recall",
        figures / "backend_model_comparison_recall_regenerated",
    )
    _maybe_plot(
        results / "evidence_family_ablation_metrics.csv",
        "excluded_family",
        ["recall_at_10_absolute_drop", "pr_auc_absolute_drop"],
        "Leave-One-Evidence-Family-Out Validation",
        "Performance Drop",
        figures / "evidence_family_ablation_regenerated",
    )
    _maybe_plot(
        results / "final_model_vs_evidence_families_metrics.csv",
        "model_name",
        ["recall_at_10", "pr_auc"],
        "Final Model Versus Individual Evidence Families",
        "Metric Value",
        figures / "final_model_vs_evidence_families_regenerated",
    )
    _maybe_plot(
        results / "published_model_comparison_metrics.csv",
        "model_name",
        ["recall_at_10", "pr_auc"],
        "Published Model Comparison",
        "Metric Value",
        figures / "published_model_comparison_regenerated",
    )
    external_metrics = results / "external_model_comparison_metrics.csv"
    if external_metrics.exists():
        external_df = pd.read_csv(external_metrics)
        for metric_name in ("recall_at_10", "precision_at_10", "pr_auc", "auroc"):
            save_method_metric_bar_figure(
                external_df,
                metric_column=metric_name,
                output_prefix=figures / f"external_model_comparison_{metric_name}_regenerated",
                title=f"External Model Comparison: {metric_name.replace('_', ' ').title()}",
            )


if __name__ == "__main__":
    main()
