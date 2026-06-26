from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.utils import json_dump  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    return ap.parse_args()


def _safe_read_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _extract_k_metric(frame: pd.DataFrame | None, k_col: str, value_col: str, k_value: int) -> float | None:
    if frame is None or frame.empty or k_col not in frame.columns or value_col not in frame.columns:
        return None
    subset = frame[pd.to_numeric(frame[k_col], errors="coerce") == int(k_value)]
    if subset.empty:
        return None
    value = pd.to_numeric(subset.iloc[0][value_col], errors="coerce")
    return None if pd.isna(value) else float(value)


def _summary_row(
    *,
    method: str,
    model_name: str,
    feature_set: str,
    missingness: str,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "method": method,
        "model_name": model_name,
        "feature_set": feature_set,
        "missingness_setting": missingness,
        "mrr": metrics.get("mrr"),
        "median_best_positive_rank": metrics.get("median_best_positive_rank"),
        "auroc": metrics.get("auroc"),
        "auprc": metrics.get("auprc"),
        "precision_at_1": metrics.get("precision_at_1"),
        "precision_at_5": metrics.get("precision_at_5"),
        "precision_at_10": metrics.get("precision_at_10"),
        "recall_at_10": metrics.get("recall_at_10"),
        "recall_at_25": metrics.get("recall_at_25"),
        "recall_at_50": metrics.get("recall_at_50"),
        "recall_at_100": metrics.get("recall_at_100"),
        "random_enrichment_fold_change": metrics.get("random_enrichment_fold_change"),
        "notes": metrics.get("notes", ""),
    }


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    reports_dir = run_root / "reports"

    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []

    baseline_metrics = _safe_read_json(reports_dir / "baseline_metrics" / "metrics_summary.json")
    baseline_precision = _safe_read_csv(reports_dir / "baseline_metrics" / "precision_at_k.csv")
    baseline_recall = _safe_read_csv(reports_dir / "baseline_metrics" / "recall_at_k.csv")
    if baseline_metrics is not None:
        rows.append(
            _summary_row(
                method="blinded_baseline_retrieval",
                model_name="baseline",
                feature_set="full",
                missingness="n/a",
                metrics={
                    "mrr": baseline_metrics.get("mrr"),
                    "median_best_positive_rank": baseline_metrics.get("median_best_positive_rank"),
                    "auroc": baseline_metrics.get("auroc"),
                    "auprc": baseline_metrics.get("auprc"),
                    "precision_at_1": _extract_k_metric(baseline_precision, "k", "precision_at_k", 1),
                    "precision_at_5": _extract_k_metric(baseline_precision, "k", "precision_at_k", 5),
                    "precision_at_10": _extract_k_metric(baseline_precision, "k", "precision_at_k", 10),
                    "recall_at_10": _extract_k_metric(baseline_recall, "k", "recall_at_k", 10),
                    "recall_at_25": _extract_k_metric(baseline_recall, "k", "recall_at_k", 25),
                    "recall_at_50": _extract_k_metric(baseline_recall, "k", "recall_at_k", 50),
                    "recall_at_100": _extract_k_metric(baseline_recall, "k", "recall_at_k", 100),
                },
            )
        )
    else:
        warnings.append("Missing baseline_metrics/metrics_summary.json")

    random_summary = _safe_read_json(reports_dir / "random_baseline" / "random_baseline_summary.json")
    random_recall = _safe_read_csv(reports_dir / "random_baseline" / "observed_vs_random_recall_at_k.csv")
    random_precision = _safe_read_csv(reports_dir / "random_baseline" / "observed_vs_random_precision_at_k.csv")
    random_mrr = _safe_read_csv(reports_dir / "random_baseline" / "observed_vs_random_mrr.csv")
    if random_summary is not None:
        enrichment = None
        if random_recall is not None and not random_recall.empty and "fold_enrichment" in random_recall.columns:
            row10 = random_recall[pd.to_numeric(random_recall["k"], errors="coerce") == 10]
            if not row10.empty:
                value = pd.to_numeric(row10.iloc[0]["fold_enrichment"], errors="coerce")
                enrichment = None if pd.isna(value) else float(value)
        rows.append(
            _summary_row(
                method="random_baseline_comparison",
                model_name="random_baseline",
                feature_set="full",
                missingness="n/a",
                metrics={
                    "mrr": random_summary.get("observed_mrr"),
                    "median_best_positive_rank": random_summary.get("observed_median_best_positive_rank"),
                    "precision_at_1": _extract_k_metric(random_precision, "k", "observed_precision_at_k", 1),
                    "precision_at_5": _extract_k_metric(random_precision, "k", "observed_precision_at_k", 5),
                    "precision_at_10": _extract_k_metric(random_precision, "k", "observed_precision_at_k", 10),
                    "recall_at_10": _extract_k_metric(random_recall, "k", "observed_recall_at_k", 10),
                    "recall_at_25": _extract_k_metric(random_recall, "k", "observed_recall_at_k", 25),
                    "recall_at_50": _extract_k_metric(random_recall, "k", "observed_recall_at_k", 50),
                    "recall_at_100": _extract_k_metric(random_recall, "k", "observed_recall_at_k", 100),
                    "random_enrichment_fold_change": enrichment,
                    "notes": (
                        f"observed_random_mrr_enrichment={float(random_mrr.iloc[0]['fold_enrichment']):.6f}"
                        if random_mrr is not None
                        and not random_mrr.empty
                        and "fold_enrichment" in random_mrr.columns
                        and pd.notna(pd.to_numeric(random_mrr.iloc[0]["fold_enrichment"], errors="coerce"))
                        else ""
                    ),
                },
            )
        )
    else:
        warnings.append("Missing random_baseline/random_baseline_summary.json")

    ablation_summary = _safe_read_csv(reports_dir / "ablation_comparison" / "ablation_metrics_summary.csv")
    ablation_recall = _safe_read_csv(reports_dir / "ablation_comparison" / "ablation_recall_at_k.csv")
    ablation_precision = _safe_read_csv(reports_dir / "ablation_comparison" / "ablation_precision_at_k.csv")
    if ablation_summary is not None:
        for _, row in ablation_summary.iterrows():
            score_mode = str(row.get("score_mode", "ablation"))
            recall_mode = ablation_recall[ablation_recall["score_mode"].astype(str) == score_mode] if ablation_recall is not None and not ablation_recall.empty else None
            precision_mode = ablation_precision[ablation_precision["score_mode"].astype(str) == score_mode] if ablation_precision is not None and not ablation_precision.empty else None
            rows.append(
                _summary_row(
                    method=f"ablation_{score_mode}",
                    model_name=score_mode,
                    feature_set="full",
                    missingness="n/a",
                    metrics={
                        "mrr": pd.to_numeric(row.get("mrr"), errors="coerce"),
                        "median_best_positive_rank": pd.to_numeric(row.get("median_best_positive_rank"), errors="coerce"),
                        "auroc": pd.to_numeric(row.get("auroc"), errors="coerce"),
                        "auprc": pd.to_numeric(row.get("auprc"), errors="coerce"),
                        "precision_at_1": _extract_k_metric(precision_mode, "k", "precision_at_k", 1),
                        "precision_at_5": _extract_k_metric(precision_mode, "k", "precision_at_k", 5),
                        "precision_at_10": _extract_k_metric(precision_mode, "k", "precision_at_k", 10),
                        "recall_at_10": _extract_k_metric(recall_mode, "k", "recall_at_k", 10),
                        "recall_at_25": _extract_k_metric(recall_mode, "k", "recall_at_k", 25),
                        "recall_at_50": _extract_k_metric(recall_mode, "k", "recall_at_k", 50),
                        "recall_at_100": _extract_k_metric(recall_mode, "k", "recall_at_k", 100),
                    },
                )
            )
    else:
        warnings.append("Missing ablation_comparison/ablation_metrics_summary.csv")

    learned_runs = [
        ("learned_ranker_xgboost_raw_missing_true", "true"),
        ("learned_ranker_xgboost_raw_missing_false", "false"),
    ]
    for dirname, missingness in learned_runs:
        summary_df = _safe_read_csv(reports_dir / dirname / "learned_ranker_metrics_summary.csv")
        recall_df = _safe_read_csv(reports_dir / dirname / "learned_ranker_recall_at_k.csv")
        precision_df = _safe_read_csv(reports_dir / dirname / "learned_ranker_precision_at_k.csv")
        if summary_df is None:
            warnings.append(f"Missing {dirname}/learned_ranker_metrics_summary.csv")
            continue
        for _, row in summary_df.iterrows():
            score_mode = str(row.get("score_mode") or row.get("model") or dirname)
            recall_mode = recall_df[recall_df["score_mode"].astype(str) == score_mode] if recall_df is not None and not recall_df.empty and "score_mode" in recall_df.columns else recall_df
            precision_mode = precision_df[precision_df["score_mode"].astype(str) == score_mode] if precision_df is not None and not precision_df.empty and "score_mode" in precision_df.columns else precision_df
            rows.append(
                _summary_row(
                    method=score_mode,
                    model_name=str(row.get("model", score_mode)),
                    feature_set=str(row.get("feature_set", "unknown")),
                    missingness=missingness,
                    metrics={
                        "mrr": pd.to_numeric(row.get("mrr"), errors="coerce"),
                        "median_best_positive_rank": pd.to_numeric(row.get("median_best_positive_rank"), errors="coerce"),
                        "auroc": pd.to_numeric(row.get("auroc"), errors="coerce"),
                        "auprc": pd.to_numeric(row.get("auprc"), errors="coerce"),
                        "precision_at_1": _extract_k_metric(precision_mode, "k", "precision_at_k", 1),
                        "precision_at_5": _extract_k_metric(precision_mode, "k", "precision_at_k", 5),
                        "precision_at_10": _extract_k_metric(precision_mode, "k", "precision_at_k", 10),
                        "recall_at_10": _extract_k_metric(recall_mode, "k", "recall_at_k", 10),
                        "recall_at_25": _extract_k_metric(recall_mode, "k", "recall_at_k", 25),
                        "recall_at_50": _extract_k_metric(recall_mode, "k", "recall_at_k", 50),
                        "recall_at_100": _extract_k_metric(recall_mode, "k", "recall_at_k", 100),
                    },
                )
            )

    model_matrix_dir = reports_dir / "learned_ranker_model_matrix"
    if model_matrix_dir.exists():
        for summary_path in sorted(model_matrix_dir.rglob("*summary*.csv")):
            summary_df = pd.read_csv(summary_path)
            if summary_df.empty:
                continue
            missingness = "unknown"
            lower_path = str(summary_path).lower()
            if "missing_true" in lower_path:
                missingness = "true"
            elif "missing_false" in lower_path:
                missingness = "false"
            for _, row in summary_df.iterrows():
                method = str(row.get("score_mode") or row.get("model") or summary_path.parent.name)
                rows.append(
                    _summary_row(
                        method=f"matrix_{method}",
                        model_name=str(row.get("model", method)),
                        feature_set=str(row.get("feature_set", summary_path.parent.name)),
                        missingness=missingness,
                        metrics={
                            "mrr": pd.to_numeric(row.get("mrr"), errors="coerce"),
                            "median_best_positive_rank": pd.to_numeric(row.get("median_best_positive_rank"), errors="coerce"),
                            "auroc": pd.to_numeric(row.get("auroc"), errors="coerce"),
                            "auprc": pd.to_numeric(row.get("auprc"), errors="coerce"),
                        },
                    )
                )

    if not rows:
        raise RuntimeError("No paper results summary rows could be assembled.")

    summary_df = pd.DataFrame(rows)
    summary_csv = reports_dir / "paper_results_summary.csv"
    summary_json = reports_dir / "paper_results_summary.json"
    summary_df.to_csv(summary_csv, index=False)
    json_dump(
        summary_json,
        {
            "run_root": str(run_root),
            "n_rows": int(len(summary_df)),
            "warnings": warnings,
            "rows": summary_df.to_dict(orient="records"),
        },
    )


if __name__ == "__main__":
    main()
