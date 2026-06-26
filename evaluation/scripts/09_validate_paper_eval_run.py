from __future__ import annotations

import argparse
import math
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


def _find_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _table_all_numeric_zero(path: Path) -> bool:
    frame = _read_table(path)
    if frame.empty:
        return True
    numeric = frame.select_dtypes(include=["number"])
    if numeric.empty:
        return False
    values = numeric.fillna(0.0)
    return bool((values.to_numpy() == 0).all())


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).resolve()

    blinded_evidence = _find_existing(
        run_root / "tables" / "blinded" / "evidence_blinded_no_mirtarbase.parquet",
        run_root / "tables" / "evidence_blinded_no_mirtarbase.parquet",
    )
    heldout_labels = _find_existing(
        run_root / "tables" / "blinded" / "heldout_mirtarbase_labels.parquet",
        run_root / "tables" / "heldout_mirtarbase_labels.parquet",
    )
    manifest_path = _find_existing(
        run_root / "tables" / "eval_queries.csv",
        run_root / "tables" / "eval_queries.parquet",
    )
    json_dir = _find_existing(run_root / "tables" / "json", run_root / "json")
    rankings_path = _find_existing(
        run_root / "tables" / "collected" / "rankings_long.parquet",
        run_root / "tables" / "rankings_long.parquet",
    )
    query_summary_path = _find_existing(
        run_root / "tables" / "collected" / "query_summary.parquet",
        run_root / "tables" / "query_summary.parquet",
    )
    label_join_path = _find_existing(
        run_root / "tables" / "collected" / "label_join_diagnostics.json",
        run_root / "tables" / "label_join_diagnostics.json",
    )

    report_path = run_root / "reports" / "evaluation_validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    critical_errors: List[str] = []
    warnings: List[str] = []
    checks: Dict[str, Any] = {}

    checks["blinded_evidence_exists"] = bool(blinded_evidence)
    checks["heldout_labels_exists"] = bool(heldout_labels)
    checks["manifest_exists"] = bool(manifest_path)
    checks["json_dir_exists"] = bool(json_dir)
    checks["rankings_long_exists"] = bool(rankings_path)
    checks["query_summary_exists"] = bool(query_summary_path)
    checks["label_join_diagnostics_exists"] = bool(label_join_path)

    for key, ok in checks.items():
        if not ok:
            critical_errors.append(f"Missing required evaluation artifact: {key}")

    manifest_rows = None
    json_count = None
    if manifest_path is not None:
        manifest_rows = int(len(_read_table(manifest_path)))
        checks["manifest_rows"] = manifest_rows
    if json_dir is not None:
        json_count = int(len(list(json_dir.glob("*.json"))))
        checks["json_file_count"] = json_count
    if manifest_rows is not None and json_count is not None and json_count != manifest_rows:
        critical_errors.append(
            f"Expected {manifest_rows} evaluation JSON files from the manifest, but found {json_count}."
        )

    if rankings_path is not None and query_summary_path is not None:
        rankings = pd.read_parquet(rankings_path)
        query_summary = pd.read_parquet(query_summary_path)
        is_positive_sum = int(pd.to_numeric(rankings.get("is_positive", 0), errors="coerce").fillna(0).sum())
        n_queries_with_positive_retrieved = int(
            (pd.to_numeric(query_summary.get("n_positives_retrieved", 0), errors="coerce").fillna(0) > 0).sum()
        )
        best_rank_series = pd.to_numeric(query_summary.get("best_positive_rank"), errors="coerce")
        median_best_positive_rank = (
            float(best_rank_series.dropna().median()) if best_rank_series.notna().any() else None
        )
        checks["is_positive_sum"] = is_positive_sum
        checks["n_queries_with_positive_retrieved"] = n_queries_with_positive_retrieved
        checks["median_best_positive_rank"] = median_best_positive_rank
        if is_positive_sum <= 0:
            critical_errors.append("rankings_long.parquet contains zero positive rows after held-out label join.")
        if n_queries_with_positive_retrieved <= 0:
            critical_errors.append("query_summary.parquet shows zero queries with retrieved positives.")
        if median_best_positive_rank is None or not math.isfinite(median_best_positive_rank):
            critical_errors.append("Median best positive rank is not finite.")

    required_reports = {
        "baseline_metrics": run_root / "reports" / "baseline_metrics" / "metrics_summary.json",
        "random_baseline": run_root / "reports" / "random_baseline" / "random_baseline_summary.json",
        "ablation_summary": run_root / "reports" / "ablation_comparison" / "ablation_metrics_summary.csv",
        "xgb_missing_true_summary": run_root / "reports" / "learned_ranker_xgboost_raw_missing_true" / "learned_ranker_metrics_summary.csv",
        "xgb_missing_false_summary": run_root / "reports" / "learned_ranker_xgboost_raw_missing_false" / "learned_ranker_metrics_summary.csv",
        "paper_results_summary_csv": run_root / "reports" / "paper_results_summary.csv",
        "paper_results_summary_json": run_root / "reports" / "paper_results_summary.json",
        "paper_figure_manifest": run_root / "paper_figures" / "figure_manifest.csv",
    }
    for key, path in required_reports.items():
        checks[key] = path.exists()
        if not path.exists():
            critical_errors.append(f"Missing required report artifact: {path}")

    figure_count = int(len(list((run_root / "paper_figures").glob("*.png"))))
    checks["paper_figure_png_count"] = figure_count
    if figure_count <= 0:
        critical_errors.append("No paper figure PNGs were generated.")

    numeric_tables_to_check = [
        run_root / "reports" / "ablation_comparison" / "ablation_metrics_summary.csv",
        run_root / "reports" / "learned_ranker_xgboost_raw_missing_true" / "learned_ranker_metrics_summary.csv",
        run_root / "reports" / "learned_ranker_xgboost_raw_missing_false" / "learned_ranker_metrics_summary.csv",
        run_root / "reports" / "random_baseline" / "observed_vs_random_recall_at_k.csv",
        run_root / "reports" / "baseline_metrics" / "recall_at_k.csv",
        run_root / "reports" / "baseline_metrics" / "precision_at_k.csv",
    ]
    for path in numeric_tables_to_check:
        if not path.exists():
            continue
        if _table_all_numeric_zero(path):
            critical_errors.append(f"Metric table appears to be entirely zero: {path}")

    model_matrix_dir = run_root / "reports" / "learned_ranker_model_matrix"
    checks["model_matrix_dir_exists"] = model_matrix_dir.exists()
    if not model_matrix_dir.exists():
        critical_errors.append(f"Missing learned ranker model matrix directory: {model_matrix_dir}")
    else:
        model_matrix_summaries = list(model_matrix_dir.rglob("*summary*.csv"))
        checks["model_matrix_summary_count"] = int(len(model_matrix_summaries))
        if not model_matrix_summaries:
            critical_errors.append("No model-matrix summary CSVs were found.")

    report = {
        "run_root": str(run_root),
        "checks": checks,
        "warnings": warnings,
        "critical_errors": critical_errors,
        "status": "ok" if not critical_errors else "failed",
    }
    json_dump(report_path, report)

    if critical_errors:
        raise RuntimeError("Paper evaluation validation failed. See evaluation_validation_report.json for details.")


if __name__ == "__main__":
    main()
