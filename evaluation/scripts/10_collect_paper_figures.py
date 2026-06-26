from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import sys
from typing import Dict, List

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


EXACT_RENAMES = {
    "observed_vs_random_mrr.png": "fig_random_mrr.png",
    "observed_vs_random_precision_at_k.png": "fig_random_precision_at_k.png",
    "observed_vs_random_recall_at_k.png": "fig_random_recall_at_k.png",
    "ablation_mrr.png": "fig_ablation_mrr.png",
    "ablation_auprc.png": "fig_ablation_auprc.png",
    "ablation_recall_at_10_25_50.png": "fig_ablation_recall_at_k.png",
    "ablation_precision_at_10_25_50.png": "fig_ablation_precision_at_k.png",
    "learned_ranker_mrr.png": "fig_xgb_raw_missing_true_mrr.png",
    "learned_ranker_auprc.png": "fig_xgb_raw_missing_true_auprc.png",
    "top_feature_importance_xgboost.png": "fig_xgb_raw_missing_true_feature_importance.png",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    return ap.parse_args()


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


def _default_prefix(source_dir: Path) -> str:
    name = source_dir.name
    if name == "random_baseline":
        return "fig_random"
    if name == "ablation_comparison":
        return "fig_ablation"
    if name == "learned_ranker_xgboost_raw_missing_true":
        return "fig_xgb_raw_missing_true"
    if name == "learned_ranker_xgboost_raw_missing_false":
        return "fig_xgb_raw_missing_false"
    if "model_matrix" in name:
        return "fig_model_matrix"
    return f"fig_{_slugify(name)}"


def _rename_png(source_path: Path, source_dir: Path) -> str:
    exact = EXACT_RENAMES.get(source_path.name)
    if exact and source_dir.name == "learned_ranker_xgboost_raw_missing_true":
        return exact
    if source_path.name == "learned_ranker_mrr.png" and source_dir.name == "learned_ranker_xgboost_raw_missing_false":
        return "fig_xgb_raw_missing_false_mrr.png"
    if source_path.name == "learned_ranker_auprc.png" and source_dir.name == "learned_ranker_xgboost_raw_missing_false":
        return "fig_xgb_raw_missing_false_auprc.png"
    if source_path.name == "top_feature_importance_xgboost.png" and source_dir.name == "learned_ranker_xgboost_raw_missing_false":
        return "fig_xgb_raw_missing_false_feature_importance.png"
    if exact:
        return exact
    prefix = _default_prefix(source_dir)
    return f"{prefix}_{_slugify(source_path.stem)}.png"


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    paper_dir = run_root / "paper_figures"
    paper_dir.mkdir(parents=True, exist_ok=True)

    source_dirs = [
        run_root / "reports" / "random_baseline",
        run_root / "reports" / "ablation_comparison",
        run_root / "reports" / "learned_ranker_xgboost_raw_missing_true",
        run_root / "reports" / "learned_ranker_xgboost_raw_missing_false",
        run_root / "reports" / "learned_ranker_model_matrix",
    ]

    manifest_rows: List[Dict[str, str]] = []
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        for source_path in sorted(source_dir.rglob("*.png")):
            dest_name = _rename_png(source_path, source_dir)
            dest_path = paper_dir / dest_name
            if dest_path.exists():
                stem = dest_path.stem
                suffix = 2
                while (paper_dir / f"{stem}_{suffix}.png").exists():
                    suffix += 1
                dest_path = paper_dir / f"{stem}_{suffix}.png"
            shutil.copy2(source_path, dest_path)
            manifest_rows.append(
                {
                    "source_dir": str(source_dir),
                    "source_path": str(source_path),
                    "copied_path": str(dest_path),
                    "copied_name": dest_path.name,
                }
            )

    pd.DataFrame(manifest_rows).to_csv(paper_dir / "figure_manifest.csv", index=False)


if __name__ == "__main__":
    main()
