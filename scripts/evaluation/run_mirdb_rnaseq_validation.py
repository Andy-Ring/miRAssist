from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_utils import (  # noqa: E402
    clean_plot_label,
    load_pickle,
    make_output_dirs,
    normalize_gene_symbol,
    normalize_mirna_name,
    prepare_keys,
    read_table,
    require_matplotlib,
)

try:
    from scipy.stats import pearsonr, spearmanr  # type: ignore
except ModuleNotFoundError:
    pearsonr = None
    spearmanr = None


METHOD_ORDER = [
    "miRAssist",
    "TargetScan",
    "miRDB",
    "DIANA-MicroT",
    "miRanda",
    "RNA22",
]

METHOD_SPECS: Dict[str, Dict[str, Any]] = {
    "miRAssist": {
        "score_column": "mirassist_xgboost_score",
        "type": "final_model",
        "color": "#0b5d8f",
    },
    "TargetScan": {
        "score_column": "targetscan_score",
        "type": "published_model",
        "color": "#a06300",
    },
    "miRDB": {
        "score_column": "mirdb_score",
        "type": "published_model",
        "aligned_path": Path("mirdb/parsed/mirdb_scores_aligned_to_evidence.csv.gz"),
        "aligned_score_column": "mirdb_score",
        "aligned_has_prediction_column": "has_prediction",
        "color": "#3f7d20",
    },
    "DIANA-MicroT": {
        "score_column": "diana_microt_score",
        "type": "published_model",
        "aligned_path": Path("diana_microt/parsed/diana_microt_scores_aligned_to_evidence.csv.gz"),
        "aligned_score_column": "diana_microt_score",
        "aligned_has_prediction_column": "has_prediction",
        "color": "#8a3ffc",
    },
    "miRanda": {
        "score_column": "miranda_best_score",
        "type": "published_model",
        "aligned_path": Path("miranda/parsed/miranda_scores_aligned_to_evidence.csv.gz"),
        "aligned_score_column": "miranda_best_score",
        "aligned_has_prediction_column": "has_prediction",
        "color": "#c73e1d",
    },
    "RNA22": {
        "score_column": "rna22_best_energy_strength",
        "type": "published_model",
        "aligned_path": Path("rna22/parsed/rna22_scores_aligned_to_evidence.csv.gz"),
        "aligned_score_column": "rna22_best_energy_strength",
        "aligned_has_prediction_column": "has_prediction",
        "color": "#00897b",
    },
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run miRDB-style RNA-seq repression validation for miRAssist and external models.")
    ap.add_argument("--rnaseq", default="evaluation/data/miRDB_RNAseq_data.xlsx")
    ap.add_argument("--evidence", default="data/processed/mirassist_clean_evidence.parquet")
    ap.add_argument("--best-model-metadata", default="evaluation/clean_evidence_eval/models/best_backend_model_metadata.json")
    ap.add_argument("--best-model-pickle", default="evaluation/clean_evidence_eval/models/best_backend_model.pkl")
    ap.add_argument("--external-root", default="evaluation/clean_evidence_eval/external_models")
    ap.add_argument("--output-root", default="evaluation/clean_evidence_eval")
    ap.add_argument("--top-k", type=int, default=100, choices=[50, 100, 200])
    ap.add_argument("--pseudocount", type=float, default=1.0)
    return ap.parse_args()


def _load_json(path: str | Path) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Required JSON file not found: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def _normalize_rnaseq_mirna_name(value: Any) -> str:
    text = str(value or "").strip().replace("_", "-")
    return normalize_mirna_name(text)


def _safe_score_array(values: Iterable[Any]) -> np.ndarray:
    return (
        pd.to_numeric(pd.Series(list(values)), errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )


def _pearson_and_pvalue(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    xs = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)
    ys = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mask = xs.notna() & ys.notna()
    if int(mask.sum()) < 3:
        return float("nan"), float("nan")
    xv = xs[mask].to_numpy(dtype=float)
    yv = ys[mask].to_numpy(dtype=float)
    if np.std(xv) == 0 or np.std(yv) == 0:
        return float("nan"), float("nan")
    if pearsonr is not None:
        corr, pval = pearsonr(xv, yv)
        return float(corr), float(pval)
    corr = float(np.corrcoef(xv, yv)[0, 1])
    return corr, float("nan")


def _spearman_and_pvalue(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    xs = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)
    ys = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mask = xs.notna() & ys.notna()
    if int(mask.sum()) < 3:
        return float("nan"), float("nan")
    xv = xs[mask].to_numpy(dtype=float)
    yv = ys[mask].to_numpy(dtype=float)
    if len(np.unique(xv)) < 2 or len(np.unique(yv)) < 2:
        return float("nan"), float("nan")
    if spearmanr is not None:
        corr, pval = spearmanr(xv, yv)
        return float(corr), float(pval)
    corr = float(pd.Series(xv).rank(method="average").corr(pd.Series(yv).rank(method="average"), method="pearson"))
    return corr, float("nan")


def _read_rnaseq_replicates(rnaseq_path: str | Path, *, pseudocount: float) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    resolved = Path(rnaseq_path).resolve()
    workbook = pd.ExcelFile(resolved)
    print(f"RNA-seq sheets: {workbook.sheet_names}")
    replicate_long_frames: List[pd.DataFrame] = []
    diagnostics: Dict[str, Any] = {"sheets": []}

    for sheet_name in workbook.sheet_names:
        sheet_df = pd.read_excel(workbook, sheet_name=sheet_name)
        print(f"Sheet {sheet_name}: shape={sheet_df.shape}")
        if sheet_df.shape[1] < 2:
            print(f"[WARN] Sheet {sheet_name} has fewer than 2 columns; skipping.")
            continue
        gene_col = sheet_df.columns[0]
        sample_columns = list(sheet_df.columns[1:])
        normalized_sample_columns = [_normalize_rnaseq_mirna_name(column) for column in sample_columns]
        print(f"Detected miRNA columns for {sheet_name}: {normalized_sample_columns}")

        expr_df = sheet_df.copy()
        expr_df[gene_col] = expr_df[gene_col].map(normalize_gene_symbol)
        expr_df = expr_df[expr_df[gene_col].astype(str).str.strip() != ""].copy()
        expr_df = expr_df.rename(columns={gene_col: "gene_symbol"})
        expr_numeric = expr_df.loc[:, sample_columns].apply(pd.to_numeric, errors="coerce")

        long_rows: List[pd.DataFrame] = []
        for idx, sample_column in enumerate(sample_columns):
            other_columns = [column for column in sample_columns if column != sample_column]
            if len(other_columns) >= 1:
                baseline = expr_numeric.loc[:, other_columns].median(axis=1, skipna=True)
            else:
                baseline = expr_numeric.loc[:, sample_columns].median(axis=1, skipna=True)
            current = expr_numeric[sample_column]
            log2fc = np.log2((current + float(pseudocount)) / (baseline + float(pseudocount)))
            sample_frame = pd.DataFrame(
                {
                    "normalized_mirna": normalized_sample_columns[idx],
                    "gene_symbol": expr_df["gene_symbol"].astype(str),
                    "replicate_name": sheet_name,
                    "sample_column": str(sample_column),
                    "log2_expression_change": pd.to_numeric(log2fc, errors="coerce"),
                }
            )
            sample_frame["repression_strength"] = -1.0 * sample_frame["log2_expression_change"]
            long_rows.append(sample_frame)

        if long_rows:
            replicate_long_frames.append(pd.concat(long_rows, ignore_index=True))
        diagnostics["sheets"].append(
            {
                "sheet_name": sheet_name,
                "shape": [int(sheet_df.shape[0]), int(sheet_df.shape[1])],
                "n_genes": int(expr_df["gene_symbol"].nunique()),
                "miRNA_columns": normalized_sample_columns,
            }
        )

    if not replicate_long_frames:
        raise RuntimeError("No usable RNA-seq replicate sheets were found.")

    long_df = pd.concat(replicate_long_frames, ignore_index=True)
    long_df = long_df.dropna(subset=["log2_expression_change"])
    summary_df = (
        long_df.groupby(["normalized_mirna", "gene_symbol"], as_index=False)
        .agg(
            n_replicates=("replicate_name", "nunique"),
            mean_log2_expression_change=("log2_expression_change", "mean"),
            mean_repression_strength=("repression_strength", "mean"),
        )
    )
    return summary_df, diagnostics


def _score_mirassist_rows(evidence_df: pd.DataFrame, metadata: Dict[str, Any], model_pickle: str | Path) -> pd.Series:
    model_path = Path(model_pickle).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Best backend model pickle not found: {model_path}")
    estimator = load_pickle(model_path)
    feature_columns = metadata.get("metrics", {}).get("feature_columns")
    if isinstance(feature_columns, str):
        feature_columns = [part.strip() for part in feature_columns.split(",") if part.strip()]
    if not isinstance(feature_columns, list) or not feature_columns:
        raise RuntimeError("best_backend_model_metadata.json does not contain feature_columns needed for scoring.")
    feature_df = evidence_df.copy()
    missing_columns = [column for column in feature_columns if column not in feature_df.columns]
    for column in missing_columns:
        feature_df[column] = np.nan
    if missing_columns:
        print(f"[WARN] Added missing feature columns as NaN before miRAssist scoring: {missing_columns}")
    feature_matrix = feature_df.loc[:, list(feature_columns)].copy()
    if hasattr(estimator, "predict_proba"):
        raw_scores = estimator.predict_proba(feature_matrix)[:, 1]
    elif hasattr(estimator, "decision_function"):
        decision = estimator.decision_function(feature_matrix)
        raw_scores = 1.0 / (1.0 + np.exp(-np.asarray(decision, dtype=float)))
    else:
        raw_scores = estimator.predict(feature_matrix).astype(float)
    return pd.Series(_safe_score_array(raw_scores), index=evidence_df.index, dtype=float)


def _collapse_gene_level_scores(
    evidence_df: pd.DataFrame,
    external_root: str | Path,
    metadata: Dict[str, Any],
    best_model_pickle: str | Path,
) -> pd.DataFrame:
    evidence = prepare_keys(evidence_df).copy()
    evidence["normalized_mirna"] = evidence["mirna_name_normalized"].fillna("").astype(str)
    evidence["gene_symbol"] = evidence["gene_symbol_normalized"].fillna("").astype(str)
    evidence["evidence_row_id"] = evidence["evidence_row_id"].fillna("").astype(str)
    evidence = evidence[(evidence["normalized_mirna"] != "") & (evidence["gene_symbol"] != "")].copy()

    base_gene_level = evidence.groupby(["normalized_mirna", "gene_symbol"], as_index=False).agg(
        n_transcript_rows=("evidence_row_id", "size")
    )

    evidence["mirassist_xgboost_score"] = _score_mirassist_rows(evidence, metadata, best_model_pickle)
    evidence["targetscan_score"] = (
        -1.0 * pd.to_numeric(evidence.get("targetscan_context_score"), errors="coerce").replace([np.inf, -np.inf], np.nan)
    ).fillna(0.0)

    gene_level = base_gene_level.copy()
    mirassist_gene = evidence.groupby(["normalized_mirna", "gene_symbol"], as_index=False)["mirassist_xgboost_score"].max()
    targetscan_gene = evidence.groupby(["normalized_mirna", "gene_symbol"], as_index=False)["targetscan_score"].max()
    gene_level = gene_level.merge(mirassist_gene, how="left", on=["normalized_mirna", "gene_symbol"])
    gene_level = gene_level.merge(targetscan_gene, how="left", on=["normalized_mirna", "gene_symbol"])

    external_root_path = Path(external_root).resolve()
    evidence_key_df = evidence[["evidence_row_id", "normalized_mirna", "gene_symbol"]].drop_duplicates(subset=["evidence_row_id"]).copy()

    for method_name in ["miRDB", "DIANA-MicroT", "miRanda", "RNA22"]:
        spec = METHOD_SPECS[method_name]
        aligned_path = (external_root_path / spec["aligned_path"]).resolve()
        aligned_df = read_table(aligned_path)
        if "evidence_row_id" not in aligned_df.columns:
            raise RuntimeError(f"{aligned_path} does not contain evidence_row_id.")
        score_column = str(spec["aligned_score_column"])
        if score_column not in aligned_df.columns:
            raise RuntimeError(f"{aligned_path} does not contain required score column {score_column}.")
        aligned = aligned_df.copy()
        aligned["evidence_row_id"] = aligned["evidence_row_id"].fillna("").astype(str)
        selected_columns = ["evidence_row_id", score_column]
        has_prediction_column = spec.get("aligned_has_prediction_column")
        if has_prediction_column and has_prediction_column in aligned.columns:
            selected_columns.append(str(has_prediction_column))
        aligned = aligned.loc[:, list(dict.fromkeys(selected_columns))].drop_duplicates(subset=["evidence_row_id"])
        merged = evidence_key_df.merge(aligned, how="left", on="evidence_row_id")
        merged[score_column] = pd.to_numeric(merged[score_column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        agg_map: Dict[str, Any] = {score_column: "max"}
        native_flag_output = None
        if has_prediction_column and has_prediction_column in merged.columns:
            merged[has_prediction_column] = pd.to_numeric(merged[has_prediction_column], errors="coerce").fillna(0).astype(int)
            native_flag_output = f"{score_column}_has_prediction"
            merged[native_flag_output] = merged[has_prediction_column]
            agg_map[native_flag_output] = "max"
        gene_method = merged.groupby(["normalized_mirna", "gene_symbol"], as_index=False).agg(agg_map)
        gene_level = gene_level.merge(gene_method, how="left", on=["normalized_mirna", "gene_symbol"])
        if native_flag_output and native_flag_output not in gene_level.columns:
            gene_level[native_flag_output] = 0

    for method_name in METHOD_ORDER:
        score_column = METHOD_SPECS[method_name]["score_column"]
        if score_column in gene_level.columns:
            gene_level[score_column] = pd.to_numeric(gene_level[score_column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    native_flag_columns = [column for column in gene_level.columns if column.endswith("_has_prediction")]
    for column in native_flag_columns:
        gene_level[column] = pd.to_numeric(gene_level[column], errors="coerce").fillna(0).astype(int)
    return gene_level


def _compute_method_summary(
    matched_base: pd.DataFrame,
    method_name: str,
    top_k: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    score_column = METHOD_SPECS[method_name]["score_column"]
    frame = matched_base[["normalized_mirna", "gene_symbol", "mean_log2_expression_change", "mean_repression_strength", score_column]].copy()
    frame = frame.rename(columns={score_column: "score"})
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    frame = frame.sort_values(["normalized_mirna", "score", "gene_symbol"], ascending=[True, False, True]).copy()
    frame["rank_within_mirna"] = frame.groupby("normalized_mirna").cumcount() + 1
    frame["is_top_k"] = (frame["rank_within_mirna"] <= int(top_k)).astype(int)
    frame["method_name"] = method_name

    n_matched_pairs = int(len(frame))
    n_matched_mirnas = int(frame["normalized_mirna"].nunique())
    pearson_log2fc, pearson_log2fc_p = _pearson_and_pvalue(frame["score"], frame["mean_log2_expression_change"])
    spearman_log2fc, spearman_log2fc_p = _spearman_and_pvalue(frame["score"], frame["mean_log2_expression_change"])
    pearson_repression, pearson_repression_p = _pearson_and_pvalue(frame["score"], frame["mean_repression_strength"])
    spearman_repression, spearman_repression_p = _spearman_and_pvalue(frame["score"], frame["mean_repression_strength"])

    topk_frame = frame[frame["is_top_k"] == 1].copy()
    topk_n_pairs = int(len(topk_frame))
    topk_mean_log2fc = float(topk_frame["mean_log2_expression_change"].mean()) if topk_n_pairs else float("nan")
    topk_mean_repression = float(topk_frame["mean_repression_strength"].mean()) if topk_n_pairs else float("nan")

    print(
        f"{method_name}: matched_pairs={n_matched_pairs}, matched_miRNAs={n_matched_mirnas}, "
        f"score_vs_log2FC={spearman_log2fc:.4f}, score_vs_repression_strength={spearman_repression:.4f}"
    )
    if n_matched_pairs < 10:
        print(f"[WARN] {method_name} has too few matched pairs for stable correlation estimates.")
    print(
        f"  Expected direction: score vs log2FC should be negative; observed Spearman={spearman_log2fc:.4f}. "
        f"score vs repression_strength should be positive; observed Spearman={spearman_repression:.4f}."
    )

    summary = {
        "method_name": method_name,
        "score_column": score_column,
        "n_matched_pairs": n_matched_pairs,
        "n_matched_mirnas": n_matched_mirnas,
        "pearson_score_vs_log2fc": pearson_log2fc,
        "pearson_score_vs_log2fc_pvalue": pearson_log2fc_p,
        "spearman_score_vs_log2fc": spearman_log2fc,
        "spearman_score_vs_log2fc_pvalue": spearman_log2fc_p,
        "pearson_score_vs_repression_strength": pearson_repression,
        "pearson_score_vs_repression_strength_pvalue": pearson_repression_p,
        "spearman_score_vs_repression_strength": spearman_repression,
        "spearman_score_vs_repression_strength_pvalue": spearman_repression_p,
        "top_k": int(top_k),
        "topk_n_pairs": topk_n_pairs,
        "topk_mean_log2_expression_change": topk_mean_log2fc,
        "topk_mean_repression_strength": topk_mean_repression,
    }
    return frame, summary


def _compute_native_prediction_summary(matched_base: pd.DataFrame, method_name: str) -> Optional[Dict[str, Any]]:
    score_column = METHOD_SPECS[method_name]["score_column"]
    native_flag_column = f"{score_column}_has_prediction"
    if native_flag_column not in matched_base.columns:
        return None
    positive_frame = matched_base[pd.to_numeric(matched_base[native_flag_column], errors="coerce").fillna(0).astype(int) > 0].copy()
    if positive_frame.empty:
        return {
            "method_name": method_name,
            "score_column": score_column,
            "native_prediction_column": native_flag_column,
            "n_native_positive_pairs": 0,
            "mean_log2_expression_change": float("nan"),
            "mean_repression_strength": float("nan"),
        }
    return {
        "method_name": method_name,
        "score_column": score_column,
        "native_prediction_column": native_flag_column,
        "n_native_positive_pairs": int(len(positive_frame)),
        "mean_log2_expression_change": float(positive_frame["mean_log2_expression_change"].mean()),
        "mean_repression_strength": float(positive_frame["mean_repression_strength"].mean()),
    }


def _plot_metric_bars(
    summary_df: pd.DataFrame,
    *,
    metric_column: str,
    title: str,
    ylabel: str,
    output_prefix: Path,
) -> None:
    plt = require_matplotlib()
    chart_df = summary_df.copy()
    order_map = {name: idx for idx, name in enumerate(METHOD_ORDER)}
    chart_df["_method_order"] = chart_df["method_name"].map(order_map)
    chart_df = chart_df.sort_values("_method_order").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.5, 6))
    values = pd.to_numeric(chart_df[metric_column], errors="coerce")
    colors = [METHOD_SPECS[str(name)]["color"] for name in chart_df["method_name"]]
    ax.bar(np.arange(len(chart_df)), values.astype(float), color=colors)
    ax.axhline(0.0, linestyle="--", color="gray", linewidth=1)
    ax.set_xticks(np.arange(len(chart_df)))
    ax.set_xticklabels([clean_plot_label(name) for name in chart_df["method_name"]], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dirs = make_output_dirs(args.output_root)

    rnaseq_changes_df, rnaseq_info = _read_rnaseq_replicates(args.rnaseq, pseudocount=args.pseudocount)
    rnaseq_changes_path = output_dirs["results"] / "mirdb_rnaseq_expression_changes.csv.gz"
    rnaseq_changes_df.to_csv(rnaseq_changes_path, index=False, compression="gzip")

    evidence_df = read_table(args.evidence)
    evidence_df = prepare_keys(evidence_df)
    model_metadata = _load_json(args.best_model_metadata)
    gene_level_scores_df = _collapse_gene_level_scores(
        evidence_df,
        args.external_root,
        model_metadata,
        args.best_model_pickle,
    )
    model_scores_path = output_dirs["results"] / "mirdb_rnaseq_model_scores_gene_level.csv.gz"
    gene_level_scores_df.to_csv(model_scores_path, index=False, compression="gzip")

    rnaseq_mirnas = set(rnaseq_changes_df["normalized_mirna"].astype(str).unique())
    evidence_mirnas = set(gene_level_scores_df["normalized_mirna"].astype(str).unique())
    rnaseq_genes = set(rnaseq_changes_df["gene_symbol"].astype(str).unique())
    evidence_genes = set(gene_level_scores_df["gene_symbol"].astype(str).unique())
    print(f"Overlapping miRNAs between RNA-seq and evidence: {len(rnaseq_mirnas & evidence_mirnas)}")
    print(f"Overlapping genes between RNA-seq and evidence: {len(rnaseq_genes & evidence_genes)}")

    matched_base = rnaseq_changes_df.merge(
        gene_level_scores_df,
        how="inner",
        on=["normalized_mirna", "gene_symbol"],
    )
    print(f"Matched miRNA-gene pairs across RNA-seq and model score table: {len(matched_base)}")

    summary_rows: List[Dict[str, Any]] = []
    matched_method_frames: List[pd.DataFrame] = []
    native_summary_rows: List[Dict[str, Any]] = []

    for method_name in METHOD_ORDER:
        score_column = METHOD_SPECS[method_name]["score_column"]
        if score_column not in matched_base.columns:
            print(f"[WARN] Missing score column for {method_name}: {score_column}")
            continue
        nonfinite_count = int((~np.isfinite(pd.to_numeric(matched_base[score_column], errors="coerce").fillna(0.0))).sum())
        if nonfinite_count > 0:
            print(f"[WARN] {method_name} had {nonfinite_count} non-finite scores before sanitization.")
        matched_base[score_column] = pd.to_numeric(matched_base[score_column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        frame, summary = _compute_method_summary(matched_base, method_name, args.top_k)
        matched_method_frames.append(frame)
        summary_rows.append(summary)
        native_summary = _compute_native_prediction_summary(matched_base, method_name)
        if native_summary is not None:
            native_summary_rows.append(native_summary)

    summary_df = pd.DataFrame(summary_rows)
    order_map = {name: idx for idx, name in enumerate(METHOD_ORDER)}
    if not summary_df.empty:
        summary_df["_method_order"] = summary_df["method_name"].map(order_map)
        summary_df = summary_df.sort_values("_method_order").drop(columns=["_method_order"]).reset_index(drop=True)
    matched_long_df = pd.concat(matched_method_frames, ignore_index=True) if matched_method_frames else pd.DataFrame()
    native_summary_df = pd.DataFrame(native_summary_rows)
    if not native_summary_df.empty:
        native_summary_df["_method_order"] = native_summary_df["method_name"].map(order_map)
        native_summary_df = native_summary_df.sort_values("_method_order").drop(columns=["_method_order"]).reset_index(drop=True)

    summary_path = output_dirs["results"] / "mirdb_rnaseq_validation_summary.csv"
    matched_path = output_dirs["results"] / "mirdb_rnaseq_validation_matched_pairs.csv.gz"
    native_path = output_dirs["results"] / "mirdb_rnaseq_validation_native_prediction_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    matched_long_df.to_csv(matched_path, index=False, compression="gzip")
    native_summary_df.to_csv(native_path, index=False)

    _plot_metric_bars(
        summary_df,
        metric_column="spearman_score_vs_repression_strength",
        title="miRDB-Style RNA-seq Validation: Spearman Correlation With Repression Strength",
        ylabel="Spearman correlation",
        output_prefix=output_dirs["figures"] / "mirdb_rnaseq_spearman_repression_strength",
    )
    _plot_metric_bars(
        summary_df,
        metric_column="topk_mean_repression_strength",
        title=f"miRDB-Style RNA-seq Validation: Mean Repression Strength Among Top {args.top_k}",
        ylabel="Mean repression strength",
        output_prefix=output_dirs["figures"] / "mirdb_rnaseq_topk_mean_repression",
    )
    _plot_metric_bars(
        summary_df,
        metric_column="topk_mean_log2_expression_change",
        title=f"miRDB-Style RNA-seq Validation: Mean log2 Expression Change Among Top {args.top_k}",
        ylabel="Mean log2 expression change",
        output_prefix=output_dirs["figures"] / "mirdb_rnaseq_topk_mean_log2fc",
    )

    print(f"Saved RNA-seq expression changes to: {rnaseq_changes_path}")
    print(f"Saved gene-level model scores to: {model_scores_path}")
    print(f"Saved RNA-seq validation summary to: {summary_path}")
    print(f"Saved matched-pairs table to: {matched_path}")
    print(f"Saved native-prediction summary to: {native_path}")


if __name__ == "__main__":
    main()
