from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_SOURCE_CANDIDATES: Tuple[str, ...] = (
    "mirassist_backend_features.parquet",
    "mirassist_backend_features.csv",
    "evidence_interactions.parquet",
    "evidence_interactions.csv",
    "mirassist_evidence_pairs_full.csv",
    "mirassist_evidence_pairs_test.csv",
)

RNAPLFOLD_SITE_COLUMNS: Tuple[str, ...] = (
    "rnaplfold_seed_unpaired_prob",
    "rnaplfold_site_unpaired_prob",
    "rnaplfold_flank_unpaired_prob",
    "rnaplfold_seed_accessibility_score",
    "rnaplfold_site_accessibility_score",
    "rnaplfold_window_length",
    "rnaplfold_region_start",
    "rnaplfold_region_end",
)
RNAPLFOLD_TRANSCRIPT_COLUMNS: Tuple[str, ...] = (
    "rnaplfold_best_seed_unpaired_prob",
    "rnaplfold_mean_seed_unpaired_prob",
    "rnaplfold_best_site_unpaired_prob",
    "rnaplfold_mean_site_unpaired_prob",
    "rnaplfold_best_flank_unpaired_prob",
    "rnaplfold_mean_flank_unpaired_prob",
    "rnaplfold_n_sites_scored",
    "rnaplfold_n_accessible_sites",
)
FAMILY_FLAG_COLUMNS: Tuple[str, ...] = (
    "has_seed_evidence",
    "has_rnahybrid_evidence",
    "has_targetscan_evidence",
    "has_rnaplfold_evidence",
    "has_clip_evidence",
    "has_tcga_evidence",
)
LEAKAGE_NAME_TOKENS: Tuple[str, ...] = (
    "mirtarbase",
    "validated",
    "label",
    "manual",
    "weighted",
    "old_score",
    "ground_truth",
    "heldout",
)
DEFAULT_RNAPLFOLD_ACCESSIBLE_THRESHOLD = 0.5

_MIRNA_HYPHEN_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212\uFE58\uFE63\uFF0D]")
_MIRNA_SPACE_RE = re.compile(r"[\s_]+")
_MIRNA_REPEAT_DASH_RE = re.compile(r"-{2,}")
_MIRNA_MICRORNA_RE = re.compile(r"micro[\s_-]*rna", re.IGNORECASE)
_MIRNA_MIRNA_RE = re.compile(r"mi[\s_-]*rna", re.IGNORECASE)
_MIRNA_PREFIX_RE = re.compile(r"^(?:hsa-)+", re.IGNORECASE)
_MIRNA_CORE_RE = re.compile(r"^(?:mir-?)(.+)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a clean miRAssist evidence table for blinded evaluation.")
    ap.add_argument("--input-dir", default="data/processed", help="Directory containing the source evidence table.")
    ap.add_argument("--input-path", default=None, help="Optional explicit source file path.")
    ap.add_argument(
        "--rnaplfold-features",
        default=None,
        help="Optional RNAplfold accessibility feature table to join into the clean evidence table.",
    )
    ap.add_argument(
        "--rnaplfold-accessible-threshold",
        type=float,
        default=DEFAULT_RNAPLFOLD_ACCESSIBLE_THRESHOLD,
        help="Seed-region unpaired probability threshold used to count accessible RNAplfold sites.",
    )
    ap.add_argument(
        "--output",
        default="data/processed/mirassist_clean_evidence.csv",
        help="Output clean evidence table (.csv or .parquet).",
    )
    ap.add_argument(
        "--report",
        default="data/processed/clean_evidence_validation_report.txt",
        help="Validation report output path.",
    )
    ap.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke testing.")
    return ap.parse_args()


def _log(message: str) -> None:
    print(f"[build_clean_evidence_db] {message}")


def _read_table(path: Path, limit: Optional[int] = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, nrows=limit)
    if suffix == ".parquet":
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read parquet file {path}. Install pyarrow or fastparquet in the active environment."
            ) from exc
        return df.head(limit).copy() if limit is not None else df
    raise ValueError(f"Unsupported input format for {path}. Expected .csv or .parquet.")


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
        return
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
        return
    raise ValueError(f"Unsupported output format for {path}. Use .csv or .parquet.")


def _discover_input_files(input_dir: Path) -> List[Path]:
    matches: List[Path] = []
    if not input_dir.exists():
        return matches
    for candidate in DEFAULT_SOURCE_CANDIDATES:
        path = input_dir / candidate
        if path.exists():
            matches.append(path.resolve())
    return matches


def _resolve_input_path(input_dir: Path, explicit_path: Optional[str]) -> Path:
    if explicit_path:
        path = Path(explicit_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")
        return path

    discovered = _discover_input_files(input_dir)
    if discovered:
        return discovered[0]

    repo_fallbacks = [Path("mirassist_evidence_pairs_full.csv"), Path("mirassist_evidence_pairs_test.csv")]
    for fallback in repo_fallbacks:
        path = fallback.resolve()
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find a source evidence table. Provide --input-path or place a supported file in --input-dir."
    )


def _normalize_mirna_name(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass

    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "nat"}:
        return ""
    text = _MIRNA_HYPHEN_RE.sub("-", text)
    text = _MIRNA_SPACE_RE.sub("-", text)
    text = _MIRNA_MICRORNA_RE.sub("mir", text)
    text = _MIRNA_MIRNA_RE.sub("mir", text)
    text = _MIRNA_PREFIX_RE.sub("", text)
    text = _MIRNA_REPEAT_DASH_RE.sub("-", text).strip("-")
    if not text:
        return ""
    match = _MIRNA_CORE_RE.match(text)
    if match:
        core = match.group(1)
    elif "mir" in text:
        core = text.split("mir", 1)[1].lstrip("-")
    else:
        core = text
    core = _MIRNA_REPEAT_DASH_RE.sub("-", core).strip("-")
    return f"mir-{core}" if core else ""


def _normalize_gene_symbol(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip().upper()


def _normalize_series_cached(series: pd.Series, normalizer) -> pd.Series:
    values = series.fillna("").astype(str)
    unique_values = pd.Index(values.drop_duplicates())
    lookup = {value: normalizer(value) for value in unique_values}
    return values.map(lookup)


def _safe_float_series(df: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _safe_int_series(df: pd.DataFrame, column: str, default: int = 0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=int)
    return pd.to_numeric(df[column], errors="coerce").fillna(default).astype(int)


def _first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    for column in candidates:
        if column in df.columns:
            return df[column]
    return pd.Series(pd.NA, index=df.index, dtype="object")


def _min_across(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    present = [pd.to_numeric(df[col], errors="coerce") for col in columns if col in df.columns]
    if not present:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.concat(present, axis=1).min(axis=1, skipna=True)


def _with_normalized_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "mirna_name_normalized" not in out.columns:
        if "mirna_name_norm" in out.columns:
            out["mirna_name_normalized"] = out["mirna_name_norm"].fillna("").astype(str)
        elif "mirna_name" in out.columns:
            out["mirna_name_normalized"] = _normalize_series_cached(out["mirna_name"], _normalize_mirna_name)
        else:
            out["mirna_name_normalized"] = ""
    if "gene_symbol_normalized" not in out.columns:
        if "gene_symbol_norm" in out.columns:
            out["gene_symbol_normalized"] = out["gene_symbol_norm"].fillna("").astype(str)
        elif "gene_symbol" in out.columns:
            out["gene_symbol_normalized"] = _normalize_series_cached(out["gene_symbol"], _normalize_gene_symbol)
        else:
            out["gene_symbol_normalized"] = ""
    return out


def _derive_unit_of_analysis(df: pd.DataFrame) -> str:
    if "transcript_id" in df.columns:
        return "transcript-level miRNA-target candidate"
    if "site_id" in df.columns:
        return "site-level candidate"
    return "miRNA-gene candidate"


def _aggregate_rnaplfold_to_transcript(
    features_df: pd.DataFrame,
    join_keys: Sequence[str],
    accessible_threshold: float,
) -> pd.DataFrame:
    features = features_df.copy()
    for column in RNAPLFOLD_SITE_COLUMNS:
        if column not in features.columns:
            features[column] = np.nan
    seed_prob = pd.to_numeric(features.get("rnaplfold_seed_unpaired_prob"), errors="coerce")
    site_prob = pd.to_numeric(features.get("rnaplfold_site_unpaired_prob"), errors="coerce")
    flank_prob = pd.to_numeric(features.get("rnaplfold_flank_unpaired_prob"), errors="coerce")
    features["_rnaplfold_site_scored"] = (seed_prob.notna() | site_prob.notna() | flank_prob.notna()).astype(int)
    features["_rnaplfold_accessible_site"] = (seed_prob >= float(accessible_threshold)).fillna(False).astype(int)

    grouped = (
        features.groupby(list(join_keys), dropna=False)
        .agg(
            rnaplfold_best_seed_unpaired_prob=("rnaplfold_seed_unpaired_prob", "max"),
            rnaplfold_mean_seed_unpaired_prob=("rnaplfold_seed_unpaired_prob", "mean"),
            rnaplfold_best_site_unpaired_prob=("rnaplfold_site_unpaired_prob", "max"),
            rnaplfold_mean_site_unpaired_prob=("rnaplfold_site_unpaired_prob", "mean"),
            rnaplfold_best_flank_unpaired_prob=("rnaplfold_flank_unpaired_prob", "max"),
            rnaplfold_mean_flank_unpaired_prob=("rnaplfold_flank_unpaired_prob", "mean"),
            rnaplfold_n_sites_scored=("_rnaplfold_site_scored", "sum"),
            rnaplfold_n_accessible_sites=("_rnaplfold_accessible_site", "sum"),
        )
        .reset_index()
    )
    return grouped


def _join_rnaplfold_features(
    clean_df: pd.DataFrame,
    features_df: pd.DataFrame,
    *,
    unit_of_analysis: str,
    accessible_threshold: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    clean = _with_normalized_keys(clean_df)
    features = _with_normalized_keys(features_df)

    if "evidence_row_id" not in features.columns and "eval_row_id" in features.columns:
        features = features.rename(columns={"eval_row_id": "evidence_row_id"})

    if unit_of_analysis == "transcript-level miRNA-target candidate":
        key_options: Tuple[Tuple[str, ...], ...] = (
            ("evidence_row_id",),
            ("mirna_name_normalized", "gene_symbol_normalized", "transcript_id"),
            ("mirna_name", "gene_symbol", "transcript_id"),
        )
    else:
        key_options = (
            ("evidence_row_id",),
            ("site_id",),
            ("mirna_name_normalized", "gene_symbol_normalized", "transcript_id"),
            ("mirna_name", "gene_symbol", "transcript_id"),
        )

    selected_keys: Optional[Tuple[str, ...]] = None
    for keys in key_options:
        if all(key in clean.columns for key in keys) and all(key in features.columns for key in keys):
            selected_keys = keys
            break
    if selected_keys is None:
        raise RuntimeError(
            "Could not join RNAplfold features back to the evidence table. "
            "Provide one of: evidence_row_id, site_id, or mirna/gene/transcript keys."
        )

    keep_columns = [key for key in selected_keys]
    if unit_of_analysis == "transcript-level miRNA-target candidate":
        keep_columns.extend(col for col in RNAPLFOLD_SITE_COLUMNS if col in features.columns)
        aggregated = _aggregate_rnaplfold_to_transcript(
            features[keep_columns].copy(),
            selected_keys,
            accessible_threshold,
        )
        merged = clean.merge(aggregated, how="left", on=list(selected_keys))
        joined_signal_rows = int(
            merged[[col for col in RNAPLFOLD_TRANSCRIPT_COLUMNS if col in merged.columns]]
            .notna()
            .any(axis=1)
            .sum()
            if any(col in merged.columns for col in RNAPLFOLD_TRANSCRIPT_COLUMNS)
            else 0
        )
    else:
        keep_columns.extend(col for col in RNAPLFOLD_SITE_COLUMNS if col in features.columns)
        merged = clean.merge(
            features[keep_columns].drop_duplicates(subset=list(selected_keys)),
            how="left",
            on=list(selected_keys),
        )
        joined_signal_rows = int(
            merged[[col for col in RNAPLFOLD_SITE_COLUMNS if col in merged.columns]].notna().any(axis=1).sum()
            if any(col in merged.columns for col in RNAPLFOLD_SITE_COLUMNS)
            else 0
        )
    diagnostics = {
        "rnaplfold_join_keys": list(selected_keys),
        "rnaplfold_input_level": "site-level",
        "rnaplfold_output_level": "transcript-level summary"
        if unit_of_analysis == "transcript-level miRNA-target candidate"
        else "site-level",
        "rnaplfold_feature_rows": int(len(features)),
        "rnaplfold_joined_rows_with_signal": joined_signal_rows,
        "rnaplfold_accessible_threshold": float(accessible_threshold),
        "rnaplfold_aggregation": "arithmetic mean for transcript summaries",
    }
    return merged, diagnostics


def _add_evidence_family_flags(clean: pd.DataFrame) -> pd.DataFrame:
    out = clean.copy()
    out["has_seed_evidence"] = (
        (pd.to_numeric(out["n_seed_sites"], errors="coerce").fillna(0) > 0)
        | (out["seed_match_type"].fillna("").astype(str).str.strip() != "")
        | (pd.to_numeric(out["is_8mer"], errors="coerce").fillna(0) > 0)
        | (pd.to_numeric(out["is_7mer_m8"], errors="coerce").fillna(0) > 0)
        | (pd.to_numeric(out["is_7mer_a1"], errors="coerce").fillna(0) > 0)
        | (pd.to_numeric(out["is_6mer"], errors="coerce").fillna(0) > 0)
    ).astype(int)
    out["has_rnahybrid_evidence"] = (
        pd.to_numeric(out["rnahybrid_mfe"], errors="coerce").notna()
        | pd.to_numeric(out["rnahybrid_seed_mfe"], errors="coerce").notna()
        | (out["rnahybrid_site_start"].fillna("").astype(str).str.strip() != "")
    ).astype(int)
    out["has_targetscan_evidence"] = (
        pd.to_numeric(out["targetscan_context_score"], errors="coerce").notna()
        | pd.to_numeric(out["targetscan_context_score_percentile"], errors="coerce").notna()
        | pd.to_numeric(out["targetscan_pct"], errors="coerce").notna()
        | (pd.to_numeric(out["targetscan_conserved_site"], errors="coerce").fillna(0) > 0)
    ).astype(int)
    if "rnaplfold_n_sites_scored" in out.columns:
        out["has_rnaplfold_evidence"] = (
            pd.to_numeric(out["rnaplfold_n_sites_scored"], errors="coerce").fillna(0) > 0
        ).astype(int)
    else:
        seed_prob = pd.to_numeric(out["rnaplfold_seed_unpaired_prob"], errors="coerce") if "rnaplfold_seed_unpaired_prob" in out.columns else pd.Series(np.nan, index=out.index, dtype=float)
        site_prob = pd.to_numeric(out["rnaplfold_site_unpaired_prob"], errors="coerce") if "rnaplfold_site_unpaired_prob" in out.columns else pd.Series(np.nan, index=out.index, dtype=float)
        flank_prob = pd.to_numeric(out["rnaplfold_flank_unpaired_prob"], errors="coerce") if "rnaplfold_flank_unpaired_prob" in out.columns else pd.Series(np.nan, index=out.index, dtype=float)
        out["has_rnaplfold_evidence"] = (
            seed_prob.notna()
            | site_prob.notna()
            | flank_prob.notna()
        ).astype(int)
    out["has_clip_evidence"] = (
        (pd.to_numeric(out["clip_any_support"], errors="coerce").fillna(0) > 0)
        | (pd.to_numeric(out["clip_max_score"], errors="coerce").fillna(0) > 0)
        | (pd.to_numeric(out["clip_n_experiments"], errors="coerce").fillna(0) > 0)
    ).astype(int)
    out["has_tcga_evidence"] = (
        (pd.to_numeric(out["tcga_n_supported_contexts"], errors="coerce").fillna(0) > 0)
        | (pd.to_numeric(out["tcga_any_anticorrelated"], errors="coerce").fillna(0) > 0)
        | pd.to_numeric(out["tcga_mean_spearman_rho"], errors="coerce").notna()
    ).astype(int)
    out["n_evidence_families_present"] = out[list(FAMILY_FLAG_COLUMNS)].fillna(0).astype(int).sum(axis=1)
    return out


def _find_leakage_columns(columns: Iterable[str]) -> List[str]:
    blocked: List[str] = []
    for column in columns:
        lower = str(column).lower()
        if any(token in lower for token in LEAKAGE_NAME_TOKENS):
            blocked.append(str(column))
    return blocked


def build_clean_evidence_table(
    source_df: pd.DataFrame,
    *,
    rnaplfold_df: Optional[pd.DataFrame] = None,
    rnaplfold_accessible_threshold: float = DEFAULT_RNAPLFOLD_ACCESSIBLE_THRESHOLD,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    source = source_df.loc[:, ~source_df.columns.duplicated()].copy()
    source = _with_normalized_keys(source)
    unit_of_analysis = _derive_unit_of_analysis(source)

    clean = pd.DataFrame(index=source.index)
    if "eval_row_id" in source.columns:
        clean["evidence_row_id"] = source["eval_row_id"]
    elif "evidence_row_id" in source.columns:
        clean["evidence_row_id"] = source["evidence_row_id"]
    else:
        clean["evidence_row_id"] = np.arange(len(source), dtype=int)

    clean["mirna_name"] = _first_existing(source, ["mirna_name", "mirna"])
    clean["mirna_name_normalized"] = source["mirna_name_normalized"]
    clean["gene_symbol"] = _first_existing(source, ["gene_symbol", "gene"])
    clean["gene_symbol_normalized"] = source["gene_symbol_normalized"]
    clean["transcript_id"] = _first_existing(source, ["transcript_id"])
    clean["site_id"] = _first_existing(source, ["site_id"])
    clean["chrom"] = _first_existing(source, ["chrom", "chr"])
    clean["start"] = _first_existing(source, ["start", "site_start", "best_site_start", "best_site_start_by_mfe"])
    clean["end"] = _first_existing(source, ["end", "site_end", "best_site_end", "best_site_end_by_mfe"])
    clean["strand"] = _first_existing(source, ["strand"])
    clean["site_sequence"] = _first_existing(source, ["site_sequence", "target_site_sequence"])
    clean["window_sequence"] = _first_existing(source, ["window_sequence", "target_window_sequence", "utr_window_sequence"])

    clean["seed_match_type"] = _first_existing(source, ["best_seed_class", "best_site_class_by_mfe"])
    clean["is_8mer"] = (_safe_float_series(source, "n_sites_8mer", default=0).fillna(0) > 0).astype(int)
    clean["is_7mer_m8"] = (_safe_float_series(source, "n_sites_7mer_m8", default=0).fillna(0) > 0).astype(int)
    clean["is_7mer_a1"] = (_safe_float_series(source, "n_sites_7mer_a1", default=0).fillna(0) > 0).astype(int)
    clean["is_6mer"] = (_safe_float_series(source, "n_sites_6mer", default=0).fillna(0) > 0).astype(int)
    clean["seed_pairing_score"] = _safe_float_series(source, "best_seed_rank")
    clean["n_seed_sites"] = _safe_float_series(source, "n_total_sites")
    clean["best_seed_site_type"] = _first_existing(source, ["best_seed_class", "best_site_class_by_mfe"])

    clean["rnahybrid_mfe"] = _safe_float_series(source, "best_mfe")
    clean["rnahybrid_mfe_best_site"] = _safe_float_series(source, "best_mfe")
    clean["rnahybrid_site_start"] = _first_existing(source, ["best_site_start_by_mfe", "best_site_start", "site_start"])
    clean["rnahybrid_site_end"] = _first_existing(source, ["best_site_end_by_mfe", "best_site_end", "site_end"])
    clean["rnahybrid_alignment"] = _first_existing(source, ["rnahybrid_alignment"])
    clean["rnahybrid_seed_mfe"] = _min_across(source, ["best_8mer_mfe", "best_7mer_m8_mfe"])
    clean["rnahybrid_strength"] = -1.0 * clean["rnahybrid_mfe"]

    clean["targetscan_context_score"] = _safe_float_series(source, "ts_best_contextpp")
    clean["targetscan_context_score_percentile"] = _safe_float_series(source, "ts_best_percentile")
    clean["targetscan_aggregate_context_score"] = _safe_float_series(source, "ts_weighted_context_score")
    clean["targetscan_conserved_site"] = _safe_int_series(source, "targetscan_conserved_site", default=0)
    clean["targetscan_pct"] = _safe_float_series(source, "targetscan_pct")
    clean["targetscan_branch_length_score"] = _safe_float_series(source, "targetscan_branch_length_score")

    clean["clip_any_support"] = (
        (_safe_int_series(source, "support_encori", default=0) > 0)
        | (_safe_float_series(source, "n_clip_sites", default=0).fillna(0) > 0)
        | (_safe_float_series(source, "clip_exp_sum", default=0).fillna(0) > 0)
        | (_safe_float_series(source, "clip_exp_max", default=0).fillna(0) > 0)
    ).astype(int)
    clean["clip_max_score"] = _safe_float_series(source, "clip_exp_max")
    clean["clip_n_experiments"] = _safe_float_series(source, "clip_exp_sum")
    clean["clip_n_cell_lines"] = _safe_float_series(source, "clip_n_cell_lines")
    clean["clip_source"] = pd.Series(
        np.where(clean["clip_any_support"].astype(int) > 0, "ENCORI", pd.NA),
        index=clean.index,
        dtype="object",
    )
    clean["encori_clip_score"] = _safe_float_series(source, "clip_exp_sum")

    for tcga in ("BRCA", "PRAD", "COAD"):
        clean[f"{tcga}_spearman_rho"] = _safe_float_series(source, f"{tcga}_spearman_rho")
        clean[f"{tcga}_repression_evidence"] = _safe_int_series(source, f"{tcga}_repression_evidence", default=0)
        anticorr_col = f"{tcga}_anticorrelated"
        if anticorr_col in source.columns:
            clean[anticorr_col] = _safe_int_series(source, anticorr_col, default=0)
        else:
            clean[anticorr_col] = (clean[f"{tcga}_spearman_rho"].fillna(0) < 0).astype(int)
        support_col = f"{tcga}_support_tcga"
        if support_col in source.columns:
            clean[support_col] = _safe_int_series(source, support_col, default=0)
        else:
            clean[support_col] = (
                (clean[anticorr_col].fillna(0).astype(int) > 0)
                | (clean[f"{tcga}_repression_evidence"].fillna(0).astype(int) > 0)
            ).astype(int)

    anticorr_cols = [f"{tcga}_anticorrelated" for tcga in ("BRCA", "PRAD", "COAD")]
    support_cols = [f"{tcga}_support_tcga" for tcga in ("BRCA", "PRAD", "COAD")]
    repression_cols = [f"{tcga}_repression_evidence" for tcga in ("BRCA", "PRAD", "COAD")]
    rho_cols = [f"{tcga}_spearman_rho" for tcga in ("BRCA", "PRAD", "COAD")]
    clean["tcga_any_anticorrelated"] = clean[anticorr_cols].fillna(0).astype(int).max(axis=1)
    clean["tcga_n_supported_contexts"] = clean[support_cols].fillna(0).astype(int).sum(axis=1)
    clean["tcga_best_repression_evidence"] = clean[repression_cols].fillna(0).astype(int).max(axis=1)
    clean["tcga_mean_spearman_rho"] = clean[rho_cols].mean(axis=1, skipna=True)

    rnaplfold_diagnostics: Dict[str, Any] = {"rnaplfold_join_keys": [], "rnaplfold_feature_rows": 0, "rnaplfold_joined_rows_with_signal": 0}
    if rnaplfold_df is not None:
        clean, rnaplfold_diagnostics = _join_rnaplfold_features(
            clean,
            rnaplfold_df,
            unit_of_analysis=unit_of_analysis,
            accessible_threshold=rnaplfold_accessible_threshold,
        )
    rnaplfold_output_columns = (
        RNAPLFOLD_TRANSCRIPT_COLUMNS
        if unit_of_analysis == "transcript-level miRNA-target candidate"
        else RNAPLFOLD_SITE_COLUMNS
    )
    for column in rnaplfold_output_columns:
        if column not in clean.columns:
            clean[column] = np.nan

    clean = _add_evidence_family_flags(clean)
    clean = clean.loc[:, ~clean.columns.duplicated()].copy()

    metadata = {
        "unit_of_analysis": unit_of_analysis,
        "rnaplfold_output_columns": list(rnaplfold_output_columns),
        "rnaplfold_accessible_threshold": float(rnaplfold_accessible_threshold),
        "source_columns": [str(col) for col in source.columns],
        "rnaplfold": rnaplfold_diagnostics,
    }
    return clean, metadata


def _missingness_fraction(df: pd.DataFrame, columns: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if len(df) == 0:
        return {column: 0.0 for column in columns}
    for column in columns:
        if column not in df.columns:
            out[column] = 1.0
        else:
            out[column] = float(df[column].isna().mean())
    return out


def _numeric_summary(df: pd.DataFrame, columns: Sequence[str]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for column in columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().sum() == 0:
            continue
        summary[column] = {
            "min": float(values.min()),
            "p25": float(values.quantile(0.25)),
            "median": float(values.quantile(0.5)),
            "mean": float(values.mean()),
            "p75": float(values.quantile(0.75)),
            "max": float(values.max()),
        }
    return summary


def build_validation_report(
    clean_df: pd.DataFrame,
    *,
    source_path: Path,
    discovered_inputs: Sequence[Path],
    clean_metadata: Dict[str, Any],
) -> str:
    numeric_columns = [
        "seed_pairing_score",
        "n_seed_sites",
        "rnahybrid_mfe",
        "rnahybrid_strength",
        "targetscan_context_score",
        "targetscan_context_score_percentile",
        "targetscan_aggregate_context_score",
        "clip_max_score",
        "clip_n_experiments",
        "BRCA_spearman_rho",
        "PRAD_spearman_rho",
        "COAD_spearman_rho",
        "tcga_n_supported_contexts",
        "tcga_mean_spearman_rho",
        "rnaplfold_best_seed_unpaired_prob",
        "rnaplfold_mean_seed_unpaired_prob",
        "rnaplfold_best_site_unpaired_prob",
        "rnaplfold_mean_site_unpaired_prob",
    ]
    missingness_groups = {
        "seed_region": [
            "seed_match_type",
            "n_seed_sites",
            "best_seed_site_type",
        ],
        "rnahybrid": [
            "rnahybrid_mfe",
            "rnahybrid_site_start",
            "rnahybrid_site_end",
        ],
        "targetscan": [
            "targetscan_context_score",
            "targetscan_context_score_percentile",
            "targetscan_aggregate_context_score",
        ],
        "rnaplfold": list(clean_metadata.get("rnaplfold_output_columns", RNAPLFOLD_TRANSCRIPT_COLUMNS)),
        "clip": [
            "clip_any_support",
            "clip_max_score",
            "encori_clip_score",
        ],
        "tcga": [
            "BRCA_spearman_rho",
            "PRAD_spearman_rho",
            "COAD_spearman_rho",
            "tcga_n_supported_contexts",
        ],
    }

    support_counts = {
        "seed_support_rows": int(((clean_df["n_seed_sites"].fillna(0) > 0) | (clean_df["seed_match_type"].fillna("") != "")).sum()),
        "rnahybrid_support_rows": int(clean_df["rnahybrid_mfe"].notna().sum()),
        "targetscan_support_rows": int(clean_df["targetscan_context_score"].notna().sum()),
        "rnaplfold_support_rows": int(clean_df["has_rnaplfold_evidence"].fillna(0).astype(int).sum()),
        "clip_support_rows": int(clean_df["clip_any_support"].fillna(0).astype(int).sum()),
        "tcga_support_rows": int(clean_df["has_tcga_evidence"].fillna(0).astype(int).sum()),
    }

    leakage_columns = _find_leakage_columns(clean_df.columns)
    family_presence_summary = {
        column: {
            "rows": int(clean_df[column].fillna(0).astype(int).sum()),
            "percent": float(clean_df[column].fillna(0).astype(int).mean() * 100.0) if len(clean_df) else 0.0,
        }
        for column in FAMILY_FLAG_COLUMNS
    }
    evidence_family_histogram = {
        n: int((clean_df["n_evidence_families_present"].fillna(0).astype(int) == n).sum())
        for n in range(7)
    }

    lines: List[str] = []
    lines.append("miRAssist Clean Evidence Validation Report")
    lines.append("=" * 44)
    lines.append(f"Source path: {source_path}")
    lines.append(f"Discovered input files: {', '.join(str(path) for path in discovered_inputs) if discovered_inputs else 'none'}")
    lines.append(f"Unit of analysis: {clean_metadata.get('unit_of_analysis', 'unknown')}")
    lines.append(
        "RNAplfold handling: "
        + (
            "site-level RNAplfold rows are aggregated to transcript-level summaries using arithmetic means and best-site maxima"
            if clean_metadata.get("unit_of_analysis") == "transcript-level miRNA-target candidate"
            else "site-level RNAplfold rows are preserved as site-level features"
        )
    )
    lines.append(f"Rows: {len(clean_df)}")
    lines.append(f"Unique miRNAs: {clean_df['mirna_name_normalized'].replace('', pd.NA).nunique(dropna=True)}")
    lines.append(f"Unique genes: {clean_df['gene_symbol_normalized'].replace('', pd.NA).nunique(dropna=True)}")
    lines.append(
        "Unique miRNA-gene pairs: "
        + str(
            clean_df[["mirna_name_normalized", "gene_symbol_normalized"]]
            .drop_duplicates()
            .shape[0]
        )
    )
    lines.append("")
    lines.append("Discovered source columns:")
    for column in clean_metadata.get("source_columns", []):
        lines.append(f"- {column}")
    lines.append("")
    lines.append("Missingness by evidence category:")
    for group_name, columns in missingness_groups.items():
        lines.append(f"{group_name}:")
        for column, fraction in _missingness_fraction(clean_df, columns).items():
            lines.append(f"  {column}: {fraction:.3f}")
    lines.append("")
    lines.append("Numeric evidence summaries:")
    for column, stats in _numeric_summary(clean_df, numeric_columns).items():
        lines.append(
            f"- {column}: min={stats['min']:.4g}, p25={stats['p25']:.4g}, median={stats['median']:.4g}, "
            f"mean={stats['mean']:.4g}, p75={stats['p75']:.4g}, max={stats['max']:.4g}"
        )
    lines.append("")
    lines.append("Evidence-family presence summary:")
    for column, stats in family_presence_summary.items():
        lines.append(f"- {column}: {stats['rows']} rows ({stats['percent']:.1f}%)")
    lines.append("")
    lines.append("Evidence-family completeness summary:")
    for n in range(7):
        count = evidence_family_histogram[n]
        percent = float((count / len(clean_df)) * 100.0) if len(clean_df) else 0.0
        lines.append(f"- {n} families present: {count} rows ({percent:.1f}%)")
    lines.append("")
    lines.append("Supported row counts by evidence family:")
    for key, value in support_counts.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("RNAplfold join diagnostics:")
    rnaplfold_info = clean_metadata.get("rnaplfold", {})
    lines.append(f"- join_keys: {rnaplfold_info.get('rnaplfold_join_keys', [])}")
    lines.append(f"- input_level: {rnaplfold_info.get('rnaplfold_input_level', 'unknown')}")
    lines.append(f"- output_level: {rnaplfold_info.get('rnaplfold_output_level', 'unknown')}")
    lines.append(f"- feature_rows: {rnaplfold_info.get('rnaplfold_feature_rows', 0)}")
    lines.append(f"- joined_rows_with_signal: {rnaplfold_info.get('rnaplfold_joined_rows_with_signal', 0)}")
    lines.append(f"- accessible_threshold: {rnaplfold_info.get('rnaplfold_accessible_threshold', clean_metadata.get('rnaplfold_accessible_threshold'))}")
    lines.append(f"- aggregation: {rnaplfold_info.get('rnaplfold_aggregation', 'n/a')}")
    lines.append("")
    lines.append(
        "Leakage-column validation: "
        + ("PASS" if not leakage_columns else f"FAIL ({', '.join(leakage_columns)})")
    )
    lines.append(
        "Confirmation that no miRTarBase evidence columns are present: "
        + ("PASS" if not any("mirtarbase" in str(col).lower() for col in clean_df.columns) else "FAIL")
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    source_path = _resolve_input_path(input_dir, args.input_path)
    discovered_inputs = _discover_input_files(input_dir)
    _log(f"Using source evidence table: {source_path}")
    _log(f"Discovered input files: {[str(path) for path in discovered_inputs]}")

    source_df = _read_table(source_path, limit=args.limit)
    _log(f"Loaded source shape: {source_df.shape}")
    inferred_unit = _derive_unit_of_analysis(source_df)
    _log(f"Final unit of analysis inferred from source schema: {inferred_unit}")
    _log("Discovered source columns:")
    for column in source_df.columns:
        _log(f"  - {column}")

    rnaplfold_df: Optional[pd.DataFrame] = None
    if args.rnaplfold_features:
        rnaplfold_path = Path(args.rnaplfold_features).resolve()
        _log(f"Loading RNAplfold features from: {rnaplfold_path}")
        rnaplfold_df = _read_table(rnaplfold_path, limit=args.limit)
        _log(f"RNAplfold feature shape: {rnaplfold_df.shape}")
        for column in rnaplfold_df.columns:
            _log(f"  RNAplfold column: {column}")

    if rnaplfold_df is not None and inferred_unit == "transcript-level miRNA-target candidate":
        _log(
            "RNAplfold input is treated as site-level and will be aggregated back to transcript-level summaries "
            f"with accessible-site threshold {args.rnaplfold_accessible_threshold:.3f}."
        )

    clean_df, clean_metadata = build_clean_evidence_table(
        source_df,
        rnaplfold_df=rnaplfold_df,
        rnaplfold_accessible_threshold=args.rnaplfold_accessible_threshold,
    )

    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    _write_table(clean_df, output_path)
    report_text = build_validation_report(
        clean_df,
        source_path=source_path,
        discovered_inputs=discovered_inputs,
        clean_metadata=clean_metadata,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")

    _log(f"Wrote clean evidence table to: {output_path}")
    _log(f"Wrote validation report to: {report_path}")


if __name__ == "__main__":
    main()
