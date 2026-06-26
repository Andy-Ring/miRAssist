from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/processed/mirassist_clean_evidence.parquet")
OUTPUT_DIR = Path("data/supabase_export")

LEAKAGE_TOKENS: tuple[str, ...] = (
    "mirtarbase",
    "validated",
    "label",
    "manual",
    "weighted",
    "old_score",
    "ground_truth",
    "heldout",
)

FAMILY_CONFIGS: Dict[str, Dict[str, Any]] = {
    "sequence_complementarity": {
        "columns": [
            "seed_match_type",
            "is_8mer",
            "is_7mer_m8",
            "is_7mer_a1",
            "is_6mer",
            "seed_pairing_score",
            "n_seed_sites",
            "best_seed_site_type",
            "has_seed_evidence",
        ],
        "flag_columns": ["has_seed_evidence"],
        "percentile_bases": ["seed_pairing_score", "n_seed_sites"],
    },
    "thermodynamic_stability": {
        "columns": [
            "rnahybrid_mfe",
            "rnahybrid_mfe_best_site",
            "rnahybrid_site_start",
            "rnahybrid_site_end",
            "rnahybrid_seed_mfe",
            "rnahybrid_strength",
            "has_rnahybrid_evidence",
        ],
        "flag_columns": ["has_rnahybrid_evidence"],
        "percentile_bases": [
            "rnahybrid_mfe",
            "rnahybrid_mfe_best_site",
            "rnahybrid_seed_mfe",
            "rnahybrid_strength",
        ],
    },
    "sequence_conservation": {
        "columns": [
            "targetscan_context_score",
            "targetscan_context_score_percentile",
            "targetscan_aggregate_context_score",
            "targetscan_conserved_site",
            "targetscan_pct",
            "targetscan_branch_length_score",
            "has_targetscan_evidence",
        ],
        "flag_columns": ["has_targetscan_evidence"],
        "percentile_bases": [
            "targetscan_context_score",
            "targetscan_context_score_percentile",
            "targetscan_aggregate_context_score",
            "targetscan_pct",
            "targetscan_branch_length_score",
        ],
    },
    "target_site_accessibility": {
        "columns": [
            "rnaplfold_best_seed_unpaired_prob",
            "rnaplfold_mean_seed_unpaired_prob",
            "rnaplfold_best_site_unpaired_prob",
            "rnaplfold_mean_site_unpaired_prob",
            "rnaplfold_best_flank_unpaired_prob",
            "rnaplfold_mean_flank_unpaired_prob",
            "rnaplfold_n_sites_scored",
            "rnaplfold_n_accessible_sites",
            "has_rnaplfold_evidence",
        ],
        "flag_columns": ["has_rnaplfold_evidence"],
        "percentile_bases": [
            "rnaplfold_best_seed_unpaired_prob",
            "rnaplfold_mean_seed_unpaired_prob",
            "rnaplfold_best_site_unpaired_prob",
            "rnaplfold_mean_site_unpaired_prob",
            "rnaplfold_best_flank_unpaired_prob",
            "rnaplfold_mean_flank_unpaired_prob",
            "rnaplfold_n_sites_scored",
            "rnaplfold_n_accessible_sites",
        ],
    },
    "functional_binding": {
        "columns": [
            "clip_any_support",
            "clip_max_score",
            "clip_n_experiments",
            "clip_n_cell_lines",
            "encori_clip_score",
            "has_clip_evidence",
        ],
        "flag_columns": ["clip_any_support", "has_clip_evidence"],
        "percentile_bases": [
            "clip_max_score",
            "clip_n_experiments",
            "clip_n_cell_lines",
            "encori_clip_score",
        ],
    },
    "functional_repression": {
        "columns": [
            "BRCA_spearman_rho",
            "BRCA_repression_evidence",
            "BRCA_anticorrelated",
            "BRCA_support_tcga",
            "PRAD_spearman_rho",
            "PRAD_repression_evidence",
            "PRAD_anticorrelated",
            "PRAD_support_tcga",
            "COAD_spearman_rho",
            "COAD_repression_evidence",
            "COAD_anticorrelated",
            "COAD_support_tcga",
            "tcga_any_anticorrelated",
            "tcga_n_supported_contexts",
            "tcga_best_repression_evidence",
            "tcga_mean_spearman_rho",
            "has_tcga_evidence",
        ],
        "flag_columns": [
            "BRCA_repression_evidence",
            "BRCA_anticorrelated",
            "BRCA_support_tcga",
            "PRAD_repression_evidence",
            "PRAD_anticorrelated",
            "PRAD_support_tcga",
            "COAD_repression_evidence",
            "COAD_anticorrelated",
            "COAD_support_tcga",
            "tcga_any_anticorrelated",
            "has_tcga_evidence",
        ],
        "percentile_bases": [
            "BRCA_spearman_rho",
            "BRCA_repression_evidence",
            "PRAD_spearman_rho",
            "PRAD_repression_evidence",
            "COAD_spearman_rho",
            "COAD_repression_evidence",
            "tcga_n_supported_contexts",
            "tcga_best_repression_evidence",
            "tcga_mean_spearman_rho",
        ],
    },
}

NUMERIC_PERCENTILE_SPECS: Dict[str, Dict[str, Any]] = {
    "seed_pairing_score": {"invert": False},
    "n_seed_sites": {"invert": False},
    "rnahybrid_mfe": {"invert": True},
    "rnahybrid_mfe_best_site": {"invert": True},
    "rnahybrid_seed_mfe": {"invert": True},
    "rnahybrid_strength": {"invert": False},
    "targetscan_context_score": {
        "invert": True,
        "output_column": "targetscan_context_score_support_percentile",
    },
    "targetscan_context_score_percentile": {"invert": False},
    "targetscan_aggregate_context_score": {"invert": True},
    "targetscan_pct": {"invert": False},
    "targetscan_branch_length_score": {"invert": False},
    "rnaplfold_best_seed_unpaired_prob": {"invert": False},
    "rnaplfold_mean_seed_unpaired_prob": {"invert": False},
    "rnaplfold_best_site_unpaired_prob": {"invert": False},
    "rnaplfold_mean_site_unpaired_prob": {"invert": False},
    "rnaplfold_best_flank_unpaired_prob": {"invert": False},
    "rnaplfold_mean_flank_unpaired_prob": {"invert": False},
    "rnaplfold_n_sites_scored": {"invert": False},
    "rnaplfold_n_accessible_sites": {"invert": False},
    "clip_max_score": {"invert": False},
    "clip_n_experiments": {"invert": False},
    "clip_n_cell_lines": {"invert": False},
    "encori_clip_score": {"invert": False},
    "BRCA_spearman_rho": {"invert": True},
    "BRCA_repression_evidence": {"invert": False},
    "PRAD_spearman_rho": {"invert": True},
    "PRAD_repression_evidence": {"invert": False},
    "COAD_spearman_rho": {"invert": True},
    "COAD_repression_evidence": {"invert": False},
    "tcga_n_supported_contexts": {"invert": False},
    "tcga_best_repression_evidence": {"invert": False},
    "tcga_mean_spearman_rho": {"invert": True},
}

CORE_ID_COLUMNS: tuple[str, ...] = (
    "evidence_row_id",
    "mirna_name",
    "mirna_name_normalized",
    "mirna_name_norm",
    "gene_symbol",
    "gene_symbol_normalized",
    "gene_symbol_norm",
    "transcript_id",
    "gene_id",
    "entrez_id",
    "ensembl_gene_id",
    "site_id",
    "chrom",
    "start",
    "end",
    "strand",
)

SCHEMA_FAMILY_OVERRIDES: Dict[str, str] = {
    "targetscan_context_score_support_percentile": "sequence_conservation",
    "support_count": "app_summary",
    "support_targetscan": "sequence_conservation",
    "support_encori": "functional_binding",
    "support_rnahybrid": "thermodynamic_stability",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a six-family miRAssist evidence table for Supabase/app use."
    )
    parser.add_argument("--input", default=str(INPUT_PATH), help="Input clean evidence parquet path.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory.")
    return parser.parse_args()


def _safe_float_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _safe_boolish_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)

    series = df[column]
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).gt(0)

    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "yes", "y", "present"})


def _is_text_available(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    lowered = df[column].fillna("").astype(str).str.strip().str.lower()
    return ~lowered.isin({"", "nan", "none", "nat"})


def _compute_percentile(values: pd.Series, invert: bool) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.nan, index=numeric.index, dtype=float)
    valid = numeric.notna()
    if not valid.any():
        return out
    ranked = numeric.loc[valid].rank(method="average", pct=True, ascending=not invert) * 100.0
    out.loc[valid] = ranked.round(3)
    return out


def _find_leakage_columns(columns: Iterable[str]) -> List[str]:
    blocked: List[str] = []
    for column in columns:
        lower = str(column).lower()
        if any(token in lower for token in LEAKAGE_TOKENS):
            blocked.append(str(column))
    return blocked


def _family_missing_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    return {
        family: [column for column in cfg["columns"] if column not in df.columns]
        for family, cfg in FAMILY_CONFIGS.items()
    }


def _example_values(series: pd.Series, limit: int = 3) -> str:
    values: List[str] = []
    for value in series.dropna().head(50):
        if isinstance(value, (list, tuple, set, dict)):
            text = json.dumps(value, sort_keys=True)
        else:
            text = str(value)
        if text not in values:
            values.append(text)
        if len(values) >= limit:
            break
    return " | ".join(values)


def _mean_percentile(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    present = [column for column in columns if column in df.columns]
    if not present:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return df[present].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True).round(3)


def _count_family_evidence(df: pd.DataFrame, family: str, columns: Sequence[str], flag_columns: Sequence[str]) -> pd.Series:
    count = pd.Series(0, index=df.index, dtype=int)
    for column in columns:
        if column not in df.columns:
            continue
        if column in flag_columns:
            count = count.add(_safe_boolish_series(df, column).astype(int), fill_value=0)
            continue

        series = df[column]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            if column.endswith("_evidence") or column.endswith("_support") or column.endswith("_anticorrelated"):
                count = count.add(numeric.fillna(0).gt(0).astype(int), fill_value=0)
            else:
                count = count.add(numeric.notna().astype(int), fill_value=0)
        else:
            count = count.add(_is_text_available(df, column).astype(int), fill_value=0)
    return count.astype(int)


def _build_schema(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    family_columns: Dict[str, str] = {}
    for family, cfg in FAMILY_CONFIGS.items():
        for column in cfg["columns"]:
            family_columns[column] = family
        for base_name in cfg["percentile_bases"]:
            output_column = NUMERIC_PERCENTILE_SPECS.get(base_name, {}).get(
                "output_column",
                f"{base_name}_percentile",
            )
            family_columns[output_column] = family
    family_columns.update(SCHEMA_FAMILY_OVERRIDES)

    for column in df.columns:
        lower = str(column).lower()
        if column in CORE_ID_COLUMNS or lower.endswith("_normalized"):
            role = "id/entity"
        elif lower.endswith("_support_percentile") or (
            lower.endswith("_percentile") and not lower.startswith("overall_")
        ):
            role = "percentile"
        elif lower.endswith("_available") or lower.endswith("_evidence_count"):
            role = "family_summary"
        elif lower in {
            "overall_evidence_support_percentile",
            "evidence_family_count",
            "evidence_family_summary_json",
            "support_count",
        }:
            role = "app_summary"
        else:
            role = "raw_feature"

        rows.append(
            {
                "column_name": column,
                "dtype": str(df[column].dtype),
                "non_null_count": int(df[column].notna().sum()),
                "null_fraction": float(df[column].isna().mean()) if len(df) else 0.0,
                "example_values": _example_values(df[column]),
                "evidence_family": family_columns.get(column, ""),
                "role": role,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_out = output_dir / "mirassist_supabase_evidence.parquet"
    csv_out = output_dir / "mirassist_supabase_evidence.csv.gz"
    schema_out = output_dir / "mirassist_supabase_evidence_schema.csv"
    report_out = output_dir / "mirassist_supabase_evidence_export_report.txt"

    if not input_path.exists():
        raise FileNotFoundError(f"Input clean evidence table not found: {input_path}")

    df = pd.read_parquet(input_path)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    print(f"Input path: {input_path}")
    print(f"Input shape: {df.shape}")
    print(f"Input columns ({len(df.columns)}):")
    for column in df.columns:
        print(f"  - {column}")

    missing_by_family = _family_missing_columns(df)
    leakage_columns = _find_leakage_columns(df.columns)
    export_df = df.drop(columns=leakage_columns, errors="ignore").copy()

    added_percentile_columns: List[str] = []
    for base_name, spec in NUMERIC_PERCENTILE_SPECS.items():
        if base_name not in export_df.columns:
            continue
        numeric = pd.to_numeric(export_df[base_name], errors="coerce")
        if numeric.notna().sum() == 0:
            continue
        output_column = spec.get("output_column", f"{base_name}_percentile")
        if output_column in export_df.columns and output_column != "targetscan_context_score_support_percentile":
            continue
        export_df[output_column] = _compute_percentile(numeric, invert=bool(spec.get("invert", False)))
        added_percentile_columns.append(output_column)

    # Mean is used for family support because it reflects consistent support across
    # the available evidence fields rather than letting one extreme metric dominate.
    family_summary_columns: List[str] = []
    family_coverage_summary: Dict[str, Dict[str, Any]] = {}
    family_json_columns: Dict[str, List[str]] = {}
    for family, cfg in FAMILY_CONFIGS.items():
        available_column = f"{family}_available"
        support_column = f"{family}_support_percentile"
        count_column = f"{family}_evidence_count"

        percentile_columns: List[str] = []
        for base_name in cfg["percentile_bases"]:
            if base_name == "targetscan_context_score" and "targetscan_context_score_support_percentile" in export_df.columns:
                percentile_columns.append("targetscan_context_score_support_percentile")
            candidate = NUMERIC_PERCENTILE_SPECS.get(base_name, {}).get(
                "output_column",
                f"{base_name}_percentile",
            )
            if candidate in export_df.columns and candidate not in percentile_columns:
                percentile_columns.append(candidate)
            if base_name in export_df.columns and str(base_name).endswith("_percentile") and base_name not in percentile_columns:
                percentile_columns.append(base_name)

        evidence_count = _count_family_evidence(
            export_df,
            family=family,
            columns=cfg["columns"],
            flag_columns=cfg["flag_columns"],
        )
        support_percentile = _mean_percentile(export_df, percentile_columns)
        available_from_flags = pd.Series(False, index=export_df.index, dtype=bool)
        for flag_column in cfg["flag_columns"]:
            available_from_flags = available_from_flags | _safe_boolish_series(export_df, flag_column)
        export_df[available_column] = (available_from_flags | evidence_count.gt(0)).astype(bool)
        export_df[support_column] = support_percentile.where(export_df[available_column])
        export_df[count_column] = evidence_count.astype(int)

        family_summary_columns.extend([available_column, support_column, count_column])
        family_json_columns[family] = [available_column, support_column, count_column]
        family_coverage_summary[family] = {
            "available_rows": int(export_df[available_column].sum()),
            "available_percent": float(export_df[available_column].mean() * 100.0) if len(export_df) else 0.0,
            "mean_support_percentile": float(export_df[support_column].mean(skipna=True))
            if export_df[support_column].notna().any()
            else float("nan"),
        }

    family_support_columns = [f"{family}_support_percentile" for family in FAMILY_CONFIGS]
    family_available_columns = [f"{family}_available" for family in FAMILY_CONFIGS]
    export_df["overall_evidence_support_percentile"] = _mean_percentile(export_df, family_support_columns)
    export_df["evidence_family_count"] = export_df[family_available_columns].fillna(False).astype(bool).sum(axis=1).astype(int)
    export_df["support_count"] = export_df["evidence_family_count"].astype(int)

    if "has_targetscan_evidence" in export_df.columns:
        export_df["support_targetscan"] = _safe_boolish_series(export_df, "has_targetscan_evidence").astype(int)
    if "has_clip_evidence" in export_df.columns:
        export_df["support_encori"] = _safe_boolish_series(export_df, "has_clip_evidence").astype(int)
    if "has_rnahybrid_evidence" in export_df.columns:
        export_df["support_rnahybrid"] = _safe_boolish_series(export_df, "has_rnahybrid_evidence").astype(int)

    def build_family_json(row: pd.Series) -> str:
        payload: Dict[str, Dict[str, Any]] = {}
        for family, columns in family_json_columns.items():
            payload[family] = {
                "available": bool(row[columns[0]]),
                "support_percentile": None if pd.isna(row[columns[1]]) else float(row[columns[1]]),
                "evidence_count": int(row[columns[2]]),
            }
        return json.dumps(payload, sort_keys=True)

    export_df["evidence_family_summary_json"] = export_df.apply(build_family_json, axis=1)
    family_summary_columns.extend(
        [
            "overall_evidence_support_percentile",
            "evidence_family_count",
            "evidence_family_summary_json",
            "support_count",
        ]
    )

    schema_df = _build_schema(export_df)
    export_df.to_parquet(parquet_out, index=False)
    export_df.to_csv(csv_out, index=False, compression="gzip")
    schema_df.to_csv(schema_out, index=False)

    report_lines: List[str] = [
        "miRAssist Supabase evidence export report",
        "=" * 40,
        f"input_path: {input_path}",
        f"parquet_output: {parquet_out}",
        f"csv_output: {csv_out}",
        f"schema_output: {schema_out}",
        f"input_shape: {df.shape}",
        f"output_shape: {export_df.shape}",
        "",
        "dropped_leakage_columns:",
    ]
    if leakage_columns:
        report_lines.extend(f"- {column}" for column in leakage_columns)
    else:
        report_lines.append("- none")

    report_lines.extend(["", "missing_expected_family_columns:"])
    for family, missing_columns in missing_by_family.items():
        if missing_columns:
            report_lines.append(f"- {family}: {', '.join(missing_columns)}")
        else:
            report_lines.append(f"- {family}: none")

    report_lines.extend(["", "added_percentile_columns:"])
    report_lines.extend(f"- {column}" for column in added_percentile_columns or ["none"])
    report_lines.extend(["", "added_family_summary_columns:"])
    report_lines.extend(f"- {column}" for column in family_summary_columns)
    report_lines.extend(["", "top_level_evidence_coverage_summary:"])
    for family, stats in family_coverage_summary.items():
        mean_support = (
            "nan"
            if np.isnan(stats["mean_support_percentile"])
            else f"{stats['mean_support_percentile']:.2f}"
        )
        report_lines.append(
            f"- {family}: rows={stats['available_rows']}, "
            f"percent={stats['available_percent']:.2f}, mean_support_percentile={mean_support}"
        )
    report_out.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Output shape: {export_df.shape}")
    print("Missing expected columns by family:")
    for family, missing_columns in missing_by_family.items():
        if missing_columns:
            print(f"  - {family}: {', '.join(missing_columns)}")
        else:
            print(f"  - {family}: none")
    print("Family coverage summary:")
    for family, stats in family_coverage_summary.items():
        print(
            f"  - {family}: available_rows={stats['available_rows']}, "
            f"available_percent={stats['available_percent']:.2f}, "
            f"mean_support_percentile={stats['mean_support_percentile']}"
        )
    print(f"Wrote parquet: {parquet_out}")
    print(f"Wrote csv.gz: {csv_out}")
    print(f"Wrote schema: {schema_out}")
    print(f"Wrote report: {report_out}")


if __name__ == "__main__":
    main()
