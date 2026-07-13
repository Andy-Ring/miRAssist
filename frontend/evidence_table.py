from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd


EVIDENCE_SHORTLIST_COLUMNS = [
    "rank",
    "candidate",
    "mirna_name",
    "gene_symbol",
    "transcript_id",
    "mirassist_xgboost_score",
    "overall_evidence_support_percentile",
    "evidence_family_count",
    "sequence_complementarity_support_percentile",
    "thermodynamic_stability_support_percentile",
    "sequence_conservation_support_percentile",
    "target_site_accessibility_support_percentile",
    "functional_binding_support_percentile",
    "functional_repression_support_percentile",
    "seed_match_type",
    "best_seed_site_type",
    "n_seed_sites",
    "rnahybrid_mfe",
    "targetscan_context_score",
    "clip_max_score",
    "clip_n_experiments",
    "rnaplfold_best_seed_unpaired_prob",
    "tcga_mean_spearman_rho",
]


def normalize_direction_label(direction: Any) -> str:
    value = str(direction or "").strip()
    if value == "gene_to_mirnas":
        return "gene -> miRNAs"
    if value == "mirna_to_targets":
        return "miRNA -> targets"
    return value or "unknown"


def candidate_column_for_direction(direction: Any) -> str:
    return "mirna_name" if str(direction or "") == "gene_to_mirnas" else "gene_symbol"


def build_evidence_shortlist_table(shortlist: list[dict], direction: Any) -> pd.DataFrame:
    if not isinstance(shortlist, list) or not shortlist:
        return pd.DataFrame(columns=EVIDENCE_SHORTLIST_COLUMNS)

    df = pd.DataFrame(shortlist).copy()
    if "rank" not in df.columns:
        df.insert(0, "rank", range(1, len(df) + 1))

    candidate_col = candidate_column_for_direction(direction)
    if candidate_col in df.columns and "candidate" not in df.columns:
        df.insert(1, "candidate", df[candidate_col])
    elif "candidate" not in df.columns:
        df.insert(1, "candidate", "")

    ordered_columns = [col for col in EVIDENCE_SHORTLIST_COLUMNS if col in df.columns]
    ordered_columns.extend([col for col in df.columns if col not in ordered_columns])
    return df.loc[:, ordered_columns]


def evidence_shortlist_csv_bytes(shortlist_df: pd.DataFrame) -> bytes:
    if shortlist_df is None:
        shortlist_df = pd.DataFrame()
    return shortlist_df.to_csv(index=False).encode("utf-8")


def evidence_shortlist_filename(query_id: Any) -> str:
    value = str(query_id or "").strip()
    if not value:
        value = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return f"mirassist_evidence_shortlist_{safe}.csv"
