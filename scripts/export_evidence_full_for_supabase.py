# scripts/export_evidence_full_for_supabase.py

import pandas as pd
import numpy as np
from pathlib import Path

parquet_path = Path(r"C:\Users\andym\OneDrive - University of Georgia\Documents\miRAssist\data\processed\mirassist_backend_features.parquet")
out_path = Path("mirassist_evidence_pairs_full.csv")

df = pd.read_parquet(parquet_path)
df = df.loc[:, ~df.columns.duplicated()].copy()

# Compatibility evidence columns
df["support_targetscan"] = (
    df["ts_best_contextpp"].notna()
    | df["ts_best_percentile"].notna()
    | df["has_seed_features"].fillna(0).astype(int).eq(1)
).astype(int)

df["support_mirdb"] = (
    df["mirdb_best_score"].notna()
    & df["mirdb_best_score"].gt(0)
).astype(int)

df["support_encori"] = (
    df["n_clip_sites"].fillna(0).gt(0)
    | df["clip_exp_sum"].fillna(0).gt(0)
).astype(int)

df["support_rnahybrid"] = (
    df["has_rnahybrid"].fillna(0).astype(int).eq(1)
    | df["n_rnahybrid_sites"].fillna(0).gt(0)
).astype(int)

for tcga in ["BRCA", "COAD", "PRAD"]:
    rho_col = f"{tcga}_spearman_rho"
    rep_col = f"{tcga}_repression_evidence"
    anticorr_col = f"{tcga}_anticorrelated"
    support_col = f"{tcga}_support_tcga"

    if rho_col in df.columns:
        df[anticorr_col] = df[rho_col].lt(0).fillna(False).astype(int)
    else:
        df[anticorr_col] = 0

    if rep_col in df.columns:
        df[support_col] = (
            df[rep_col].fillna(0).astype(int).eq(1)
            | df[anticorr_col].eq(1)
        ).astype(int)
    else:
        df[support_col] = df[anticorr_col].astype(int)

support_cols = [
    "mirtarbase_pos",
    "support_targetscan",
    "support_mirdb",
    "support_encori",
    "support_rnahybrid",
    "BRCA_support_tcga",
    "COAD_support_tcga",
    "PRAD_support_tcga",
]

for c in support_cols:
    if c not in df.columns:
        df[c] = 0

df["support_count"] = df[support_cols].fillna(0).astype(int).sum(axis=1)

# Convert list-like object columns safely
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].apply(
            lambda x: ";".join(map(str, x)) if isinstance(x, (list, tuple, set)) else x
        )

# Ensure typo is not present
df = df.rename(columns={"PRAD_suppor0t_tcga": "PRAD_support_tcga"})

df.to_csv(out_path, index=False)
print("Wrote:", out_path)
print("Shape:", df.shape)
print("Columns:", len(df.columns))