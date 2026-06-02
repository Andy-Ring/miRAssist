# scripts/export_evidence_for_supabase.py

import pandas as pd
import numpy as np
from pathlib import Path

parquet_path = Path(r"C:\Users\andym\OneDrive - University of Georgia\Documents\miRAssist\data\processed\mirassist_backend_features.parquet")
out_path = Path("mirassist_evidence_pairs_test.csv")

df = pd.read_parquet(parquet_path)
df = df.loc[:, ~df.columns.duplicated()].copy()

# -----------------------------
# Compatibility evidence columns
# -----------------------------

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

# TCGA support flags
for tcga in ["BRCA", "COAD", "PRAD"]:
    rho_col = f"{tcga}_spearman_rho"
    rep_col = f"{tcga}_repression_evidence"
    support_col = f"{tcga}_support_tcga"
    anticorr_col = f"{tcga}_anticorrelated"

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

# Overall support count
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

# Ensure expected normalized names exist
if "mirna_name_norm" not in df.columns:
    df["mirna_name_norm"] = df["mirna_name"].astype(str).str.lower()

if "gene_symbol_norm" not in df.columns:
    df["gene_symbol_norm"] = df["gene_symbol"].astype(str).str.upper()

# Optional: export a small test file first
test = df.head(5000).copy()

# Convert any object/list-like values safely for CSV
for col in test.columns:
    if test[col].dtype == "object":
        test[col] = test[col].apply(
            lambda x: ";".join(map(str, x)) if isinstance(x, (list, tuple, set)) else x
        )

test.to_csv(out_path, index=False)
print("Wrote:", out_path)
print("Shape:", test.shape)
print("Columns:")
for c in test.columns:
    print(c)