#!/usr/bin/env python3
"""Verify and optionally install the frozen Variant A/RF v1 production table.

This command never scores candidates or loads the model for inference. It checks
the already-approved release and uses a byte-for-byte copy for local cutover.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "outputs/sequence_defined_candidates/variant_a_rf_v1_scored_release"
DEFAULT_SOURCE = RELEASE / "tables/variant_a_rf_v1_scored_evidence_table.parquet"
DEFAULT_INSTALLED = ROOT / "data/processed/mirassist_evidence_variant_a_rf_v1.parquet"
PREMODEL = ROOT / "outputs/sequence_defined_candidates/variant_a_label_alignment_and_splits/tables/variant_a_final_premodel_evidence_table.parquet"
MODEL = RELEASE / "model/mirassist_rf_variant_a_v1.joblib"
REPRODUCTION = RELEASE / "reproducibility/approved_train_test_reproduction.json"

EXPECTED_TABLE_SHA256 = "2fc1b25af55c22c7e44e4587ac586942c0b5f3eb47afe0e36ccf5cab0512a9ee"
EXPECTED_MODEL_SHA256 = "c765ff90ef841d05e976f8948318cd644f60bbd94c1e3466eca197f35dceeb94"
EXPECTED_PREMODEL_SHA256 = "47ac51e4b4f5837f7791426bad798034ef6d0fdc2f72192f534881f2a77b3030"
EXPECTED_ROWS = 280_917
EXPECTED_COLUMNS = 130
EXPECTED_KNOWN_POSITIVES = 2_583
EXPECTED_MODEL_VERSION = "mirassist_rf_variant_a_v1"
EXPECTED_AUROC = 0.8462379826608851
EXPECTED_PR_AUC = 0.13475229166015865
APPENDED_COLUMNS = [
    "mirassist_model_score",
    "mirassist_model_version",
    "mirassist_score_rank_within_mirna",
    "mirassist_score_percentile_within_mirna",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_release_checksums() -> int:
    checked = 0
    for raw in (RELEASE / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        expected, relative = raw.split(maxsplit=1)
        path = RELEASE / relative.strip()
        if sha256(path) != expected:
            raise AssertionError(f"Frozen release checksum mismatch: {path}")
        checked += 1
    return checked


def _bool(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)


def verify_source(source: Path) -> dict[str, Any]:
    source = source.resolve()
    if sha256(source) != EXPECTED_TABLE_SHA256:
        raise AssertionError(f"Approved table checksum mismatch: {source}")
    if sha256(MODEL) != EXPECTED_MODEL_SHA256:
        raise AssertionError("Approved model checksum mismatch")
    if sha256(PREMODEL) != EXPECTED_PREMODEL_SHA256:
        raise AssertionError("Frozen 126-column premodel checksum mismatch")

    checksum_entries = verify_release_checksums()
    scored_parquet = pq.ParquetFile(source)
    original_parquet = pq.ParquetFile(PREMODEL)
    if (scored_parquet.metadata.num_rows, scored_parquet.metadata.num_columns) != (
        EXPECTED_ROWS,
        EXPECTED_COLUMNS,
    ):
        raise AssertionError("Approved scored shape mismatch")
    original_columns = original_parquet.schema_arrow.names
    if len(original_columns) != 126:
        raise AssertionError("Frozen original schema is not 126 columns")
    if scored_parquet.schema_arrow.names[:126] != original_columns:
        raise AssertionError("Original 126-column order/schema was not preserved")
    if scored_parquet.schema_arrow.names[-4:] != APPENDED_COLUMNS:
        raise AssertionError("Versioned score/rank fields do not match the approved schema")

    rows_compared = 0
    for original_batch, scored_batch in zip(
        original_parquet.iter_batches(batch_size=10_000, columns=original_columns),
        scored_parquet.iter_batches(batch_size=10_000, columns=original_columns),
        strict=True,
    ):
        if not original_batch.equals(scored_batch):
            raise AssertionError("An original evidence value changed in the scored release")
        rows_compared += original_batch.num_rows
    if rows_compared != EXPECTED_ROWS:
        raise AssertionError("Original-column comparison did not cover every row")

    verification_columns = [
        "evidence_row_id",
        "mirna_name",
        "mirna_name_normalized",
        "gene_symbol",
        "gene_symbol_normalized",
        "transcript_id",
        "has_seed_evidence",
        "has_targetscan_evidence",
        "has_clip_evidence",
        "tcga_any_anticorrelated",
        "overall_evidence_support_percentile",
        "evidence_family_count",
        "mirtarbase_known_positive",
        "known_validated_mti",
        "mirassist_xgboost_score",
        *APPENDED_COLUMNS,
    ]
    frame = pd.read_parquet(source, columns=verification_columns)
    keys = ["mirna_name_normalized", "gene_symbol_normalized", "transcript_id"]
    if frame.duplicated(keys).any():
        raise AssertionError("Duplicate biological keys are present")
    scores = pd.to_numeric(frame["mirassist_model_score"], errors="coerce")
    if scores.isna().any() or not np.isfinite(scores.to_numpy()).all():
        raise AssertionError("RF score coverage is missing or non-finite")
    if not frame["mirassist_model_version"].eq(EXPECTED_MODEL_VERSION).all():
        raise AssertionError("Unexpected model version")
    if frame["mirassist_xgboost_score"].notna().any():
        raise AssertionError("RF values were written into the legacy XGBoost field")
    if int(_bool(frame["mirtarbase_known_positive"]).sum()) != EXPECTED_KNOWN_POSITIVES:
        raise AssertionError("Known-positive annotation count changed")
    if not _bool(frame["mirtarbase_known_positive"]).equals(
        _bool(frame["known_validated_mti"])
    ):
        raise AssertionError("Retained known-positive label fields disagree")

    external_route = (
        _bool(frame["has_targetscan_evidence"])
        | _bool(frame["has_clip_evidence"])
        | _bool(frame["tcga_any_anticorrelated"])
    )
    if not _bool(frame["has_seed_evidence"]).all() or not external_route.all():
        raise AssertionError("A row violates the approved Variant A eligibility definition")

    ranking_keys = pd.read_parquet(
        RELEASE / "tables/variant_a_rf_v1_keyed_scores.parquet",
        columns=["evidence_row_id", "mirna_identifier", "gene_identifier"],
    )
    ranked_input = frame.merge(
        ranking_keys,
        on="evidence_row_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if ranked_input[["mirna_identifier", "gene_identifier"]].isna().any().any():
        raise AssertionError("Approved ranking identifiers are incomplete")
    ordered = ranked_input.sort_values(
        [
            "mirna_identifier",
            "mirassist_model_score",
            "overall_evidence_support_percentile",
            "evidence_family_count",
            "gene_identifier",
            "transcript_id",
            "evidence_row_id",
        ],
        ascending=[True, False, False, False, True, True, True],
        kind="mergesort",
        na_position="last",
    ).copy()
    reproduced_rank = (
        ordered.groupby("mirna_identifier", observed=True, sort=False).cumcount() + 1
    )
    if not reproduced_rank.to_numpy().astype(np.int64).tolist() == pd.to_numeric(
        ordered["mirassist_score_rank_within_mirna"], errors="raise"
    ).to_numpy().astype(np.int64).tolist():
        raise AssertionError("Stored global within-miRNA ranks do not reproduce")
    counts = ordered.groupby("mirna_identifier", observed=True)["evidence_row_id"].transform("size")
    reproduced_percentile = np.where(
        counts.gt(1),
        1.0 - (reproduced_rank.astype(float) - 1.0) / (counts.astype(float) - 1.0),
        1.0,
    )
    maximum_percentile_difference = float(
        np.max(
            np.abs(
                reproduced_percentile
                - pd.to_numeric(
                    ordered["mirassist_score_percentile_within_mirna"], errors="raise"
                ).to_numpy()
            )
        )
    )
    if maximum_percentile_difference > 1e-15:
        raise AssertionError("Stored within-miRNA score percentiles do not reproduce")

    reproduction = json.loads(REPRODUCTION.read_text(encoding="utf-8"))
    if not np.isclose(reproduction["heldout_auroc"], EXPECTED_AUROC, atol=1e-15, rtol=0):
        raise AssertionError("Held-out AUROC mismatch")
    if not np.isclose(reproduction["heldout_pr_auc"], EXPECTED_PR_AUC, atol=1e-15, rtol=0):
        raise AssertionError("Held-out PR-AUC mismatch")

    sample_ids = [0, int(frame.iloc[len(frame) // 2]["evidence_row_id"]), int(frame.iloc[-1]["evidence_row_id"])]
    samples = frame.loc[frame["evidence_row_id"].isin(sample_ids), [
        "evidence_row_id",
        "mirna_name",
        "gene_symbol",
        "transcript_id",
        "mirassist_model_score",
        "mirassist_score_rank_within_mirna",
    ]].to_dict(orient="records")
    return {
        "status": "passed",
        "source": str(source),
        "source_sha256": EXPECTED_TABLE_SHA256,
        "model": str(MODEL.resolve()),
        "model_sha256": EXPECTED_MODEL_SHA256,
        "rows": EXPECTED_ROWS,
        "columns": EXPECTED_COLUMNS,
        "original_columns_exact": True,
        "unique_biological_keys": True,
        "finite_complete_scores": True,
        "model_version": EXPECTED_MODEL_VERSION,
        "known_positive_count": EXPECTED_KNOWN_POSITIVES,
        "variant_a_eligibility_all_rows": True,
        "variant_d_rows": 0,
        "legacy_xgboost_field_populated_rows": 0,
        "ranks_exact": True,
        "maximum_percentile_difference": maximum_percentile_difference,
        "heldout_auroc": reproduction["heldout_auroc"],
        "heldout_pr_auc": reproduction["heldout_pr_auc"],
        "release_checksum_entries_verified": checksum_entries,
        "fixed_samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--installed", type=Path, default=DEFAULT_INSTALLED)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    source_checksum_before = sha256(args.source)
    report = verify_source(args.source)
    if args.install:
        args.installed.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.source, args.installed)
        if sha256(args.installed) != EXPECTED_TABLE_SHA256:
            raise AssertionError("Installed production table is not byte-identical to the approved source")
        report["installed"] = str(args.installed.resolve())
        report["installed_sha256"] = sha256(args.installed)
    if sha256(args.source) != source_checksum_before:
        raise AssertionError("Approved scored-release source changed during validation")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
