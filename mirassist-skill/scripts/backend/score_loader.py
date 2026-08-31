"""Canonical, model-agnostic miRAssist score loading.

The production contract prefers the versioned model score and retains explicit
compatibility with legacy evidence snapshots.  It never blends different model
outputs row-by-row or treats a prioritization score as a biological probability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import warnings

import numpy as np
import pandas as pd


MODEL_SCORE_COLUMN = "mirassist_model_score"
MODEL_VERSION_COLUMN = "mirassist_model_version"
LEGACY_SCORE_COLUMN = "mirassist_xgboost_score"
CANONICAL_SCORE_COLUMN = "mirassist_score"
GLOBAL_RANK_COLUMN = "mirassist_score_rank_within_mirna"
SCORE_PERCENTILE_COLUMN = "mirassist_score_percentile_within_mirna"
FILTERED_RANK_COLUMN = "mirassist_filtered_rank"

APPROVED_MODEL_VERSION = "mirassist_rf_variant_a_v1"
SCHEMA_VERSION = "mirassist_evidence_variant_a_rf_v1"
CANDIDATE_UNIVERSE_VERSION = "variant_a"
LEGACY_MODEL_VERSION = "legacy_xgboost_unspecified"
SCORE_SEMANTICS = (
    "raw uncalibrated random-forest positive-class vote fraction used solely as a "
    "relative prioritization score; no biological probability interpretation"
)
LEGACY_SCORE_SEMANTICS = "legacy XGBoost relative prioritization score"


@dataclass(frozen=True)
class LoadedScoreTable:
    frame: pd.DataFrame
    metadata: Mapping[str, Any]


def _numeric_score(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce").astype(float)
    finite = values.dropna()
    if not np.isfinite(finite.to_numpy()).all():
        raise ValueError(f"{column} contains non-finite values")
    return values


def load_compatible_scores(
    frame: pd.DataFrame,
    *,
    source_name: str = "in-memory evidence table",
    conflict_tolerance: float = 1e-12,
    require_complete: bool = True,
) -> LoadedScoreTable:
    """Resolve a versioned or legacy persisted score into ``mirassist_score``.

    Populated versioned and legacy fields may coexist only if every overlapping
    value agrees within tolerance. Mixed partial coverage is rejected because it
    would silently combine scores from different models in one ranked list.
    """
    has_model = MODEL_SCORE_COLUMN in frame.columns
    has_legacy = LEGACY_SCORE_COLUMN in frame.columns
    if not has_model and not has_legacy:
        raise ValueError("No supported miRAssist score column is present")

    model = (
        _numeric_score(frame, MODEL_SCORE_COLUMN)
        if has_model
        else pd.Series(np.nan, index=frame.index, dtype=float)
    )
    legacy = (
        _numeric_score(frame, LEGACY_SCORE_COLUMN)
        if has_legacy
        else pd.Series(np.nan, index=frame.index, dtype=float)
    )
    model_present = model.notna()
    legacy_present = legacy.notna()
    warning_messages: list[str] = []

    if model_present.any():
        if MODEL_VERSION_COLUMN not in frame.columns:
            raise ValueError(f"{MODEL_SCORE_COLUMN} requires {MODEL_VERSION_COLUMN}")
        versions = frame[MODEL_VERSION_COLUMN].astype("string")
        populated_versions = versions.loc[model_present]
        if populated_versions.isna().any() or populated_versions.str.strip().eq("").any():
            raise ValueError("Populated model scores require nonmissing model versions")

        overlap = model_present & legacy_present
        maximum_difference = None
        if overlap.any():
            maximum_difference = float(
                np.max(np.abs(model.loc[overlap].to_numpy() - legacy.loc[overlap].to_numpy()))
            )
            if maximum_difference > conflict_tolerance:
                raise ValueError(
                    f"Conflicting new and legacy score columns in {source_name}; "
                    f"maximum overlap difference={maximum_difference:.6g}"
                )
            message = (
                "Both persisted score columns are populated and agree within tolerance; "
                "mirassist_model_score takes precedence."
            )
            warnings.warn(message, UserWarning, stacklevel=2)
            warning_messages.append(message)
        if (legacy_present & ~model_present).any():
            raise ValueError("Mixed model/legacy row coverage is not allowed")

        canonical = model
        model_versions = sorted(populated_versions.dropna().unique().tolist())
        source_column = MODEL_SCORE_COLUMN
        contract = "versioned_model_agnostic"
        semantics = SCORE_SEMANTICS
    elif legacy_present.any():
        message = (
            f"Using legacy {LEGACY_SCORE_COLUMN}; assigned model version "
            f"{LEGACY_MODEL_VERSION}."
        )
        warnings.warn(message, UserWarning, stacklevel=2)
        warning_messages.append(message)
        canonical = legacy
        model_versions = [LEGACY_MODEL_VERSION]
        source_column = LEGACY_SCORE_COLUMN
        contract = "legacy_xgboost_fallback"
        semantics = LEGACY_SCORE_SEMANTICS
        maximum_difference = None
    else:
        raise ValueError("Supported score columns are present but contain no scores")

    if require_complete and canonical.isna().any():
        raise ValueError(f"Resolved score column has {int(canonical.isna().sum())} missing values")

    output = frame.copy()
    output[CANONICAL_SCORE_COLUMN] = canonical
    metadata = {
        "source_name": source_name,
        "score_contract": contract,
        "score_source_column": source_column,
        "canonical_score_column": CANONICAL_SCORE_COLUMN,
        "model_versions": model_versions,
        "model_version": model_versions[0] if len(model_versions) == 1 else None,
        "new_score_precedence": True,
        "conflict_tolerance": conflict_tolerance,
        "maximum_overlap_difference": maximum_difference,
        "warnings": warning_messages,
        "score_semantics": semantics,
        "candidate_universe_version": (
            CANDIDATE_UNIVERSE_VERSION
            if source_column == MODEL_SCORE_COLUMN
            else "legacy_unspecified"
        ),
        "schema_version": SCHEMA_VERSION if source_column == MODEL_SCORE_COLUMN else "legacy_126_column",
    }
    output.attrs["mirassist_score_metadata"] = metadata
    return LoadedScoreTable(frame=output, metadata=metadata)


def score_metadata(frame: pd.DataFrame) -> Mapping[str, Any]:
    """Return score metadata already attached to a loaded frame, if available."""
    value = frame.attrs.get("mirassist_score_metadata", {})
    return dict(value) if isinstance(value, Mapping) else {}
