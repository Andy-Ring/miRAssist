from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd


FeatureTransform = Callable[[pd.DataFrame], pd.Series]


def percentile_label(percentile: float | None) -> str:
    if percentile is None or pd.isna(percentile):
        return "not available"
    if percentile >= 95:
        return "exceptional"
    if percentile >= 90:
        return "very high"
    if percentile >= 75:
        return "high"
    if percentile >= 50:
        return "above average"
    if percentile >= 25:
        return "typical"
    return "low"


def _numeric_column(df: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.full(len(df), default, dtype=float), index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def _negative_to_strength(series: pd.Series) -> pd.Series:
    return pd.to_numeric(-series, errors="coerce")


FEATURE_SPECS: Dict[str, Dict[str, Any]] = {
    "support_count": {"transform": lambda df: _numeric_column(df, "support_count")},
    "mirdb_best_score": {"transform": lambda df: _numeric_column(df, "mirdb_best_score")},
    "mirdb_mean_score": {"transform": lambda df: _numeric_column(df, "mirdb_mean_score")},
    "ts_context_strength": {
        "transform": lambda df: (
            _numeric_column(df, "ts_context_strength")
            if "ts_context_strength" in df.columns
            else _negative_to_strength(_numeric_column(df, "ts_best_contextpp"))
        )
    },
    "ts_best_percentile": {"transform": lambda df: _numeric_column(df, "ts_best_percentile")},
    "n_clip_sites": {"transform": lambda df: _numeric_column(df, "n_clip_sites")},
    "clip_exp_sum": {"transform": lambda df: _numeric_column(df, "clip_exp_sum")},
    "clip_exp_max": {"transform": lambda df: _numeric_column(df, "clip_exp_max")},
    "n_total_sites": {
        "transform": lambda df: (
            _numeric_column(df, "n_total_sites")
            if "n_total_sites" in df.columns
            else (
                _numeric_column(df, "n_sites_6mer", 0)
                + _numeric_column(df, "n_sites_7mer_a1", 0)
                + _numeric_column(df, "n_sites_7mer_m8", 0)
                + _numeric_column(df, "n_sites_8mer", 0)
            )
        )
    },
    "site_density_per_kb": {"transform": lambda df: _numeric_column(df, "site_density_per_kb")},
    "n_sites_6mer": {"transform": lambda df: _numeric_column(df, "n_sites_6mer")},
    "n_sites_7mer_a1": {"transform": lambda df: _numeric_column(df, "n_sites_7mer_a1")},
    "n_sites_7mer_m8": {"transform": lambda df: _numeric_column(df, "n_sites_7mer_m8")},
    "n_sites_8mer": {"transform": lambda df: _numeric_column(df, "n_sites_8mer")},
    "best_local_au": {"transform": lambda df: _numeric_column(df, "best_local_au")},
    "n_rnahybrid_sites": {"transform": lambda df: _numeric_column(df, "n_rnahybrid_sites")},
    "mfe_strength": {
        "transform": lambda df: (
            _numeric_column(df, "mfe_strength")
            if "mfe_strength" in df.columns
            else _negative_to_strength(_numeric_column(df, "best_mfe"))
        )
    },
    "mean_top3_mfe_strength": {
        "transform": lambda df: (
            _numeric_column(df, "mean_top3_mfe_strength")
            if "mean_top3_mfe_strength" in df.columns
            else _negative_to_strength(_numeric_column(df, "mean_top3_mfe"))
        )
    },
    "n_sites_mfe_lt_-20": {"transform": lambda df: _numeric_column(df, "n_sites_mfe_lt_-20")},
    "n_sites_mfe_lt_-25": {"transform": lambda df: _numeric_column(df, "n_sites_mfe_lt_-25")},
    "best_local_au_by_mfe": {"transform": lambda df: _numeric_column(df, "best_local_au_by_mfe")},
    "BRCA_anticorrelation_strength": {
        "transform": lambda df: np.maximum(0.0, -_numeric_column(df, "BRCA_spearman_rho"))
    },
    "COAD_anticorrelation_strength": {
        "transform": lambda df: np.maximum(0.0, -_numeric_column(df, "COAD_spearman_rho"))
    },
    "PRAD_anticorrelation_strength": {
        "transform": lambda df: np.maximum(0.0, -_numeric_column(df, "PRAD_spearman_rho"))
    },
}

_FEATURE_CACHE_KEY: Any = None
_FEATURE_CACHE: Dict[str, Dict[str, Any]] | None = None


def _ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for feature_name, spec in FEATURE_SPECS.items():
        try:
            out[feature_name] = spec["transform"](out)
        except Exception:
            out[feature_name] = pd.Series(np.full(len(out), np.nan), index=out.index)
    return out


def load_feature_reference_distribution(ev: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    global _FEATURE_CACHE_KEY, _FEATURE_CACHE

    key = (id(ev), len(ev), tuple(str(col) for col in ev.columns))
    if _FEATURE_CACHE_KEY == key and _FEATURE_CACHE is not None:
        return _FEATURE_CACHE

    ev_features = _ensure_feature_columns(ev)
    stats: Dict[str, Dict[str, Any]] = {}
    for feature_name in FEATURE_SPECS:
        values = pd.to_numeric(ev_features[feature_name], errors="coerce")
        finite_values = values[np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))]
        sorted_values = np.sort(finite_values.to_numpy(dtype=float, copy=True)) if len(finite_values) else np.array([])
        stats[feature_name] = {"sorted_values": sorted_values, "count": int(sorted_values.size)}

    _FEATURE_CACHE_KEY = key
    _FEATURE_CACHE = stats
    return stats


def _percentile_from_sorted(sorted_values: np.ndarray, value: Any) -> float | None:
    try:
        numeric_value = float(value)
    except Exception:
        return None

    if not np.isfinite(numeric_value) or sorted_values.size == 0:
        return None

    rank = np.searchsorted(sorted_values, numeric_value, side="right")
    return float(rank / sorted_values.size * 100.0)


def annotate_feature_percentiles(shortlist: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    if shortlist is None:
        return shortlist

    annotated = _ensure_feature_columns(shortlist)
    reference = load_feature_reference_distribution(ev)

    for feature_name in FEATURE_SPECS:
        percentile_col = f"{feature_name}_percentile"
        label_col = f"{feature_name}_label"
        sorted_values = reference[feature_name]["sorted_values"]

        annotated[percentile_col] = annotated[feature_name].apply(
            lambda value: _percentile_from_sorted(sorted_values, value)
        )
        annotated[label_col] = annotated[percentile_col].apply(percentile_label)

    return annotated
