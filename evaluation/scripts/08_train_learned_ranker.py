from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.utils import average_precision_manual, json_dump, roc_auc_score_manual  # noqa: E402


DEFAULT_KS = (1, 5, 10, 25, 50, 100)
FEATURE_SET_CHOICES = ("raw", "components", "all")
COMPONENT_FEATURE_COLUMNS = {
    "retrieval_support",
    "retrieval_ts_contrib",
    "retrieval_clip_contrib",
    "retrieval_mirdb_contrib",
    "retrieval_tcga_contrib",
    "retrieval_seed_contrib",
    "retrieval_rnahybrid_contrib",
    "retrieval_local_au_contrib",
    "retrieval_structure_contrib",
}
LEAKAGE_PATTERNS = (
    "mirtarbase",
    "label_mirtarbase",
    "is_positive",
    "positive",
    "validated",
    "functional",
    "heldout",
    "eval_row_id",
)
EXPLICIT_DROP_COLUMNS = {
    "query_id",
    "mirna",
    "mirna_name",
    "mirna_name_normalized",
    "gene_symbol",
    "gene_symbol_normalized",
    "gene",
    "transcript_id",
    "rank",
    "original_rank",
    "ablation_rank",
    "matched_query_tokens",
    "pathway_selected_names",
}
SEED_CLASS_RANKS = {
    "8mer": 4.0,
    "7mer-m8": 3.0,
    "7mer_a1": 2.0,
    "7mer-a1": 2.0,
    "6mer": 1.0,
}
IMPORTANT_MISSINGNESS_FEATURES = (
    "retrieval_score",
    "retrieval_support",
    "retrieval_ts_contrib",
    "retrieval_clip_contrib",
    "retrieval_mirdb_contrib",
    "retrieval_tcga_contrib",
    "retrieval_seed_contrib",
    "retrieval_rnahybrid_contrib",
    "retrieval_local_au_contrib",
    "retrieval_structure_contrib",
    "support_count",
    "ts_best_contextpp",
    "ts_best_percentile",
    "ts_context_strength",
    "mirdb_best_score",
    "mirdb_mean_score",
    "clip_exp_sum",
    "clip_exp_max",
    "n_clip_sites",
    "best_local_au",
    "best_local_au_by_mfe",
    "best_mfe",
    "mfe_strength",
    "mean_top3_mfe_strength",
    "n_rnahybrid_sites",
)
SUPPORT_COUNT_SOURCE_COLUMNS = (
    "support_targetscan",
    "support_mirdb",
    "support_encori",
    "support_rnahybrid",
    "BRCA_support_tcga",
    "COAD_support_tcga",
    "PRAD_support_tcga",
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rankings", required=True)
    ap.add_argument("--query-summary", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--models", default="logistic,xgboost")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--ks", default="1,5,10,25,50,100")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--split-mode", default="group_by_mirna")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--balance-mode", default="class_weight")
    ap.add_argument("--negative-sample-ratio", type=int, default=20)
    ap.add_argument("--use-calibration", action="store_true")
    ap.add_argument("--feature-set", default="all")
    ap.add_argument("--include-missingness-indicators", default="true")
    ap.add_argument("--save-model-artifact", default=None)
    ap.add_argument("--model-name", default="xgb_raw_v1")
    ap.add_argument("--refit-full-data", default="false")
    return ap.parse_args()


def parse_ks(text: str) -> List[int]:
    values: List[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise ValueError("At least one K value is required.")
    return sorted(set(values))


def parse_bool_text(value: Any, *, name: str) -> bool:
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true/false, 1/0, yes/no, or on/off.")


def parse_feature_sets(text: str) -> List[str]:
    values = [part.strip().lower() for part in str(text or "").split(",") if part.strip()]
    if not values:
        values = ["all"]
    invalid = [value for value in values if value not in FEATURE_SET_CHOICES]
    if invalid:
        raise ValueError(f"Unsupported feature set(s): {invalid}. Choose from {list(FEATURE_SET_CHOICES)}.")
    return sorted(set(values), key=values.index)


def safe_float_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def safe_binary_series(df: pd.DataFrame, col: str) -> pd.Series:
    values = safe_float_series(df, col, default=0.0)
    return (values > 0).astype(float)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def seed_class_rank(value: Any) -> float:
    text = str(value or "").strip().lower().replace("_", "-")
    if not text:
        return 0.0
    for key, score in SEED_CLASS_RANKS.items():
        if key in text:
            return score
    return 0.0


def engineer_features(df: pd.DataFrame, include_missingness_indicators: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = df.copy()
    notes: Dict[str, Any] = {
        "derived_features": [],
        "missingness_indicators": [],
        "dropped_non_numeric_columns": [],
        "created_missing_columns": [],
    }

    if "support_count" not in out.columns:
        available_support_sources = [col for col in SUPPORT_COUNT_SOURCE_COLUMNS if col in out.columns]
        if available_support_sources:
            support_total = pd.Series(0.0, index=out.index, dtype=float)
            for col in available_support_sources:
                support_total += safe_binary_series(out, col)
            out["support_count"] = support_total.astype(float)
            notes["derived_features"].append("support_count")

    rho_columns = [col for col in out.columns if str(col).lower().endswith("_spearman_rho")]
    for col in rho_columns:
        prefix = str(col)[: -len("_spearman_rho")]
        derived_col = f"{prefix}_anticorrelation_strength"
        out[derived_col] = np.clip(-safe_float_series(out, col, default=0.0), 0.0, None)
        notes["derived_features"].append(derived_col)

    if "mfe_strength" not in out.columns and "best_mfe" in out.columns:
        out["mfe_strength_from_best_mfe"] = np.clip(-safe_float_series(out, "best_mfe", default=0.0), 0.0, None)
        notes["derived_features"].append("mfe_strength_from_best_mfe")

    if "best_seed_class" in out.columns:
        out["best_seed_class_rank_encoded"] = out["best_seed_class"].map(seed_class_rank).fillna(0.0).astype(float)
        notes["derived_features"].append("best_seed_class_rank_encoded")

    if include_missingness_indicators:
        for col in IMPORTANT_MISSINGNESS_FEATURES:
            if col in out.columns:
                out[f"{col}_is_missing"] = pd.to_numeric(out[col], errors="coerce").isna().astype(int)
                notes["missingness_indicators"].append(f"{col}_is_missing")

    non_numeric_columns = [
        col
        for col in out.columns
        if not pd.api.types.is_numeric_dtype(out[col])
        and col not in {"best_seed_class"}
    ]
    notes["dropped_non_numeric_columns"] = sorted(non_numeric_columns)
    return out, notes


def detect_leakage_columns(columns: Iterable[str]) -> List[str]:
    leaked: List[str] = []
    for col in columns:
        text = str(col).strip().lower()
        if col in EXPLICIT_DROP_COLUMNS:
            leaked.append(str(col))
            continue
        if any(pattern in text for pattern in LEAKAGE_PATTERNS):
            leaked.append(str(col))
    return sorted(set(leaked))


def select_feature_columns(df: pd.DataFrame) -> Tuple[List[str], Dict[str, Any]]:
    leakage_columns = detect_leakage_columns(df.columns)
    allowed: List[str] = []
    dropped_non_numeric: List[str] = []
    dropped_list_like: List[str] = []

    for col in df.columns:
        if col in leakage_columns or col in EXPLICIT_DROP_COLUMNS:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            allowed.append(col)
            continue
        sample_nonnull = next((value for value in series if value is not None and not (isinstance(value, float) and np.isnan(value))), None)
        if isinstance(sample_nonnull, (list, tuple, set, dict, np.ndarray)):
            dropped_list_like.append(str(col))
        else:
            dropped_non_numeric.append(str(col))

    if not allowed:
        raise ValueError("No numeric non-leaking feature columns were available for model training.")

    return sorted(set(allowed)), {
        "dropped_leakage_columns": leakage_columns,
        "dropped_non_numeric_columns": sorted(set(dropped_non_numeric)),
        "dropped_list_like_columns": sorted(set(dropped_list_like)),
    }


def _base_feature_name(column_name: str) -> str:
    if str(column_name).endswith("_is_missing"):
        return str(column_name)[: -len("_is_missing")]
    return str(column_name)


def filter_feature_columns_by_set(feature_columns: Sequence[str], feature_set: str) -> List[str]:
    mode = (feature_set or "all").strip().lower()
    if mode not in FEATURE_SET_CHOICES:
        raise ValueError(f"Unsupported feature set: {feature_set}")

    if mode == "all":
        return list(feature_columns)

    filtered: List[str] = []
    for column_name in feature_columns:
        base_name = _base_feature_name(column_name)
        if mode == "components":
            if base_name in COMPONENT_FEATURE_COLUMNS:
                filtered.append(column_name)
            continue

        if base_name == "retrieval_score":
            continue
        if base_name.startswith("retrieval_"):
            continue
        filtered.append(column_name)

    return filtered


def prepare_feature_frame(
    input_df: pd.DataFrame,
    *,
    feature_set: str = "all",
    include_missingness_indicators: bool = True,
    selected_feature_columns: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    engineered_df, notes = engineer_features(input_df, include_missingness_indicators=include_missingness_indicators)
    feature_columns, selection_notes = select_feature_columns(engineered_df)
    feature_columns = filter_feature_columns_by_set(feature_columns, feature_set)
    if selected_feature_columns is not None:
        selected_feature_columns = list(selected_feature_columns)
        leaked_selected_columns = detect_leakage_columns(selected_feature_columns)
        if leaked_selected_columns:
            raise ValueError(
                f"Artifact-selected feature columns included leakage-like fields and were refused: {leaked_selected_columns}"
            )
        created_missing_columns: List[str] = []
        for col in selected_feature_columns:
            if col in engineered_df.columns:
                continue
            base_name = _base_feature_name(col)
            if col.endswith("_is_missing"):
                if base_name not in engineered_df.columns:
                    base_was_recomputed = False
                    if base_name == "support_count":
                        available_support_sources = [name for name in SUPPORT_COUNT_SOURCE_COLUMNS if name in engineered_df.columns]
                        if available_support_sources:
                            support_total = pd.Series(0.0, index=engineered_df.index, dtype=float)
                            for name in available_support_sources:
                                support_total += safe_binary_series(engineered_df, name)
                            engineered_df[base_name] = support_total.astype(float)
                            notes["derived_features"].append(base_name)
                            base_was_recomputed = True
                        else:
                            engineered_df[base_name] = pd.Series(0.0, index=engineered_df.index, dtype=float)
                            created_missing_columns.append(base_name)
                    elif base_name in {"pathway_match_count", "pathway_selected_gene"}:
                        engineered_df[base_name] = pd.Series(0.0, index=engineered_df.index, dtype=float)
                        created_missing_columns.append(base_name)
                    else:
                        engineered_df[base_name] = pd.Series(0.0, index=engineered_df.index, dtype=float)
                        created_missing_columns.append(base_name)
                    engineered_df[col] = pd.Series(0.0 if base_was_recomputed else 1.0, index=engineered_df.index, dtype=float)
                else:
                    engineered_df[col] = pd.Series(0.0, index=engineered_df.index, dtype=float)
                created_missing_columns.append(col)
                continue

            if col == "support_count":
                available_support_sources = [name for name in SUPPORT_COUNT_SOURCE_COLUMNS if name in engineered_df.columns]
                if available_support_sources:
                    support_total = pd.Series(0.0, index=engineered_df.index, dtype=float)
                    for name in available_support_sources:
                        support_total += safe_binary_series(engineered_df, name)
                    engineered_df[col] = support_total.astype(float)
                    notes["derived_features"].append(col)
                else:
                    engineered_df[col] = pd.Series(0.0, index=engineered_df.index, dtype=float)
                    created_missing_columns.append(col)
                continue

            if col in {"pathway_match_count", "pathway_selected_gene"}:
                engineered_df[col] = pd.Series(0.0, index=engineered_df.index, dtype=float)
                created_missing_columns.append(col)
                continue

            engineered_df[col] = pd.Series(0.0, index=engineered_df.index, dtype=float)
            created_missing_columns.append(col)

        if created_missing_columns:
            notes["created_missing_columns"] = sorted(set(notes.get("created_missing_columns", [])) | set(created_missing_columns))
        feature_columns = selected_feature_columns
    if not feature_columns:
        raise ValueError(f"No features remained after applying feature_set={feature_set!r}.")
    feature_df = engineered_df.copy()
    for col in feature_columns:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce").fillna(0.0).astype(float)
    notes.update(selection_notes)
    notes["feature_set"] = feature_set
    notes["include_missingness_indicators"] = bool(include_missingness_indicators)
    notes["selected_feature_columns"] = list(feature_columns)
    return feature_df, list(feature_columns), notes


def prepare_learning_frame(
    rankings_df: pd.DataFrame,
    *,
    feature_set: str = "all",
    include_missingness_indicators: bool = True,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    model_df, feature_columns, notes = prepare_feature_frame(
        rankings_df,
        feature_set=feature_set,
        include_missingness_indicators=include_missingness_indicators,
    )
    model_df["is_positive"] = pd.to_numeric(model_df["is_positive"], errors="coerce").fillna(0).astype(int)
    return model_df, feature_columns, notes


def choose_group_column(df: pd.DataFrame, split_mode: str) -> str:
    mode = (split_mode or "group_by_mirna").strip().lower()
    if mode == "group_by_query_id":
        return "query_id"
    if "mirna" in df.columns and df["mirna"].fillna("").astype(str).str.strip().ne("").any():
        return "mirna"
    return "query_id"


def maybe_limit_rows(df: pd.DataFrame, max_rows: Optional[int], seed: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= int(max_rows):
        return df
    rng = np.random.default_rng(seed)
    query_ids = df["query_id"].dropna().astype(str).unique().tolist()
    rng.shuffle(query_ids)
    keep_queries: List[str] = []
    current = 0
    for query_id in query_ids:
        n_rows = int((df["query_id"].astype(str) == str(query_id)).sum())
        if keep_queries and current + n_rows > int(max_rows):
            break
        keep_queries.append(str(query_id))
        current += n_rows
        if current >= int(max_rows):
            break
    if not keep_queries:
        keep_queries = query_ids[:1]
    return df[df["query_id"].astype(str).isin(keep_queries)].copy().reset_index(drop=True)


def split_groups(values: Sequence[str], test_size: float, seed: int) -> Tuple[set[str], set[str]]:
    unique_values = sorted({str(value) for value in values if str(value).strip()})
    if len(unique_values) < 2:
        raise ValueError("At least two unique groups are required for a grouped train/test split.")
    rng = np.random.default_rng(seed)
    shuffled = list(unique_values)
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * float(test_size))))
    n_test = min(n_test, len(shuffled) - 1)
    test_groups = set(shuffled[:n_test])
    train_groups = set(shuffled[n_test:])
    return train_groups, test_groups


def compute_scale_pos_weight(y: pd.Series) -> float:
    positives = max(1, int(pd.to_numeric(y, errors="coerce").fillna(0).astype(int).sum()))
    negatives = max(1, int(len(y) - positives))
    return float(negatives / positives)


def downsample_negatives_per_query(
    frame: pd.DataFrame,
    negative_sample_ratio: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sampled_parts: List[pd.DataFrame] = []
    for _, group in frame.groupby("query_id", sort=True):
        positives = group[group["is_positive"] == 1]
        negatives = group[group["is_positive"] == 0]
        if positives.empty:
            continue
        keep_negatives = min(len(negatives), int(max(0, negative_sample_ratio)) * len(positives))
        if keep_negatives > 0:
            chosen = rng.choice(negatives.index.to_numpy(), size=keep_negatives, replace=False)
            sampled_parts.append(pd.concat([positives, negatives.loc[chosen]], axis=0))
        else:
            sampled_parts.append(positives.copy())
    if not sampled_parts:
        return frame.copy()
    return pd.concat(sampled_parts, axis=0).sort_index().reset_index(drop=True)


def build_model_spec(model_name: str, balance_mode: str, seed: int, y_train: pd.Series) -> Tuple[Optional[Any], Optional[str], Optional[str]]:
    model_key = (model_name or "").strip().lower()
    if model_key == "logistic":
        try:
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return None, None, "scikit-learn is required for logistic regression."

        class_weight = "balanced" if balance_mode == "class_weight" else None
        estimator = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000, class_weight=class_weight, solver="lbfgs", random_state=seed)),
            ]
        )
        return estimator, "logistic", None

    if model_key == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            try:
                from sklearn.ensemble import RandomForestClassifier
            except ImportError:
                return None, None, "Neither xgboost nor a sklearn fallback model is available."
            class_weight = "balanced_subsample" if balance_mode == "class_weight" else None
            estimator = RandomForestClassifier(
                n_estimators=400,
                max_depth=6,
                min_samples_leaf=1,
                random_state=seed,
                n_jobs=-1,
                class_weight=class_weight,
            )
            return estimator, "random_forest_fallback", "xgboost was not installed; used RandomForestClassifier fallback."

        scale_pos_weight = compute_scale_pos_weight(y_train) if balance_mode == "class_weight" else 1.0
        estimator = XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            n_jobs=1,
            verbosity=0,
        )
        return estimator, "xgboost", None

    return None, None, f"Unsupported model name: {model_name}"


def fit_predict_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    balance_mode: str,
    seed: int,
    use_calibration: bool,
) -> Tuple[np.ndarray, str, Optional[pd.DataFrame], List[str]]:
    warnings: List[str] = []
    estimator, resolved_name, setup_warning = build_model_spec(model_name, balance_mode, seed, y_train)
    if estimator is None or resolved_name is None:
        raise RuntimeError(setup_warning or f"Unable to initialize model {model_name}.")
    if setup_warning:
        warnings.append(setup_warning)

    from sklearn.base import clone

    importance_estimator = clone(estimator)
    importance_estimator.fit(X_train, y_train)
    predictor = importance_estimator

    if use_calibration:
        try:
            from sklearn.calibration import CalibratedClassifierCV
        except ImportError:
            warnings.append("Calibration was requested but sklearn calibration utilities were unavailable; used uncalibrated predictions.")
        else:
            predictor = CalibratedClassifierCV(estimator=clone(estimator), method="sigmoid", cv=3)
            predictor.fit(X_train, y_train)

    if hasattr(predictor, "predict_proba"):
        predicted_scores = predictor.predict_proba(X_test)[:, 1]
    elif hasattr(predictor, "decision_function"):
        values = predictor.decision_function(X_test)
        predicted_scores = 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=float)))
    else:
        predicted_scores = np.asarray(predictor.predict(X_test), dtype=float)

    feature_importance_df = extract_feature_importance(importance_estimator, list(X_train.columns), resolved_name)
    return np.asarray(predicted_scores, dtype=float), resolved_name, feature_importance_df, warnings


def fit_model_for_artifact(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    balance_mode: str,
    seed: int,
) -> Tuple[Any, str, List[str]]:
    warnings: List[str] = []
    estimator, resolved_name, setup_warning = build_model_spec(model_name, balance_mode, seed, y_train)
    if estimator is None or resolved_name is None:
        raise RuntimeError(setup_warning or f"Unable to initialize model {model_name}.")
    if setup_warning:
        warnings.append(setup_warning)
    estimator.fit(X_train, y_train)
    return estimator, resolved_name, warnings


def extract_feature_importance(estimator: Any, feature_columns: Sequence[str], model_name: str) -> Optional[pd.DataFrame]:
    if model_name == "logistic" and hasattr(estimator, "named_steps") and "model" in estimator.named_steps:
        coeffs = np.asarray(estimator.named_steps["model"].coef_[0], dtype=float)
        return pd.DataFrame(
            {
                "model": model_name,
                "feature": list(feature_columns),
                "coefficient": coeffs,
                "abs_coefficient": np.abs(coeffs),
            }
        ).sort_values("abs_coefficient", ascending=False)

    raw_importance = None
    if hasattr(estimator, "feature_importances_"):
        raw_importance = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "named_steps") and "model" in estimator.named_steps and hasattr(estimator.named_steps["model"], "feature_importances_"):
        raw_importance = np.asarray(estimator.named_steps["model"].feature_importances_, dtype=float)

    if raw_importance is None:
        return None

    return pd.DataFrame(
        {
            "model": model_name,
            "feature": list(feature_columns),
            "importance": raw_importance,
        }
    ).sort_values("importance", ascending=False)


def label_score_mode(name: str, feature_set: str) -> str:
    return f"{name}__{feature_set}"


def get_package_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for package_name in ("scikit-learn", "xgboost", "joblib", "pandas", "numpy"):
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def save_model_artifact(
    artifact_path: str | Path,
    *,
    estimator: Any,
    model_type: str,
    model_name: str,
    feature_set: str,
    include_missingness_indicators: bool,
    feature_names: Sequence[str],
    leakage_excluded_columns: Sequence[str],
    selected_feature_columns: Sequence[str],
) -> Path:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("joblib is required to save model artifacts.") from exc

    artifact_path = Path(artifact_path).resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": estimator,
        "feature_names": list(feature_names),
        "feature_set": feature_set,
        "include_missingness_indicators": bool(include_missingness_indicators),
        "model_type": model_type,
        "model_name": model_name,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "leakage_excluded_columns": list(leakage_excluded_columns),
        "selected_feature_columns": list(selected_feature_columns),
        "versions": get_package_versions(),
    }
    joblib.dump(payload, artifact_path)
    metadata_path = artifact_path.with_suffix(f"{artifact_path.suffix}.metadata.json")
    json_dump(
        metadata_path,
        {
            "artifact_path": str(artifact_path),
            "model_name": model_name,
            "model_type": model_type,
            "feature_set": feature_set,
            "include_missingness_indicators": bool(include_missingness_indicators),
            "feature_names": list(feature_names),
            "leakage_excluded_columns": list(leakage_excluded_columns),
            "selected_feature_columns": list(selected_feature_columns),
            "training_timestamp": payload["training_timestamp"],
            "versions": payload["versions"],
        },
    )
    return artifact_path


def compute_query_metrics_from_scores(
    df: pd.DataFrame,
    score_col: str,
    ks: Sequence[int],
    query_summary_lookup: Dict[str, Dict[str, Any]],
    score_mode: str,
    feature_set: str,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["_score"] = pd.to_numeric(work[score_col], errors="coerce").fillna(0.0)
    work["_original_rank"] = pd.to_numeric(work.get("rank"), errors="coerce").fillna(np.inf)
    if "gene_symbol" in work.columns:
        work["_gene_tie"] = work["gene_symbol"].fillna("").astype(str)
    else:
        work["_gene_tie"] = ""
    work = work.sort_values(["query_id", "_score", "_original_rank", "_gene_tie"], ascending=[True, False, True, True]).copy()
    work["learned_rank"] = work.groupby("query_id").cumcount() + 1

    query_rows: List[Dict[str, Any]] = []
    ranking_rows: List[Dict[str, Any]] = []
    for query_id, group in work.groupby("query_id", sort=True):
        ordered = group.sort_values("learned_rank").reset_index(drop=True)
        is_positive = pd.to_numeric(ordered["is_positive"], errors="coerce").fillna(0).astype(int).to_numpy()
        positive_positions = np.flatnonzero(is_positive > 0) + 1
        summary_info = query_summary_lookup.get(str(query_id), {})
        total_positives = safe_int(summary_info.get("n_positives_total"), default=int(is_positive.sum()))
        denominator = total_positives if total_positives > 0 else int(is_positive.sum())
        row = {
            "score_mode": score_mode,
            "feature_set": feature_set,
            "query_id": query_id,
            "mirna": summary_info.get("mirna", ordered["mirna"].iloc[0] if "mirna" in ordered.columns and len(ordered) else ""),
            "n_ranked": int(len(ordered)),
            "n_positives_total": int(denominator),
            "n_positives_retrieved": int(is_positive.sum()),
            "best_positive_rank": float(positive_positions.min()) if positive_positions.size else np.nan,
            "reciprocal_rank": (1.0 / float(positive_positions.min())) if positive_positions.size else 0.0,
        }
        for k in ks:
            positives_in_topk = int(is_positive[: min(k, len(ordered))].sum())
            row[f"positive_count_at_{k}"] = positives_in_topk
            row[f"recall_at_{k}"] = positives_in_topk / float(denominator) if denominator > 0 else np.nan
            row[f"precision_at_{k}"] = positives_in_topk / float(max(1, min(k, len(ordered)))) if len(ordered) > 0 else 0.0
        query_rows.append(row)

        keep_columns = [
            col
            for col in (
                "query_id",
                "mirna",
                "gene_symbol",
                "is_positive",
                "retrieval_score",
                "retrieval_mirdb_contrib",
                "retrieval_ts_contrib",
                "retrieval_clip_contrib",
                "retrieval_structure_contrib",
            )
            if col in ordered.columns
        ]
        ranking_chunk = ordered[keep_columns].copy()
        ranking_chunk["score_mode"] = score_mode
        ranking_chunk["feature_set"] = feature_set
        ranking_chunk["predicted_score"] = ordered["_score"].to_numpy()
        ranking_chunk["learned_rank"] = ordered["learned_rank"].to_numpy()
        ranking_rows.append(ranking_chunk)

    query_metrics_df = pd.DataFrame(query_rows)
    ranking_df = pd.concat(ranking_rows, ignore_index=True) if ranking_rows else pd.DataFrame()

    labels = pd.to_numeric(work["is_positive"], errors="coerce").fillna(0).astype(int).tolist()
    scores = pd.to_numeric(work["_score"], errors="coerce").fillna(0.0).tolist()
    best_rank_series = pd.to_numeric(query_metrics_df.get("best_positive_rank"), errors="coerce")
    summary_row = {
        "score_mode": score_mode,
        "feature_set": feature_set,
        "split": "heldout_test",
        "n_queries": int(len(query_metrics_df)),
        "n_ranked_interactions": int(len(work)),
        "positives_retrieved": int(pd.to_numeric(query_metrics_df.get("n_positives_retrieved"), errors="coerce").fillna(0).sum())
        if not query_metrics_df.empty
        else 0,
        "mrr": float(np.nanmean(query_metrics_df["reciprocal_rank"])) if not query_metrics_df.empty else np.nan,
        "mean_best_positive_rank": float(best_rank_series.dropna().mean()) if best_rank_series.notna().any() else np.nan,
        "median_best_positive_rank": float(best_rank_series.dropna().median()) if best_rank_series.notna().any() else np.nan,
        "auroc": roc_auc_score_manual(labels, scores) if len(set(labels)) > 1 else None,
        "auprc": average_precision_manual(labels, scores) if any(labels) else None,
    }

    recall_rows = [
        {
            "score_mode": score_mode,
            "feature_set": feature_set,
            "k": int(k),
            "recall_at_k": float(np.nanmean(query_metrics_df[f"recall_at_{k}"])) if not query_metrics_df.empty else np.nan,
        }
        for k in ks
    ]
    precision_rows = [
        {
            "score_mode": score_mode,
            "feature_set": feature_set,
            "k": int(k),
            "precision_at_k": float(np.nanmean(query_metrics_df[f"precision_at_{k}"])) if not query_metrics_df.empty else np.nan,
        }
        for k in ks
    ]
    return query_metrics_df, summary_row, pd.DataFrame(recall_rows), pd.DataFrame(precision_rows), ranking_df


def build_baseline_scores(df: pd.DataFrame) -> Tuple[Dict[str, pd.Series], Dict[str, Any]]:
    baselines: Dict[str, pd.Series] = {}
    warnings = {"skipped_baselines": [], "missing_columns": {}}

    def add(name: str, required_cols: Sequence[str], builder) -> None:
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            warnings["skipped_baselines"].append(name)
            warnings["missing_columns"][name] = missing
            return
        baselines[name] = builder()

    add("retrieval_score", ["retrieval_score"], lambda: safe_float_series(df, "retrieval_score", default=0.0))
    if "mirdb_best_score" in df.columns:
        add("mirdb_best_score", ["mirdb_best_score"], lambda: safe_float_series(df, "mirdb_best_score", default=0.0))
    else:
        add("retrieval_mirdb_contrib", ["retrieval_mirdb_contrib"], lambda: safe_float_series(df, "retrieval_mirdb_contrib", default=0.0))
    if "ts_context_strength" in df.columns:
        add("ts_context_strength", ["ts_context_strength"], lambda: safe_float_series(df, "ts_context_strength", default=0.0))
    else:
        add("retrieval_ts_contrib", ["retrieval_ts_contrib"], lambda: safe_float_series(df, "retrieval_ts_contrib", default=0.0))
    if "clip_exp_sum" in df.columns:
        add("clip_exp_sum", ["clip_exp_sum"], lambda: safe_float_series(df, "clip_exp_sum", default=0.0))
    else:
        add("retrieval_clip_contrib", ["retrieval_clip_contrib"], lambda: safe_float_series(df, "retrieval_clip_contrib", default=0.0))
    add("retrieval_structure_contrib", ["retrieval_structure_contrib"], lambda: safe_float_series(df, "retrieval_structure_contrib", default=0.0))
    add(
        "mirdb_targetscan",
        ["retrieval_mirdb_contrib", "retrieval_ts_contrib"],
        lambda: safe_float_series(df, "retrieval_mirdb_contrib", default=0.0) + safe_float_series(df, "retrieval_ts_contrib", default=0.0),
    )
    add(
        "mirdb_targetscan_structure",
        ["retrieval_mirdb_contrib", "retrieval_ts_contrib", "retrieval_structure_contrib"],
        lambda: (
            safe_float_series(df, "retrieval_mirdb_contrib", default=0.0)
            + safe_float_series(df, "retrieval_ts_contrib", default=0.0)
            + safe_float_series(df, "retrieval_structure_contrib", default=0.0)
        ),
    )
    return baselines, warnings


def save_bar_plot(df: pd.DataFrame, metric_col: str, title: str, ylabel: str, path: Path) -> None:
    plot_df = df.copy()
    if plot_df.empty or metric_col not in plot_df.columns:
        return
    plt.figure(figsize=(9, 4))
    plt.bar(plot_df["score_mode"].astype(str), plot_df[metric_col].fillna(0.0))
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_grouped_k_plot(df: pd.DataFrame, value_col: str, title: str, ylabel: str, path: Path, ks_to_plot: Sequence[int]) -> None:
    if df.empty:
        return
    subset = df[df["k"].isin(list(ks_to_plot))].copy()
    if subset.empty:
        return
    pivot = subset.pivot(index="score_mode", columns="k", values=value_col).fillna(0.0)
    ax = pivot.plot(kind="bar", figsize=(9, 4))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Score mode")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_feature_plot(df: pd.DataFrame, model_name: str, path: Path) -> None:
    subset = df[df["model"] == model_name].copy()
    if subset.empty:
        return
    if "abs_coefficient" in subset.columns:
        subset = subset.sort_values("abs_coefficient", ascending=False).head(15)
        x_col = "abs_coefficient"
        title = f"Top Feature Importance: {model_name}"
        ylabel = "|Coefficient|"
    elif "importance" in subset.columns:
        subset = subset.sort_values("importance", ascending=False).head(15)
        x_col = "importance"
        title = f"Top Feature Importance: {model_name}"
        ylabel = "Importance"
    else:
        return
    plt.figure(figsize=(8, 4))
    plt.bar(subset["feature"].astype(str), subset[x_col].fillna(0.0))
    plt.xticks(rotation=60, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    ks = parse_ks(args.ks)
    feature_sets = parse_feature_sets(args.feature_set)
    include_missingness_indicators = parse_bool_text(
        args.include_missingness_indicators,
        name="--include-missingness-indicators",
    )
    refit_full_data = parse_bool_text(args.refit_full_data, name="--refit-full-data")
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rankings_df = pd.read_parquet(Path(args.rankings).resolve())
    query_summary_df = pd.read_parquet(Path(args.query_summary).resolve())
    if "is_positive" not in rankings_df.columns:
        raise ValueError("rankings_long must include an is_positive column.")
    if "query_id" not in rankings_df.columns:
        raise ValueError("rankings_long must include a query_id column.")

    rankings_df = maybe_limit_rows(rankings_df, args.max_rows, args.seed)
    group_source_df = rankings_df.copy()
    group_col = choose_group_column(group_source_df, args.split_mode)
    if group_col == "mirna":
        group_source_df[group_col] = np.where(
            group_source_df[group_col].fillna("").astype(str).str.strip().ne(""),
            group_source_df[group_col].fillna("").astype(str),
            group_source_df["query_id"].fillna("").astype(str),
        )
    else:
        group_source_df[group_col] = group_source_df[group_col].fillna("").astype(str)
    if pd.Series(group_source_df[group_col]).eq("").all():
        group_col = "query_id"
        group_source_df[group_col] = group_source_df[group_col].fillna("").astype(str)

    train_groups, test_groups = split_groups(group_source_df[group_col].tolist(), args.test_size, args.seed)

    query_summary_lookup: Dict[str, Dict[str, Any]] = {}
    if not query_summary_df.empty and "query_id" in query_summary_df.columns:
        query_summary_lookup = {
            str(key): value
            for key, value in query_summary_df.set_index("query_id").to_dict(orient="index").items()
        }

    models_requested = [item.strip() for item in str(args.models).split(",") if item.strip()]
    if args.save_model_artifact and len(models_requested) != 1:
        raise ValueError("--save-model-artifact currently requires exactly one model in --models.")
    if args.save_model_artifact and len(feature_sets) != 1:
        raise ValueError("--save-model-artifact currently requires exactly one feature set in --feature-set.")
    warnings_payload: Dict[str, Any] = {
        "skipped_models": [],
        "model_warnings": [],
        "feature_sets": feature_sets,
        "include_missingness_indicators": bool(include_missingness_indicators),
        "save_model_artifact": str(args.save_model_artifact) if args.save_model_artifact else None,
        "model_artifact_name": args.model_name,
        "refit_full_data": bool(refit_full_data),
        "prep_notes_by_feature_set": {},
        "split": {
            "group_column": group_col,
            "n_rows_total": int(len(group_source_df)),
            "n_train_groups": int(len(train_groups)),
            "n_test_groups": int(len(test_groups)),
            "train_groups": sorted(train_groups),
            "test_groups": sorted(test_groups),
            "balance_mode": args.balance_mode,
            "negative_sample_ratio": int(args.negative_sample_ratio),
        },
        "baseline_warnings": {},
        "missing_feature_warnings": [],
    }

    summary_rows: List[Dict[str, Any]] = []
    recall_frames: List[pd.DataFrame] = []
    precision_frames: List[pd.DataFrame] = []
    ranking_frames: List[pd.DataFrame] = []
    feature_importance_frames: List[pd.DataFrame] = []
    cv_rows: List[Dict[str, Any]] = []
    artifact_saved_path: Optional[str] = None

    for feature_set in feature_sets:
        model_df, feature_columns, prep_notes = prepare_learning_frame(
            rankings_df,
            feature_set=feature_set,
            include_missingness_indicators=include_missingness_indicators,
        )
        if group_col == "mirna":
            model_df[group_col] = np.where(
                model_df[group_col].fillna("").astype(str).str.strip().ne(""),
                model_df[group_col].fillna("").astype(str),
                model_df["query_id"].fillna("").astype(str),
            )
        else:
            model_df[group_col] = model_df[group_col].fillna("").astype(str)
        warnings_payload["prep_notes_by_feature_set"][feature_set] = prep_notes

        train_df = model_df[model_df[group_col].isin(train_groups)].copy().reset_index(drop=True)
        test_df = model_df[model_df[group_col].isin(test_groups)].copy().reset_index(drop=True)
        if train_df.empty or test_df.empty:
            raise ValueError(f"The grouped split produced an empty train or test set for feature_set={feature_set}.")

        X_train = train_df[feature_columns].copy()
        y_train = train_df["is_positive"].astype(int)
        X_test = test_df[feature_columns].copy()
        y_test = test_df["is_positive"].astype(int)

        training_frame_for_fit = train_df.copy()
        if args.balance_mode == "downsample_negatives":
            training_frame_for_fit = downsample_negatives_per_query(train_df, args.negative_sample_ratio, args.seed)
            X_train = training_frame_for_fit[feature_columns].copy()
            y_train = training_frame_for_fit["is_positive"].astype(int)

        baseline_scores, baseline_warnings = build_baseline_scores(test_df)
        warnings_payload["baseline_warnings"][feature_set] = baseline_warnings

        for model_name in models_requested:
            try:
                predicted_scores, resolved_name, feature_importance_df, model_warnings = fit_predict_model(
                    model_name=model_name,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    balance_mode=args.balance_mode,
                    seed=args.seed,
                    use_calibration=args.use_calibration,
                )
            except Exception as exc:
                warnings_payload["skipped_models"].append({"model": model_name, "feature_set": feature_set, "reason": str(exc)})
                continue

            warnings_payload["model_warnings"].extend([f"{feature_set}:{warning}" for warning in model_warnings])
            scored_test_df = test_df.copy()
            scored_test_df["predicted_score"] = predicted_scores
            score_mode = label_score_mode(resolved_name, feature_set)
            _, summary_row, recall_df, precision_df, ranking_df = compute_query_metrics_from_scores(
                scored_test_df,
                "predicted_score",
                ks,
                query_summary_lookup,
                score_mode,
                feature_set,
            )
            summary_row.update(
                {
                    "model": resolved_name,
                    "n_train_rows": int(len(X_train)),
                    "n_test_rows": int(len(X_test)),
                    "n_train_positives": int(y_train.sum()),
                    "n_test_positives": int(y_test.sum()),
                    "n_test_queries": int(test_df["query_id"].astype(str).nunique()),
                }
            )
            summary_rows.append(summary_row)
            recall_frames.append(recall_df)
            precision_frames.append(precision_df)
            ranking_frames.append(ranking_df)
            if feature_importance_df is not None and not feature_importance_df.empty:
                feature_importance_df = feature_importance_df.copy()
                feature_importance_df["feature_set"] = feature_set
                feature_importance_df["score_mode"] = score_mode
                feature_importance_frames.append(feature_importance_df)

            if args.save_model_artifact:
                artifact_train_df = train_df if not refit_full_data else model_df
                artifact_X = artifact_train_df[feature_columns].copy()
                artifact_y = artifact_train_df["is_positive"].astype(int)
                if args.balance_mode == "downsample_negatives":
                    artifact_source_df = artifact_train_df if not refit_full_data else model_df
                    artifact_source_df = downsample_negatives_per_query(
                        artifact_source_df,
                        args.negative_sample_ratio,
                        args.seed,
                    )
                    artifact_X = artifact_source_df[feature_columns].copy()
                    artifact_y = artifact_source_df["is_positive"].astype(int)
                artifact_estimator, artifact_model_type, artifact_warnings = fit_model_for_artifact(
                    model_name=model_name,
                    X_train=artifact_X,
                    y_train=artifact_y,
                    balance_mode=args.balance_mode,
                    seed=args.seed,
                )
                warnings_payload["model_warnings"].extend([f"{feature_set}:{warning}" for warning in artifact_warnings])
                artifact_saved = save_model_artifact(
                    args.save_model_artifact,
                    estimator=artifact_estimator,
                    model_type=artifact_model_type,
                    model_name=args.model_name,
                    feature_set=feature_set,
                    include_missingness_indicators=include_missingness_indicators,
                    feature_names=feature_columns,
                    leakage_excluded_columns=prep_notes.get("dropped_leakage_columns", []),
                    selected_feature_columns=prep_notes.get("selected_feature_columns", feature_columns),
                )
                artifact_saved_path = str(artifact_saved)

        for baseline_name, baseline_scores_series in baseline_scores.items():
            baseline_df = test_df.copy()
            baseline_df["predicted_score"] = pd.to_numeric(baseline_scores_series, errors="coerce").fillna(0.0).to_numpy()
            score_mode = label_score_mode(baseline_name, feature_set)
            _, summary_row, recall_df, precision_df, ranking_df = compute_query_metrics_from_scores(
                baseline_df,
                "predicted_score",
                ks,
                query_summary_lookup,
                score_mode,
                feature_set,
            )
            summary_row.update(
                {
                    "model": "baseline",
                    "n_train_rows": int(len(X_train)),
                    "n_test_rows": int(len(X_test)),
                    "n_train_positives": int(y_train.sum()),
                    "n_test_positives": int(y_test.sum()),
                    "n_test_queries": int(test_df["query_id"].astype(str).nunique()),
                }
            )
            summary_rows.append(summary_row)
            recall_frames.append(recall_df)
            precision_frames.append(precision_df)
            ranking_frames.append(ranking_df)

        if args.cv_folds > 1:
            try:
                from sklearn.model_selection import GroupKFold
            except ImportError:
                warnings_payload["model_warnings"].append("GroupKFold was unavailable; skipped cross-validation metrics.")
            else:
                cv_groups = model_df[group_col].astype(str)
                n_groups = int(cv_groups.nunique())
                n_splits = min(int(args.cv_folds), n_groups)
                if n_splits >= 2:
                    splitter = GroupKFold(n_splits=n_splits)
                    for model_name in models_requested:
                        fold_index = 0
                        for train_index, valid_index in splitter.split(model_df[feature_columns], model_df["is_positive"], groups=cv_groups):
                            fold_index += 1
                            cv_train = model_df.iloc[train_index].copy()
                            cv_valid = model_df.iloc[valid_index].copy()
                            if args.balance_mode == "downsample_negatives":
                                cv_train = downsample_negatives_per_query(cv_train, args.negative_sample_ratio, args.seed + fold_index)
                            try:
                                predicted_scores, resolved_name, _, model_warnings = fit_predict_model(
                                    model_name=model_name,
                                    X_train=cv_train[feature_columns].copy(),
                                    y_train=cv_train["is_positive"].astype(int),
                                    X_test=cv_valid[feature_columns].copy(),
                                    balance_mode=args.balance_mode,
                                    seed=args.seed + fold_index,
                                    use_calibration=args.use_calibration,
                                )
                            except Exception as exc:
                                warnings_payload["model_warnings"].append(f"CV skipped for {model_name} {feature_set} fold {fold_index}: {exc}")
                                break
                            warnings_payload["model_warnings"].extend([f"{feature_set}:{warning}" for warning in model_warnings])
                            cv_valid = cv_valid.copy()
                            cv_valid["predicted_score"] = predicted_scores
                            score_mode = label_score_mode(resolved_name, feature_set)
                            _, summary_row, _, _, _ = compute_query_metrics_from_scores(
                                cv_valid,
                                "predicted_score",
                                ks,
                                query_summary_lookup,
                                score_mode,
                                feature_set,
                            )
                            cv_rows.append(
                                {
                                    "model": resolved_name,
                                    "feature_set": feature_set,
                                    "score_mode": score_mode,
                                    "fold": int(fold_index),
                                    "split": f"cv_fold_{fold_index}",
                                    "n_train_rows": int(len(cv_train)),
                                    "n_valid_rows": int(len(cv_valid)),
                                    "n_train_positives": int(cv_train["is_positive"].sum()),
                                    "n_valid_positives": int(cv_valid["is_positive"].sum()),
                                    "n_valid_queries": int(cv_valid["query_id"].astype(str).nunique()),
                                    "positives_retrieved": summary_row["positives_retrieved"],
                                    "mrr": summary_row["mrr"],
                                    "mean_best_positive_rank": summary_row["mean_best_positive_rank"],
                                    "median_best_positive_rank": summary_row["median_best_positive_rank"],
                                    "auroc": summary_row["auroc"],
                                    "auprc": summary_row["auprc"],
                                }
                            )

    summary_df = pd.DataFrame(summary_rows)
    recall_df = pd.concat(recall_frames, ignore_index=True) if recall_frames else pd.DataFrame()
    precision_df = pd.concat(precision_frames, ignore_index=True) if precision_frames else pd.DataFrame()
    rankings_out_df = pd.concat(ranking_frames, ignore_index=True) if ranking_frames else pd.DataFrame()
    feature_importance_df = pd.concat(feature_importance_frames, ignore_index=True) if feature_importance_frames else pd.DataFrame()
    cv_metrics_df = pd.DataFrame(cv_rows)

    summary_df.to_csv(outdir / "learned_ranker_metrics_summary.csv", index=False)
    recall_df.to_csv(outdir / "learned_ranker_recall_at_k.csv", index=False)
    precision_df.to_csv(outdir / "learned_ranker_precision_at_k.csv", index=False)
    rankings_out_df.to_parquet(outdir / "learned_ranker_test_rankings.parquet", index=False)
    rankings_out_df.to_csv(outdir / "learned_ranker_test_rankings.csv", index=False)
    feature_importance_df.to_csv(outdir / "learned_ranker_feature_importance.csv", index=False)
    cv_metrics_df.to_csv(outdir / "learned_ranker_cv_metrics.csv", index=False)
    warnings_payload["saved_model_artifact"] = artifact_saved_path
    json_dump(outdir / "learned_ranker_warnings.json", warnings_payload)

    if not summary_df.empty:
        save_bar_plot(summary_df, "mrr", "Learned Ranker Comparison: MRR", "MRR", outdir / "learned_ranker_mrr.png")
        save_bar_plot(summary_df, "auprc", "Learned Ranker Comparison: AUPRC", "AUPRC", outdir / "learned_ranker_auprc.png")
    save_grouped_k_plot(
        recall_df,
        "recall_at_k",
        "Learned Ranker Comparison: Recall@10/25/50",
        "Recall",
        outdir / "learned_ranker_recall_at_10_25_50.png",
        ks_to_plot=(10, 25, 50),
    )
    save_grouped_k_plot(
        precision_df,
        "precision_at_k",
        "Learned Ranker Comparison: Precision@10/25/50",
        "Precision",
        outdir / "learned_ranker_precision_at_10_25_50.png",
        ks_to_plot=(10, 25, 50),
    )
    save_feature_plot(feature_importance_df, "logistic", outdir / "top_feature_importance_logistic.png")
    save_feature_plot(feature_importance_df, "xgboost", outdir / "top_feature_importance_xgboost.png")
    save_feature_plot(feature_importance_df, "random_forest_fallback", outdir / "top_feature_importance_xgboost.png")


if __name__ == "__main__":
    main()
