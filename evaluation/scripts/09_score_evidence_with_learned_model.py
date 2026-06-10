from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import sys
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--model-artifact", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--score-column", default="learned_score_xgb_raw_v1")
    ap.add_argument("--batch-size", type=int, default=250000)
    return ap.parse_args()


def _load_learned_ranker_module():
    module_path = REPO_ROOT / "evaluation" / "scripts" / "08_train_learned_ranker.py"
    spec = importlib.util.spec_from_file_location("eval_learned_ranker", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load learned-ranker module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path).resolve()
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format for {path}. Use .parquet or .csv.")


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return path
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return path
    raise ValueError(f"Unsupported output format for {path}. Use .parquet or .csv.")


def predict_in_batches(estimator: Any, feature_df: pd.DataFrame, batch_size: int) -> np.ndarray:
    outputs: list[np.ndarray] = []
    n_rows = int(len(feature_df))
    batch_size = max(1, int(batch_size))
    for start in range(0, n_rows, batch_size):
        batch = feature_df.iloc[start : start + batch_size]
        if hasattr(estimator, "predict_proba"):
            outputs.append(np.asarray(estimator.predict_proba(batch)[:, 1], dtype=float))
        elif hasattr(estimator, "decision_function"):
            values = estimator.decision_function(batch)
            outputs.append(1.0 / (1.0 + np.exp(-np.asarray(values, dtype=float))))
        else:
            outputs.append(np.asarray(estimator.predict(batch), dtype=float))
    if not outputs:
        return np.zeros(0, dtype=float)
    return np.concatenate(outputs)


def main() -> None:
    args = parse_args()
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("joblib is required to load learned model artifacts.") from exc

    learned = _load_learned_ranker_module()
    evidence_df = read_table(args.evidence)
    artifact = joblib.load(Path(args.model_artifact).resolve())

    estimator = artifact["model"]
    feature_names: Sequence[str] = artifact["feature_names"]
    feature_set = artifact["feature_set"]
    include_missingness_indicators = bool(artifact.get("include_missingness_indicators", True))
    model_name = str(artifact.get("model_name") or "unknown_model")

    feature_df, prepared_feature_names, prep_notes = learned.prepare_feature_frame(
        evidence_df,
        feature_set=feature_set,
        include_missingness_indicators=include_missingness_indicators,
        selected_feature_columns=feature_names,
    )
    if list(prepared_feature_names) != list(feature_names):
        raise RuntimeError("Prepared feature names did not match artifact feature order.")

    predictions = predict_in_batches(estimator, feature_df[list(feature_names)], batch_size=args.batch_size)
    timestamp = datetime.now(timezone.utc).isoformat()

    scored_df = evidence_df.copy()
    scored_df[args.score_column] = pd.Series(predictions, index=scored_df.index, dtype=float)
    scored_df["learned_score_model_version"] = model_name
    scored_df["learned_score_feature_set"] = feature_set
    scored_df["learned_score_updated_at"] = timestamp

    write_table(scored_df, args.out)

    companion_metadata = Path(args.out).resolve().with_suffix(f"{Path(args.out).suffix}.metadata.json")
    learned.json_dump(
        companion_metadata,
        {
            "input_path": str(Path(args.evidence).resolve()),
            "output_path": str(Path(args.out).resolve()),
            "model_artifact": str(Path(args.model_artifact).resolve()),
            "score_column": args.score_column,
            "model_name": model_name,
            "feature_set": feature_set,
            "include_missingness_indicators": include_missingness_indicators,
            "selected_feature_columns": list(feature_names),
            "leakage_excluded_columns": artifact.get("leakage_excluded_columns", []),
            "created_missing_columns": prep_notes.get("created_missing_columns", []),
            "prep_notes": prep_notes,
            "timestamp_utc": timestamp,
        },
    )
    print(f"Wrote scored evidence to {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
