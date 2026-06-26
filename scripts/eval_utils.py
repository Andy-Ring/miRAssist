from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import pickle
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from evaluation.utils import average_precision_manual, roc_auc_score_manual
except ModuleNotFoundError:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import evaluation.utils or sklearn.metrics. "
            "Install scikit-learn or restore evaluation/utils.py."
        ) from exc

    def average_precision_manual(y_true, y_score):
        return float(average_precision_score(y_true, y_score))

    def roc_auc_score_manual(y_true, y_score):
        return float(roc_auc_score(y_true, y_score))


DEFAULT_K_VALUES: Tuple[int, ...] = (1, 3, 5, 10)
DEFAULT_OUTPUT_ROOT = Path("evaluation/clean_evidence_eval")
DEFAULT_CLEAN_EVIDENCE_PATH = Path("data/processed/mirassist_clean_evidence.parquet")
LEAKAGE_TOKENS: Tuple[str, ...] = (
    "mirtarbase",
    "validated",
    "label",
    "manual",
    "weighted",
    "old_score",
    "ground_truth",
    "heldout",
)
IDENTIFIER_COLUMNS: Tuple[str, ...] = (
    "evidence_row_id",
    "eval_row_id",
    "mirna_name",
    "mirna_name_normalized",
    "mirna_name_norm",
    "gene_symbol",
    "gene_symbol_normalized",
    "gene_symbol_norm",
    "transcript_id",
    "site_id",
    "chrom",
    "start",
    "end",
    "strand",
    "site_sequence",
    "window_sequence",
)
SEED_CLASS_STRENGTH = {
    "8mer": 1.0,
    "7mer-m8": 0.8,
    "7mer_a1": 0.7,
    "7mer-a1": 0.7,
    "6mer": 0.5,
}
MIRDB_SOURCE_CANDIDATES: Tuple[str, ...] = (
    "data/processed/mirassist_backend_features.parquet",
    "data/processed/mirassist_backend_features.csv",
    "mirassist_evidence_pairs_full.csv",
    "mirassist_evidence_pairs_test.csv",
)
LABEL_SOURCE_CANDIDATES: Tuple[str, ...] = (
    "evaluation/data/heldout_mirtarbase_labels.parquet",
    "evaluation/data/heldout_mirtarbase_labels.csv",
    "evaluation/runs/paper_learned_ranker_v3_local_ps/tables/blinded/heldout_mirtarbase_labels.parquet",
    "evaluation/runs/paper_learned_ranker_v3_local_ps/tables/blinded/heldout_mirtarbase_labels.csv",
)
EXTERNAL_MODEL_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "diana_microt": ("*diana*microt*.csv", "*diana*microt*.tsv", "*diana*microt*.parquet"),
    "miranda": ("*miranda*.csv", "*miranda*.tsv", "*miranda*.parquet"),
    "pita": ("*pita*.csv", "*pita*.tsv", "*pita*.parquet"),
}
ALIGNED_EXTERNAL_MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "mirdb": {
        "relative_path": Path("mirdb/parsed/mirdb_scores_aligned_to_evidence.csv.gz"),
        "score_candidates": (("mirdb_score", True),),
        "prediction_flag": "has_prediction",
    },
    "miranda": {
        "relative_path": Path("miranda/parsed/miranda_scores_aligned_to_evidence.csv.gz"),
        "score_candidates": (
            ("miranda_best_score", True),
            ("miranda_best_energy_strength", True),
            ("miranda_n_sites", True),
        ),
        "prediction_flag": "has_prediction",
    },
    "rna22": {
        "relative_path": Path("rna22/parsed/rna22_scores_aligned_to_evidence.csv.gz"),
        "score_candidates": (
            ("rna22_best_energy_strength", True),
            ("rna22_mean_energy_strength", True),
            ("rna22_best_pvalue", False),
            ("rna22_n_3utr_sites", True),
        ),
        "prediction_flag": "has_prediction",
    },
    "diana_microt": {
        "relative_path": Path("diana_microt/parsed/diana_microt_scores_aligned_to_evidence.csv.gz"),
        "score_candidates": (("diana_microt_score", True),),
        "prediction_flag": "has_prediction",
    },
}
EXTERNAL_SCORE_CANDIDATES: Dict[str, Tuple[Tuple[str, bool], ...]] = {
    "diana_microt": (
        ("score", True),
        ("diana_score", True),
        ("microt_score", True),
        ("prediction_score", True),
    ),
    "miranda": (
        ("miranda_score", True),
        ("alignment_score", True),
        ("score", True),
        ("energy", False),
        ("dg_duplex", False),
    ),
    "pita": (
        ("score", True),
        ("pita_score", True),
        ("ddg", False),
        ("delta_g", False),
    ),
}
PLOT_LABELS = {
    "logistic": "Logistic Regression",
    "xgboost": "XGBoost",
    "svm": "SVM",
    "random_forest": "Random Forest",
    "mlp": "MLP",
    "naive_bayes": "Naive Bayes",
    "sequence_complementarity": "Sequence Complementarity",
    "thermodynamic_stability": "Thermodynamic Stability",
    "sequence_conservation": "Sequence Conservation",
    "target_site_accessibility": "Target-Site Accessibility",
    "functional_binding": "Functional Binding",
    "functional_repression": "Functional Repression",
    "full_model": "Full Model",
    "final_model": "miRAssist Final Model",
    "miRAssist": "miRAssist",
    "diana_microt": "DIANA-microT",
    "DIANA-microT": "DIANA-microT",
    "DIANA-MicroT": "DIANA-MicroT",
    "miranda": "miRanda",
    "miRanda": "miRanda",
    "pita": "PITA",
    "mirdb": "miRDB",
    "miRDB": "miRDB",
    "rna22": "RNA22",
    "RNA22": "RNA22",
    "targetscan": "TargetScan",
    "TargetScan": "TargetScan",
}
METHOD_TYPE_COLORS = {
    "final_model": "#0b5d8f",
    "evidence_family": "#c97b00",
    "published_model": "#4a8f29",
}


def clean_plot_label(value: Any) -> str:
    text = str(value or "").strip()
    if text in PLOT_LABELS:
        return PLOT_LABELS[text]
    text = text.replace("_", " ").strip()
    return " ".join(part.capitalize() if part else part for part in text.split())


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def json_dump(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(_sanitize_json(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _sanitize_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _sanitize_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return _sanitize_json(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _sanitize_json(value.tolist())
    if isinstance(value, np.ndarray):
        return _sanitize_json(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return _sanitize_json(value.item())
    return str(value)


def module_available(module_name: str) -> bool:
    return bool(importlib.util.find_spec(module_name))


def read_table(path: str | Path) -> pd.DataFrame:
    resolved = Path(path).resolve()
    suffix = resolved.suffix.lower()
    suffixes = [part.lower() for part in resolved.suffixes]
    if suffix == ".csv" or suffixes[-2:] == [".csv", ".gz"]:
        return pd.read_csv(resolved)
    if suffix == ".tsv" or suffixes[-2:] == [".tsv", ".gz"]:
        return pd.read_csv(resolved, sep="\t")
    if suffix == ".parquet":
        try:
            return pd.read_parquet(resolved)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read parquet file {resolved}. Install pyarrow or fastparquet in the active environment."
            ) from exc
    raise ValueError(f"Unsupported table format: {resolved}")


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    suffix = resolved.suffix.lower()
    if suffix == ".csv":
        df.to_csv(resolved, index=False)
        return
    if suffix == ".parquet":
        df.to_parquet(resolved, index=False)
        return
    raise ValueError(f"Unsupported output format: {resolved}")


_MIRNA_HYPHEN_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212\uFE58\uFE63\uFF0D]")
_MIRNA_SPACE_RE = re.compile(r"[\s_]+")
_MIRNA_REPEAT_DASH_RE = re.compile(r"-{2,}")
_MIRNA_MICRORNA_RE = re.compile(r"micro[\s_-]*rna", re.IGNORECASE)
_MIRNA_MIRNA_RE = re.compile(r"mi[\s_-]*rna", re.IGNORECASE)
_MIRNA_PREFIX_RE = re.compile(r"^(?:hsa-)+", re.IGNORECASE)
_MIRNA_CORE_RE = re.compile(r"^(?:mir-?)(.+)$", re.IGNORECASE)


def normalize_mirna_name(value: Any) -> str:
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


def normalize_gene_symbol(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip().upper()


def normalize_series_cached(series: pd.Series, normalizer: Callable[[Any], str]) -> pd.Series:
    values = series.fillna("").astype(str)
    unique_values = pd.Index(values.drop_duplicates())
    lookup = {value: normalizer(value) for value in unique_values}
    return values.map(lookup)


def safe_float_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default).astype(float)


def safe_binary_series(df: pd.DataFrame, column: str) -> pd.Series:
    return (safe_float_series(df, column, default=0.0) > 0).astype(int)


def compute_rank_percentile(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    ranks = numeric.rank(method="average", pct=True, ascending=not higher_is_better)
    return ranks.fillna(0.0).astype(float)


def aggregate_directional_score(df: pd.DataFrame, column_specs: Sequence[Tuple[str, bool]]) -> pd.Series:
    components: List[pd.Series] = []
    for column_name, higher_is_better in column_specs:
        if column_name not in df.columns:
            continue
        numeric = pd.to_numeric(df[column_name], errors="coerce")
        if numeric.notna().sum() == 0:
            continue
        components.append(compute_rank_percentile(numeric, higher_is_better=higher_is_better))
    if not components:
        return pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    return pd.concat(components, axis=1).mean(axis=1).fillna(0.0).astype(float)


def seed_type_score(series: pd.Series) -> pd.Series:
    def _one(value: Any) -> float:
        text = str(value or "").strip().lower().replace("_", "-")
        for token, score in SEED_CLASS_STRENGTH.items():
            if token in text:
                return score
        return 0.0

    return series.map(_one).fillna(0.0).astype(float)


def get_feature_family_map() -> Dict[str, List[str]]:
    return {
        "sequence_complementarity": [
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
        "thermodynamic_stability": [
            "rnahybrid_mfe",
            "rnahybrid_mfe_best_site",
            "rnahybrid_site_start",
            "rnahybrid_site_end",
            "rnahybrid_seed_mfe",
            "rnahybrid_strength",
            "has_rnahybrid_evidence",
        ],
        "sequence_conservation": [
            "targetscan_context_score",
            "targetscan_context_score_percentile",
            "targetscan_aggregate_context_score",
            "targetscan_conserved_site",
            "targetscan_pct",
            "targetscan_branch_length_score",
            "has_targetscan_evidence",
        ],
        "target_site_accessibility": [
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
        "functional_binding": [
            "clip_any_support",
            "clip_max_score",
            "clip_n_experiments",
            "clip_n_cell_lines",
            "encori_clip_score",
            "has_clip_evidence",
        ],
        "functional_repression": [
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
    }


def get_feature_columns_for_families(
    df: pd.DataFrame,
    *,
    include_families: Optional[Sequence[str]] = None,
    exclude_families: Optional[Sequence[str]] = None,
) -> List[str]:
    family_map = get_feature_family_map()
    families = list(include_families) if include_families is not None else list(family_map.keys())
    if exclude_families:
        families = [family for family in families if family not in set(exclude_families)]
    selected: List[str] = []
    for family in families:
        for column in family_map.get(family, []):
            if column in df.columns and column not in selected:
                selected.append(column)
    return selected


def detect_leakage_columns(columns: Iterable[str]) -> List[str]:
    leaked: List[str] = []
    for column in columns:
        lower = str(column).lower()
        if any(token in lower for token in LEAKAGE_TOKENS):
            leaked.append(str(column))
    return sorted(set(leaked))


def resolve_clean_evidence_path(explicit: Optional[str] = None) -> Path:
    if explicit:
        path = Path(explicit).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Clean evidence input does not exist: {path}")
        return path
    path = DEFAULT_CLEAN_EVIDENCE_PATH.resolve()
    if path.exists():
        return path
    raise FileNotFoundError(
        f"Clean evidence input was not found at {path}. Provide --evidence explicitly."
    )


def resolve_label_path(explicit: Optional[str] = None) -> Path:
    if explicit:
        path = Path(explicit).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Label input does not exist: {path}")
        return path
    for candidate in LABEL_SOURCE_CANDIDATES:
        path = Path(candidate).resolve()
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find a held-out miRTarBase label table. Provide --labels explicitly."
    )


def resolve_source_feature_path(explicit: Optional[str] = None) -> Optional[Path]:
    if explicit:
        path = Path(explicit).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Source feature table does not exist: {path}")
        return path
    for candidate in MIRDB_SOURCE_CANDIDATES:
        path = Path(candidate).resolve()
        if path.exists():
            return path
    return None


def prepare_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "mirna_name_normalized" not in out.columns:
        if "mirna_name_norm" in out.columns:
            out["mirna_name_normalized"] = out["mirna_name_norm"].fillna("").astype(str)
        elif "mirna_name" in out.columns:
            out["mirna_name_normalized"] = normalize_series_cached(out["mirna_name"], normalize_mirna_name)
        else:
            out["mirna_name_normalized"] = ""
    if "gene_symbol_normalized" not in out.columns:
        if "gene_symbol_norm" in out.columns:
            out["gene_symbol_normalized"] = out["gene_symbol_norm"].fillna("").astype(str)
        elif "gene_symbol" in out.columns:
            out["gene_symbol_normalized"] = normalize_series_cached(out["gene_symbol"], normalize_gene_symbol)
        else:
            out["gene_symbol_normalized"] = ""
    if "query_group" not in out.columns:
        out["query_group"] = out["mirna_name_normalized"].fillna("").astype(str)
    return out


def build_label_lookup(labels_df: pd.DataFrame) -> pd.DataFrame:
    labels = prepare_keys(labels_df)
    transcript_col = "transcript_id" if "transcript_id" in labels.columns else None
    labels["heldout_mirtarbase_pos"] = safe_float_series(labels, "mirtarbase_pos", default=0.0).astype(int)
    labels["heldout_label_mirtarbase"] = safe_float_series(labels, "label_mirtarbase", default=0.0).astype(int)
    labels["is_positive"] = (
        (labels["heldout_mirtarbase_pos"] > 0)
        | (labels["heldout_label_mirtarbase"] > 0)
    ).astype(int)
    group_keys = ["mirna_name_normalized", "gene_symbol_normalized"]
    if transcript_col is not None:
        labels["transcript_id"] = labels["transcript_id"].fillna("").astype(str)
        group_keys.append("transcript_id")
    grouped = labels.groupby(group_keys, dropna=False, as_index=False)[
        ["heldout_mirtarbase_pos", "heldout_label_mirtarbase", "is_positive"]
    ].max()
    return grouped


def build_label_lookup_by_row_id(labels_df: pd.DataFrame) -> pd.DataFrame:
    labels = labels_df.copy()
    if "eval_row_id" not in labels.columns:
        return pd.DataFrame(columns=["eval_row_id", "heldout_mirtarbase_pos", "heldout_label_mirtarbase", "is_positive"])
    labels["eval_row_id"] = labels["eval_row_id"].fillna("").astype(str)
    labels["heldout_mirtarbase_pos"] = safe_float_series(labels, "mirtarbase_pos", default=0.0).astype(int)
    labels["heldout_label_mirtarbase"] = safe_float_series(labels, "label_mirtarbase", default=0.0).astype(int)
    labels["is_positive"] = (
        (labels["heldout_mirtarbase_pos"] > 0)
        | (labels["heldout_label_mirtarbase"] > 0)
    ).astype(int)
    return (
        labels.groupby("eval_row_id", as_index=False)[
            ["heldout_mirtarbase_pos", "heldout_label_mirtarbase", "is_positive"]
        ]
        .max()
    )


def attach_labels(evidence_df: pd.DataFrame, labels_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    evidence = prepare_keys(evidence_df)
    row_id_lookup = build_label_lookup_by_row_id(labels_df)
    label_lookup = build_label_lookup(labels_df)
    join_mode = "normalized_keys"
    row_id_matches = 0

    if "evidence_row_id" in evidence.columns and not row_id_lookup.empty:
        evidence["evidence_row_id"] = evidence["evidence_row_id"].fillna("").astype(str)
        merged = evidence.merge(
            row_id_lookup,
            how="left",
            left_on="evidence_row_id",
            right_on="eval_row_id",
        )
        row_id_matches = int(merged["eval_row_id"].fillna("").astype(str).ne("").sum())
        join_mode = "evidence_row_id_to_eval_row_id"
    elif "transcript_id" in evidence.columns and "transcript_id" in label_lookup.columns:
        evidence["transcript_id"] = evidence["transcript_id"].fillna("").astype(str)
        merged = evidence.merge(
            label_lookup,
            how="left",
            on=["mirna_name_normalized", "gene_symbol_normalized", "transcript_id"],
        )
    else:
        merged = evidence.merge(
            label_lookup,
            how="left",
            on=["mirna_name_normalized", "gene_symbol_normalized"],
        )

    if "heldout_mirtarbase_pos" not in merged.columns:
        merged["heldout_mirtarbase_pos"] = 0
    if "heldout_label_mirtarbase" not in merged.columns:
        merged["heldout_label_mirtarbase"] = 0
    if "is_positive" not in merged.columns:
        merged["is_positive"] = 0

    if row_id_matches < len(evidence):
        if "eval_row_id" in merged.columns:
            missing_mask = merged["eval_row_id"].fillna("").astype(str).eq("")
        else:
            missing_mask = safe_float_series(merged, "is_positive", default=0.0).astype(int) <= 0
        fallback_merged = None
        if "transcript_id" in evidence.columns and "transcript_id" in label_lookup.columns:
            fallback_lookup = label_lookup.copy()
            evidence["transcript_id"] = evidence["transcript_id"].fillna("").astype(str)
            fallback_merged = evidence.merge(
                fallback_lookup,
                how="left",
                on=["mirna_name_normalized", "gene_symbol_normalized", "transcript_id"],
            )
        else:
            fallback_merged = evidence.merge(
                label_lookup,
                how="left",
                on=["mirna_name_normalized", "gene_symbol_normalized"],
            )
        for column in ("heldout_mirtarbase_pos", "heldout_label_mirtarbase", "is_positive"):
            merged.loc[missing_mask, column] = safe_float_series(fallback_merged.loc[missing_mask], column, default=0.0).astype(int)
        if row_id_matches > 0:
            join_mode = "row_id_with_normalized_key_fallback"

    merged["heldout_mirtarbase_pos"] = safe_float_series(merged, "heldout_mirtarbase_pos", default=0.0).astype(int)
    merged["heldout_label_mirtarbase"] = safe_float_series(merged, "heldout_label_mirtarbase", default=0.0).astype(int)
    merged["is_positive"] = (
        (merged["heldout_mirtarbase_pos"] > 0) | (merged["heldout_label_mirtarbase"] > 0)
    ).astype(int)

    diagnostics = {
        "n_rows": int(len(merged)),
        "n_positives": int(merged["is_positive"].sum()),
        "positive_rate": float(merged["is_positive"].mean()) if len(merged) else 0.0,
        "n_unique_queries": int(merged["query_group"].replace("", pd.NA).nunique(dropna=True)),
        "label_join_mode": join_mode,
        "row_id_label_matches": int(row_id_matches),
        "preferred_label_column": "mirtarbase_pos" if "mirtarbase_pos" in labels_df.columns else (
            "label_mirtarbase" if "label_mirtarbase" in labels_df.columns else None
        ),
    }
    return merged, diagnostics


def load_labeled_clean_evidence(
    evidence_path: Optional[str] = None,
    labels_path: Optional[str] = None,
    *,
    limit_rows: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    evidence_resolved = resolve_clean_evidence_path(evidence_path)
    labels_resolved = resolve_label_path(labels_path)
    evidence_df = read_table(evidence_resolved)
    if limit_rows is not None and limit_rows > 0:
        evidence_df = evidence_df.head(int(limit_rows)).copy()
    labels_df = read_table(labels_resolved)
    merged, diagnostics = attach_labels(evidence_df, labels_df)
    diagnostics["evidence_path"] = str(evidence_resolved)
    diagnostics["labels_path"] = str(labels_resolved)
    diagnostics["limit_rows"] = int(limit_rows) if limit_rows is not None else None
    return merged, diagnostics


def split_train_test_groups(
    df: pd.DataFrame,
    *,
    group_column: str = "query_group",
    test_size: float = 0.2,
    seed: int = 2026,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if not module_available("sklearn"):
        raise RuntimeError("scikit-learn is required to perform grouped train/test splits.")
    from sklearn.model_selection import GroupShuffleSplit

    groups = df[group_column].fillna("").astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    diagnostics = {
        "split_strategy": f"grouped by {group_column}",
        "group_column": group_column,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_queries": int(train_df[group_column].nunique()),
        "test_queries": int(test_df[group_column].nunique()),
    }
    return train_df, test_df, diagnostics


def build_feature_frame(
    df: pd.DataFrame,
    *,
    include_families: Optional[Sequence[str]] = None,
    exclude_families: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    feature_columns = get_feature_columns_for_families(df, include_families=include_families, exclude_families=exclude_families)
    leakage_columns = detect_leakage_columns(feature_columns)
    if leakage_columns:
        raise RuntimeError(
            "Leakage features were selected for the clean evidence evaluation: " + ", ".join(leakage_columns)
        )
    if not feature_columns:
        raise RuntimeError("No clean evidence feature columns were available for the selected evidence families.")
    feature_df = df.loc[:, feature_columns].copy()
    categorical_columns = [
        column
        for column in feature_columns
        if feature_df[column].dtype == "object" or pd.api.types.is_string_dtype(feature_df[column])
    ]
    boolean_columns = [
        column
        for column in feature_columns
        if column not in categorical_columns and (
            pd.api.types.is_bool_dtype(feature_df[column])
            or set(pd.to_numeric(feature_df[column], errors="coerce").dropna().unique()).issubset({0, 1})
        )
    ]
    numeric_columns = [column for column in feature_columns if column not in categorical_columns]
    feature_info = {
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "boolean_columns": boolean_columns,
        "evidence_families": list(include_families) if include_families is not None else [
            family for family in get_feature_family_map().keys() if family not in set(exclude_families or [])
        ],
    }
    return feature_df, feature_info


def require_backend_model_dependencies(model_name: str) -> None:
    if not module_available("sklearn"):
        raise RuntimeError("scikit-learn is required for backend model comparison.")
    if model_name == "xgboost" and not module_available("xgboost"):
        raise RuntimeError("xgboost is required for the XGBoost backend model.")


def build_estimator(model_name: str, *, seed: int = 2026, positive_rate: float = 0.5):
    require_backend_model_dependencies(model_name)
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    def _preprocessor(feature_df: pd.DataFrame):
        categorical_columns = [
            column
            for column in feature_df.columns
            if feature_df[column].dtype == "object" or pd.api.types.is_string_dtype(feature_df[column])
        ]
        numeric_columns = [column for column in feature_df.columns if column not in categorical_columns]
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, numeric_columns),
                ("categorical", categorical_transformer, categorical_columns),
            ],
            sparse_threshold=0.0,
        )

    if model_name == "logistic":
        from sklearn.linear_model import LogisticRegression

        def _factory(feature_df: pd.DataFrame):
            return Pipeline(
                steps=[
                    ("preprocessor", _preprocessor(feature_df)),
                    ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
                ]
            )

        return _factory
    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        def _factory(feature_df: pd.DataFrame):
            return Pipeline(
                steps=[
                    ("preprocessor", _preprocessor(feature_df)),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=300,
                            max_depth=16,
                            min_samples_leaf=10,
                            max_features="sqrt",
                            bootstrap=True,
                            max_samples=0.5,
                            class_weight="balanced_subsample",
                            n_jobs=-1,
                            random_state=seed,
                        ),
                    ),
                ]
            )

        return _factory
    if model_name == "svm":
        from sklearn.svm import LinearSVC

        def _factory(feature_df: pd.DataFrame):
            return Pipeline(
                steps=[
                    ("preprocessor", _preprocessor(feature_df)),
                    (
                        "model",
                        LinearSVC(
                            class_weight="balanced",
                            C=1.0,
                            random_state=seed,
                            max_iter=10000,
                        ),
                    ),
                ]
            )

        return _factory
    if model_name == "mlp":
        from sklearn.neural_network import MLPClassifier

        def _factory(feature_df: pd.DataFrame):
            return Pipeline(
                steps=[
                    ("preprocessor", _preprocessor(feature_df)),
                    ("model", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=seed)),
                ]
            )

        return _factory
    if model_name == "naive_bayes":
        from sklearn.naive_bayes import GaussianNB

        def _factory(feature_df: pd.DataFrame):
            return Pipeline(
                steps=[
                    ("preprocessor", _preprocessor(feature_df)),
                    ("model", GaussianNB()),
                ]
            )

        return _factory
    if model_name == "xgboost":
        from xgboost import XGBClassifier

        scale_pos_weight = float((1.0 - positive_rate) / max(positive_rate, 1e-6))

        def _factory(feature_df: pd.DataFrame):
            return Pipeline(
                steps=[
                    ("preprocessor", _preprocessor(feature_df)),
                    (
                        "model",
                        XGBClassifier(
                            n_estimators=300,
                            max_depth=4,
                            learning_rate=0.05,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            objective="binary:logistic",
                            eval_metric="logloss",
                            random_state=seed,
                            scale_pos_weight=scale_pos_weight,
                        ),
                    ),
                ]
            )

        return _factory
    raise ValueError(f"Unsupported backend model: {model_name}")


def fit_and_score_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    model_name: str,
    include_families: Optional[Sequence[str]] = None,
    exclude_families: Optional[Sequence[str]] = None,
    seed: int = 2026,
) -> Tuple[Any, pd.DataFrame, Dict[str, Any]]:
    feature_train, feature_info = build_feature_frame(
        train_df,
        include_families=include_families,
        exclude_families=exclude_families,
    )
    feature_test = test_df.loc[:, feature_info["feature_columns"]].copy()

    builder = build_estimator(
        model_name,
        seed=seed,
        positive_rate=float(train_df["is_positive"].mean()) if len(train_df) else 0.5,
    )
    estimator = builder(feature_train)
    estimator.fit(feature_train, train_df["is_positive"].astype(int))

    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(feature_test)[:, 1]
    elif hasattr(estimator, "decision_function"):
        decision = estimator.decision_function(feature_test)
        scores = 1.0 / (1.0 + np.exp(-np.asarray(decision, dtype=float)))
    else:
        scores = estimator.predict(feature_test).astype(float)
    predicted = (scores >= 0.5).astype(int)

    predictions = prepare_keys(test_df).copy()
    predictions["model_name"] = model_name
    predictions["score"] = np.asarray(scores, dtype=float)
    predictions["predicted_label"] = np.asarray(predicted, dtype=int)
    predictions["evaluation_role"] = "test"
    predictions["evidence_family_mode"] = (
        ",".join(include_families)
        if include_families is not None
        else "all_minus_" + ",".join(exclude_families or [])
        if exclude_families
        else "all"
    )

    metrics = compute_classification_and_ranking_metrics(predictions, score_column="score")
    metrics["model_name"] = model_name
    metrics["n_train_rows"] = int(len(train_df))
    metrics["n_test_rows"] = int(len(test_df))
    metrics["n_train_positives"] = int(train_df["is_positive"].sum())
    metrics["n_test_positives"] = int(test_df["is_positive"].sum())
    metrics["feature_column_count"] = int(len(feature_info["feature_columns"]))
    return estimator, predictions, {**metrics, **feature_info}


def compute_classification_and_ranking_metrics(
    df: pd.DataFrame,
    *,
    score_column: str = "score",
    label_column: str = "is_positive",
    group_column: str = "query_group",
    ks: Sequence[int] = DEFAULT_K_VALUES,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    y_true = df[label_column].astype(int).tolist()
    y_score = pd.to_numeric(df[score_column], errors="coerce").fillna(0.0).tolist()
    out["auroc"] = roc_auc_score_manual(y_true, y_score)
    out["pr_auc"] = average_precision_manual(y_true, y_score)

    rankings = compute_group_ranking_metrics(
        df,
        score_column=score_column,
        label_column=label_column,
        group_column=group_column,
        ks=ks,
    )
    out.update(rankings["summary"])
    return out


def compute_group_ranking_metrics(
    df: pd.DataFrame,
    *,
    score_column: str = "score",
    label_column: str = "is_positive",
    group_column: str = "query_group",
    ks: Sequence[int] = DEFAULT_K_VALUES,
) -> Dict[str, Any]:
    ranked = prepare_keys(df).copy()
    ranked = ranked.sort_values([group_column, score_column], ascending=[True, False]).copy()
    ranked["rank"] = ranked.groupby(group_column).cumcount() + 1
    query_rows: List[Dict[str, Any]] = []
    for query_id, query_df in ranked.groupby(group_column, dropna=False):
        total_positives = int(query_df[label_column].sum())
        row: Dict[str, Any] = {
            "query_group": query_id,
            "n_ranked": int(len(query_df)),
            "n_positives_total": total_positives,
        }
        for k in ks:
            topk = query_df[query_df["rank"] <= k]
            positives_in_topk = int(topk[label_column].sum())
            row[f"recall_at_{k}"] = (
                positives_in_topk / total_positives if total_positives > 0 else np.nan
            )
            row[f"precision_at_{k}"] = positives_in_topk / float(min(k, len(query_df))) if len(query_df) else np.nan
        query_rows.append(row)
    query_metrics = pd.DataFrame(query_rows)
    summary: Dict[str, Any] = {}
    for k in ks:
        summary[f"recall_at_{k}"] = float(pd.to_numeric(query_metrics[f"recall_at_{k}"], errors="coerce").dropna().mean()) if not query_metrics.empty else np.nan
        summary[f"precision_at_{k}"] = float(pd.to_numeric(query_metrics[f"precision_at_{k}"], errors="coerce").dropna().mean()) if not query_metrics.empty else np.nan
    return {"ranked": ranked, "query_metrics": query_metrics, "summary": summary}


def sort_models_for_selection(metrics_df: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "recall_at_10",
        "precision_at_1",
        "precision_at_3",
        "precision_at_5",
        "precision_at_10",
        "pr_auc",
        "auroc",
    ]
    available = [column for column in sort_columns if column in metrics_df.columns]
    return metrics_df.sort_values(available, ascending=[False] * len(available)).reset_index(drop=True)


def pick_best_model(metrics_df: pd.DataFrame) -> pd.Series:
    available = metrics_df[metrics_df["status"] == "ok"].copy() if "status" in metrics_df.columns else metrics_df.copy()
    if available.empty:
        raise RuntimeError("No successful backend models were available for selection.")
    ranked = sort_models_for_selection(available)
    return ranked.iloc[0]


def save_pickle(path: str | Path, payload: Any) -> None:
    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("wb") as handle:
        pickle.dump(payload, handle)


def load_pickle(path: str | Path) -> Any:
    with Path(path).resolve().open("rb") as handle:
        return pickle.load(handle)


def make_output_dirs(root: str | Path = DEFAULT_OUTPUT_ROOT) -> Dict[str, Path]:
    root_path = ensure_dir(root)
    return {
        "root": root_path,
        "results": ensure_dir(root_path / "results"),
        "figures": ensure_dir(root_path / "figures"),
        "models": ensure_dir(root_path / "models"),
        "logs": ensure_dir(root_path / "logs"),
        "external_models": ensure_dir(root_path / "external_models"),
    }


def summarise_dataset(df: pd.DataFrame, split_strategy: str) -> Dict[str, Any]:
    return {
        "rows": int(len(df)),
        "positives": int(df["is_positive"].sum()),
        "positive_rate": float(df["is_positive"].mean()) if len(df) else 0.0,
        "unique_queries": int(df["query_group"].replace("", pd.NA).nunique(dropna=True)),
        "split_strategy": split_strategy,
    }


def score_estimator_on_frame(
    estimator: Any,
    test_df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    model_name: str,
    evidence_family_mode: str = "all",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    feature_test = test_df.loc[:, list(feature_columns)].copy()
    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(feature_test)[:, 1]
    elif hasattr(estimator, "decision_function"):
        decision = estimator.decision_function(feature_test)
        scores = 1.0 / (1.0 + np.exp(-np.asarray(decision, dtype=float)))
    else:
        scores = estimator.predict(feature_test).astype(float)
    predicted = (np.asarray(scores, dtype=float) >= 0.5).astype(int)

    predictions = prepare_keys(test_df).copy()
    predictions["model_name"] = model_name
    predictions["score"] = np.asarray(scores, dtype=float)
    predictions["predicted_label"] = np.asarray(predicted, dtype=int)
    predictions["evaluation_role"] = "test"
    predictions["evidence_family_mode"] = evidence_family_mode

    metrics = compute_classification_and_ranking_metrics(predictions, score_column="score")
    metrics["model_name"] = model_name
    metrics["n_test_rows"] = int(len(test_df))
    metrics["n_test_positives"] = int(test_df["is_positive"].sum())
    metrics["feature_column_count"] = int(len(feature_columns))
    return predictions, metrics


def _select_first_available_score(
    df: pd.DataFrame,
    candidates: Sequence[Tuple[str, bool]],
    *,
    method_name: str,
    source_path: Optional[str] = None,
    extra_notes: Optional[Sequence[str]] = None,
) -> Tuple[pd.Series, Dict[str, Any]]:
    for column_name, higher_is_stronger in candidates:
        if column_name not in df.columns:
            continue
        numeric = pd.to_numeric(df[column_name], errors="coerce")
        if numeric.notna().sum() == 0:
            continue
        score = numeric.astype(float)
        if not higher_is_stronger:
            score = -1.0 * score
        diagnostics = {
            "method_name": method_name,
            "score_column": column_name,
            "score_columns": [column_name],
            "higher_is_stronger": True,
            "raw_higher_is_stronger": bool(higher_is_stronger),
            "source_path": source_path,
            "notes": list(extra_notes or []),
            "n_non_missing_raw": int(numeric.notna().sum()),
        }
        return score.fillna(0.0), diagnostics
    raise RuntimeError(f"No usable score column was found for {method_name}. Looked for {[name for name, _ in candidates]}.")


def select_targetscan_primary_score(df: pd.DataFrame) -> Tuple[pd.Series, Dict[str, Any]]:
    return _select_first_available_score(
        df,
        [
            ("targetscan_context_score", False),
            ("targetscan_context_score_percentile", True),
            ("targetscan_pct", True),
            ("targetscan_branch_length_score", True),
            ("targetscan_aggregate_context_score", True),
        ],
        method_name="targetscan",
        extra_notes=[
            "Higher standardized scores indicate stronger TargetScan support.",
            "If the raw context score is used, the sign is flipped because more negative is stronger.",
        ],
    )


def select_family_score(df: pd.DataFrame, family_name: str) -> Tuple[pd.Series, Dict[str, Any]]:
    if family_name == "sequence_complementarity":
        components: List[pd.Series] = []
        selected_columns: List[str] = []
        notes = ["Composite seed score using strongest available seed-class and site-support columns."]
        seed_col = "best_seed_site_type" if "best_seed_site_type" in df.columns else "seed_match_type" if "seed_match_type" in df.columns else None
        if seed_col is not None:
            components.append(seed_type_score(df[seed_col]))
            selected_columns.append(seed_col)
        for column_name, weight in [("is_8mer", 1.0), ("is_7mer_m8", 0.8), ("is_7mer_a1", 0.7), ("is_6mer", 0.5)]:
            if column_name in df.columns:
                components.append(safe_float_series(df, column_name) * weight)
                selected_columns.append(column_name)
        for column_name in ("seed_pairing_score", "n_seed_sites"):
            if column_name in df.columns:
                components.append(compute_rank_percentile(safe_float_series(df, column_name), higher_is_better=True))
                selected_columns.append(column_name)
        if not components:
            raise RuntimeError("No usable sequence-complementarity columns were found.")
        return (
            pd.concat(components, axis=1).mean(axis=1).fillna(0.0),
            {
                "method_name": family_name,
                "score_column": "|".join(selected_columns),
                "score_columns": selected_columns,
                "higher_is_stronger": True,
                "raw_higher_is_stronger": True,
                "notes": notes,
                "source_path": None,
                "n_non_missing_raw": int(len(df)),
            },
        )
    if family_name == "thermodynamic_stability":
        score, info = _select_first_available_score(
            df,
            [
                ("rnahybrid_strength", True),
                ("rnahybrid_mfe", False),
                ("rnahybrid_seed_mfe", False),
            ],
            method_name=family_name,
        )
        info["method_name"] = family_name
        info["notes"] = ["Higher standardized scores indicate stronger RNAhybrid thermodynamic support."]
        return score, info
    if family_name == "sequence_conservation":
        score, info = select_targetscan_primary_score(df)
        info["method_name"] = family_name
        info["notes"] = ["TargetScan-based conservation score reuses the strongest available clean-evidence TargetScan column."]
        return score, info
    if family_name == "target_site_accessibility":
        score, info = _select_first_available_score(
            df,
            [
                ("rnaplfold_best_seed_unpaired_prob", True),
                ("rnaplfold_best_site_unpaired_prob", True),
                ("rnaplfold_best_flank_unpaired_prob", True),
                ("rnaplfold_mean_seed_unpaired_prob", True),
                ("rnaplfold_mean_site_unpaired_prob", True),
                ("rnaplfold_mean_flank_unpaired_prob", True),
                ("rnaplfold_n_accessible_sites", True),
            ],
            method_name=family_name,
        )
        info["method_name"] = family_name
        info["notes"] = ["Higher unpaired probability or accessible-site count means stronger accessibility support."]
        return score, info
    if family_name == "functional_binding":
        score, info = _select_first_available_score(
            df,
            [
                ("clip_max_score", True),
                ("encori_clip_score", True),
                ("clip_n_experiments", True),
                ("clip_n_cell_lines", True),
                ("clip_any_support", True),
            ],
            method_name=family_name,
        )
        info["method_name"] = family_name
        info["notes"] = ["Higher CLIP score, experiment count, or support flag indicates stronger functional binding support."]
        return score, info
    if family_name == "functional_repression":
        score, info = _select_first_available_score(
            df,
            [
                ("tcga_best_repression_evidence", True),
                ("tcga_n_supported_contexts", True),
                ("tcga_mean_spearman_rho", False),
                ("BRCA_spearman_rho", False),
                ("PRAD_spearman_rho", False),
                ("COAD_spearman_rho", False),
                ("BRCA_repression_evidence", True),
                ("PRAD_repression_evidence", True),
                ("COAD_repression_evidence", True),
            ],
            method_name=family_name,
        )
        info["method_name"] = family_name
        info["notes"] = ["More negative repression-consistent Spearman rho values are sign-flipped so higher means stronger repression support."]
        return score, info
    raise ValueError(f"Unsupported evidence family: {family_name}")


def family_only_score(df: pd.DataFrame, family_name: str) -> pd.Series:
    score, _ = select_family_score(df, family_name)
    return score


def evaluate_score_series(
    df: pd.DataFrame,
    score: pd.Series,
    *,
    comparator_name: str,
    ks: Sequence[int] = DEFAULT_K_VALUES,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    pred_df = prepare_keys(df).copy()
    pred_df["model_name"] = comparator_name
    pred_df["score"] = pd.to_numeric(score, errors="coerce").fillna(0.0).astype(float)
    pred_df["evaluation_role"] = "test"
    metrics = compute_classification_and_ranking_metrics(pred_df, score_column="score", ks=ks)
    metrics["model_name"] = comparator_name
    return pred_df, metrics


def resolve_external_model_paths(
    external_model_dir: str | Path,
    *,
    explicit_paths: Optional[Dict[str, Optional[str]]] = None,
) -> Tuple[Dict[str, Optional[Path]], Dict[str, Any]]:
    base_dir = ensure_dir(external_model_dir)
    paths: Dict[str, Optional[Path]] = {}
    status: Dict[str, Any] = {}
    for model_name, patterns in EXTERNAL_MODEL_PATTERNS.items():
        explicit = None if explicit_paths is None else explicit_paths.get(model_name)
        looked_at: List[str] = []
        resolved: Optional[Path] = None
        if explicit:
            candidate = Path(explicit).resolve()
            looked_at.append(str(candidate))
            if candidate.exists():
                resolved = candidate
        if resolved is None:
            for pattern in patterns:
                for candidate in base_dir.rglob(pattern):
                    looked_at.append(str(candidate.resolve()))
                    resolved = candidate.resolve()
                    break
                if resolved is not None:
                    break
        paths[model_name] = resolved
        status[model_name] = {
            "path": str(resolved) if resolved is not None else None,
            "looked_at": looked_at or [str(base_dir.resolve() / pattern) for pattern in patterns],
            "expected_formats": [".csv", ".tsv", ".parquet"],
            "available": resolved is not None,
        }
    return paths, status


def resolve_aligned_external_model_paths(
    external_root: str | Path,
    *,
    explicit_paths: Optional[Dict[str, Optional[str]]] = None,
) -> Tuple[Dict[str, Optional[Path]], Dict[str, Any]]:
    root = Path(external_root).resolve()
    paths: Dict[str, Optional[Path]] = {}
    status: Dict[str, Any] = {}
    for model_name, spec in ALIGNED_EXTERNAL_MODEL_SPECS.items():
        explicit = None if explicit_paths is None else explicit_paths.get(model_name)
        if explicit:
            resolved = Path(explicit).resolve()
            available = resolved.exists()
            paths[model_name] = resolved if available else None
            status[model_name] = {
                "path": str(resolved),
                "available": available,
                "looked_at": [str(resolved)],
                "score_candidates": [name for name, _ in spec["score_candidates"]],
                "prediction_flag": spec.get("prediction_flag"),
            }
            continue
        candidate = (root / spec["relative_path"]).resolve()
        available = candidate.exists()
        paths[model_name] = candidate if available else None
        status[model_name] = {
            "path": str(candidate),
            "available": available,
            "looked_at": [str(candidate)],
            "score_candidates": [name for name, _ in spec["score_candidates"]],
            "prediction_flag": spec.get("prediction_flag"),
        }
    return paths, status


def join_external_score_table(
    test_df: pd.DataFrame,
    external_df: pd.DataFrame,
    *,
    model_name: str,
    score_candidates: Sequence[Tuple[str, bool]],
) -> Tuple[pd.Series, Dict[str, Any]]:
    test_keys = prepare_keys(test_df)
    ext = prepare_keys(external_df)
    if "transcript_id" in ext.columns:
        ext["transcript_id"] = ext["transcript_id"].fillna("").astype(str)
    if "transcript_id" in test_keys.columns:
        test_keys["transcript_id"] = test_keys["transcript_id"].fillna("").astype(str)

    selected_score_col = None
    higher_is_better = True
    for candidate, direction in score_candidates:
        if candidate in ext.columns:
            selected_score_col = candidate
            higher_is_better = direction
            break
    if selected_score_col is None:
        raise RuntimeError(
            f"No expected score column was found for {model_name}. Looked for {[name for name, _ in score_candidates]}."
        )

    join_keys = ["mirna_name_normalized", "gene_symbol_normalized"]
    if "transcript_id" in ext.columns and "transcript_id" in test_keys.columns:
        join_keys.append("transcript_id")
    ext_score = ext.loc[:, list(dict.fromkeys(join_keys + [selected_score_col]))].copy()
    ext_score = ext_score.drop_duplicates(subset=join_keys)
    merged = test_keys.merge(ext_score, how="left", on=join_keys)
    score = pd.to_numeric(merged[selected_score_col], errors="coerce")
    if not higher_is_better:
        score = -1.0 * score
    diagnostics = {
        "join_keys": join_keys,
        "score_column": selected_score_col,
        "higher_is_better": higher_is_better,
        "rows_with_score": int(score.notna().sum()),
    }
    return score.fillna(0.0), diagnostics


def join_row_aligned_score_table(
    test_df: pd.DataFrame,
    external_df: pd.DataFrame,
    *,
    model_name: str,
    score_candidates: Sequence[Tuple[str, bool]],
    prediction_flag_column: Optional[str] = None,
) -> Tuple[pd.Series, Dict[str, Any]]:
    if "evidence_row_id" not in test_df.columns:
        raise RuntimeError("Test dataframe does not contain evidence_row_id for row-aligned external score joins.")
    if "evidence_row_id" not in external_df.columns:
        raise RuntimeError(f"External score table for {model_name} does not contain evidence_row_id.")

    test_keys = test_df[["evidence_row_id"]].copy()
    test_keys["evidence_row_id"] = test_keys["evidence_row_id"].fillna("").astype(str)
    ext = external_df.copy()
    ext["evidence_row_id"] = ext["evidence_row_id"].fillna("").astype(str)

    selected_score_col = None
    higher_is_better = True
    for candidate, direction in score_candidates:
        if candidate in ext.columns:
            selected_score_col = candidate
            higher_is_better = direction
            break
    if selected_score_col is None:
        raise RuntimeError(
            f"No expected row-aligned score column was found for {model_name}. "
            f"Looked for {[name for name, _ in score_candidates]}."
        )

    selected_columns = ["evidence_row_id", selected_score_col]
    if prediction_flag_column and prediction_flag_column in ext.columns:
        selected_columns.append(prediction_flag_column)
    ext = ext.loc[:, list(dict.fromkeys(selected_columns))].drop_duplicates(subset=["evidence_row_id"])
    merged = test_keys.merge(ext, how="left", on="evidence_row_id")

    raw_score = pd.to_numeric(merged[selected_score_col], errors="coerce")
    if prediction_flag_column and prediction_flag_column in merged.columns:
        has_prediction = safe_float_series(merged, prediction_flag_column, default=0.0).astype(int) > 0
    else:
        has_prediction = raw_score.notna()
    score = raw_score.astype(float)
    if not higher_is_better:
        score = -1.0 * score

    diagnostics = {
        "join_keys": ["evidence_row_id"],
        "score_column": selected_score_col,
        "score_columns": [selected_score_col],
        "higher_is_stronger": True,
        "raw_higher_is_stronger": bool(higher_is_better),
        "prediction_flag_column": prediction_flag_column if prediction_flag_column in merged.columns else None,
        "n_predicted_rows": int(has_prediction.sum()),
        "prediction_coverage": float(has_prediction.mean()) if len(has_prediction) else 0.0,
        "rows_with_score": int(raw_score.notna().sum()),
    }
    return score.fillna(0.0), diagnostics


def load_mirdb_scores(
    test_df: pd.DataFrame,
    *,
    source_feature_path: Optional[str] = None,
) -> Tuple[pd.Series, Dict[str, Any]]:
    source_path = resolve_source_feature_path(source_feature_path)
    if source_path is None:
        raise RuntimeError(
            "Could not find a source backend feature table for miRDB comparison. "
            "Provide --source-features explicitly."
        )
    source_df = read_table(source_path)
    if "mirdb_best_score" not in source_df.columns and "mirdb_mean_score" not in source_df.columns:
        raise RuntimeError(
            f"The source feature table {source_path} does not contain miRDB score columns."
        )
    candidates = [("mirdb_best_score", True), ("mirdb_mean_score", True)]
    score, diagnostics = join_external_score_table(test_df, source_df, model_name="mirdb", score_candidates=candidates)
    diagnostics["source_path"] = str(source_path)
    return score, diagnostics


def load_row_aligned_external_scores(
    test_df: pd.DataFrame,
    *,
    model_name: str,
    path: str | Path,
) -> Tuple[pd.Series, Dict[str, Any]]:
    if model_name not in ALIGNED_EXTERNAL_MODEL_SPECS:
        raise RuntimeError(f"Unsupported row-aligned external model: {model_name}")
    spec = ALIGNED_EXTERNAL_MODEL_SPECS[model_name]
    external_df = read_table(path)
    score, diagnostics = join_row_aligned_score_table(
        test_df,
        external_df,
        model_name=model_name,
        score_candidates=tuple(spec["score_candidates"]),
        prediction_flag_column=spec.get("prediction_flag"),
    )
    diagnostics["source_path"] = str(Path(path).resolve())
    return score, diagnostics


def save_text(path: str | Path, text: str) -> None:
    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(text, encoding="utf-8")


def format_metric_block(metrics: Dict[str, Any], ks: Sequence[int] = DEFAULT_K_VALUES) -> str:
    lines = [
        f"AUROC: {metrics.get('auroc')}",
        f"PR-AUC: {metrics.get('pr_auc')}",
    ]
    for k in ks:
        lines.append(f"Recall at {k}: {metrics.get(f'recall_at_{k}')}")
        lines.append(f"Precision at {k}: {metrics.get(f'precision_at_{k}')}")
    return "\n".join(lines)


def require_matplotlib():
    if not module_available("matplotlib"):
        raise RuntimeError("matplotlib is required to generate publication-ready figures.")
    import matplotlib.pyplot as plt

    return plt


def save_bar_figure(
    df: pd.DataFrame,
    *,
    category_column: str,
    value_columns: Sequence[str],
    title: str,
    ylabel: str,
    output_prefix: str | Path,
) -> None:
    plt = require_matplotlib()
    if df.empty:
        return
    chart_df = df.copy()
    chart_df[category_column] = chart_df[category_column].map(clean_plot_label)
    x = np.arange(len(chart_df))
    width = 0.8 / max(1, len(value_columns))
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, column in enumerate(value_columns):
        positions = x + (idx - (len(value_columns) - 1) / 2.0) * width
        ax.bar(positions, chart_df[column].astype(float), width=width, label=clean_plot_label(column))
    ax.set_xticks(x)
    ax.set_xticklabels(chart_df[category_column], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    prefix = Path(output_prefix).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_method_metric_bar_figure(
    df: pd.DataFrame,
    *,
    metric_column: str,
    output_prefix: str | Path,
    title: str,
    category_column: str = "method_name",
    type_column: str = "method_type",
) -> None:
    plt = require_matplotlib()
    if df.empty or metric_column not in df.columns:
        return
    chart_df = df.copy()
    chart_df[metric_column] = pd.to_numeric(chart_df[metric_column], errors="coerce")
    chart_df = chart_df.dropna(subset=[metric_column]).sort_values(metric_column, ascending=False).reset_index(drop=True)
    if chart_df.empty:
        return
    labels = chart_df[category_column].map(clean_plot_label)
    colors = chart_df[type_column].map(lambda value: METHOD_TYPE_COLORS.get(str(value), "#7a7a7a"))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(np.arange(len(chart_df)), chart_df[metric_column].astype(float), color=colors)
    ax.set_xticks(np.arange(len(chart_df)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(clean_plot_label(metric_column))
    ax.set_title(title)
    legend_handles = []
    for method_type, color in METHOD_TYPE_COLORS.items():
        if (chart_df[type_column] == method_type).any():
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color, label=clean_plot_label(method_type)))
    if legend_handles:
        ax.legend(handles=legend_handles, frameon=False)
    fig.tight_layout()
    prefix = Path(output_prefix).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_pr_curve_figure(
    prediction_frames: Sequence[pd.DataFrame],
    *,
    title: str,
    output_prefix: str | Path,
    label_column: str = "is_positive",
    score_column: str = "score",
    name_column: str = "model_name",
    include_pr_auc_in_legend: bool = False,
) -> None:
    plt = require_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 6))
    for frame in prediction_frames:
        if frame.empty:
            continue
        scored = frame[[label_column, score_column, name_column]].copy()
        scored[score_column] = pd.to_numeric(scored[score_column], errors="coerce").fillna(0.0)
        scored = scored.sort_values(score_column, ascending=False)
        y_true = scored[label_column].astype(int).to_numpy()
        y_score = scored[score_column].to_numpy(dtype=float)
        positives = max(int(y_true.sum()), 1)
        tp = np.cumsum(y_true)
        fp = np.cumsum(1 - y_true)
        precision = tp / np.maximum(tp + fp, 1)
        recall = tp / positives
        label = clean_plot_label(scored[name_column].iloc[0])
        if include_pr_auc_in_legend:
            pr_auc = float(average_precision_manual(y_true.tolist(), y_score.tolist()))
            label = f"{label} (PR-AUC={pr_auc:.3f})"
        ax.plot(recall, precision, label=label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    prefix = Path(output_prefix).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_roc_curve_figure(
    prediction_frames: Sequence[pd.DataFrame],
    *,
    title: str,
    output_prefix: str | Path,
    label_column: str = "is_positive",
    score_column: str = "score",
    name_column: str = "model_name",
    include_auroc_in_legend: bool = False,
) -> None:
    plt = require_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 6))
    for frame in prediction_frames:
        if frame.empty:
            continue
        scored = frame[[label_column, score_column, name_column]].copy()
        scored[score_column] = pd.to_numeric(scored[score_column], errors="coerce").fillna(0.0)
        scored = scored.sort_values(score_column, ascending=False)
        y_true = scored[label_column].astype(int).to_numpy()
        y_score = scored[score_column].to_numpy(dtype=float)
        positives = max(int(y_true.sum()), 1)
        negatives = max(int((1 - y_true).sum()), 1)
        tp = np.cumsum(y_true)
        fp = np.cumsum(1 - y_true)
        tpr = tp / positives
        fpr = fp / negatives
        label = clean_plot_label(scored[name_column].iloc[0])
        if include_auroc_in_legend:
            auroc = float(roc_auc_score_manual(y_true.tolist(), y_score.tolist()))
            label = f"{label} (AUROC={auroc:.3f})"
        ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    prefix = Path(output_prefix).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
