from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return int(value)
    except Exception:
        return default


def _as_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [str(v) for v in value.tolist() if v not in (None, "", [])]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if v not in (None, "", [])]
    return [str(value)]


def _feature_phrase(row: pd.Series, feature: str, label: Optional[str] = None, fmt: str = "{value:g}") -> Optional[str]:
    if feature not in row or pd.isna(row.get(feature)):
        return None
    value = row.get(feature)
    text = fmt.format(value=float(value) if isinstance(value, (np.floating, float, int, np.integer)) else value)
    percentile = row.get(f"{feature}_percentile")
    percentile_label = row.get(f"{feature}_label")
    if percentile is not None and not pd.isna(percentile) and percentile_label and percentile_label != "not available":
        return f"{label or feature} {text} ({int(round(float(percentile)))}th percentile; {percentile_label})"
    return f"{label or feature} {text}"


def _dedupe_keep_order(values: List[str]) -> List[str]:
    return list(dict.fromkeys(v for v in values if v))


def _raw_value_if_present(row: pd.Series, key: str) -> Any:
    value = row.get(key)
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _mirdb_label(score: float) -> str:
    if score >= 90:
        return "very strong miRDB support"
    if score >= 80:
        return "strong miRDB support"
    if score >= 60:
        return "moderate miRDB support"
    return "weak miRDB support"


def _best_seed_summary(row: pd.Series) -> Optional[str]:
    best_seed = str(row.get("best_seed_class") or "").strip()
    if not best_seed:
        return None
    if best_seed == "8mer":
        return "best seed class 8mer (strong canonical seed support)"
    if best_seed == "7mer-m8":
        return "best seed class 7mer-m8 (strong canonical seed support)"
    if best_seed == "7mer-a1":
        return "best seed class 7mer-A1 (moderate canonical seed support)"
    if best_seed == "6mer":
        return "best seed class 6mer (weaker seed support)"
    return f"best seed class {best_seed}"


def build_evidence_sections(row: pd.Series, tcga: Optional[str] = None) -> Dict[str, Any]:
    target_evidence: List[str] = []
    published_model_evidence: List[str] = []
    clip_binding_evidence: List[str] = []
    seed_site_evidence: List[str] = []
    structure_evidence: List[str] = []
    tcga_context_evidence: List[str] = []
    pathway_evidence: List[str] = []
    strongest_features: List[str] = []
    caveats: List[str] = []

    support_count = _as_int(row.get("support_count"), 0)
    support_phrase = _feature_phrase(row, "support_count", "support_count", "{value:.0f}")
    if support_phrase:
        strongest_features.append(support_phrase)

    if _as_int(row.get("mirtarbase_pos"), 0) == 1:
        target_evidence.append("miRTarBase: functional=1, known curated interaction")
    else:
        caveats.append("No curated miRTarBase functional validation in this record")

    mirdb_score = _as_float(row.get("mirdb_best_score"))
    if np.isfinite(mirdb_score):
        published_model_evidence.append(f"miRDB score {mirdb_score:g}, {_mirdb_label(mirdb_score)}")
        phrase = _feature_phrase(row, "mirdb_best_score", "miRDB score", "{value:g}")
        if phrase:
            strongest_features.append(phrase)
    mirdb_mean_phrase = _feature_phrase(row, "mirdb_mean_score", "miRDB mean score", "{value:g}")
    if mirdb_mean_phrase:
        published_model_evidence.append(mirdb_mean_phrase)

    ts_strength = _as_float(row.get("ts_context_strength"))
    if np.isfinite(ts_strength):
        published_model_evidence.append(
            f"TargetScan context strength {ts_strength:g}, {row.get('ts_context_strength_label', 'not available')}"
        )
        phrase = _feature_phrase(row, "ts_context_strength", "TargetScan context strength", "{value:g}")
        if phrase:
            strongest_features.append(phrase)
    elif np.isfinite(_as_float(row.get("ts_best_contextpp"))):
        published_model_evidence.append(f"TargetScan context++ {float(row.get('ts_best_contextpp')):.3f}")
    ts_percentile_phrase = _feature_phrase(row, "ts_best_percentile", "TargetScan best-site percentile", "{value:g}")
    if ts_percentile_phrase:
        published_model_evidence.append(ts_percentile_phrase)

    clip_sites = _as_int(row.get("n_clip_sites"), 0)
    clip_sum = _as_float(row.get("clip_exp_sum"))
    if clip_sites > 0 or np.isfinite(clip_sum) and clip_sum > 0:
        clip_phrase = _feature_phrase(row, "n_clip_sites", "CLIP sites", "{value:.0f}")
        if clip_phrase:
            clip_binding_evidence.append(clip_phrase)
            strongest_features.append(clip_phrase)
        clip_sum_phrase = _feature_phrase(row, "clip_exp_sum", "CLIP signal", "{value:g}")
        if clip_sum_phrase:
            clip_binding_evidence.append(clip_sum_phrase)
            strongest_features.append(clip_sum_phrase)
        clip_max_phrase = _feature_phrase(row, "clip_exp_max", "max CLIP signal", "{value:g}")
        if clip_max_phrase:
            clip_binding_evidence.append(clip_max_phrase)
        caveats.append("CLIP supports binding potential, not necessarily functional repression")

    seed_summary = _best_seed_summary(row)
    if seed_summary:
        seed_site_evidence.append(seed_summary)
    total_sites_phrase = _feature_phrase(row, "n_total_sites", "total predicted sites", "{value:.0f}")
    if total_sites_phrase:
        seed_site_evidence.append(total_sites_phrase)
    density_phrase = _feature_phrase(row, "site_density_per_kb", "site density per kb", "{value:g}")
    if density_phrase:
        seed_site_evidence.append(density_phrase)
    for site_col, site_label in [
        ("n_sites_8mer", "8mer sites"),
        ("n_sites_7mer_m8", "7mer-m8 sites"),
        ("n_sites_7mer_a1", "7mer-A1 sites"),
        ("n_sites_6mer", "6mer sites"),
    ]:
        site_phrase = _feature_phrase(row, site_col, site_label, "{value:.0f}")
        if site_phrase:
            seed_site_evidence.append(site_phrase)

    if _as_int(row.get("has_rnahybrid"), 0) == 1 or _as_int(row.get("n_rnahybrid_sites"), 0) > 0:
        structure_evidence.append("RNAhybrid support present")
        n_rnahybrid_phrase = _feature_phrase(row, "n_rnahybrid_sites", "RNAhybrid sites", "{value:.0f}")
        if n_rnahybrid_phrase:
            structure_evidence.append(n_rnahybrid_phrase)
        mfe_phrase = _feature_phrase(row, "mfe_strength", "mfe_strength", "{value:g}")
        if mfe_phrase:
            structure_evidence.append(mfe_phrase)
            strongest_features.append(mfe_phrase)
        mean_top3_phrase = _feature_phrase(
            row,
            "mean_top3_mfe_strength",
            "mean top-3 mfe strength",
            "{value:g}",
        )
        if mean_top3_phrase:
            structure_evidence.append(mean_top3_phrase)
        if np.isfinite(_as_float(row.get("best_mfe"))):
            structure_evidence.append(f"best MFE {float(row.get('best_mfe')):.1f} kcal/mol")
        stable20 = _as_int(row.get("n_sites_mfe_lt_-20"), 0)
        stable25 = _as_int(row.get("n_sites_mfe_lt_-25"), 0)
        if stable20 > 0:
            structure_evidence.append(f"n_sites_mfe_lt_-20 = {stable20}")
        if stable25 > 0:
            structure_evidence.append(f"n_sites_mfe_lt_-25 = {stable25}")
        au_by_mfe_phrase = _feature_phrase(row, "best_local_au_by_mfe", "local AU at best-MFE site", "{value:.2f}")
        if au_by_mfe_phrase:
            structure_evidence.append(au_by_mfe_phrase)
        caveats.append("Structure evidence reflects predicted duplex stability, not structural confirmation")

    local_au_phrase = _feature_phrase(row, "best_local_au", "local AU", "{value:.2f}")
    if local_au_phrase:
        seed_site_evidence.append(local_au_phrase)

    if tcga:
        rho_col = f"{tcga}_spearman_rho"
        support_col = f"{tcga}_support_tcga"
        rho = _as_float(row.get(rho_col))
        support_tcga = _as_int(row.get(support_col), 0)
        anti_strength_phrase = _feature_phrase(
            row,
            f"{tcga}_anticorrelation_strength",
            f"{tcga} anticorrelation strength",
            "{value:g}",
        )
        if np.isfinite(rho):
            if rho < 0:
                tcga_context_evidence.append(f"{tcga} rho {rho:.3f}, consistent with repression")
            else:
                tcga_context_evidence.append(f"{tcga} rho {rho:.3f}, not supportive of repression in this context")
        if support_tcga == 1:
            tcga_context_evidence.append(f"{tcga} repression support present")
        if anti_strength_phrase and rho < 0:
            strongest_features.append(anti_strength_phrase)
        caveats.append("TCGA correlation is context evidence, not direct binding evidence")

    pathway_names: List[str] = []
    if _as_int(row.get("pathway_selected_gene"), 0) == 1:
        pathway_names = _as_list(row.get("pathway_selected_names"))
        if pathway_names:
            pathway_evidence.append("retained by strict pathway filter")
            pathway_evidence.append("selected pathways: " + "; ".join(pathway_names[:6]))
        else:
            pathway_evidence.append("retained by strict pathway filter")

    published_model_evidence = _dedupe_keep_order(published_model_evidence)
    clip_binding_evidence = _dedupe_keep_order(clip_binding_evidence)
    seed_site_evidence = _dedupe_keep_order(seed_site_evidence)
    structure_evidence = _dedupe_keep_order(structure_evidence)
    tcga_context_evidence = _dedupe_keep_order(tcga_context_evidence)
    pathway_evidence = _dedupe_keep_order(pathway_evidence)
    caveats = _dedupe_keep_order(caveats)
    strongest_features = _dedupe_keep_order(strongest_features)

    raw_key_values: Dict[str, Any] = {}
    raw_keys = [
        "support_count",
        "mirdb_best_score",
        "mirdb_mean_score",
        "ts_context_strength",
        "ts_best_contextpp",
        "ts_best_percentile",
        "n_clip_sites",
        "clip_exp_sum",
        "clip_exp_max",
        "best_seed_class",
        "n_total_sites",
        "site_density_per_kb",
        "best_local_au",
        "n_rnahybrid_sites",
        "best_mfe",
        "mfe_strength",
        "mean_top3_mfe_strength",
        "n_sites_mfe_lt_-20",
        "n_sites_mfe_lt_-25",
        "best_local_au_by_mfe",
        "pathway_selected_gene",
    ]
    if tcga:
        raw_keys.extend([f"{tcga}_spearman_rho", f"{tcga}_support_tcga", f"{tcga}_anticorrelation_strength"])
    for key in raw_keys:
        value = _raw_value_if_present(row, key)
        if value is not None:
            raw_key_values[key] = value

    sections = {
        "support_count": support_count,
        "number_of_features_supporting_interaction": sum(
            1 for section in [
                target_evidence,
                published_model_evidence,
                clip_binding_evidence,
                seed_site_evidence,
                structure_evidence,
                tcga_context_evidence,
                pathway_evidence,
            ]
            if section
        ),
        "target_evidence": target_evidence,
        "published_model_evidence": published_model_evidence,
        "clip_binding_evidence": clip_binding_evidence,
        "seed_site_evidence": seed_site_evidence,
        "structure_evidence": structure_evidence,
        "tcga_context_evidence": tcga_context_evidence,
        "pathway_evidence": pathway_evidence,
        "pathway_names": pathway_names[:6],
        "strongest_features": strongest_features[:8],
        "caveats": caveats,
        "raw_key_values": raw_key_values,
    }
    return sections
