from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

EVIDENCE_CATEGORY_LABELS: Dict[str, str] = {
    "curated_validation": "miRTarBase",
    "mirdb_model": "miRDB",
    "targetscan_model": "TargetScan",
    "clip_binding": "CLIP",
    "seed_site": "Seed/site",
    "structure_rnahybrid": "RNAhybrid/structure",
    "tcga_context": "TCGA context",
    "pathway_membership": "Pathway",
}


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
        return f"{label or feature} {text} ({format_percentile(percentile)}; {percentile_label})"
    return f"{label or feature} {text}"


def format_percentile(percentile: float | int | None) -> str:
    if percentile is None or pd.isna(percentile):
        return "percentile not available"
    n = int(round(float(percentile)))
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix} percentile"


def _annotation_suffix(row: pd.Series, feature: str, extras: Optional[List[str]] = None) -> str:
    parts: List[str] = []
    percentile = row.get(f"{feature}_percentile")
    percentile_label = row.get(f"{feature}_label")
    if percentile is not None and not pd.isna(percentile) and percentile_label and percentile_label != "not available":
        parts.append(format_percentile(percentile))
        parts.append(str(percentile_label))
    for extra in extras or []:
        text = str(extra or "").strip()
        if text:
            parts.append(text)
    if not parts:
        return ""
    return " (" + "; ".join(parts) + ")"


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


def _label_rank(label: Any) -> int:
    mapping = {
        "exceptional": 5,
        "very high": 4,
        "high": 3,
        "above average": 2,
        "typical": 1,
        "low": 0,
        "not available": -1,
    }
    return mapping.get(str(label or "").strip().lower(), -1)


def _has_high_strength_signal(row: pd.Series) -> bool:
    checks = [
        _label_rank(row.get("mirdb_best_score_label")) >= 3,
        _label_rank(row.get("ts_context_strength_label")) >= 3,
        _label_rank(row.get("clip_exp_sum_label")) >= 3,
        _label_rank(row.get("n_clip_sites_label")) >= 3,
        _label_rank(row.get("mfe_strength_label")) >= 3,
        _label_rank(row.get("BRCA_anticorrelation_strength_label")) >= 3,
        _label_rank(row.get("COAD_anticorrelation_strength_label")) >= 3,
        _label_rank(row.get("PRAD_anticorrelation_strength_label")) >= 3,
        _as_float(row.get("mirdb_best_score")) >= 80,
    ]
    return any(checks)


def _has_very_strong_signal(row: pd.Series) -> bool:
    checks = [
        _label_rank(row.get("mirdb_best_score_label")) >= 4,
        _label_rank(row.get("ts_context_strength_label")) >= 4,
        _label_rank(row.get("clip_exp_sum_label")) >= 4,
        _label_rank(row.get("n_clip_sites_label")) >= 4,
        _label_rank(row.get("mfe_strength_label")) >= 4,
        _as_float(row.get("mirdb_best_score")) >= 90,
    ]
    return any(checks)


def _context_signal_state(row: pd.Series, tcga: Optional[str]) -> str:
    if not tcga:
        return "not_requested"
    rho = _as_float(row.get(f"{tcga}_spearman_rho"))
    support_tcga = _as_int(row.get(f"{tcga}_support_tcga"), 0)
    repression_flag = _as_int(row.get(f"{tcga}_repression_evidence"), 0)
    if support_tcga == 1 or repression_flag == 1 or (np.isfinite(rho) and rho < 0):
        return "supportive"
    if np.isfinite(rho) and rho >= 0:
        return "not_supportive"
    return "missing"


def _compute_priority_fields(
    row: pd.Series,
    tcga: Optional[str],
    evidence_categories: Dict[str, bool],
    evidence_support_count: int,
) -> Dict[str, str]:
    context_state = _context_signal_state(row, tcga)
    high_strength = _has_high_strength_signal(row)
    very_strong = _has_very_strong_signal(row)
    has_curated = bool(evidence_categories.get("curated_validation"))
    has_model = bool(evidence_categories.get("mirdb_model") or evidence_categories.get("targetscan_model"))
    has_binding = bool(evidence_categories.get("clip_binding"))

    if context_state == "not_supportive" and (has_model or has_binding or has_curated):
        overall_priority = "Conflicting context"
        context_strength = "Not supportive"
        strength_tier = "Moderate" if high_strength else "Weak"
        summary = "strong model or binding support, but the requested cancer-context evidence is not supportive"
    elif tcga and context_state == "missing" and (has_model or has_binding):
        overall_priority = "Context-limited"
        context_strength = "Unavailable"
        strength_tier = "Moderate" if high_strength else "Weak"
        summary = "model or binding support is present, but context-specific evidence is limited"
    elif has_curated or (very_strong and evidence_support_count >= 3 and (has_binding or evidence_categories.get("tcga_context"))):
        overall_priority = "Strong"
        context_strength = "Supportive" if context_state == "supportive" else "Not requested"
        strength_tier = "Strong"
        summary = "broad support with strong values across key evidence categories"
    elif evidence_support_count >= 3 and high_strength:
        overall_priority = "Moderate"
        context_strength = "Supportive" if context_state == "supportive" else ("Unavailable" if context_state == "missing" else "Not requested")
        strength_tier = "Strong"
        summary = "fewer categories than the top tier, but stronger values in the available evidence"
    elif evidence_support_count >= 4:
        overall_priority = "Exploratory"
        context_strength = "Supportive" if context_state == "supportive" else ("Unavailable" if context_state == "missing" else "Not requested")
        strength_tier = "Moderate"
        summary = "broad support across categories, but most values are weak or typical"
    elif evidence_support_count >= 2:
        overall_priority = "Exploratory"
        context_strength = "Supportive" if context_state == "supportive" else ("Unavailable" if context_state == "missing" else "Not requested")
        strength_tier = "Moderate" if high_strength else "Weak"
        summary = "some support is present, but the evidence remains exploratory"
    else:
        overall_priority = "Weak/context-limited"
        context_strength = "Supportive" if context_state == "supportive" else ("Unavailable" if context_state == "missing" else "Not requested")
        strength_tier = "Weak"
        summary = "limited support is available in this record"

    return {
        "evidence_strength_summary": summary,
        "evidence_strength_tier": strength_tier,
        "context_strength_tier": context_strength,
        "overall_priority_tier": overall_priority,
    }


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
    primary_evidence_by_category: Dict[str, Optional[str]] = {
        "curated": None,
        "mirdb": None,
        "targetscan": None,
        "clip": None,
        "seed_site": None,
        "structure": None,
        "tcga": None,
        "pathway": None,
    }

    support_count = _as_int(row.get("support_count"), 0)
    evidence_categories: Dict[str, bool] = {
        "curated_validation": False,
        "mirdb_model": False,
        "targetscan_model": False,
        "clip_binding": False,
        "seed_site": False,
        "structure_rnahybrid": False,
        "tcga_context": False,
        "pathway_membership": False,
    }

    if _as_int(row.get("mirtarbase_pos"), 0) == 1 or _as_int(row.get("label_mirtarbase"), 0) == 1:
        curated_line = "miRTarBase functional interaction present"
        target_evidence.append(curated_line)
        primary_evidence_by_category["curated"] = curated_line
        evidence_categories["curated_validation"] = True
    else:
        caveats.append("No curated miRTarBase functional validation in this record")

    mirdb_score = _as_float(row.get("mirdb_best_score"))
    if _as_int(row.get("support_mirdb"), 0) == 1:
        evidence_categories["mirdb_model"] = True
    if np.isfinite(mirdb_score):
        support_label = _mirdb_label(mirdb_score).replace("miRDB support", "model support")
        mirdb_line = f"miRDB score {mirdb_score:g}{_annotation_suffix(row, 'mirdb_best_score', [support_label])}"
        published_model_evidence.append(mirdb_line)
        primary_evidence_by_category["mirdb"] = mirdb_line
        strongest_features.append(mirdb_line)
        if mirdb_score >= 60 or _as_int(row.get("support_mirdb"), 0) == 1:
            evidence_categories["mirdb_model"] = True
    elif _as_float(row.get("mirdb_mean_score")) >= 60:
        mirdb_mean = _as_float(row.get("mirdb_mean_score"))
        support_label = _mirdb_label(mirdb_mean).replace("miRDB support", "model support")
        mirdb_line = f"miRDB score {mirdb_mean:g}{_annotation_suffix(row, 'mirdb_mean_score', [support_label])}"
        published_model_evidence.append(mirdb_line)
        primary_evidence_by_category["mirdb"] = mirdb_line
        strongest_features.append(mirdb_line)
        evidence_categories["mirdb_model"] = True

    ts_strength = _as_float(row.get("ts_context_strength"))
    ts_best_contextpp = _as_float(row.get("ts_best_contextpp"))
    if np.isfinite(ts_best_contextpp):
        targetscan_line = (
            f"TargetScan context++ {ts_best_contextpp:.3f}"
            f"{_annotation_suffix(row, 'ts_context_strength', ['more negative is stronger'])}"
        )
        published_model_evidence.append(targetscan_line)
        primary_evidence_by_category["targetscan"] = targetscan_line
        strongest_features.append(targetscan_line)
        evidence_categories["targetscan_model"] = True
    elif np.isfinite(ts_strength):
        targetscan_line = (
            f"TargetScan context strength {ts_strength:g}"
            f"{_annotation_suffix(row, 'ts_context_strength')}"
        )
        published_model_evidence.append(targetscan_line)
        primary_evidence_by_category["targetscan"] = targetscan_line
        strongest_features.append(targetscan_line)
        evidence_categories["targetscan_model"] = True
    elif np.isfinite(_as_float(row.get("ts_best_percentile"))):
        targetscan_line = f"TargetScan best-site percentile {float(row.get('ts_best_percentile')):g}{_annotation_suffix(row, 'ts_best_percentile')}"
        published_model_evidence.append(targetscan_line)
        primary_evidence_by_category["targetscan"] = targetscan_line
        strongest_features.append(targetscan_line)
        evidence_categories["targetscan_model"] = True
    if _as_int(row.get("support_targetscan"), 0) == 1:
        evidence_categories["targetscan_model"] = True

    clip_sites = _as_int(row.get("n_clip_sites"), 0)
    clip_sum = _as_float(row.get("clip_exp_sum"))
    clip_max = _as_float(row.get("clip_exp_max"))
    if clip_sites > 0 or np.isfinite(clip_sum) and clip_sum > 0 or np.isfinite(clip_max) and clip_max > 0 or _as_int(row.get("support_encori"), 0) == 1:
        evidence_categories["clip_binding"] = True
        clip_line = None
        if np.isfinite(clip_sum) and clip_sum > 0:
            clip_line = f"CLIP signal {clip_sum:g}{_annotation_suffix(row, 'clip_exp_sum')}"
        elif clip_sites > 0:
            clip_line = f"CLIP sites {clip_sites:.0f}{_annotation_suffix(row, 'n_clip_sites')}"
        elif np.isfinite(clip_max) and clip_max > 0:
            clip_line = f"CLIP signal {clip_max:g}{_annotation_suffix(row, 'clip_exp_max')}"
        if clip_line:
            clip_binding_evidence.append(clip_line)
            primary_evidence_by_category["clip"] = clip_line
            strongest_features.append(clip_line)
        caveats.append("CLIP supports binding potential, not necessarily functional repression")

    seed_summary = _best_seed_summary(row)
    total_sites = _as_int(row.get("n_total_sites"), 0)
    if seed_summary or total_sites > 0 or _as_int(row.get("has_seed_features"), 0) == 1:
        evidence_categories["seed_site"] = True
        seed_parts: List[str] = []
        if seed_summary:
            seed_parts.append(seed_summary)
        if total_sites > 0:
            seed_parts.append(f"total predicted sites {total_sites}")
        if seed_parts:
            seed_line = "; ".join(seed_parts)
            seed_site_evidence.append(seed_line)
            primary_evidence_by_category["seed_site"] = seed_line
            strongest_features.append(seed_line)

    has_structure_signal = (
        _as_int(row.get("has_rnahybrid"), 0) == 1
        or _as_int(row.get("n_rnahybrid_sites"), 0) > 0
        or np.isfinite(_as_float(row.get("mfe_strength")))
        or np.isfinite(_as_float(row.get("best_mfe")))
        or _as_int(row.get("n_sites_mfe_lt_-20"), 0) > 0
        or _as_int(row.get("n_sites_mfe_lt_-25"), 0) > 0
    )
    if has_structure_signal:
        evidence_categories["structure_rnahybrid"] = True
        best_mfe = _as_float(row.get("best_mfe"))
        mfe_strength = _as_float(row.get("mfe_strength"))
        structure_line = None
        if np.isfinite(best_mfe):
            structure_line = f"RNAhybrid best MFE {best_mfe:.1f} kcal/mol"
            if np.isfinite(mfe_strength):
                structure_line += _annotation_suffix(row, "mfe_strength")
        elif np.isfinite(mfe_strength):
            structure_line = f"RNAhybrid mfe strength {mfe_strength:g}{_annotation_suffix(row, 'mfe_strength')}"
        elif _as_int(row.get("n_rnahybrid_sites"), 0) > 0:
            sites = _as_int(row.get("n_rnahybrid_sites"), 0)
            structure_line = f"RNAhybrid sites {sites:.0f}{_annotation_suffix(row, 'n_rnahybrid_sites')}"
        elif _as_int(row.get("has_rnahybrid"), 0) == 1:
            structure_line = "RNAhybrid support present"
        if structure_line:
            structure_evidence.append(structure_line)
            primary_evidence_by_category["structure"] = structure_line
            strongest_features.append(structure_line)
        caveats.append("Structure evidence reflects predicted duplex stability, not structural confirmation")

    if tcga:
        rho_col = f"{tcga}_spearman_rho"
        support_col = f"{tcga}_support_tcga"
        rho = _as_float(row.get(rho_col))
        support_tcga = _as_int(row.get(support_col), 0)
        tcga_bits: List[str] = []
        if np.isfinite(rho):
            if rho < 0:
                tcga_bits.append(f"{tcga} rho {rho:.3f}, consistent with repression")
            else:
                tcga_bits.append(f"{tcga} rho {rho:.3f}, not supportive of repression in this context")
        if support_tcga == 1:
            tcga_bits.append(f"{tcga} repression support present")
            evidence_categories["tcga_context"] = True
        if np.isfinite(rho) and rho < 0:
            evidence_categories["tcga_context"] = True
        if _as_int(row.get(f"{tcga}_repression_evidence"), 0) == 1:
            tcga_bits.append(f"{tcga} repression support present")
            evidence_categories["tcga_context"] = True
        if tcga_bits:
            tcga_line = "; ".join(_dedupe_keep_order(tcga_bits))
            tcga_context_evidence.append(tcga_line)
            primary_evidence_by_category["tcga"] = tcga_line
            strongest_features.append(tcga_line)
        caveats.append("TCGA correlation is context evidence, not direct binding evidence")

    pathway_names: List[str] = []
    if _as_int(row.get("pathway_selected_gene"), 0) == 1:
        pathway_names = _as_list(row.get("pathway_selected_names"))
        evidence_categories["pathway_membership"] = True
        if pathway_names:
            pathway_line = "Retained by strict pathway filter: " + "; ".join(pathway_names[:6])
            pathway_evidence.append(pathway_line)
            primary_evidence_by_category["pathway"] = pathway_line
            strongest_features.append(pathway_line)
        else:
            pathway_line = "Retained by strict pathway filter"
            pathway_evidence.append(pathway_line)
            primary_evidence_by_category["pathway"] = pathway_line
            strongest_features.append(pathway_line)

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

    evidence_categories_present = [
        label for key, label in EVIDENCE_CATEGORY_LABELS.items() if evidence_categories.get(key)
    ]
    evidence_support_count = int(sum(1 for present in evidence_categories.values() if present))
    priority_fields = _compute_priority_fields(row, tcga, evidence_categories, evidence_support_count)

    sections = {
        "support_count": support_count,
        "evidence_categories": evidence_categories,
        "evidence_categories_present": evidence_categories_present,
        "evidence_support_count": evidence_support_count,
        "evidence_strength_summary": priority_fields["evidence_strength_summary"],
        "evidence_strength_tier": priority_fields["evidence_strength_tier"],
        "context_strength_tier": priority_fields["context_strength_tier"],
        "overall_priority_tier": priority_fields["overall_priority_tier"],
        "number_of_features_supporting_interaction": evidence_support_count,
        "primary_curated_evidence": primary_evidence_by_category["curated"],
        "primary_mirdb_evidence": primary_evidence_by_category["mirdb"],
        "primary_targetscan_evidence": primary_evidence_by_category["targetscan"],
        "primary_clip_evidence": primary_evidence_by_category["clip"],
        "primary_seed_evidence": primary_evidence_by_category["seed_site"],
        "primary_structure_evidence": primary_evidence_by_category["structure"],
        "primary_tcga_evidence": primary_evidence_by_category["tcga"],
        "primary_pathway_evidence": primary_evidence_by_category["pathway"],
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
