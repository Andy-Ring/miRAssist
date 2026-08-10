from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.config import use_mirtarbase_evidence
from backend.retrieval import MIRTARBASE_KNOWN_POSITIVE_COLUMNS

EVIDENCE_CATEGORY_LABELS: Dict[str, str] = {
    "sequence_complementarity": "Sequence complementarity",
    "thermodynamic_stability": "Thermodynamic stability",
    "sequence_conservation": "Sequence conservation",
    "target_site_accessibility": "Target site accessibility",
    "functional_binding": "Functional binding",
    "functional_repression": "Functional repression",
}

FAMILY_PERCENTILE_COLUMNS: Dict[str, List[str]] = {
    "sequence_complementarity": [
        "sequence_complementarity_support_percentile",
        "seed_pairing_score_percentile",
        "n_seed_sites_percentile",
    ],
    "thermodynamic_stability": [
        "thermodynamic_stability_support_percentile",
        "rnahybrid_mfe_percentile",
        "rnahybrid_mfe_best_site_percentile",
        "rnahybrid_seed_mfe_percentile",
        "rnahybrid_strength_percentile",
        "mfe_strength_percentile",
    ],
    "sequence_conservation": [
        "sequence_conservation_support_percentile",
        "targetscan_context_score_support_percentile",
        "targetscan_context_score_percentile",
        "targetscan_aggregate_context_score_percentile",
        "targetscan_pct_percentile",
        "targetscan_branch_length_score_percentile",
        "ts_context_strength_percentile",
        "ts_best_percentile_percentile",
    ],
    "target_site_accessibility": [
        "target_site_accessibility_support_percentile",
        "rnaplfold_best_seed_unpaired_prob_percentile",
        "rnaplfold_mean_seed_unpaired_prob_percentile",
        "rnaplfold_best_site_unpaired_prob_percentile",
        "rnaplfold_mean_site_unpaired_prob_percentile",
        "rnaplfold_best_flank_unpaired_prob_percentile",
        "rnaplfold_mean_flank_unpaired_prob_percentile",
        "rnaplfold_n_sites_scored_percentile",
        "rnaplfold_n_accessible_sites_percentile",
    ],
    "functional_binding": [
        "functional_binding_support_percentile",
        "clip_max_score_percentile",
        "clip_n_experiments_percentile",
        "clip_n_cell_lines_percentile",
        "encori_clip_score_percentile",
        "clip_exp_sum_percentile",
        "clip_exp_max_percentile",
        "n_clip_sites_percentile",
    ],
    "functional_repression": [
        "functional_repression_support_percentile",
        "BRCA_spearman_rho_percentile",
        "BRCA_repression_evidence_percentile",
        "PRAD_spearman_rho_percentile",
        "PRAD_repression_evidence_percentile",
        "COAD_spearman_rho_percentile",
        "COAD_repression_evidence_percentile",
        "tcga_n_supported_contexts_percentile",
        "tcga_best_repression_evidence_percentile",
        "tcga_mean_spearman_rho_percentile",
        "BRCA_anticorrelation_strength_percentile",
        "COAD_anticorrelation_strength_percentile",
        "PRAD_anticorrelation_strength_percentile",
    ],
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


def _as_bool(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off", ""}:
            return False
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass
    return bool(value)


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [str(v) for v in value.tolist() if v not in (None, "", [])]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if v not in (None, "", [])]
    return [str(value)]


def format_percentile(percentile: float | int | None) -> str:
    if percentile is None or pd.isna(percentile):
        return "percentile not available"
    n = int(round(float(percentile)))
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix} percentile"


def _percentile_label(percentile: float | int | None) -> str:
    if percentile is None or pd.isna(percentile):
        return "not available"
    pct = float(percentile)
    if pct >= 95:
        return "exceptional"
    if pct >= 90:
        return "very high"
    if pct >= 75:
        return "high"
    if pct >= 50:
        return "above average"
    if pct >= 25:
        return "typical"
    return "low"


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


def _first_present_value(row: pd.Series, columns: List[str]) -> Any:
    for column in columns:
        value = row.get(column)
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except TypeError:
            pass
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _first_present_numeric(row: pd.Series, columns: List[str]) -> float:
    for column in columns:
        value = _as_float(row.get(column))
        if np.isfinite(value):
            return value
    return np.nan


def _format_support_suffix(
    row: pd.Series,
    percentile_columns: List[str],
    extras: Optional[List[str]] = None,
) -> str:
    percentile_value = _first_present_numeric(row, percentile_columns)
    parts: List[str] = []
    if np.isfinite(percentile_value):
        parts.append(format_percentile(percentile_value))
        label = _percentile_label(percentile_value)
        if label != "not available":
            parts.append(label)
    for extra in extras or []:
        text = str(extra or "").strip()
        if text:
            parts.append(text)
    if not parts:
        return ""
    return " (" + "; ".join(parts) + ")"


def _append_line(lines: List[str], text: Optional[str]) -> None:
    if not text:
        return
    text = str(text).strip()
    if text and text not in lines:
        lines.append(text)


def _family_support_percentile(row: pd.Series, family: str) -> float:
    explicit = _as_float(row.get(f"{family}_support_percentile"))
    if np.isfinite(explicit):
        return explicit
    values = [
        _as_float(row.get(column))
        for column in FAMILY_PERCENTILE_COLUMNS.get(family, [])
        if np.isfinite(_as_float(row.get(column)))
    ]
    if not values:
        return np.nan
    return float(np.mean(values))


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


def _best_seed_summary(row: pd.Series) -> Optional[str]:
    best_seed = _first_present_value(row, ["seed_match_type", "best_seed_site_type", "best_seed_class"])
    if not best_seed:
        return None
    best_seed = str(best_seed).strip()
    if best_seed == "8mer":
        return "seed match 8mer"
    if best_seed == "7mer-m8":
        return "seed match 7mer-m8"
    if best_seed == "7mer-a1":
        return "seed match 7mer-A1"
    if best_seed == "6mer":
        return "seed match 6mer"
    return f"seed match {best_seed}"


def _sequence_key_evidence(row: pd.Series) -> List[str]:
    lines: List[str] = []
    _append_line(lines, _best_seed_summary(row))

    seed_pairing_score = _first_present_numeric(row, ["seed_pairing_score", "best_seed_rank"])
    if np.isfinite(seed_pairing_score):
        _append_line(
            lines,
            f"seed pairing score {seed_pairing_score:g}"
            + _format_support_suffix(row, ["seed_pairing_score_percentile"]),
        )

    n_seed_sites = _first_present_numeric(row, ["n_seed_sites", "n_total_sites"])
    if np.isfinite(n_seed_sites) and n_seed_sites > 0:
        _append_line(
            lines,
            f"predicted seed sites {n_seed_sites:g}"
            + _format_support_suffix(row, ["n_seed_sites_percentile"]),
        )

    if _as_int(row.get("is_8mer"), 0) == 1 and "seed match 8mer" not in lines:
        _append_line(lines, "canonical 8mer site present")
    return lines[:3]


def _thermodynamic_key_evidence(row: pd.Series) -> List[str]:
    lines: List[str] = []
    mfe = _first_present_numeric(row, ["rnahybrid_mfe_best_site", "rnahybrid_mfe", "best_mfe"])
    if np.isfinite(mfe):
        _append_line(
            lines,
            f"RNAhybrid MFE {mfe:.3g} kcal/mol"
            + _format_support_suffix(
                row,
                ["rnahybrid_mfe_best_site_percentile", "rnahybrid_mfe_percentile", "mfe_strength_percentile"],
                ["more negative is stronger"],
            ),
        )

    seed_mfe = _as_float(row.get("rnahybrid_seed_mfe"))
    if np.isfinite(seed_mfe):
        _append_line(
            lines,
            f"seed-region MFE {seed_mfe:.3g} kcal/mol"
            + _format_support_suffix(row, ["rnahybrid_seed_mfe_percentile"], ["more negative is stronger"]),
        )

    strength = _first_present_numeric(row, ["rnahybrid_strength", "mfe_strength"])
    if np.isfinite(strength):
        _append_line(
            lines,
            f"duplex stability score {strength:g}"
            + _format_support_suffix(row, ["rnahybrid_strength_percentile", "mfe_strength_percentile"]),
        )
    return lines[:3]


def _conservation_key_evidence(row: pd.Series) -> List[str]:
    lines: List[str] = []
    context_score = _first_present_numeric(row, ["targetscan_context_score", "ts_best_contextpp"])
    if np.isfinite(context_score):
        _append_line(
            lines,
            f"TargetScan context score {context_score:.3g}"
            + _format_support_suffix(
                row,
                [
                    "targetscan_context_score_support_percentile",
                    "targetscan_context_score_percentile",
                    "ts_context_strength_percentile",
                ],
                ["more negative is stronger"],
            ),
        )

    raw_percentile = _as_float(row.get("targetscan_context_score_percentile"))
    if np.isfinite(raw_percentile):
        _append_line(
            lines,
            f"TargetScan published percentile {raw_percentile:g}",
        )

    if _as_int(row.get("targetscan_conserved_site"), 0) == 1:
        _append_line(lines, "TargetScan conserved site present")
    return lines[:3]


def _accessibility_key_evidence(row: pd.Series) -> List[str]:
    lines: List[str] = []
    best_seed_prob = _as_float(row.get("rnaplfold_best_seed_unpaired_prob"))
    if np.isfinite(best_seed_prob):
        _append_line(
            lines,
            f"best seed-region unpaired probability {best_seed_prob:.3g}"
            + _format_support_suffix(row, ["rnaplfold_best_seed_unpaired_prob_percentile"]),
        )

    mean_seed_prob = _first_present_numeric(row, ["rnaplfold_mean_seed_unpaired_prob", "rnaplfold_mean_site_unpaired_prob"])
    if np.isfinite(mean_seed_prob):
        _append_line(
            lines,
            f"mean seed-region unpaired probability {mean_seed_prob:.3g}"
            + _format_support_suffix(
                row,
                ["rnaplfold_mean_seed_unpaired_prob_percentile", "rnaplfold_mean_site_unpaired_prob_percentile"],
            ),
        )

    sites_scored = _as_float(row.get("rnaplfold_n_sites_scored"))
    if np.isfinite(sites_scored) and sites_scored > 0:
        _append_line(
            lines,
            f"RNAplfold sites scored {sites_scored:g}"
            + _format_support_suffix(row, ["rnaplfold_n_sites_scored_percentile"]),
        )

    accessible_sites = _as_float(row.get("rnaplfold_n_accessible_sites"))
    if np.isfinite(accessible_sites) and accessible_sites > 0:
        _append_line(
            lines,
            f"accessible sites scored {accessible_sites:g}"
            + _format_support_suffix(row, ["rnaplfold_n_accessible_sites_percentile"]),
        )
    return lines[:3]


def _binding_key_evidence(row: pd.Series) -> List[str]:
    lines: List[str] = []
    clip_max_score = _first_present_numeric(row, ["clip_max_score", "clip_exp_max"])
    if np.isfinite(clip_max_score) and clip_max_score > 0:
        _append_line(
            lines,
            f"CLIP max score {clip_max_score:g}"
            + _format_support_suffix(row, ["clip_max_score_percentile", "clip_exp_max_percentile"]),
        )

    clip_n_experiments = _first_present_numeric(row, ["clip_n_experiments", "encori_clip_score", "clip_exp_sum"])
    if np.isfinite(clip_n_experiments) and clip_n_experiments > 0:
        _append_line(
            lines,
            f"CLIP-supported experiments {clip_n_experiments:g}"
            + _format_support_suffix(row, ["clip_n_experiments_percentile", "encori_clip_score_percentile", "clip_exp_sum_percentile"]),
        )

    clip_n_cell_lines = _as_float(row.get("clip_n_cell_lines"))
    if np.isfinite(clip_n_cell_lines) and clip_n_cell_lines > 0:
        _append_line(
            lines,
            f"supporting cell lines {clip_n_cell_lines:g}"
            + _format_support_suffix(row, ["clip_n_cell_lines_percentile"]),
        )
    return lines[:3]


def _repression_key_evidence(row: pd.Series, tcga: Optional[str]) -> List[str]:
    lines: List[str] = []

    if tcga:
        rho = _as_float(row.get(f"{tcga}_spearman_rho"))
        if np.isfinite(rho):
            direction_note = "more negative is stronger"
            if rho < 0:
                text = f"{tcga} Spearman rho {rho:.3g}"
            else:
                text = f"{tcga} Spearman rho {rho:.3g}"
            _append_line(
                lines,
                text
                + _format_support_suffix(
                    row,
                    [f"{tcga}_spearman_rho_percentile", f"{tcga}_anticorrelation_strength_percentile"],
                    [direction_note],
                ),
            )
        if _as_int(row.get(f"{tcga}_support_tcga"), 0) == 1 or _as_int(row.get(f"{tcga}_repression_evidence"), 0) == 1:
            _append_line(lines, f"{tcga} repression support present")
    else:
        for context in ["BRCA", "COAD", "PRAD"]:
            rho = _as_float(row.get(f"{context}_spearman_rho"))
            support = _as_bool(row.get(f"{context}_support_tcga")) or _as_bool(row.get(f"{context}_repression_evidence"))
            anticorrelated = _as_bool(row.get(f"{context}_anticorrelated"))
            if support or anticorrelated or (np.isfinite(rho) and rho < 0):
                if np.isfinite(rho):
                    _append_line(lines, f"{context} TCGA rho {rho:.3g}")
                else:
                    _append_line(lines, f"{context} TCGA repression support present")

    supported_contexts = _as_float(row.get("tcga_n_supported_contexts"))
    if np.isfinite(supported_contexts) and supported_contexts > 0:
        _append_line(
            lines,
            f"supported TCGA contexts {supported_contexts:g}"
            + _format_support_suffix(row, ["tcga_n_supported_contexts_percentile"]),
        )

    mean_rho = _as_float(row.get("tcga_mean_spearman_rho"))
    if np.isfinite(mean_rho):
        _append_line(
            lines,
            f"mean TCGA rho {mean_rho:.3g}"
            + _format_support_suffix(row, ["tcga_mean_spearman_rho_percentile"], ["more negative is stronger"]),
        )
    return lines[:3]


def _family_key_evidence(row: pd.Series, family: str, tcga: Optional[str]) -> List[str]:
    if family == "sequence_complementarity":
        return _sequence_key_evidence(row)
    if family == "thermodynamic_stability":
        return _thermodynamic_key_evidence(row)
    if family == "sequence_conservation":
        return _conservation_key_evidence(row)
    if family == "target_site_accessibility":
        return _accessibility_key_evidence(row)
    if family == "functional_binding":
        return _binding_key_evidence(row)
    if family == "functional_repression":
        return _repression_key_evidence(row, tcga)
    return []


def _family_available(row: pd.Series, family: str, key_evidence: List[str]) -> bool:
    explicit = row.get(f"{family}_available")
    if explicit is not None and not pd.isna(explicit):
        return _as_bool(explicit)

    if family == "sequence_complementarity":
        return bool(
            key_evidence
            or _as_int(row.get("has_seed_evidence"), 0) == 1
            or _as_int(row.get("has_seed_features"), 0) == 1
        )
    if family == "thermodynamic_stability":
        return bool(
            key_evidence
            or _as_int(row.get("has_rnahybrid_evidence"), 0) == 1
            or _as_int(row.get("has_rnahybrid"), 0) == 1
        )
    if family == "sequence_conservation":
        return bool(
            key_evidence
            or _as_int(row.get("has_targetscan_evidence"), 0) == 1
            or _as_int(row.get("support_targetscan"), 0) == 1
        )
    if family == "target_site_accessibility":
        return bool(key_evidence or _as_int(row.get("has_rnaplfold_evidence"), 0) == 1)
    if family == "functional_binding":
        return bool(
            key_evidence
            or _as_int(row.get("has_clip_evidence"), 0) == 1
            or _as_int(row.get("support_encori"), 0) == 1
        )
    if family == "functional_repression":
        return bool(
            key_evidence
            or _as_int(row.get("has_tcga_evidence"), 0) == 1
            or _as_int(row.get("tcga_any_anticorrelated"), 0) == 1
        )
    return bool(key_evidence)


def build_family_evidence_summary(row: pd.Series, tcga: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for family, display_name in EVIDENCE_CATEGORY_LABELS.items():
        key_evidence = _family_key_evidence(row, family, tcga)
        support_percentile = _family_support_percentile(row, family)
        available = _family_available(row, family, key_evidence)
        evidence_count = _as_int(row.get(f"{family}_evidence_count"), len(key_evidence) if available else 0)
        summary[family] = {
            "label": display_name,
            "available": bool(available),
            "support_percentile": support_percentile if np.isfinite(support_percentile) else np.nan,
            "key_evidence": key_evidence,
            "evidence_count": evidence_count,
        }
    return summary


def _compute_priority_fields(
    row: pd.Series,
    tcga: Optional[str],
    family_summary: Dict[str, Dict[str, Any]],
) -> Dict[str, str]:
    context_state = _context_signal_state(row, tcga)
    available_families = [info for info in family_summary.values() if info["available"]]
    family_count = len(available_families)
    support_values = [float(info["support_percentile"]) for info in available_families if np.isfinite(info["support_percentile"])]
    overall_support = float(np.mean(support_values)) if support_values else np.nan
    binding_or_sequence = any(
        family_summary[name]["available"]
        for name in ("sequence_complementarity", "sequence_conservation", "functional_binding")
    )

    if context_state == "not_supportive" and binding_or_sequence:
        overall_priority = "Conflicting context"
        context_strength = "Not supportive"
        strength_tier = "Moderate" if family_count >= 2 else "Weak"
        summary = "stronger sequence or binding support is present, but the requested cancer-context repression signal is not supportive"
    elif tcga and context_state == "missing" and binding_or_sequence:
        overall_priority = "Context-limited"
        context_strength = "Unavailable"
        strength_tier = "Moderate" if family_count >= 3 else "Weak"
        summary = "support is present outside the requested cancer context, but context-specific repression evidence is limited"
    elif family_count >= 4 and np.isfinite(overall_support) and overall_support >= 75:
        overall_priority = "Strong"
        context_strength = "Supportive" if context_state == "supportive" else "Not requested"
        strength_tier = "Strong"
        summary = "broad support is present across the evidence families, with strong percentile support in the available signals"
    elif family_count >= 3 and (not np.isfinite(overall_support) or overall_support >= 60):
        overall_priority = "Moderate"
        context_strength = "Supportive" if context_state == "supportive" else ("Unavailable" if context_state == "missing" else "Not requested")
        strength_tier = "Strong" if np.isfinite(overall_support) and overall_support >= 70 else "Moderate"
        summary = "multiple evidence families support the interaction, with interpretable percentile support for the available metrics"
    elif family_count >= 2:
        overall_priority = "Exploratory"
        context_strength = "Supportive" if context_state == "supportive" else ("Unavailable" if context_state == "missing" else "Not requested")
        strength_tier = "Moderate" if np.isfinite(overall_support) and overall_support >= 60 else "Weak"
        summary = "some support is present, but the evidence remains exploratory because breadth or strength is limited"
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
    allow_mirtarbase = use_mirtarbase_evidence()
    family_summary = build_family_evidence_summary(row, tcga=tcga)
    evidence_categories = {family: bool(info["available"]) for family, info in family_summary.items()}
    evidence_categories_present = [
        info["label"] for info in family_summary.values() if info["available"]
    ]
    evidence_support_count = _as_int(
        row.get("evidence_family_count"),
        sum(1 for info in family_summary.values() if info["available"]),
    )
    support_count = _as_int(row.get("support_count"), evidence_support_count)
    overall_evidence_support_percentile = _as_float(row.get("overall_evidence_support_percentile"))
    if not np.isfinite(overall_evidence_support_percentile):
        support_values = [
            float(info["support_percentile"])
            for info in family_summary.values()
            if np.isfinite(info["support_percentile"])
        ]
        overall_evidence_support_percentile = float(np.mean(support_values)) if support_values else np.nan

    pathway_names: List[str] = []
    pathway_evidence: List[str] = []
    if _as_int(row.get("pathway_selected_gene"), 0) == 1:
        pathway_names = _as_list(row.get("pathway_selected_names"))[:6]
        if pathway_names:
            pathway_evidence.append("Retained by strict pathway filter: " + "; ".join(pathway_names))
        else:
            pathway_evidence.append("Retained by strict pathway filter")

    predicted_target_change = str(row.get("predicted_mirna_effect_on_target") or "unknown")
    expected_target_role = str(row.get("expected_target_effect_on_phenotype") or "unknown")
    target_role_evidence = _as_list(row.get("target_role_evidence"))
    positive_regulator_evidence = _as_list(row.get("target_role_positive_pathways"))
    negative_regulator_evidence = _as_list(row.get("target_role_negative_pathways"))
    target_role_status = str(row.get("target_role_evidence_status") or "unknown")
    directionally_consistent = (
        True if target_role_status == "consistent" else
        False if target_role_status == "conflicting" else None
    )

    caveats: List[str] = []
    if family_summary["functional_binding"]["available"]:
        caveats.append("Binding evidence supports physical association, not necessarily functional repression")
    if family_summary["thermodynamic_stability"]["available"]:
        caveats.append("Thermodynamic stability is computational support, not structural confirmation")
    if family_summary["functional_repression"]["available"]:
        caveats.append("TCGA correlation is context evidence, not direct binding evidence")
    if target_role_status == "absent":
        caveats.append("No explicit positive- or negative-regulator pathway annotation was found")
    elif target_role_status == "conflicting":
        caveats.append("Positive- and negative-regulator pathway annotations conflict")

    curated_evidence = None
    if allow_mirtarbase and any(_as_int(row.get(col), 0) == 1 for col in MIRTARBASE_KNOWN_POSITIVE_COLUMNS):
        curated_evidence = "Curated prior interaction support present in source data"
        caveats.append("Curated prior evidence is background support and is not one of the six family summaries")

    primary_evidence_by_category = {
        "curated": curated_evidence,
        "mirdb": None,
        "targetscan": family_summary["sequence_conservation"]["key_evidence"][0]
        if family_summary["sequence_conservation"]["key_evidence"]
        else None,
        "clip": family_summary["functional_binding"]["key_evidence"][0]
        if family_summary["functional_binding"]["key_evidence"]
        else None,
        "seed_site": family_summary["sequence_complementarity"]["key_evidence"][0]
        if family_summary["sequence_complementarity"]["key_evidence"]
        else None,
        "structure": family_summary["thermodynamic_stability"]["key_evidence"][0]
        if family_summary["thermodynamic_stability"]["key_evidence"]
        else None,
        "tcga": family_summary["functional_repression"]["key_evidence"][0]
        if family_summary["functional_repression"]["key_evidence"]
        else None,
        "pathway": pathway_evidence[0] if pathway_evidence else None,
    }

    strongest_features: List[str] = []
    for family in EVIDENCE_CATEGORY_LABELS:
        strongest_features.extend(family_summary[family]["key_evidence"][:1])
    strongest_features.extend(pathway_evidence[:1])
    strongest_features = _dedupe_keep_order(strongest_features)
    caveats = _dedupe_keep_order(caveats)

    raw_key_values: Dict[str, Any] = {}
    raw_keys = [
        "support_count",
        "mirassist_score",
        "mirassist_model_score",
        "mirassist_model_version",
        "mirassist_score_rank_within_mirna",
        "mirassist_filtered_rank",
        "mirassist_score_percentile_within_mirna",
        "mirassist_xgboost_score",
        "learned_score_used",
        "_learned_score_missing",
        "score_column_used",
        "overall_evidence_support_percentile",
        "evidence_family_count",
        "evidence_family_summary_json",
        "sequence_complementarity_available",
        "sequence_complementarity_support_percentile",
        "sequence_complementarity_evidence_count",
        "thermodynamic_stability_available",
        "thermodynamic_stability_support_percentile",
        "thermodynamic_stability_evidence_count",
        "sequence_conservation_available",
        "sequence_conservation_support_percentile",
        "sequence_conservation_evidence_count",
        "target_site_accessibility_available",
        "target_site_accessibility_support_percentile",
        "target_site_accessibility_evidence_count",
        "functional_binding_available",
        "functional_binding_support_percentile",
        "functional_binding_evidence_count",
        "functional_repression_available",
        "functional_repression_support_percentile",
        "functional_repression_evidence_count",
        "seed_match_type",
        "best_seed_site_type",
        "seed_pairing_score",
        "n_seed_sites",
        "rnahybrid_mfe",
        "rnahybrid_mfe_best_site",
        "rnahybrid_seed_mfe",
        "rnahybrid_strength",
        "targetscan_context_score",
        "targetscan_context_score_percentile",
        "targetscan_context_score_support_percentile",
        "targetscan_pct",
        "targetscan_conserved_site",
        "targetscan_aggregate_context_score",
        "clip_any_support",
        "clip_max_score",
        "clip_n_experiments",
        "clip_n_cell_lines",
        "encori_clip_score",
        "rnaplfold_best_seed_unpaired_prob",
        "rnaplfold_mean_seed_unpaired_prob",
        "rnaplfold_mean_site_unpaired_prob",
        "rnaplfold_n_sites_scored",
        "rnaplfold_n_accessible_sites",
        "tcga_n_supported_contexts",
        "tcga_mean_spearman_rho",
        "pathway_selected_gene",
        "predicted_mirna_effect_on_target",
        "expected_target_effect_on_phenotype",
        "target_role_evidence_status",
        "directionally_consistent",
        "directional_consistency_score",
    ]
    tcga_contexts = [tcga] if tcga else ["BRCA", "COAD", "PRAD"]
    for context in [str(item).upper() for item in tcga_contexts if item]:
        raw_keys.extend(
            [
                f"{context}_spearman_rho",
                f"{context}_repression_evidence",
                f"{context}_anticorrelated",
                f"{context}_support_tcga",
            ]
        )
    for key in raw_keys:
        value = _raw_value_if_present(row, key)
        if value is not None:
            raw_key_values[key] = value

    priority_fields = _compute_priority_fields(row, tcga, family_summary)

    sections = {
        "support_count": support_count,
        "evidence_categories": evidence_categories,
        "evidence_categories_present": evidence_categories_present,
        "evidence_families_present": evidence_categories_present,
        "evidence_support_count": evidence_support_count,
        "evidence_family_count": evidence_support_count,
        "overall_evidence_support_percentile": overall_evidence_support_percentile,
        "evidence_strength_summary": priority_fields["evidence_strength_summary"],
        "evidence_strength_tier": priority_fields["evidence_strength_tier"],
        "context_strength_tier": priority_fields["context_strength_tier"],
        "overall_priority_tier": priority_fields["overall_priority_tier"],
        "number_of_features_supporting_interaction": evidence_support_count,
        "family_evidence_summary": family_summary,
        "primary_curated_evidence": primary_evidence_by_category["curated"],
        "primary_mirdb_evidence": primary_evidence_by_category["mirdb"],
        "primary_targetscan_evidence": primary_evidence_by_category["targetscan"],
        "primary_clip_evidence": primary_evidence_by_category["clip"],
        "primary_seed_evidence": primary_evidence_by_category["seed_site"],
        "primary_structure_evidence": primary_evidence_by_category["structure"],
        "primary_tcga_evidence": primary_evidence_by_category["tcga"],
        "primary_pathway_evidence": primary_evidence_by_category["pathway"],
        "target_evidence": [curated_evidence] if curated_evidence else [],
        "published_model_evidence": family_summary["sequence_conservation"]["key_evidence"],
        "clip_binding_evidence": family_summary["functional_binding"]["key_evidence"],
        "seed_site_evidence": family_summary["sequence_complementarity"]["key_evidence"],
        "structure_evidence": family_summary["thermodynamic_stability"]["key_evidence"],
        "tcga_context_evidence": family_summary["functional_repression"]["key_evidence"],
        "pathway_evidence": pathway_evidence,
        "pathway_names": pathway_names,
        "predicted_mirna_effect_on_target": predicted_target_change,
        "expected_target_effect_on_phenotype": expected_target_role,
        "target_role_evidence": target_role_evidence,
        "positive_regulator_evidence": positive_regulator_evidence,
        "negative_regulator_evidence": negative_regulator_evidence,
        "target_role_evidence_status": target_role_status,
        "directionally_consistent": directionally_consistent,
        "strongest_features": strongest_features[:8],
        "caveats": caveats,
        "raw_key_values": raw_key_values,
    }
    return sections
