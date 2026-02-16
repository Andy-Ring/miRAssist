from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def _as_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default
        return int(x)
    except Exception:
        return default


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default


def _as_str_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x if v is not None]
    if isinstance(x, np.ndarray):
        return [str(v) for v in x.tolist() if v is not None]
    return [str(x)]


def cards_from_dataframe(df: pd.DataFrame, tcga: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Convert shortlist dataframe -> compact evidence cards consumed by prompting.py.
    """
    cards: List[Dict[str, Any]] = []
    if df is None or len(df) == 0:
        return cards

    tcga = tcga.upper() if tcga else None

    for _, row in df.iterrows():
        mirna = str(row.get("mirna_name", "") or "")
        gene = str(row.get("gene_symbol", "") or "")

        name = f"{gene} (← {mirna})" if gene and mirna else (gene or mirna or "UNKNOWN")
        support_count = _as_int(row.get("support_count", 0), 0)

        evidence_bits: List[str] = []

        if "mirtarbase_pos" in row:
            evidence_bits.append(f"miRTarBase(functional={_as_int(row.get('mirtarbase_pos', 0))})")

        if _as_int(row.get("support_encori", 0)) == 1:
            clip_sum = _as_float(row.get("clip_exp_sum", 0.0))
            n_sites = _as_int(row.get("n_clip_sites", 0))
            evidence_bits.append(f"ENCORI(CLIP_sum={clip_sum:g}, sites={n_sites})")

        if _as_int(row.get("support_targetscan", 0)) == 1:
            ctxpp_f = _as_float(row.get("ts_best_contextpp", float("nan")), default=float("nan"))
            ts_sites = _as_int(row.get("ts_n_sites", 0))
            if np.isnan(ctxpp_f):
                evidence_bits.append(f"TargetScan(sites={ts_sites})")
            else:
                evidence_bits.append(f"TargetScan(best_context++={ctxpp_f:.3f}, sites={ts_sites})")

        if _as_int(row.get("support_mirdb", 0)) == 1:
            score = _as_float(row.get("mirdb_best_score", 0.0))
            evidence_bits.append(f"miRDB(best={score:g})")

        # TCGA evidence is a separate line:
        # - anti_corr flag (preferred)
        # - rho + p shown as context/strength
        if tcga:
            rho_col = f"{tcga}_spearman_rho"
            p_col = f"{tcga}_spearman_p"
            antic_col = f"{tcga}_anticorrelated"
            rep_col = f"{tcga}_repression_evidence"

            antic = _as_int(row.get(antic_col, 0)) if antic_col in row else 0
            rep = _as_int(row.get(rep_col, 0)) if rep_col in row else 0

            rho = None
            p = None
            if rho_col in row:
                rho = _as_float(row.get(rho_col, float("nan")), default=float("nan"))
                if np.isnan(rho):
                    rho = None
            if p_col in row:
                p = _as_float(row.get(p_col, float("nan")), default=float("nan"))
                if np.isnan(p):
                    p = None

            tcga_bits = []
            if antic is not None:
                tcga_bits.append(f"anti_corr={antic}")
            if rho is not None:
                tcga_bits.append(f"rho={rho:.3f}")
            if p is not None:
                tcga_bits.append(f"p={p:.2g}")
            if rep == 1:
                tcga_bits.append("repression_flag=1")

            if tcga_bits:
                evidence_bits.append(f"TCGA({tcga} " + ", ".join(tcga_bits) + ")")

        notes_bits: List[str] = []
        if "cellline_tissue_set" in row and _as_int(row.get("support_encori", 0)) == 1:
            tissues = _as_str_list(row.get("cellline_tissue_set"))
            if tissues:
                notes_bits.append("ENCORI tissues: " + ", ".join(tissues[:6]) + (" ..." if len(tissues) > 6 else ""))

        evidence_line = f"support_count={support_count}; " + "; ".join(evidence_bits) if evidence_bits else f"support_count={support_count}"

        cards.append(
            {
                "name": name,
                "evidence": evidence_line,
                "notes": " | ".join(notes_bits) if notes_bits else None,
                "mirna_name": mirna,
                "gene_symbol": gene,
            }
        )

    return cards
