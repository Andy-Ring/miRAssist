from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.evidence_interpretation import build_evidence_sections


def _as_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default
        return int(x)
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
    Convert shortlist dataframe -> structured evidence cards consumed by prompting.py.
    """
    cards: List[Dict[str, Any]] = []
    if df is None or len(df) == 0:
        return cards

    tcga = tcga.upper() if tcga else None

    for _, row in df.iterrows():
        mirna = str(row.get("mirna_name", "") or "")
        gene = str(row.get("gene_symbol", "") or "")
        name = f"{gene} (<- {mirna})" if gene and mirna else (gene or mirna or "UNKNOWN")

        sections = build_evidence_sections(row, tcga=tcga)

        notes_bits: List[str] = []
        if "cellline_tissue_set" in row and _as_int(row.get("support_encori", 0)) == 1:
            tissues = _as_str_list(row.get("cellline_tissue_set"))
            if tissues:
                notes_bits.append(
                    "ENCORI tissues: "
                    + ", ".join(tissues[:6])
                    + (" ..." if len(tissues) > 6 else "")
                )

        evidence_line_parts: List[str] = []
        for section_name in [
            "target_evidence",
            "published_model_evidence",
            "clip_binding_evidence",
            "seed_site_evidence",
            "structure_evidence",
            "tcga_context_evidence",
            "pathway_evidence",
        ]:
            evidence_line_parts.extend(sections.get(section_name, []))

        evidence_line = (
            f"support_count={sections['support_count']}; " + "; ".join(evidence_line_parts)
            if evidence_line_parts
            else f"support_count={sections['support_count']}"
        )
        if sections.get("strongest_features"):
            evidence_line = evidence_line + "; strongest=" + "; ".join(sections["strongest_features"][:4])

        cards.append(
            {
                "candidate_name": name,
                "name": name,
                "evidence": evidence_line,
                "notes": " | ".join(notes_bits + list(sections.get("caveats") or [])) if (notes_bits or sections.get("caveats")) else None,
                "mirna_name": mirna,
                "gene_symbol": gene,
                "support_count": sections["support_count"],
                "number_of_features_supporting_interaction": sections["number_of_features_supporting_interaction"],
                "target_evidence": sections["target_evidence"],
                "published_model_evidence": sections["published_model_evidence"],
                "clip_binding_evidence": sections["clip_binding_evidence"],
                "seed_site_evidence": sections["seed_site_evidence"],
                "structure_evidence": sections["structure_evidence"],
                "tcga_context_evidence": sections["tcga_context_evidence"],
                "pathway_evidence": sections["pathway_evidence"],
                "pathway_names": sections["pathway_names"],
                "strongest_features": sections["strongest_features"],
                "caveats": sections["caveats"],
                "raw_key_values": sections["raw_key_values"],
            }
        )

    return cards


def cards_from_dataframe_with_diagnostics(
    df: pd.DataFrame,
    tcga: Optional[str] = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {
        "n_shortlist_rows": int(0 if df is None else len(df)),
        "n_cards_generated": 0,
        "card_errors": [],
    }
    cards: List[Dict[str, Any]] = []
    if df is None or len(df) == 0:
        return cards, diagnostics

    tcga = tcga.upper() if tcga else None

    for idx, (_, row) in enumerate(df.iterrows()):
        try:
            row_df = pd.DataFrame([row.to_dict()])
            row_cards = cards_from_dataframe(row_df, tcga=tcga)
            if row_cards:
                cards.extend(row_cards)
            else:
                diagnostics["card_errors"].append(
                    f"Row {idx} produced no card for {row.get('gene_symbol') or row.get('mirna_name') or 'UNKNOWN'}."
                )
        except Exception as exc:
            diagnostics["card_errors"].append(
                f"Row {idx} failed during card generation: {exc}"
            )

    diagnostics["n_cards_generated"] = int(len(cards))
    return cards, diagnostics
