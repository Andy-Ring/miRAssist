from __future__ import annotations

import pandas as pd

from backend.cards import cards_from_dataframe
from backend.prompting import build_prompt_bundle
from backend.retrieval import apply_learned_score_ranking


def _mir210_row(gene: str, score: float, family_count: int = 5) -> dict:
    return {
        "mirna_name": "hsa-miR-210-5p",
        "mirna_name_normalized": "hsa-mir-210-5p",
        "gene_symbol": gene,
        "support_count": family_count,
        "retrieval_score": 0.0,
        "mirassist_xgboost_score": score,
        "evidence_family_count": family_count,
        "overall_evidence_support_percentile": 80.0,
        "sequence_complementarity_available": True,
        "sequence_complementarity_support_percentile": 70.0,
        "sequence_complementarity_evidence_count": 3,
        "thermodynamic_stability_available": True,
        "thermodynamic_stability_support_percentile": 85.0,
        "thermodynamic_stability_evidence_count": 3,
        "sequence_conservation_available": True,
        "sequence_conservation_support_percentile": 65.0,
        "sequence_conservation_evidence_count": 2,
        "target_site_accessibility_available": True,
        "target_site_accessibility_support_percentile": 55.0,
        "target_site_accessibility_evidence_count": 2,
        "functional_binding_available": False,
        "functional_binding_support_percentile": None,
        "functional_binding_evidence_count": 0,
        "functional_repression_available": True,
        "functional_repression_support_percentile": 90.0,
        "functional_repression_evidence_count": 4,
        "seed_match_type": "7mer-m8",
        "best_seed_site_type": "7mer-m8",
        "n_seed_sites": 2,
        "seed_pairing_score": 0.8,
        "rnahybrid_mfe": -34.7,
        "rnahybrid_strength": 34.7,
        "rnahybrid_seed_mfe": -8.2,
        "targetscan_context_score": -0.22,
        "targetscan_context_score_support_percentile": 74.0,
        "targetscan_pct": 0.62,
        "targetscan_conserved_site": True,
        "rnaplfold_best_seed_unpaired_prob": 0.0354885513749999,
        "rnaplfold_mean_seed_unpaired_prob": 0.051,
        "rnaplfold_n_sites_scored": 2,
        "rnaplfold_n_accessible_sites": 1,
        "BRCA_spearman_rho": -0.18,
        "BRCA_repression_evidence": True,
        "BRCA_anticorrelated": True,
        "BRCA_support_tcga": True,
        "COAD_spearman_rho": -0.11,
        "COAD_repression_evidence": True,
        "COAD_anticorrelated": True,
        "COAD_support_tcga": True,
        "PRAD_spearman_rho": -0.07,
        "PRAD_repression_evidence": False,
        "PRAD_anticorrelated": True,
        "PRAD_support_tcga": False,
        "tcga_mean_spearman_rho": -0.12,
        "tcga_n_supported_contexts": 2,
    }


def test_mir210_mocked_supabase_ranking_and_cards_preserve_xgboost_order() -> None:
    df = pd.DataFrame(
        [
            _mir210_row("APBB3", 0.521383225917816),
            _mir210_row("PDRG1", 0.18, family_count=1),
            _mir210_row("SCN7A", 0.482584953308105),
            _mir210_row("NIPAL1", 0.12, family_count=1),
            _mir210_row("GIGYF1", 0.323723077774048),
            _mir210_row("MOB4", 0.10, family_count=1),
            _mir210_row("FBXO28", 0.253126889467239),
            _mir210_row("BTG2", 0.247091025114059),
        ]
    )

    ranked, diagnostics = apply_learned_score_ranking(
        df,
        learned_score_column="learned_score_xgb_raw_v1",
        enabled=True,
    )
    top_five = ranked.head(5).reset_index(drop=True)

    assert top_five["gene_symbol"].tolist() == ["APBB3", "SCN7A", "GIGYF1", "FBXO28", "BTG2"]
    assert diagnostics["score_column_used"] == "mirassist_xgboost_score"

    cards = cards_from_dataframe(top_five)
    assert [card["gene_symbol"] for card in cards] == ["APBB3", "SCN7A", "GIGYF1", "FBXO28", "BTG2"]
    assert cards[0]["evidence_family_count"] == 5
    assert "Sequence complementarity" in cards[0]["evidence_families_present"]
    assert "Thermodynamic stability" in cards[0]["evidence_families_present"]
    assert "Target site accessibility" in cards[0]["evidence_families_present"]
    assert "Functional repression" in cards[0]["evidence_families_present"]

    bundle = build_prompt_bundle(
        queryspec={
            "original_question": "What are the targets of miR-210",
            "mode": "mirna_to_targets",
            "mirna": "hsa-miR-210-5p",
            "k": 10,
        },
        shortlist=top_five,
        direction="mirna_to_targets",
        cards=cards,
        retrieval_diagnostics={"score_column_used": "mirassist_xgboost_score"},
    )
    sent_order = bundle["meta"]["candidate_order_sent_to_llm"]
    assert [item["gene_symbol"] for item in sent_order[:5]] == ["APBB3", "SCN7A", "GIGYF1", "FBXO28", "BTG2"]
    assert "Preserve the available evidence card order" in bundle["user_prompt"]
    assert "Cancer context: No specific cancer context requested" in bundle["user_prompt"]
