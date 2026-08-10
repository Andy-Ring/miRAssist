from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from backend.cards import cards_from_dataframe
from backend.evidence_interpretation import build_evidence_sections, format_percentile
from backend.feature_stats import annotate_feature_percentiles
from backend.prompting import build_prompt_bundle


class FeaturePercentileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ev = pd.DataFrame(
            [
                {
                    "mirna_name": "hsa-let-7a-2-3p",
                    "gene_symbol": f"GENE{i}",
                    "support_count": i,
                    "mirdb_best_score": 50 + (i * 10),
                    "mirdb_mean_score": 45 + (i * 10),
                    "ts_best_contextpp": -0.10 * i,
                    "n_clip_sites": i,
                    "clip_exp_sum": i * 2.0,
                    "clip_exp_max": i * 1.5,
                    "n_sites_8mer": 1 if i >= 4 else 0,
                    "n_sites_7mer_m8": 1 if i >= 3 else 0,
                    "n_sites_7mer_a1": 1 if i >= 2 else 0,
                    "n_sites_6mer": i,
                    "best_local_au": 0.10 * i,
                    "has_rnahybrid": 1,
                    "n_rnahybrid_sites": i,
                    "best_mfe": -10.0 - (i * 3.0),
                    "mean_top3_mfe": -8.0 - (i * 2.0),
                    "n_sites_mfe_lt_-20": 1 if i >= 4 else 0,
                    "n_sites_mfe_lt_-25": 1 if i >= 5 else 0,
                    "BRCA_spearman_rho": -0.05 * i,
                }
                for i in range(1, 6)
            ]
        )

    def test_annotate_feature_percentiles_marks_high_values(self) -> None:
        shortlist = self.ev.iloc[[4]].copy()
        annotated = annotate_feature_percentiles(shortlist, self.ev)
        row = annotated.iloc[0]

        self.assertGreaterEqual(row["support_count_percentile"], 95.0)
        self.assertEqual(row["support_count_label"], "exceptional")
        self.assertGreaterEqual(row["n_clip_sites_percentile"], 95.0)
        self.assertEqual(row["n_clip_sites_label"], "exceptional")
        self.assertGreaterEqual(row["mfe_strength_percentile"], 95.0)
        self.assertEqual(row["mfe_strength_label"], "exceptional")

    def test_annotate_feature_percentiles_handles_nan(self) -> None:
        shortlist = pd.DataFrame(
            [
                {
                    "mirna_name": "hsa-let-7a-2-3p",
                    "gene_symbol": "GENE_NA",
                    "support_count": np.nan,
                    "mirdb_best_score": np.nan,
                    "n_clip_sites": np.nan,
                    "best_mfe": np.nan,
                }
            ]
        )
        annotated = annotate_feature_percentiles(shortlist, self.ev)
        row = annotated.iloc[0]

        self.assertTrue(pd.isna(row["support_count_percentile"]))
        self.assertEqual(row["support_count_label"], "not available")
        self.assertTrue(pd.isna(row["n_clip_sites_percentile"]))
        self.assertEqual(row["n_clip_sites_label"], "not available")

    def test_cards_include_raw_values_and_labels(self) -> None:
        shortlist = self.ev.iloc[[4]].copy()
        shortlist["mirtarbase_pos"] = 1
        shortlist["pathway_selected_gene"] = 1
        shortlist["pathway_selected_names"] = [["apoptosis", "negative regulation of apoptotic process"]]
        shortlist["BRCA_support_tcga"] = 1
        annotated = annotate_feature_percentiles(shortlist, self.ev)

        card = cards_from_dataframe(annotated, tcga="BRCA")[0]

        self.assertFalse(card["primary_mirdb_evidence"])
        self.assertTrue(any("TargetScan context score" in line for line in card["published_model_evidence"]))
        self.assertTrue(any("CLIP max score 7.5" in line for line in card["clip_binding_evidence"]))
        self.assertTrue(any("Retained by strict pathway filter" in line for line in card["pathway_evidence"]))
        self.assertEqual(card["pathway_names"][0], "apoptosis")
        self.assertEqual(card["evidence_support_count"], 5)
        self.assertEqual(
            card["evidence_categories_present"],
            [
                "Sequence complementarity",
                "Thermodynamic stability",
                "Sequence conservation",
                "Functional binding",
                "Functional repression",
            ],
        )
        self.assertIn("TargetScan context score", card["primary_targetscan_evidence"])
        self.assertIn("CLIP max score", card["primary_clip_evidence"])

    def test_prompt_bundle_uses_required_headings(self) -> None:
        shortlist = self.ev.iloc[[4, 3]].copy()
        shortlist["mirtarbase_pos"] = [0, 1]
        shortlist["pathway_selected_gene"] = [1, 1]
        shortlist["pathway_selected_names"] = [
            ["apoptosis"],
            ["apoptosis", "negative regulation of apoptotic process"],
        ]
        shortlist["BRCA_support_tcga"] = [1, 0]
        annotated = annotate_feature_percentiles(shortlist, self.ev)

        bundle = build_prompt_bundle(
            queryspec={
                "original_question": "what genes are regulated by hsa-let-7a-2-3p?",
                "mode": "mirna_to_targets",
                "mirna": "hsa-let-7a-2-3p",
                "cancer": {"name": "breast cancer", "tcga": "BRCA"},
                "phenotype_context": {
                    "phenotype": "proliferation",
                    "observed_change": "increased",
                    "miRNA_perturbation": "overexpression",
                    "direction": "increases",
                    "raw_phrase": "I overexpressed hsa-let-7a-2-3p and proliferation increased.",
                },
                "target_role_inference": {
                    "enabled": True,
                    "assumption": "miRNAs usually repress target gene expression",
                    "expected_target_effect_on_phenotype": "negative_regulator",
                    "reasoning": "The user reported miRNA overexpression increased proliferation. Since miRNAs usually repress target genes, miRAssist prioritized negative regulators of proliferation.",
                },
                "novel": True,
                "k": 10,
                "pathway_selection": {
                    "enabled": True,
                    "selected_pathways": [{"pathway_name": "apoptosis"}],
                },
            },
            shortlist=annotated,
            direction="mirna_to_targets",
            meta={"queryspec": {"mirna": "hsa-let-7a-2-3p"}},
        )

        self.assertIn("## Interpretation", bundle["system_prompt"])
        self.assertIn("## Results", bundle["system_prompt"])
        self.assertIn("## Final recommendation", bundle["system_prompt"])
        self.assertIn("Requested ranked results: 5", bundle["user_prompt"])
        self.assertIn("Available evidence cards: 2", bundle["user_prompt"])
        self.assertIn("Evidence support count", bundle["user_prompt"])
        self.assertIn("Evidence families:", bundle["user_prompt"])
        self.assertIn("Key pieces of evidence", bundle["system_prompt"])
        self.assertIn("Pathways:", bundle["user_prompt"])
        self.assertIn("Overall priority", bundle["system_prompt"])
        self.assertIn("breadth, not strength", bundle["system_prompt"])
        self.assertIn("Target-role interpretation:", bundle["user_prompt"])
        self.assertIn("deterministic pathway annotations", bundle["user_prompt"])
        self.assertIn("Preserve the provided candidate order", bundle["system_prompt"])
        self.assertIn("backend ranking order", bundle["user_prompt"])

    def test_evidence_support_count_uses_categories_not_percentiles(self) -> None:
        row = pd.Series(
            {
                "n_seed_sites": 2,
                "rnahybrid_mfe": -22.0,
                "targetscan_context_score": -0.4,
                "rnaplfold_best_seed_unpaired_prob": 0.8,
                "clip_max_score": 9.0,
                "BRCA_spearman_rho": -0.2,
                "BRCA_support_tcga": 1,
                "mirtarbase_known_positive": True,
                "pathway_selected_gene": 1,
                "pathway_selected_names": ["HALLMARK_OXIDATIVE_PHOSPHORYLATION"],
            }
        )

        sections = build_evidence_sections(row, tcga="BRCA")

        self.assertEqual(sections["evidence_support_count"], 6)
        self.assertEqual(
            sections["evidence_categories_present"],
            [
                "Sequence complementarity",
                "Thermodynamic stability",
                "Sequence conservation",
                "Target site accessibility",
                "Functional binding",
                "Functional repression",
            ],
        )
        self.assertNotIn("miRTarBase", sections["evidence_categories_present"])
        self.assertNotIn("Pathway", sections["evidence_categories_present"])

    def test_clip_category_counts_once_even_with_multiple_metrics_and_percentiles(self) -> None:
        row = pd.Series(
            {
                "clip_max_score": 10.0,
                "clip_max_score_percentile": 96.0,
                "clip_n_experiments": 20.0,
                "clip_n_experiments_percentile": 97.0,
                "encori_clip_score": 5.0,
                "encori_clip_score_percentile": 95.0,
            }
        )

        sections = build_evidence_sections(row)
        self.assertEqual(sections["evidence_support_count"], 1)
        self.assertEqual(sections["evidence_categories_present"], ["Functional binding"])

    def test_targetscan_category_counts_once_even_with_multiple_fields(self) -> None:
        row = pd.Series(
            {
                "targetscan_context_score": -0.8,
                "targetscan_context_score_support_percentile": 93.0,
                "targetscan_aggregate_context_score": -1.2,
                "targetscan_conserved_site": 1,
            }
        )

        sections = build_evidence_sections(row)
        self.assertEqual(sections["evidence_support_count"], 1)
        self.assertEqual(sections["evidence_categories_present"], ["Sequence conservation"])

    def test_format_percentile_uses_correct_ordinals(self) -> None:
        self.assertEqual(format_percentile(1), "1st percentile")
        self.assertEqual(format_percentile(2), "2nd percentile")
        self.assertEqual(format_percentile(3), "3rd percentile")
        self.assertEqual(format_percentile(4), "4th percentile")
        self.assertEqual(format_percentile(11), "11th percentile")
        self.assertEqual(format_percentile(21), "21st percentile")
        self.assertEqual(format_percentile(93), "93rd percentile")

    def test_legacy_mirdb_fields_do_not_create_variant_a_family(self) -> None:
        row = pd.Series(
            {
                "mirdb_best_score": 91.5,
                "mirdb_best_score_percentile": 84.0,
                "mirdb_mean_score": 89.4,
            }
        )

        sections = build_evidence_sections(row)
        self.assertIsNone(sections["primary_mirdb_evidence"])
        self.assertEqual(sections["published_model_evidence"], [])
        self.assertEqual(sections["evidence_support_count"], 0)

    def test_clip_primary_display_uses_approved_clip_fields(self) -> None:
        row = pd.Series(
            {
                "clip_max_score": 11.0,
                "clip_max_score_percentile": 90.0,
                "clip_n_experiments": 25.0,
                "clip_n_experiments_percentile": 98.0,
            }
        )

        sections = build_evidence_sections(row)
        self.assertEqual(
            sections["primary_clip_evidence"],
            "CLIP max score 11 (90th percentile; very high)",
        )
        self.assertEqual(
            sections["clip_binding_evidence"],
            [
                "CLIP max score 11 (90th percentile; very high)",
                "CLIP-supported experiments 25 (98th percentile; exceptional)",
            ],
        )

    def test_targetscan_primary_display_uses_context_score(self) -> None:
        row = pd.Series(
            {
                "targetscan_context_score": -0.932,
                "targetscan_context_score_support_percentile": 100.0,
                "targetscan_aggregate_context_score": -1.2,
            }
        )

        sections = build_evidence_sections(row)
        self.assertEqual(
            sections["primary_targetscan_evidence"],
            "TargetScan context score -0.932 (100th percentile; exceptional; more negative is stronger)",
        )
        self.assertEqual(
            sections["published_model_evidence"],
            ["TargetScan context score -0.932 (100th percentile; exceptional; more negative is stronger)"],
        )

    def test_seed_site_zero_sites_not_presented_as_positive_key_evidence(self) -> None:
        row = pd.Series(
            {
                "n_total_sites": 0,
                "n_total_sites_percentile": 13.0,
                "n_total_sites_label": "low",
            }
        )

        sections = build_evidence_sections(row)
        self.assertIsNone(sections["primary_seed_evidence"])
        self.assertEqual(sections["seed_site_evidence"], [])

    def test_priority_fields_distinguish_breadth_and_strength(self) -> None:
        broad_but_weak = pd.Series(
            {
                "sequence_complementarity_available": True,
                "sequence_complementarity_support_percentile": 40.0,
                "thermodynamic_stability_available": True,
                "thermodynamic_stability_support_percentile": 40.0,
                "sequence_conservation_available": True,
                "sequence_conservation_support_percentile": 40.0,
                "functional_binding_available": True,
                "functional_binding_support_percentile": 40.0,
            }
        )
        broad_and_strong = pd.Series(
            {
                "sequence_complementarity_available": True,
                "sequence_complementarity_support_percentile": 90.0,
                "thermodynamic_stability_available": True,
                "thermodynamic_stability_support_percentile": 90.0,
                "sequence_conservation_available": True,
                "sequence_conservation_support_percentile": 90.0,
                "functional_binding_available": True,
                "functional_binding_support_percentile": 90.0,
            }
        )

        weak_sections = build_evidence_sections(broad_but_weak)
        strong_sections = build_evidence_sections(broad_and_strong)

        self.assertEqual(weak_sections["overall_priority_tier"], "Exploratory")
        self.assertIn("breadth or strength is limited", weak_sections["evidence_strength_summary"])
        self.assertEqual(strong_sections["overall_priority_tier"], "Strong")
        self.assertIn("broad support", strong_sections["evidence_strength_summary"])


if __name__ == "__main__":
    unittest.main()
