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

        cards = cards_from_dataframe(annotated, tcga="BRCA")
        card = cards[0]

        self.assertTrue(any("miRDB score 100" in line for line in card["published_model_evidence"]))
        self.assertTrue(any("TargetScan context strength" in line for line in card["published_model_evidence"]))
        self.assertTrue(any("CLIP signal 10" in line and "exceptional" in line for line in card["clip_binding_evidence"]))
        self.assertTrue(any("Retained by strict pathway filter" in line for line in card["pathway_evidence"]))
        self.assertEqual(card["pathway_names"][0], "apoptosis")
        self.assertIn("mirdb_best_score", card["raw_key_values"])
        self.assertEqual(card["evidence_support_count"], 8)
        self.assertIn("miRTarBase", card["evidence_categories_present"])
        self.assertIn("TargetScan", card["evidence_categories_present"])
        self.assertIn("Seed/site", card["evidence_categories_present"])
        self.assertIn("RNAhybrid/structure", card["evidence_categories_present"])
        self.assertEqual(card["primary_mirdb_evidence"], "miRDB score 100 (100th percentile; exceptional; very strong model support)")
        self.assertEqual(card["primary_clip_evidence"], "CLIP signal 10 (100th percentile; exceptional)")
        self.assertIn("TargetScan context strength", card["primary_targetscan_evidence"])
        self.assertNotIn("miRDB mean score", " ".join(card["published_model_evidence"]))

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
                "novel": True,
                "k": 2,
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
        self.assertIn("Requested ranked results: 2", bundle["user_prompt"])
        self.assertIn("Evidence support count", bundle["user_prompt"])
        self.assertIn("Evidence categories:", bundle["user_prompt"])
        self.assertIn("Key pieces of evidence", bundle["system_prompt"])
        self.assertIn("Pathways:", bundle["user_prompt"])

    def test_evidence_support_count_uses_categories_not_percentiles(self) -> None:
        row = pd.Series(
            {
                "mirtarbase_pos": 1,
                "mirdb_best_score": 91.5,
                "mirdb_best_score_percentile": 96.0,
                "mirdb_best_score_label": "exceptional",
                "ts_context_strength": 0.932,
                "ts_context_strength_percentile": 100.0,
                "ts_context_strength_label": "exceptional",
                "n_clip_sites": 7,
                "n_clip_sites_percentile": 99.0,
                "n_clip_sites_label": "exceptional",
                "clip_exp_sum": 25.0,
                "clip_exp_sum_percentile": 98.0,
                "clip_exp_sum_label": "exceptional",
                "BRCA_spearman_rho": -0.014,
                "BRCA_support_tcga": 1,
                "pathway_selected_gene": 1,
                "pathway_selected_names": ["HALLMARK_OXIDATIVE_PHOSPHORYLATION"],
            }
        )

        sections = build_evidence_sections(row, tcga="BRCA")

        self.assertEqual(sections["evidence_support_count"], 6)
        self.assertEqual(
            sections["evidence_categories_present"],
            ["miRTarBase", "miRDB", "TargetScan", "CLIP", "TCGA context", "Pathway"],
        )

    def test_clip_category_counts_once_even_with_multiple_metrics_and_percentiles(self) -> None:
        row = pd.Series(
            {
                "n_clip_sites": 4,
                "n_clip_sites_percentile": 95.0,
                "n_clip_sites_label": "exceptional",
                "clip_exp_sum": 20.0,
                "clip_exp_sum_percentile": 97.0,
                "clip_exp_sum_label": "exceptional",
                "clip_exp_max": 10.0,
                "clip_exp_max_percentile": 96.0,
                "clip_exp_max_label": "exceptional",
            }
        )

        sections = build_evidence_sections(row)
        self.assertEqual(sections["evidence_support_count"], 1)
        self.assertEqual(sections["evidence_categories_present"], ["CLIP"])

    def test_targetscan_category_counts_once_even_with_multiple_fields(self) -> None:
        row = pd.Series(
            {
                "ts_context_strength": 0.8,
                "ts_context_strength_percentile": 93.0,
                "ts_context_strength_label": "very high",
                "ts_best_percentile": 98.0,
                "ts_best_percentile_percentile": 90.0,
                "ts_best_percentile_label": "very high",
            }
        )

        sections = build_evidence_sections(row)
        self.assertEqual(sections["evidence_support_count"], 1)
        self.assertEqual(sections["evidence_categories_present"], ["TargetScan"])

    def test_format_percentile_uses_correct_ordinals(self) -> None:
        self.assertEqual(format_percentile(1), "1st percentile")
        self.assertEqual(format_percentile(2), "2nd percentile")
        self.assertEqual(format_percentile(3), "3rd percentile")
        self.assertEqual(format_percentile(4), "4th percentile")
        self.assertEqual(format_percentile(11), "11th percentile")
        self.assertEqual(format_percentile(21), "21st percentile")
        self.assertEqual(format_percentile(93), "93rd percentile")

    def test_mirdb_primary_display_uses_best_score_only(self) -> None:
        row = pd.Series(
            {
                "mirdb_best_score": 91.5,
                "mirdb_best_score_percentile": 84.0,
                "mirdb_best_score_label": "high",
                "mirdb_mean_score": 89.4,
                "mirdb_mean_score_percentile": 80.0,
                "mirdb_mean_score_label": "high",
            }
        )

        sections = build_evidence_sections(row)
        self.assertEqual(
            sections["primary_mirdb_evidence"],
            "miRDB score 91.5 (84th percentile; high; very strong model support)",
        )
        self.assertEqual(len(sections["published_model_evidence"]), 1)

    def test_clip_primary_display_uses_clip_sum_over_clip_max(self) -> None:
        row = pd.Series(
            {
                "clip_exp_sum": 25.0,
                "clip_exp_sum_percentile": 98.0,
                "clip_exp_sum_label": "exceptional",
                "clip_exp_max": 11.0,
                "clip_exp_max_percentile": 90.0,
                "clip_exp_max_label": "very high",
                "n_clip_sites": 7,
                "n_clip_sites_percentile": 99.0,
                "n_clip_sites_label": "exceptional",
            }
        )

        sections = build_evidence_sections(row)
        self.assertEqual(
            sections["primary_clip_evidence"],
            "CLIP signal 25 (98th percentile; exceptional)",
        )
        self.assertEqual(sections["clip_binding_evidence"], ["CLIP signal 25 (98th percentile; exceptional)"])

    def test_targetscan_primary_display_uses_context_strength(self) -> None:
        row = pd.Series(
            {
                "ts_context_strength": 0.932,
                "ts_context_strength_percentile": 100.0,
                "ts_context_strength_label": "exceptional",
                "ts_best_percentile": 99.0,
                "ts_best_percentile_percentile": 100.0,
                "ts_best_percentile_label": "exceptional",
            }
        )

        sections = build_evidence_sections(row)
        self.assertEqual(
            sections["primary_targetscan_evidence"],
            "TargetScan context strength 0.932 (100th percentile; exceptional)",
        )
        self.assertEqual(
            sections["published_model_evidence"],
            ["TargetScan context strength 0.932 (100th percentile; exceptional)"],
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


if __name__ == "__main__":
    unittest.main()
