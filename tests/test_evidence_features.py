from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from backend.cards import cards_from_dataframe
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
        self.assertTrue(any("CLIP sites 5" in line and "exceptional" in line for line in card["clip_binding_evidence"]))
        self.assertTrue(any("retained by strict pathway filter" in line for line in card["pathway_evidence"]))
        self.assertEqual(card["pathway_names"][0], "apoptosis")
        self.assertIn("mirdb_best_score", card["raw_key_values"])

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
        self.assertIn("Number of features supporting interaction", bundle["user_prompt"])
        self.assertIn("Key pieces of evidence", bundle["system_prompt"])
        self.assertIn("Pathways:", bundle["user_prompt"])


if __name__ == "__main__":
    unittest.main()
