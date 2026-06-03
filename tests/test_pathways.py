from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from backend.cards import cards_from_dataframe_with_diagnostics
from backend.planner import _validate_and_fill
from backend.pathways import compact_pathway_selection, resolve_pathway_selection
from backend.retrieval import retrieve_from_queryspec
from backend.worker import _apply_query_overrides


def _base_queryspec() -> dict:
    return {
        "original_question": "Does miR-34a promote apoptosis?",
        "mode": "mirna_to_targets",
        "mirna": "miR-34a",
        "gene": None,
        "cancer": {"name": None, "tcga": None},
        "phenotype_context": {
            "phenotype": "apoptosis",
            "direction": "promotes",
            "raw_phrase": "promotes apoptosis",
        },
        "pathway_selection_request": {
            "enabled": True,
            "query_terms": ["apoptosis"],
            "directional_query_terms": ["positive regulation of apoptosis"],
            "strict": True,
        },
        "phenotype_keywords": ["apoptosis"],
        "pathway_keywords": [],
        "pathway_filter": {"enabled": True, "mode": "filter", "min_gene_sets": 1},
        "novel": False,
        "k": 25,
        "filters": {
            "min_support": 1,
            "require_binding_evidence": False,
            "require_expression": False,
        },
        "needs_clarification": [],
    }


class PathwayResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pathways_df = pd.DataFrame(
            [
                {
                    "pathway_id": "GO:1",
                    "pathway_name": "positive regulation of apoptosis",
                    "description": "positive regulation of apoptotic process",
                    "category": "GO",
                },
                {
                    "pathway_id": "GO:2",
                    "pathway_name": "negative regulation of apoptosis",
                    "description": "negative regulation of apoptotic process",
                    "category": "GO",
                },
                {
                    "pathway_id": "H:1",
                    "pathway_name": "HALLMARK_GLYCOLYSIS",
                    "description": "glycolysis energy metabolism program",
                    "category": "HALLMARK",
                },
                {
                    "pathway_id": "H:2",
                    "pathway_name": "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
                    "description": "oxidative phosphorylation mitochondrial function",
                    "category": "HALLMARK",
                },
            ]
        )
        self.gene_to_pathways_df = pd.DataFrame(
            [
                {"gene_symbol": "BAX", "pathway_id": "GO:1", "pathway_name": "positive regulation of apoptosis"},
                {"gene_symbol": "CASP3", "pathway_id": "GO:1", "pathway_name": "positive regulation of apoptosis"},
                {"gene_symbol": "BCL2", "pathway_id": "GO:2", "pathway_name": "negative regulation of apoptosis"},
                {"gene_symbol": "LDHA", "pathway_id": "H:1", "pathway_name": "HALLMARK_GLYCOLYSIS"},
                {"gene_symbol": "NDUFA4", "pathway_id": "H:2", "pathway_name": "HALLMARK_OXIDATIVE_PHOSPHORYLATION"},
            ]
        )

    def test_promotes_apoptosis_selects_positive_pathways(self) -> None:
        qs = _base_queryspec()
        with patch("backend.pathways.load_pathways", return_value=self.pathways_df), patch(
            "backend.pathways.load_gene_to_pathways", return_value=self.gene_to_pathways_df
        ):
            selection = resolve_pathway_selection(qs)

        self.assertTrue(selection["enabled"])
        self.assertEqual(selection["mode"], "filter")
        self.assertEqual(selection["phenotype"], "apoptosis")
        self.assertEqual(selection["direction"], "promotes")
        self.assertGreaterEqual(selection["n_selected_pathways"], 1)
        self.assertIn("BAX", selection["_selected_gene_set"])

    def test_suppresses_apoptosis_selects_negative_pathways(self) -> None:
        qs = _base_queryspec()
        qs["phenotype_context"]["direction"] = "suppresses"
        qs["phenotype_context"]["raw_phrase"] = "suppresses apoptosis"
        qs["pathway_selection_request"]["directional_query_terms"] = [
            "negative regulation of apoptosis"
        ]
        with patch("backend.pathways.load_pathways", return_value=self.pathways_df), patch(
            "backend.pathways.load_gene_to_pathways", return_value=self.gene_to_pathways_df
        ):
            selection = resolve_pathway_selection(qs)

        selected_names = [item["pathway_name"] for item in selection["selected_pathways"]]
        self.assertIn("negative regulation of apoptosis", selected_names)
        self.assertIn("BCL2", selection["_selected_gene_set"])

    def test_missing_files_no_context_does_not_crash(self) -> None:
        qs = _base_queryspec()
        qs["phenotype_context"] = {"phenotype": None, "direction": None, "raw_phrase": None}
        qs["pathway_selection_request"] = {
            "enabled": False,
            "query_terms": [],
            "directional_query_terms": [],
            "strict": False,
        }
        qs["phenotype_keywords"] = []
        qs["pathway_filter"] = {"enabled": False, "mode": "filter", "min_gene_sets": 1}

        with patch("backend.pathways.load_pathways", side_effect=RuntimeError("missing")):
            selection = resolve_pathway_selection(qs)

        self.assertFalse(selection["enabled"])
        self.assertEqual(selection["warnings"], [])

    def test_missing_files_with_context_returns_warning(self) -> None:
        qs = _base_queryspec()
        with patch("backend.pathways.load_pathways", side_effect=RuntimeError("missing files")):
            selection = resolve_pathway_selection(qs)

        self.assertTrue(selection["enabled"])
        self.assertEqual(selection["_selected_gene_set"], set())
        self.assertTrue(selection["warnings"])

    def test_energy_metabolism_expands_to_hallmark_terms(self) -> None:
        qs = {
            "phenotype_context": {
                "phenotype": "energy metabolism",
                "direction": "associated",
                "raw_phrase": "involved in energy metabolism",
            },
            "pathway_selection_request": {
                "enabled": True,
                "query_terms": [],
                "directional_query_terms": [],
                "strict": True,
            },
            "phenotype_keywords": ["energy metabolism"],
            "pathway_keywords": [],
            "pathway_filter": {"enabled": True, "mode": "filter", "min_gene_sets": 1},
        }
        with patch("backend.pathways.load_pathways", return_value=self.pathways_df), patch(
            "backend.pathways.load_gene_to_pathways", return_value=self.gene_to_pathways_df
        ):
            selection = resolve_pathway_selection(qs)

        self.assertIn("glycolysis", [term.lower() for term in selection["query_terms"]])
        self.assertIn("oxidative phosphorylation", [term.lower() for term in selection["query_terms"]])
        selected_names = [item["pathway_name"] for item in selection["selected_pathways"]]
        self.assertIn("HALLMARK_GLYCOLYSIS", selected_names)
        self.assertIn("HALLMARK_OXIDATIVE_PHOSPHORYLATION", selected_names)
        self.assertEqual(compact_pathway_selection(selection)["selected_gene_examples"], ["LDHA", "NDUFA4"])


class RetrievalPathwayFilterTests(unittest.TestCase):
    def test_retrieval_filters_to_selected_genes(self) -> None:
        ev = pd.DataFrame(
            [
                {"mirna_name": "miR-34a", "gene_symbol": "BAX", "support_count": 3, "ts_best_contextpp": -0.3},
                {"mirna_name": "miR-34a", "gene_symbol": "TP53", "support_count": 3, "ts_best_contextpp": -0.2},
            ]
        )
        qs = _base_queryspec()
        pathway_selection = {
            "enabled": True,
            "mode": "filter",
            "_selected_gene_set": {"BAX"},
            "_selected_gene_pathways": {"BAX": ["positive regulation of apoptosis"]},
            "selected_pathways": [{"pathway_id": "GO:1", "pathway_name": "positive regulation of apoptosis", "matched_terms": ["apoptosis"]}],
            "warnings": [],
        }

        shortlist, _, diagnostics = retrieve_from_queryspec(ev, qs, pathway_selection=pathway_selection)

        self.assertEqual(shortlist["gene_symbol"].tolist(), ["BAX"])
        self.assertEqual(shortlist["pathway_selected_gene"].tolist(), [1])
        self.assertEqual(shortlist["pathway_match_count"].tolist(), [1])
        self.assertEqual(diagnostics["n_after_pathway_filter"], 1)
        self.assertEqual(diagnostics["n_candidate_genes_removed_by_pathway_filter"], 1)

    def test_mirna_210_matching_and_diagnostics(self) -> None:
        ev = pd.DataFrame(
            [
                {"mirna_name": "MiR-210", "gene_symbol": "LDHA", "support_count": 3, "ts_best_contextpp": -0.3},
                {"mirna_name": "hsa-miR-210-3p", "gene_symbol": "NDUFA4", "support_count": 2, "ts_best_contextpp": -0.25},
                {"mirna_name": "miR-21", "gene_symbol": "TP53", "support_count": 4, "ts_best_contextpp": -0.4},
            ]
        )
        qs = {
            "mode": "mirna_to_targets",
            "mirna": "miRNA-210",
            "gene": None,
            "cancer": {"name": "breast cancer", "tcga": "BRCA"},
            "pathway_filter": {"enabled": True, "mode": "filter", "min_gene_sets": 1},
            "filters": {"min_support": 1, "require_binding_evidence": False, "require_expression": False},
            "novel": False,
            "k": 10,
        }
        pathway_selection = {
            "enabled": True,
            "mode": "filter",
            "_selected_gene_set": {"LDHA", "NDUFA4"},
            "_selected_gene_pathways": {
                "LDHA": ["HALLMARK_GLYCOLYSIS"],
                "NDUFA4": ["HALLMARK_OXIDATIVE_PHOSPHORYLATION"],
            },
            "selected_pathways": [
                {"pathway_id": "H:1", "pathway_name": "HALLMARK_GLYCOLYSIS", "matched_terms": ["glycolysis"]},
                {"pathway_id": "H:2", "pathway_name": "HALLMARK_OXIDATIVE_PHOSPHORYLATION", "matched_terms": ["oxidative phosphorylation"]},
            ],
            "warnings": [],
        }

        shortlist, direction, diagnostics = retrieve_from_queryspec(ev, qs, pathway_selection=pathway_selection)
        cards, card_diagnostics = cards_from_dataframe_with_diagnostics(shortlist, tcga="BRCA")

        self.assertEqual(direction, "mirna_to_targets")
        self.assertEqual(diagnostics["query_mirna_raw"], "miRNA-210")
        self.assertEqual(diagnostics["query_mirna_normalized"], "mir-210")
        self.assertIn("MiR-210", diagnostics["matched_mirna_names"])
        self.assertGreaterEqual(diagnostics["n_final_shortlist"], 1)
        self.assertEqual(card_diagnostics["n_shortlist_rows"], len(shortlist))
        self.assertEqual(card_diagnostics["n_cards_generated"], len(shortlist))
        self.assertEqual(len(cards), len(shortlist))


class PathwayModeCompatibilityTests(unittest.TestCase):
    def test_boost_mode_is_deprecated_to_filter(self) -> None:
        qs = _base_queryspec()
        updated = _apply_query_overrides(
            queryspec=qs,
            k=25,
            min_support=1,
            novel=True,
            require_binding_evidence=False,
            require_expression=False,
            pathway_mode="boost",
        )

        self.assertEqual(updated["pathway_filter"]["mode"], "filter")
        self.assertTrue(updated["pathway_filter"]["enabled"])
        self.assertIn(
            "Pathway boost mode has been removed; using strict filter mode.",
            updated.get("debug_warnings", []),
        )


class PlannerNormalizationTests(unittest.TestCase):
    def test_breast_cancer_maps_to_brca_and_optional_clarifications(self) -> None:
        qs = _validate_and_fill(
            {
                "mode": "mirna_to_targets",
                "mirna": "miRNA-210",
                "gene": None,
                "cancer": {"name": "breast cancer cells", "tcga": None},
                "phenotype_context": {
                    "phenotype": "energy metabolism",
                    "direction": "associated",
                    "raw_phrase": "involved in energy metabolism",
                },
                "pathway_selection_request": {"enabled": True, "query_terms": [], "directional_query_terms": [], "strict": True},
                "phenotype_keywords": ["energy metabolism"],
                "pathway_keywords": [],
                "pathway_filter": {"enabled": True, "mode": "filter", "min_gene_sets": 1},
                "novel": False,
                "k": 25,
                "filters": {"min_support": 1, "require_binding_evidence": False, "require_expression": False},
                "needs_clarification": ["Could narrow to glycolysis or oxidative phosphorylation."],
            },
            "I am studying miRNA-210 in breast cancer cells. I think it might be involved in energy metabolism. What genes might it be regulating?",
        )

        self.assertEqual(qs["cancer"]["tcga"], "BRCA")
        self.assertEqual(qs["needs_clarification"], [])
        self.assertTrue(qs["optional_clarifications"])


if __name__ == "__main__":
    unittest.main()
