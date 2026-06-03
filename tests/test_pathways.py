from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from backend.pathways import resolve_pathway_selection
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
            ]
        )
        self.gene_to_pathways_df = pd.DataFrame(
            [
                {"gene_symbol": "BAX", "pathway_id": "GO:1", "pathway_name": "positive regulation of apoptosis"},
                {"gene_symbol": "CASP3", "pathway_id": "GO:1", "pathway_name": "positive regulation of apoptosis"},
                {"gene_symbol": "BCL2", "pathway_id": "GO:2", "pathway_name": "negative regulation of apoptosis"},
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
        self.assertIn("BAX", selection["selected_genes"])

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
        self.assertIn("BCL2", selection["selected_genes"])

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
        self.assertEqual(selection["selected_genes"], [])
        self.assertTrue(selection["warnings"])


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
            "selected_genes": ["BAX"],
            "selected_gene_pathways": {"BAX": ["positive regulation of apoptosis"]},
            "selected_pathways": [{"pathway_id": "GO:1", "pathway_name": "positive regulation of apoptosis", "matched_terms": ["apoptosis"]}],
            "warnings": [],
        }

        shortlist, _ = retrieve_from_queryspec(ev, qs, pathway_selection=pathway_selection)

        self.assertEqual(shortlist["gene_symbol"].tolist(), ["BAX"])
        self.assertEqual(shortlist["pathway_selected_gene"].tolist(), [1])
        self.assertEqual(shortlist["pathway_match_count"].tolist(), [1])


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


if __name__ == "__main__":
    unittest.main()
