from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from backend.cards import cards_from_dataframe_with_diagnostics
from backend.planner import (
    _validate_and_fill,
    build_directional_query_terms,
    infer_expected_target_role,
)
from backend.pathways import (
    compact_pathway_selection,
    load_gene_to_pathways,
    load_pathways,
    resolve_pathway_selection,
)
from backend.retrieval import retrieve_from_queryspec
from backend.worker import _apply_query_overrides


def _base_queryspec() -> dict:
    return {
        "original_question": "I overexpressed miR-34a and apoptosis increased.",
        "mode": "mirna_to_targets",
        "mirna": "miR-34a",
        "gene": None,
        "cancer": {"name": None, "tcga": None},
        "phenotype_context": {
            "phenotype": "apoptosis",
            "observed_change": "increased",
            "miRNA_perturbation": "overexpression",
            "direction": "increases",
            "raw_phrase": "I overexpressed miR-34a and apoptosis increased.",
        },
        "target_role_inference": {
            "enabled": True,
            "assumption": "miRNAs usually repress target gene expression",
            "expected_target_effect_on_phenotype": "negative_regulator",
            "reasoning": "The user reported miRNA overexpression increased apoptosis.",
        },
        "pathway_selection_request": {
            "enabled": True,
            "query_terms": ["apoptosis"],
            "directional_query_terms": ["negative regulation of apoptosis"],
            "strict": True,
        },
        "phenotype_keywords": ["apoptosis"],
        "pathway_keywords": [],
        "pathway_filter": {"enabled": True, "mode": "filter", "min_gene_sets": 1},
        "novel": False,
        "k": 25,
        "result_count": 5,
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

    def test_csv_gz_tables_match_parquet_sources_and_selection(self) -> None:
        import os
        from pandas.testing import assert_frame_equal

        old_path = os.environ.pop("MIRASSIST_PATHWAYS_PATH", None)
        old_gene_path = os.environ.pop("MIRASSIST_GENE_TO_PATHWAYS_PATH", None)
        old_fallback = os.environ.pop("MIRASSIST_ENABLE_PARQUET_FALLBACK", None)
        try:
            csv_pathways = load_pathways(force_reload=True)
            csv_gene_pathways = load_gene_to_pathways(force_reload=True)
            os.environ["MIRASSIST_PATHWAYS_PATH"] = "data/processed/pathways/pathways.parquet"
            os.environ["MIRASSIST_GENE_TO_PATHWAYS_PATH"] = "data/processed/pathways/gene_to_pathways.parquet"
            os.environ["MIRASSIST_ENABLE_PARQUET_FALLBACK"] = "1"
            parquet_pathways = load_pathways(force_reload=True)
            parquet_gene_pathways = load_gene_to_pathways(force_reload=True)

            self.assertEqual(csv_pathways.shape, parquet_pathways.shape)
            self.assertEqual(csv_gene_pathways.shape, parquet_gene_pathways.shape)
            self.assertListEqual(list(csv_pathways.columns), list(parquet_pathways.columns))
            self.assertListEqual(list(csv_gene_pathways.columns), list(parquet_gene_pathways.columns))
            assert_frame_equal(
                csv_pathways.drop(columns=["genes"]),
                parquet_pathways.drop(columns=["genes"]),
                check_dtype=False,
            )
            assert_frame_equal(csv_gene_pathways, parquet_gene_pathways, check_dtype=False)

            csv_selection = resolve_pathway_selection(_base_queryspec())
            os.environ["MIRASSIST_PATHWAYS_PATH"] = "data/processed/pathways/pathways.parquet"
            os.environ["MIRASSIST_GENE_TO_PATHWAYS_PATH"] = "data/processed/pathways/gene_to_pathways.parquet"
            parquet_selection = resolve_pathway_selection(_base_queryspec())
            self.assertEqual(csv_selection["selected_pathways"], parquet_selection["selected_pathways"])
            self.assertEqual(csv_selection["_selected_gene_set"], parquet_selection["_selected_gene_set"])
            self.assertEqual(csv_selection["_selected_gene_pathways"], parquet_selection["_selected_gene_pathways"])
        finally:
            for key, value in (
                ("MIRASSIST_PATHWAYS_PATH", old_path),
                ("MIRASSIST_GENE_TO_PATHWAYS_PATH", old_gene_path),
                ("MIRASSIST_ENABLE_PARQUET_FALLBACK", old_fallback),
            ):
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            load_pathways(force_reload=True)
            load_gene_to_pathways(force_reload=True)

    def test_parquet_fallback_requires_explicit_environment_flag(self) -> None:
        import os

        old_path = os.environ.get("MIRASSIST_PATHWAYS_PATH")
        old_fallback = os.environ.pop("MIRASSIST_ENABLE_PARQUET_FALLBACK", None)
        try:
            os.environ["MIRASSIST_PATHWAYS_PATH"] = "data/processed/pathways/pathways.parquet"
            with self.assertRaisesRegex(RuntimeError, "Parquet fallback is disabled"):
                load_pathways(force_reload=True)
        finally:
            if old_path is None:
                os.environ.pop("MIRASSIST_PATHWAYS_PATH", None)
            else:
                os.environ["MIRASSIST_PATHWAYS_PATH"] = old_path
            if old_fallback is not None:
                os.environ["MIRASSIST_ENABLE_PARQUET_FALLBACK"] = old_fallback
            load_pathways(force_reload=True)

    def test_overexpression_increased_apoptosis_selects_negative_regulator_pathways(self) -> None:
        qs = _base_queryspec()
        with patch("backend.pathways.load_pathways", return_value=self.pathways_df), patch(
            "backend.pathways.load_gene_to_pathways", return_value=self.gene_to_pathways_df
        ):
            selection = resolve_pathway_selection(qs)

        self.assertTrue(selection["enabled"])
        self.assertEqual(selection["mode"], "filter")
        self.assertEqual(selection["phenotype"], "apoptosis")
        self.assertEqual(selection["direction"], "increases")
        self.assertEqual(selection["observed_change"], "increased")
        self.assertEqual(selection["miRNA_perturbation"], "overexpression")
        self.assertEqual(selection["expected_target_effect_on_phenotype"], "negative_regulator")
        self.assertGreaterEqual(selection["n_selected_pathways"], 1)
        self.assertIn("BCL2", selection["_selected_gene_set"])

    def test_overexpression_decreased_apoptosis_selects_positive_regulator_pathways(self) -> None:
        qs = _base_queryspec()
        qs["phenotype_context"]["observed_change"] = "decreased"
        qs["phenotype_context"]["direction"] = "decreases"
        qs["phenotype_context"]["raw_phrase"] = "I overexpressed miR-34a and apoptosis decreased."
        qs["target_role_inference"] = infer_expected_target_role(qs["phenotype_context"])
        qs["pathway_selection_request"]["directional_query_terms"] = [
            "positive regulation of apoptosis"
        ]
        with patch("backend.pathways.load_pathways", return_value=self.pathways_df), patch(
            "backend.pathways.load_gene_to_pathways", return_value=self.gene_to_pathways_df
        ):
            selection = resolve_pathway_selection(qs)

        selected_names = [item["pathway_name"] for item in selection["selected_pathways"]]
        self.assertIn("positive regulation of apoptosis", selected_names)
        self.assertIn("BAX", selection["_selected_gene_set"])

    def test_missing_files_no_context_does_not_crash(self) -> None:
        qs = _base_queryspec()
        qs["phenotype_context"] = {
            "phenotype": None,
            "observed_change": None,
            "miRNA_perturbation": None,
            "direction": None,
            "raw_phrase": None,
        }
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
                "observed_change": "associated",
                "miRNA_perturbation": "unknown",
                "direction": "associated",
                "raw_phrase": "involved in energy metabolism",
            },
            "target_role_inference": {
                "enabled": True,
                "assumption": "miRNAs usually repress target gene expression",
                "expected_target_effect_on_phenotype": "unknown",
                "reasoning": "The user did not provide a confident miRNA perturbation direction.",
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

    def test_associated_context_uses_general_terms_not_directional_pathways(self) -> None:
        qs = {
            "phenotype_context": {
                "phenotype": "proliferation",
                "observed_change": "associated",
                "miRNA_perturbation": "unknown",
                "direction": "associated",
                "raw_phrase": "miR-X is associated with proliferation.",
            },
            "target_role_inference": {
                "enabled": True,
                "assumption": "miRNAs usually repress target gene expression",
                "expected_target_effect_on_phenotype": "unknown",
                "reasoning": "Unknown perturbation direction.",
            },
            "pathway_selection_request": {
                "enabled": True,
                "query_terms": ["proliferation"],
                "directional_query_terms": [],
                "strict": True,
            },
            "phenotype_keywords": ["proliferation"],
            "pathway_keywords": [],
            "pathway_filter": {"enabled": True, "mode": "filter", "min_gene_sets": 1},
        }

        with patch("backend.pathways.load_pathways", return_value=self.pathways_df), patch(
            "backend.pathways.load_gene_to_pathways", return_value=self.gene_to_pathways_df
        ):
            query_terms = compact_pathway_selection(resolve_pathway_selection(qs))["query_terms"]
        query_terms_lower = [term.lower() for term in query_terms]
        self.assertIn("proliferation", query_terms_lower)
        self.assertNotIn("negative regulation of cell proliferation", query_terms_lower)
        self.assertNotIn("positive regulation of cell proliferation", query_terms_lower)


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

        with patch.dict("os.environ", {"EVIDENCE_BACKEND": "parquet"}, clear=False):
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

        with patch.dict("os.environ", {"EVIDENCE_BACKEND": "parquet"}, clear=False):
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

    def test_base_mirna_query_prefers_primary_arm_without_norm_column(self) -> None:
        ev = pd.DataFrame(
            [
                {"mirna_name": "MiR-210", "gene_symbol": "EPHA2", "support_count": 1, "ts_best_contextpp": -0.1},
                {"mirna_name": "hsa-miR-210-5p", "gene_symbol": "KCMF1", "support_count": 2, "ts_best_contextpp": -0.4},
                {"mirna_name": "hsa-miR-210-3p", "gene_symbol": "ISCU", "support_count": 2, "ts_best_contextpp": -0.3},
            ]
        )
        qs = {
            "mode": "mirna_to_targets",
            "mirna": "miRNA-210",
            "gene": None,
            "cancer": {"name": None, "tcga": None},
            "pathway_filter": {"enabled": False, "mode": "filter", "min_gene_sets": 1},
            "filters": {"min_support": 1, "require_binding_evidence": False, "require_expression": False},
            "novel": False,
            "k": 10,
        }

        with patch.dict("os.environ", {"MIRASSIST_DEFAULT_MIRNA_ARM": "5p", "EVIDENCE_BACKEND": "parquet"}, clear=False):
            shortlist, direction, diagnostics = retrieve_from_queryspec(ev, qs, pathway_selection={"enabled": False, "warnings": []})

        self.assertEqual(direction, "mirna_to_targets")
        self.assertEqual(shortlist["gene_symbol"].tolist(), ["KCMF1"])
        self.assertNotIn("EPHA2", shortlist["gene_symbol"].tolist())
        self.assertEqual(diagnostics["n_rows_primary"], 1)
        self.assertEqual(diagnostics["n_rows_fallback"], 0)
        self.assertEqual(diagnostics["variants_used"], ["mir-210-5p", "hsa-mir-210-5p"])
        self.assertEqual(diagnostics["matched_mirna_names"], ["hsa-miR-210-5p"])
        self.assertIn("miR-210-5p", diagnostics.get("arm_interpretation_note") or "")


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
    def test_overexpression_increased_proliferation_infers_negative_regulator_targets(self) -> None:
        qs = _validate_and_fill(
            {
                "mode": "mirna_to_targets",
                "mirna": "miR-X",
                "phenotype_context": {"phenotype": "proliferation"},
                "pathway_selection_request": {"enabled": True, "query_terms": [], "directional_query_terms": [], "strict": True},
                "phenotype_keywords": ["proliferation"],
                "pathway_keywords": [],
                "pathway_filter": {"enabled": True, "mode": "filter", "min_gene_sets": 1},
                "filters": {"min_support": 1, "require_binding_evidence": False, "require_expression": False},
            },
            "I overexpressed miR-X and proliferation increased.",
        )

        self.assertEqual(qs["phenotype_context"]["miRNA_perturbation"], "overexpression")
        self.assertEqual(qs["phenotype_context"]["observed_change"], "increased")
        self.assertEqual(qs["phenotype_context"]["phenotype"], "proliferation")
        self.assertEqual(
            qs["target_role_inference"]["expected_target_effect_on_phenotype"],
            "negative_regulator",
        )
        self.assertIn(
            "negative regulation of cell proliferation",
            qs["pathway_selection_request"]["directional_query_terms"],
        )

    def test_overexpression_decreased_proliferation_infers_positive_regulator_targets(self) -> None:
        qs = _validate_and_fill(
            {
                "mode": "mirna_to_targets",
                "mirna": "miR-X",
                "phenotype_context": {"phenotype": "proliferation"},
                "pathway_selection_request": {"enabled": True, "query_terms": [], "directional_query_terms": [], "strict": True},
                "phenotype_keywords": ["proliferation"],
                "pathway_keywords": [],
                "pathway_filter": {"enabled": True, "mode": "filter", "min_gene_sets": 1},
                "filters": {"min_support": 1, "require_binding_evidence": False, "require_expression": False},
            },
            "I overexpressed miR-X and proliferation decreased.",
        )

        self.assertEqual(
            qs["target_role_inference"]["expected_target_effect_on_phenotype"],
            "positive_regulator",
        )
        self.assertIn(
            "positive regulation of cell proliferation",
            qs["pathway_selection_request"]["directional_query_terms"],
        )

    def test_overexpression_increased_cell_migration_replaces_conflicting_directional_terms(self) -> None:
        qs = _validate_and_fill(
            {
                "mode": "mirna_to_targets",
                "mirna": "miRNA 3065",
                "phenotype_context": {"phenotype": "cell migration"},
                "pathway_selection_request": {
                    "enabled": True,
                    "query_terms": ["cell migration", "migration", "cell motility"],
                    "directional_query_terms": [
                        "positive regulation of cell migration",
                        "positive regulation of migration",
                        "positive regulation of cell motility",
                    ],
                    "strict": True,
                },
                "phenotype_keywords": ["cell migration", "migration", "cell motility"],
                "pathway_keywords": ["cell migration", "migration", "cell motility"],
                "pathway_filter": {"enabled": True, "mode": "filter", "min_gene_sets": 2},
                "filters": {"min_support": 2, "require_binding_evidence": False, "require_expression": False},
            },
            "I'm interested in miRNA 3065 and when i overexpress it, it causes an increase in cell migration. What genes might it be regulating to cause this?",
        )

        self.assertEqual(
            qs["target_role_inference"]["expected_target_effect_on_phenotype"],
            "negative_regulator",
        )
        self.assertEqual(
            qs["pathway_selection_request"]["directional_query_terms"],
            [
                "negative regulation of cell migration",
                "negative regulation of migration",
                "negative regulation of cell motility",
                "cell migration suppressor",
                "migration inhibition",
            ],
        )
        self.assertNotIn(
            "positive regulation of cell migration",
            qs["pathway_selection_request"]["directional_query_terms"],
        )

    def test_overexpression_increased_apoptosis_infers_negative_regulator_targets(self) -> None:
        inference = infer_expected_target_role(
            {
                "phenotype": "apoptosis",
                "observed_change": "increased",
                "miRNA_perturbation": "overexpression",
                "raw_phrase": "I overexpressed miR-X and apoptosis increased.",
            }
        )
        self.assertEqual(inference["expected_target_effect_on_phenotype"], "negative_regulator")

    def test_overexpression_decreased_apoptosis_infers_positive_regulator_targets(self) -> None:
        inference = infer_expected_target_role(
            {
                "phenotype": "apoptosis",
                "observed_change": "decreased",
                "miRNA_perturbation": "overexpression",
                "raw_phrase": "I overexpressed miR-X and apoptosis decreased.",
            }
        )
        self.assertEqual(inference["expected_target_effect_on_phenotype"], "positive_regulator")

    def test_associated_query_keeps_target_role_unknown(self) -> None:
        qs = _validate_and_fill(
            {
                "mode": "mirna_to_targets",
                "mirna": "miR-X",
                "pathway_selection_request": {"enabled": True, "query_terms": [], "directional_query_terms": [], "strict": True},
                "phenotype_keywords": ["proliferation"],
                "pathway_keywords": [],
                "pathway_filter": {"enabled": True, "mode": "filter", "min_gene_sets": 1},
                "filters": {"min_support": 1, "require_binding_evidence": False, "require_expression": False},
            },
            "miR-X is associated with proliferation.",
        )

        self.assertEqual(qs["phenotype_context"]["miRNA_perturbation"], "unknown")
        self.assertEqual(qs["phenotype_context"]["observed_change"], "associated")
        self.assertEqual(
            qs["target_role_inference"]["expected_target_effect_on_phenotype"],
            "unknown",
        )
        self.assertEqual(
            build_directional_query_terms(
                qs["phenotype_context"]["phenotype"],
                qs["target_role_inference"]["expected_target_effect_on_phenotype"],
            ),
            [],
        )

    def test_breast_cancer_maps_to_brca_and_optional_clarifications(self) -> None:
        qs = _validate_and_fill(
            {
                "mode": "mirna_to_targets",
                "mirna": "miRNA-210",
                "gene": None,
                "cancer": {"name": "breast cancer cells", "tcga": None},
                "phenotype_context": {
                    "phenotype": "energy metabolism",
                    "observed_change": "associated",
                    "miRNA_perturbation": "unknown",
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


    def test_coordinated_phenotypes_are_preserved_by_deterministic_parser(self) -> None:
        cases = {
            "cell invasion and migration": {"cell invasion", "cell migration"},
            "cell proliferation and survival": {"cell proliferation", "cell survival"},
            "angiogenesis and metastasis": {"angiogenesis", "metastasis"},
            "apoptosis": {"apoptosis"},
        }
        for phrase, expected in cases.items():
            with self.subTest(phrase=phrase):
                qs = _validate_and_fill(
                    {"mode": "mirna_to_targets", "mirna": "miR-X"},
                    f"What are the targets of miR-X related to {phrase}?",
                )
                self.assertTrue(expected.issubset(set(qs["phenotype_keywords"])))
                self.assertTrue(expected.issubset(set(qs["pathway_selection_request"]["query_terms"])))

    def test_explicit_pathway_metadata_term_is_selected(self) -> None:
        qs = _base_queryspec()
        qs["phenotype_context"]["phenotype"] = None
        qs["phenotype_keywords"] = []
        qs["pathway_keywords"] = ["PI3K AKT"]
        qs["pathway_selection_request"]["query_terms"] = ["PI3K AKT"]
        pathways = pd.DataFrame([{
            "pathway_id": "P1", "pathway_name": "SIGNALING_SET",
            "description": "curated pathway", "aliases": "PI3K-AKT",
            "collection_name": "Hallmark signaling",
        }])
        genes = pd.DataFrame([{"gene_symbol": "AKT1", "pathway_id": "P1", "pathway_name": "SIGNALING_SET"}])
        with patch("backend.pathways.load_pathways", return_value=pathways), patch(
            "backend.pathways.load_gene_to_pathways", return_value=genes
        ):
            selection = resolve_pathway_selection(qs)
        self.assertEqual(selection["n_selected_pathways"], 1)
        self.assertEqual(selection["n_selected_genes"], 1)

    def test_unmatched_phenotype_keeps_strict_filter_and_warns(self) -> None:
        qs = _validate_and_fill(
            {"mode": "mirna_to_targets", "mirna": "miR-X"},
            "What are the targets of miR-X related to an unknown cellular phenomenon?",
        )
        qs["phenotype_context"]["phenotype"] = "unknown cellular phenomenon"
        qs["phenotype_keywords"] = ["unknown cellular phenomenon"]
        qs["pathway_selection_request"]["query_terms"] = ["unknown cellular phenomenon"]
        pathways = pd.DataFrame([{"pathway_id": "X", "pathway_name": "known pathway"}])
        genes = pd.DataFrame([{"gene_symbol": "X1", "pathway_id": "X", "pathway_name": "known pathway"}])
        with patch("backend.pathways.load_pathways", return_value=pathways), patch(
            "backend.pathways.load_gene_to_pathways", return_value=genes
        ):
            selection = resolve_pathway_selection(qs)
        self.assertTrue(selection["enabled"])
        self.assertEqual(selection["mode"], "filter")
        self.assertEqual(selection["n_selected_pathways"], 0)
        self.assertTrue(selection["warnings"])

    def test_real_pathways_select_invasion_migration_pathway_and_genes(self) -> None:
        qs = _validate_and_fill(
            {"mode": "mirna_to_targets", "mirna": "miRNA-93"},
            "What are the targets of miRNA-93 that relate to cell invasion and migration?",
        )
        selection = resolve_pathway_selection(qs)
        names = {item["pathway_name"] for item in selection["selected_pathways"]}
        self.assertTrue(names & {
            "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
            "HALLMARK_ANGIOGENESIS",
        })
        self.assertGreater(selection["n_selected_genes"], 0)

class PhenotypePathwayRelevanceRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pathways = pd.DataFrame(
            [
                {"pathway_id": "GO:1", "pathway_name": "Cell Migration", "description": "", "collection": "GO Biological Process"},
                {"pathway_id": "GO:2", "pathway_name": "Regulation of Cell Migration", "description": "", "collection": "GO Biological Process"},
                {"pathway_id": "GO:3", "pathway_name": "Cell Motility", "description": "", "collection": "GO Biological Process"},
                {"pathway_id": "GO:4", "pathway_name": "Chemotaxis", "description": "", "collection": "GO Biological Process"},
                {"pathway_id": "R:1", "pathway_name": "Focal Adhesion", "description": "", "collection": "Reactome"},
                {"pathway_id": "R:2", "pathway_name": "Extracellular Matrix Organization", "description": "", "collection": "Reactome"},
                {"pathway_id": "R:3", "pathway_name": "Integrin Signaling", "description": "", "collection": "Reactome"},
                {"pathway_id": "WP:1", "pathway_name": "Regulation of Actin Cytoskeleton", "description": "", "collection": "WikiPathways"},
                {"pathway_id": "H:EMT", "pathway_name": "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION", "description": "", "collection": "Hallmark"},
                {
                    "pathway_id": "H:ADIPO",
                    "pathway_name": "HALLMARK_ADIPOGENESIS",
                    "description": "Adipocyte differentiation with indirect cell migration associations.",
                    "collection": "Hallmark",
                },
                {
                    "pathway_id": "H:XENO",
                    "pathway_name": "HALLMARK_XENOBIOTIC_METABOLISM",
                    "description": "Drug detoxification with indirect cell migration associations.",
                    "collection": "Hallmark",
                },
                {
                    "pathway_id": "LOW:1",
                    "pathway_name": "GENERAL_CELLULAR_SIGNALING",
                    "description": "",
                    "aliases": "migration network",
                    "collection": "Other curated",
                },
            ]
        )
        self.genes = pd.DataFrame(
            [
                {
                    "gene_symbol": f"GENE{index}",
                    "pathway_id": row["pathway_id"],
                    "pathway_name": row["pathway_name"],
                }
                for index, row in self.pathways.iterrows()
            ]
        )

    def _migration_queryspec(self) -> dict:
        qs = _validate_and_fill(
            {"mode": "mirna_to_targets", "mirna": "miRNA-93"},
            "What are the targets of miRNA-93 that relate to cell migration?",
        )
        # A legacy minimum count must not cause padding below the relevance threshold.
        qs["pathway_filter"]["min_gene_sets"] = 25
        return qs

    def test_cell_migration_selects_only_controlled_relevant_pathways(self) -> None:
        with patch("backend.pathways.load_pathways", return_value=self.pathways), patch(
            "backend.pathways.load_gene_to_pathways", return_value=self.genes
        ):
            selection = resolve_pathway_selection(self._migration_queryspec())

        selected_names = {
            item["pathway_name"] for item in selection["selected_pathways"]
        }
        expected = {
            "Cell Migration",
            "Regulation of Cell Migration",
            "Cell Motility",
            "Chemotaxis",
            "Focal Adhesion",
            "Extracellular Matrix Organization",
            "Integrin Signaling",
            "Regulation of Actin Cytoskeleton",
            "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
        }
        self.assertEqual(selected_names, expected)
        self.assertNotIn("HALLMARK_ADIPOGENESIS", selected_names)
        self.assertNotIn("HALLMARK_XENOBIOTIC_METABOLISM", selected_names)
        self.assertNotIn("GENERAL_CELLULAR_SIGNALING", selected_names)
        self.assertEqual(selection["n_selected_pathways"], len(expected))

        required_metadata = {
            "match_type",
            "matched_term",
            "source_concept",
            "relevance_score",
            "collection",
            "rationale",
        }
        for pathway in selection["selected_pathways"]:
            self.assertTrue(required_metadata.issubset(pathway))
            self.assertGreaterEqual(
                pathway["relevance_score"], selection["minimum_relevance_score"]
            )

    def test_explicit_pathway_name_bypasses_phenotype_exclusion(self) -> None:
        qs = _base_queryspec()
        qs["phenotype_context"]["phenotype"] = None
        qs["phenotype_keywords"] = []
        qs["pathway_keywords"] = ["HALLMARK_XENOBIOTIC_METABOLISM"]
        qs["pathway_selection_request"]["query_terms"] = [
            "HALLMARK_XENOBIOTIC_METABOLISM"
        ]
        qs["pathway_selection_request"]["directional_query_terms"] = []
        with patch("backend.pathways.load_pathways", return_value=self.pathways), patch(
            "backend.pathways.load_gene_to_pathways", return_value=self.genes
        ):
            selection = resolve_pathway_selection(qs)

        self.assertEqual(selection["n_selected_pathways"], 1)
        self.assertEqual(
            selection["selected_pathways"][0]["pathway_name"],
            "HALLMARK_XENOBIOTIC_METABOLISM",
        )
        self.assertEqual(
            selection["selected_pathways"][0]["match_type"],
            "exact_normalized_name_match",
        )

    def test_observed_query_uses_real_migration_collections_without_false_positives(self) -> None:
        qs = _validate_and_fill(
            {"mode": "mirna_to_targets", "mirna": "miRNA-93"},
            "What are the targets of miRNA-93 that relate to cell migration?",
        )
        selection = resolve_pathway_selection(qs)
        selected_names = {
            item["pathway_name"] for item in selection["selected_pathways"]
        }
        normalized_names = {name.lower().replace("_", " ") for name in selected_names}

        self.assertIn(
            "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION", selected_names
        )
        self.assertNotIn("HALLMARK_ADIPOGENESIS", selected_names)
        self.assertNotIn("HALLMARK_XENOBIOTIC_METABOLISM", selected_names)
        for expected_term in (
            "migration", "motility", "focal adhesion", "extracellular matrix",
            "integrin", "actin cytoskeleton",
        ):
            self.assertTrue(
                any(expected_term in name for name in normalized_names),
                msg=f"Expected a selected pathway related to {expected_term}",
            )

    def test_bundled_database_includes_prioritized_collections(self) -> None:
        collections = set(load_pathways(force_reload=True)["collection"].dropna())
        self.assertTrue(
            {"GO Biological Process", "Reactome", "WikiPathways", "Hallmark"}.issubset(
                collections
            )
        )


if __name__ == "__main__":
    unittest.main()
