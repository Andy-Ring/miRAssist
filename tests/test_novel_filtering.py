from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import pandas as pd

from backend.retrieval import retrieve_from_queryspec


class NovelFilteringTests(unittest.TestCase):
    def _queryspec(self, novel: bool) -> dict:
        return {
            "mode": "mirna_to_targets",
            "mirna": "hsa-miR-210-5p",
            "gene": None,
            "cancer": {"name": None, "tcga": None},
            "pathway_filter": {"enabled": False, "mode": "filter", "min_gene_sets": 1},
            "filters": {
                "min_support": 1,
                "require_binding_evidence": False,
                "require_expression": False,
            },
            "novel": novel,
            "k": 10,
        }

    def test_novel_mode_excludes_mirtarbase_positive_columns(self) -> None:
        ev = pd.DataFrame(
            [
                {
                    "mirna_name": "hsa-miR-210-5p",
                    "mirna_name_normalized": "hsa-mir-210-5p",
                    "gene_symbol": "KEEP",
                    "support_count": 2,
                    "mirassist_xgboost_score": 0.9,
                    "mirtarbase_known_positive": False,
                    "mirtarbase_pos": 0,
                    "label_mirtarbase": 0,
                },
                {
                    "mirna_name": "hsa-miR-210-5p",
                    "mirna_name_normalized": "hsa-mir-210-5p",
                    "gene_symbol": "DROP_POS",
                    "support_count": 2,
                    "mirassist_xgboost_score": 0.95,
                    "mirtarbase_known_positive": False,
                    "mirtarbase_pos": 1,
                    "label_mirtarbase": 0,
                },
                {
                    "mirna_name": "hsa-miR-210-5p",
                    "mirna_name_normalized": "hsa-mir-210-5p",
                    "gene_symbol": "DROP_LABEL",
                    "support_count": 2,
                    "mirassist_xgboost_score": 0.85,
                    "mirtarbase_known_positive": False,
                    "mirtarbase_pos": 0,
                    "label_mirtarbase": 1,
                },
                {
                    "mirna_name": "hsa-miR-210-5p",
                    "mirna_name_normalized": "hsa-mir-210-5p",
                    "gene_symbol": "DROP_KNOWN_POSITIVE",
                    "support_count": 2,
                    "mirassist_xgboost_score": 0.99,
                    "mirtarbase_known_positive": True,
                    "mirtarbase_pos": 0,
                    "label_mirtarbase": 0,
                },
            ]
        )

        with patch.dict(
            os.environ,
            {
                "EVIDENCE_BACKEND": "parquet",
                "MIRASSIST_USE_MIRTARBASE_EVIDENCE": "1",
            },
            clear=False,
        ):
            shortlist, _, diagnostics = retrieve_from_queryspec(
                ev,
                self._queryspec(novel=True),
                pathway_selection={"enabled": False, "warnings": []},
            )

        self.assertEqual(shortlist["gene_symbol"].tolist(), ["KEEP"])
        self.assertEqual(diagnostics["n_after_novel_filter"], 1)

    def test_non_novel_mode_keeps_mirtarbase_positive_rows(self) -> None:
        ev = pd.DataFrame(
            [
                {
                    "mirna_name": "hsa-miR-210-5p",
                    "mirna_name_normalized": "hsa-mir-210-5p",
                    "gene_symbol": "KNOWN",
                    "support_count": 2,
                    "mirassist_xgboost_score": 0.95,
                    "mirtarbase_known_positive": True,
                    "mirtarbase_pos": 1,
                    "label_mirtarbase": 1,
                }
            ]
        )

        with patch.dict(
            os.environ,
            {
                "EVIDENCE_BACKEND": "parquet",
                "MIRASSIST_USE_MIRTARBASE_EVIDENCE": "1",
            },
            clear=False,
        ):
            shortlist, _, diagnostics = retrieve_from_queryspec(
                ev,
                self._queryspec(novel=False),
                pathway_selection={"enabled": False, "warnings": []},
            )

        self.assertEqual(shortlist["gene_symbol"].tolist(), ["KNOWN"])
        self.assertEqual(diagnostics["n_after_novel_filter"], 1)


if __name__ == "__main__":
    unittest.main()
