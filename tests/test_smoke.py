from __future__ import annotations

import os
import tempfile
import unittest
import uuid
from importlib.util import find_spec
from unittest.mock import patch

import numpy as np
import pandas as pd

from backend.jobstore import read_job, write_job
from backend.worker import run_query_job


class FileSystemJobStoreTests(unittest.TestCase):
    def test_filesystem_jobstore_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {
                "JOBSTORE_BACKEND": "filesystem",
                "MIRASSIST_JOB_DIR": tmpdir,
            },
            clear=False,
        ):
            write_job("job-fs", {"status": "queued", "stage": "queued", "value": 1})
            write_job("job-fs", {"status": "running"})
            payload = read_job("job-fs")

        self.assertEqual(payload["status"], "running")
        self.assertEqual(payload["stage"], "queued")
        self.assertEqual(payload["value"], 1)

    def test_filesystem_jobstore_sanitizes_non_json_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {
                "JOBSTORE_BACKEND": "filesystem",
                "MIRASSIST_JOB_DIR": tmpdir,
            },
            clear=False,
        ):
            write_job(
                "job-fs-sanitize",
                {
                    "status": "done",
                    "x": float("nan"),
                    "y": float("inf"),
                    "rows": [{"ts_best_contextpp": float("nan")}],
                    "np_float": np.float64(np.nan),
                    "pd_na": pd.NA,
                    "pd_nat": pd.NaT,
                },
            )
            payload = read_job("job-fs-sanitize")

        self.assertIsNone(payload["x"])
        self.assertIsNone(payload["y"])
        self.assertIsNone(payload["rows"][0]["ts_best_contextpp"])
        self.assertIsNone(payload["np_float"])
        self.assertIsNone(payload["pd_na"])
        self.assertIsNone(payload["pd_nat"])


class WorkerDiagnosticTests(unittest.TestCase):
    def test_disable_synthesis_returns_chart_only_payload_and_preserves_novel_flag(self) -> None:
        qs = {
            "original_question": "What genes are regulated by miR-21?",
            "mode": "mirna_to_targets",
            "mirna": "miR-21",
            "gene": None,
            "cancer": {},
            "phenotype_keywords": [],
            "pathway_keywords": [],
            "pathway_filter": {"enabled": False, "mode": "filter", "min_gene_sets": 1},
            "novel": False,
            "k": 10,
            "filters": {
                "min_support": 1,
                "require_binding_evidence": False,
                "require_expression": False,
            },
            "needs_clarification": [],
        }
        shortlist = pd.DataFrame(
            [
                {
                    "mirna_name": "miR-21",
                    "gene_symbol": "PTEN",
                    "support_count": 3,
                    "retrieval_score": 8.5,
                    "mirassist_xgboost_score": 0.91,
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {
                "JOBSTORE_BACKEND": "filesystem",
                "MIRASSIST_JOB_DIR": tmpdir,
                "EVIDENCE_BACKEND": "parquet",
            },
            clear=False,
        ), patch("backend.planner.run_planner", return_value=qs), patch(
            "backend.pathways.resolve_pathway_selection", return_value={"enabled": False, "warnings": []}
        ), patch(
            "backend.pathways.compact_pathway_selection", return_value={"enabled": False, "warnings": []}
        ), patch("backend.retrieval.load_evidence", return_value=shortlist), patch(
            "backend.retrieval.retrieve_from_queryspec",
            return_value=(shortlist, "mirna_to_targets", {"n_final_shortlist": 1, "warnings": []}),
        ):
            result = run_query_job(
                query_id="job-chart-only",
                question=qs["original_question"],
                k=10,
                min_support=1,
                novel=True,
                disable_synthesis=True,
                require_binding_evidence=False,
                require_expression=False,
                pathway_mode="auto",
            )

        self.assertEqual(result["status"], "done")
        self.assertTrue(result["disable_synthesis"])
        self.assertEqual(result["result_mode"], "chart_only")
        self.assertEqual(result["queryspec"]["novel"], True)
        self.assertEqual(len(result["shortlist"]), 1)
        self.assertEqual(result["shortlist"][0]["gene_symbol"], "PTEN")
        self.assertEqual(result["retrieval_diagnostics"]["n_rows_sent_to_synthesizer"], 0)

    def test_empty_shortlist_returns_backend_diagnostic_without_synth_call(self) -> None:
        qs = {
            "original_question": "I am studying miRNA-210 in breast cancer cells. I think it might be involved in energy metabolism. What genes might it be regulating?",
            "mode": "mirna_to_targets",
            "mirna": "miRNA-210",
            "gene": None,
            "cancer": {"name": "breast cancer", "tcga": "BRCA"},
            "phenotype_context": {
                "phenotype": "energy metabolism",
                "direction": "associated",
                "raw_phrase": "involved in energy metabolism",
            },
            "pathway_selection_request": {
                "enabled": True,
                "query_terms": ["energy metabolism"],
                "directional_query_terms": [],
                "strict": True,
            },
            "phenotype_keywords": ["energy metabolism"],
            "pathway_keywords": [],
            "pathway_filter": {"enabled": True, "mode": "filter", "min_gene_sets": 1},
            "novel": False,
            "k": 10,
            "filters": {
                "min_support": 1,
                "require_binding_evidence": False,
                "require_expression": False,
            },
            "needs_clarification": [],
        }
        pathway_selection = {
            "enabled": True,
            "mode": "filter",
            "phenotype": "energy metabolism",
            "direction": "associated",
            "query_terms": ["energy metabolism", "glycolysis", "oxidative phosphorylation"],
            "selected_pathways": [{"pathway_id": "H:1", "pathway_name": "HALLMARK_GLYCOLYSIS", "matched_terms": ["glycolysis"]}],
            "n_selected_pathways": 1,
            "n_selected_genes": 1,
            "selected_gene_examples": ["LDHA"],
            "warnings": [],
            "_selected_gene_set": {"LDHA"},
            "_selected_gene_pathways": {"LDHA": ["HALLMARK_GLYCOLYSIS"]},
        }
        ev = pd.DataFrame(
            [
                {"mirna_name": "miR-21", "gene_symbol": "TP53", "support_count": 2, "ts_best_contextpp": -0.2},
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {
                "JOBSTORE_BACKEND": "filesystem",
                "MIRASSIST_JOB_DIR": tmpdir,
                "EVIDENCE_BACKEND": "parquet",
            },
            clear=False,
        ), patch("backend.planner.run_planner", return_value=qs), patch(
            "backend.pathways.resolve_pathway_selection", return_value=pathway_selection
        ), patch(
            "backend.pathways.compact_pathway_selection", return_value=pathway_selection
        ), patch("backend.retrieval.load_evidence", return_value=ev):
            result = run_query_job(
                query_id="job-empty-shortlist",
                question=qs["original_question"],
                k=10,
                min_support=1,
                novel=False,
                require_binding_evidence=False,
                require_expression=False,
                pathway_mode="auto",
            )

        self.assertEqual(result["status"], "done")
        self.assertEqual(result["retrieval_diagnostics"]["n_final_shortlist"], 0)
        self.assertIn("No candidates passed the current filters.", result["answer"]["summary"])


@unittest.skipUnless(os.environ.get("DATABASE_URL"), "DATABASE_URL is not set")
@unittest.skipUnless(find_spec("sqlalchemy") is not None, "sqlalchemy is not installed")
class PostgresJobStoreTests(unittest.TestCase):
    def test_postgres_jobstore_roundtrip(self) -> None:
        query_id = f"job-pg-{uuid.uuid4().hex[:8]}"
        with patch.dict(
            os.environ,
            {
                "JOBSTORE_BACKEND": "postgres",
            },
            clear=False,
        ):
            write_job(query_id, {"status": "queued", "stage": "queued"})
            write_job(query_id, {"status": "done", "answer": {"summary": "ok"}})
            payload = read_job(query_id)

        self.assertEqual(payload["status"], "done")
        self.assertEqual(payload["stage"], "queued")
        self.assertEqual(payload["answer"]["summary"], "ok")

    def test_postgres_jobstore_sanitizes_non_json_numbers(self) -> None:
        query_id = f"job-pg-sanitize-{uuid.uuid4().hex[:8]}"
        with patch.dict(
            os.environ,
            {
                "JOBSTORE_BACKEND": "postgres",
            },
            clear=False,
        ):
            write_job(
                query_id,
                {
                    "status": "done",
                    "x": float("nan"),
                    "y": float("inf"),
                    "rows": [{"ts_best_contextpp": float("nan")}],
                    "np_float": np.float64(np.nan),
                },
            )
            payload = read_job(query_id)

        self.assertIsNone(payload["x"])
        self.assertIsNone(payload["y"])
        self.assertIsNone(payload["rows"][0]["ts_best_contextpp"])
        self.assertIsNone(payload["np_float"])


if __name__ == "__main__":
    unittest.main()
