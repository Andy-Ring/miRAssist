from __future__ import annotations

import os
import tempfile
import unittest
import uuid
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.app import app
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


class ApiSmokeTests(unittest.TestCase):
    def test_health_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {
                "JOBSTORE_BACKEND": "filesystem",
                "MIRASSIST_JOB_DIR": tmpdir,
                "WORKER_MODE": "inline",
            },
            clear=False,
        ):
            with TestClient(app) as client:
                response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["ok"])
        self.assertIn("jobstore_backend", body)
        self.assertIn("worker_mode", body)
        self.assertIn("evidence_backend", body)

    def test_query_status_result_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {
                "JOBSTORE_BACKEND": "filesystem",
                "MIRASSIST_JOB_DIR": tmpdir,
                "WORKER_MODE": "inline",
            },
            clear=False,
        ):
            def fake_run_query_job(**kwargs):
                query_id = kwargs["query_id"]
                write_job(
                    query_id,
                    {
                        "status": "done",
                        "stage": "done",
                        "queryspec": {
                            "original_question": kwargs["question"],
                            "mode": "mirna_to_targets",
                            "mirna": "miR-21",
                            "gene": None,
                            "cancer": {"name": "colon cancer", "tcga": "COAD"},
                            "phenotype_keywords": ["proliferation"],
                            "pathway_keywords": [],
                            "pathway_filter": {
                                "enabled": False,
                                "mode": "filter",
                                "min_gene_sets": 1,
                            },
                            "novel": False,
                            "k": kwargs["k"],
                            "filters": {
                                "min_support": kwargs["min_support"],
                                "require_binding_evidence": kwargs["require_binding_evidence"],
                                "require_expression": kwargs["require_expression"],
                            },
                            "needs_clarification": [],
                        },
                        "shortlist": [{"gene_symbol": "PTEN", "retrieval_score": 8.5}],
                        "bundle": {"meta": {"queryspec": {"mirna": "miR-21"}}},
                        "answer": {"summary": "PTEN is the leading candidate."},
                    },
                )

            with patch("backend.app.run_query_job", side_effect=fake_run_query_job):
                with TestClient(app) as client:
                    submit = client.post(
                        "/query",
                        json={
                            "question": "What does miR-21 regulate in colon cancer?",
                            "novel": False,
                            "k": 25,
                            "min_support": 1,
                            "require_binding_evidence": False,
                            "require_expression": False,
                            "pathway_mode": "auto",
                        },
                    )
                    self.assertEqual(submit.status_code, 200)
                    query_id = submit.json()["query_id"]

                    status_response = client.get(f"/status/{query_id}")
                    result_response = client.get(f"/result/{query_id}")

        self.assertEqual(status_response.status_code, 200)
        self.assertEqual(status_response.json()["status"], "done")
        self.assertEqual(result_response.status_code, 200)
        result = result_response.json()
        self.assertEqual(result["status"], "done")
        self.assertIn("queryspec", result)
        self.assertIn("shortlist", result)
        self.assertIn("bundle", result)
        self.assertIn("answer", result)


class WorkerDiagnosticTests(unittest.TestCase):
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
            },
            clear=False,
        ), patch("backend.worker.run_planner", return_value=qs), patch(
            "backend.worker.resolve_pathway_selection", return_value=pathway_selection
        ), patch("backend.worker.load_evidence", return_value=ev), patch(
            "backend.worker.run_synthesizer"
        ) as synth_mock:
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
        synth_mock.assert_not_called()


@unittest.skipUnless(os.environ.get("DATABASE_URL"), "DATABASE_URL is not set")
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
