from __future__ import annotations

import threading
import time
import traceback
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

from backend.config import EVIDENCE_PATH
from backend.planner import run_planner
from backend.retrieval import retrieve_from_queryspec
from backend.prompting import build_prompt_bundle
from backend.synthesizer import run_synthesizer

import subprocess
from backend.jobstore import write_job, read_job

app = FastAPI(title="miRAssist")

# ----------------------------
# In-memory job store (single process)
# ----------------------------
JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()

#-----------------------------
# Get Job Helpers
#-----------------------------
def _job_get(query_id: str) -> Dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(query_id)
        if job is None:
            raise KeyError(query_id)
        return dict(job)  # copy (small)

def _job_update(query_id: str, **fields) -> None:
    with JOBS_LOCK:
        if query_id not in JOBS:
            JOBS[query_id] = {}
        JOBS[query_id].update(fields)



# ----------------------------
# Evidence cache
# ----------------------------
_EVIDENCE_DF: Optional[pd.DataFrame] = None
_EVIDENCE_LOCK = threading.Lock()


def load_evidence_cached() -> pd.DataFrame:
    global _EVIDENCE_DF
    if _EVIDENCE_DF is not None:
        return _EVIDENCE_DF

    with _EVIDENCE_LOCK:
        if _EVIDENCE_DF is None:
            if not EVIDENCE_PATH.exists():
                raise FileNotFoundError(f"Evidence parquet not found: {EVIDENCE_PATH}")
            _EVIDENCE_DF = pd.read_parquet(EVIDENCE_PATH)
        return _EVIDENCE_DF


# ----------------------------
# Request schema
# ----------------------------
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    novel: bool = True
    k: int = 25
    min_support: int = 2
    require_binding_evidence: bool = False
    require_expression: bool = False


def _make_query_id(question: str) -> str:
    return f"miRAssist_{int(time.time())}_{abs(hash(question)) % 10**10}"


def _run_job(query_id: str, req: QueryRequest) -> None:
    try:
        _job_update(query_id, status="running", started_at=time.time())

        # ---- heavy work: NO LOCKS ----
        qs = run_planner(req.question)

        qs["novel"] = bool(req.novel)
        qs["k"] = int(req.k)
        qs.setdefault("filters", {})
        qs["filters"]["min_support"] = int(req.min_support)
        qs["filters"]["require_binding_evidence"] = bool(req.require_binding_evidence)
        qs["filters"]["require_expression"] = bool(req.require_expression)

        ev = load_evidence_cached()
        shortlist_df, direction = retrieve_from_queryspec(ev, qs)

        bundle = build_prompt_bundle(
            queryspec=qs,
            shortlist=shortlist_df,
            direction=direction,
        )

        answer_text = run_synthesizer(bundle)

        # ---- store results: BRIEF LOCK ----
        _job_update(
            query_id,
            status="complete",
            queryspec=qs,
            direction=direction,
            shortlist=shortlist_df.to_dict(orient="records"),
            answer={"raw_text": answer_text},
            finished_at=time.time(),
        )

    except Exception as e:
        tb = traceback.format_exc()
        _job_update(
            query_id,
            status="error",
            error=str(e),
            traceback=tb,
            finished_at=time.time(),
        )



@app.get("/health")
def health() -> Dict[str, Any]:
    # does not force loading the evidence (keeps it lightweight)
    return {"ok": True, "service": "miRAssist"}


@app.post("/query")
def submit_query(req: QueryRequest):
    query_id = _make_query_id(req.question)
    write_job(query_id, {"status": "queued", "created_at": time.time()})

    cmd = [
        "python", "-m", "backend.worker",
        "--query_id", query_id,
        "--question", req.question,
        "--k", str(req.k),
        "--min_support", str(req.min_support),
    ]
    if req.novel: cmd.append("--novel")
    if req.require_binding_evidence: cmd.append("--require_binding_evidence")
    if req.require_expression: cmd.append("--require_expression")

    subprocess.Popen(cmd)  # returns immediately
    return {"query_id": query_id}



@app.get("/status/{query_id}")
def get_status(query_id: str):
    try:
        job = read_job(query_id)
        return {"status": job.get("status"), "error": job.get("error")}
    except Exception as e:
        return {"status": "error", "error": f"/status failed: {type(e).__name__}: {e}"}


@app.get("/result/{query_id}")
def get_result(query_id: str):
    try:
        return read_job(query_id)
    except Exception as e:
        return {"status": "error", "error": f"/result failed: {type(e).__name__}: {e}"}

