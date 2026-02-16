import os
import subprocess
import uuid
from typing import Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel, Field

from backend.jobstore import read_job, write_job


app = FastAPI(title="miRAssist backend")


class QueryRequest(BaseModel):
    question: str
    novel: bool = True
    k: int = 200
    min_support: int = 1
    require_binding_evidence: bool = False
    require_expression: bool = False
    pathway_mode: str = Field(
        default="auto",
        description="Override pathway integration mode: auto|boost|filter",
    )


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/query")
def submit_query(req: QueryRequest) -> Dict[str, Any]:
    query_id = uuid.uuid4().hex[:16]
    write_job(query_id, {"status": "queued"})

    cmd = [
        "python", "-m", "backend.worker",
        "--query_id", query_id,
        "--question", req.question,
        "--k", str(req.k),
        "--min_support", str(req.min_support),
    ]

    if req.novel:
        cmd.append("--novel")
    if req.require_binding_evidence:
        cmd.append("--require_binding_evidence")
    if req.require_expression:
        cmd.append("--require_expression")

    pm = (req.pathway_mode or "auto").lower().strip()
    if pm in ("boost", "filter"):
        cmd += ["--pathway_mode", pm]

    # Keep worker output out of the API process
    env = os.environ.copy()
    subprocess.Popen(cmd, env=env)

    return {"query_id": query_id}


@app.get("/status/{query_id}")
def status(query_id: str) -> Dict[str, Any]:
    job = read_job(query_id)
    return {"status": job.get("status", "unknown")}


@app.get("/result/{query_id}")
def result(query_id: str) -> Dict[str, Any]:
    return read_job(query_id)
