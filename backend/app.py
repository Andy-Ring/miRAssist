import os
import sys
import time
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from backend.jobstore import read_job, write_job


app = FastAPI(title="miRAssist backend")

# Repo root = parent of backend/ (this file lives in backend/app.py)
REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_logs_dir() -> Path:
    """
    Pick a logs directory that works both locally and on Colab/Drive.

    Priority:
      1) MIASSIST_LOG_DIR (explicit override)
      2) MIASSIST_BASE/logs (if MIASSIST_BASE is set, e.g. /content/drive/MyDrive/miRAssist_colab)
      3) <repo>/logs
    """
    env = os.environ

    if env.get("MIASSIST_LOG_DIR"):
        return Path(env["MIASSIST_LOG_DIR"]).expanduser().resolve()

    if env.get("MIASSIST_BASE"):
        base = Path(env["MIASSIST_BASE"]).expanduser().resolve()
        return base / "logs"

    return (REPO_ROOT / "logs").resolve()


LOGS_DIR = _resolve_logs_dir()
LOGS_DIR.mkdir(parents=True, exist_ok=True)


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
    return {
        "ok": True,
        "repo_root": str(REPO_ROOT),
        "logs_dir": str(LOGS_DIR),
        "python": sys.executable,
        "cwd": os.getcwd(),
        "has_drive": Path("/content/drive").exists(),
    }


@app.post("/query")
def submit_query(req: QueryRequest) -> Dict[str, Any]:
    query_id = uuid.uuid4().hex[:16]

    # Create initial job record
    write_job(
        query_id,
        {
            "status": "queued",
            "stage": "queued",
            "query_id": query_id,
            "question": req.question,
            "created_at": time.time(),
        },
    )

    cmd = [
        sys.executable,
        "-m",
        "backend.worker",
        "--query_id",
        query_id,
        "--question",
        req.question,
        "--k",
        str(req.k),
        "--min_support",
        str(req.min_support),
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

    out_path = LOGS_DIR / f"worker_{query_id}.out"
    err_path = LOGS_DIR / f"worker_{query_id}.err"

    # Make sure the worker can import backend.* reliably
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    # Helpful for workers that want to know where to write logs or find data
    env.setdefault("MIASSIST_LOG_DIR", str(LOGS_DIR))
    env.setdefault("MIASSIST_REPO_ROOT", str(REPO_ROOT))

    # Launch worker, redirect output to log files
    with open(out_path, "w") as out_f, open(err_path, "w") as err_f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=out_f,
            stderr=err_f,
        )

    # Record worker metadata immediately
    write_job(
        query_id,
        {
            "status": "queued",   # worker should flip to running when it starts
            "stage": "queued",
            "worker_pid": proc.pid,
            "worker_out": str(out_path),
            "worker_err": str(err_path),
            "worker_cmd": cmd,
            "repo_root": str(REPO_ROOT),
            "logs_dir": str(LOGS_DIR),
        },
    )

    # Quick failure detection so jobs don't sit queued forever
    time.sleep(0.25)
    rc = proc.poll()
    if rc is not None:
        write_job(
            query_id,
            {
                "status": "error",
                "stage": "error",
                "error": (
                    f"Worker exited immediately (rc={rc}). "
                    f"See logs: {out_path} and {err_path}"
                ),
                "worker_rc": rc,
                "worker_pid": proc.pid,
                "worker_out": str(out_path),
                "worker_err": str(err_path),
                "worker_cmd": cmd,
                "repo_root": str(REPO_ROOT),
                "logs_dir": str(LOGS_DIR),
            },
        )

    return {"query_id": query_id}


@app.get("/status/{query_id}")
def status(query_id: str) -> Dict[str, Any]:
    job = read_job(query_id)
    return {
        "status": job.get("status", "unknown"),
        "stage": job.get("stage"),
        "worker_pid": job.get("worker_pid"),
    }


@app.get("/result/{query_id}")
def result(query_id: str) -> Dict[str, Any]:
    return read_job(query_id)

