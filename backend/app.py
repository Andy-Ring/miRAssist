from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict

from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel, Field

from backend.config import (
    ROOT_DIR,
    VERSION,
    database_configured,
    get_evidence_backend,
    get_jobstore_backend,
    get_llm_backend,
    get_model_name,
    get_planner_model,
    get_synth_model,
    get_worker_mode,
    openai_configured,
    resolve_logs_dir,
)
from backend.db import check_database_connectivity
from backend.jobstore import initialize_jobstore, read_job, write_job
from backend.worker import run_query_job


app = FastAPI(title="miRAssist backend", version=VERSION)

REPO_ROOT = ROOT_DIR
LOGS_DIR = resolve_logs_dir()


class QueryRequest(BaseModel):
    question: str
    novel: bool = True
    k: int = 200
    min_support: int = 1
    require_binding_evidence: bool = False
    require_expression: bool = False
    pathway_mode: str = Field(
        default="auto",
        description="Legacy pathway override: auto|filter; boost is accepted for backward compatibility and treated as filter",
    )


def _ensure_runtime_dirs() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    initialize_jobstore()


def _build_worker_command(query_id: str, req: QueryRequest) -> list[str]:
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

    pathway_mode = (req.pathway_mode or "auto").lower().strip()
    if pathway_mode in {"boost", "filter"}:
        cmd += ["--pathway_mode", pathway_mode]

    return cmd


def _launch_subprocess_worker(query_id: str, req: QueryRequest) -> None:
    cmd = _build_worker_command(query_id, req)
    out_path = LOGS_DIR / f"worker_{query_id}.out"
    err_path = LOGS_DIR / f"worker_{query_id}.err"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env.setdefault("MIASSIST_LOG_DIR", str(LOGS_DIR))
    env.setdefault("MIASSIST_REPO_ROOT", str(REPO_ROOT))

    with open(out_path, "w", encoding="utf-8") as out_f, open(
        err_path, "w", encoding="utf-8"
    ) as err_f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=out_f,
            stderr=err_f,
        )

    write_job(
        query_id,
        {
            "status": "queued",
            "stage": "queued",
            "worker_mode": "subprocess",
            "worker_pid": proc.pid,
            "worker_out": str(out_path),
            "worker_err": str(err_path),
            "worker_cmd": cmd,
            "repo_root": str(REPO_ROOT),
            "logs_dir": str(LOGS_DIR),
        },
    )

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
            },
        )


@app.on_event("startup")
def startup() -> None:
    _ensure_runtime_dirs()


@app.get("/health")
def health() -> Dict[str, Any]:
    _ensure_runtime_dirs()
    db_status, _ = check_database_connectivity()

    return {
        "ok": True,
        "version": VERSION,
        "repo_root": str(REPO_ROOT),
        "logs_dir": str(LOGS_DIR),
        "jobstore_backend": get_jobstore_backend(),
        "worker_mode": get_worker_mode(),
        "evidence_backend": get_evidence_backend(),
        "database_configured": database_configured(),
        "database_connectivity": db_status,
        "llm_backend": get_llm_backend(),
        "planner_model": get_planner_model(),
        "synth_model": get_synth_model(),
        "openai_configured": openai_configured(),
        "model_name": get_model_name(),
        "python_executable": sys.executable,
        "cwd": os.getcwd(),
    }


@app.post("/query")
def submit_query(req: QueryRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    _ensure_runtime_dirs()

    query_id = uuid.uuid4().hex[:16]
    worker_mode = get_worker_mode()

    write_job(
        query_id,
        {
            "status": "queued",
            "stage": "queued",
            "query_id": query_id,
            "question": req.question,
            "created_at": time.time(),
            "worker_mode": worker_mode,
            "repo_root": str(REPO_ROOT),
            "logs_dir": str(LOGS_DIR),
        },
    )

    if worker_mode == "inline":
        background_tasks.add_task(
            run_query_job,
            query_id=query_id,
            question=req.question,
            k=req.k,
            min_support=req.min_support,
            novel=req.novel,
            require_binding_evidence=req.require_binding_evidence,
            require_expression=req.require_expression,
            pathway_mode=req.pathway_mode,
        )
    else:
        _launch_subprocess_worker(query_id, req)

    return {"query_id": query_id}


@app.get("/status/{query_id}")
def status(query_id: str) -> Dict[str, Any]:
    job = read_job(query_id)
    return {
        "status": job.get("status", "unknown"),
        "stage": job.get("stage"),
        "worker_pid": job.get("worker_pid"),
        "error": job.get("error"),
    }


@app.get("/result/{query_id}")
def result(query_id: str) -> Dict[str, Any]:
    return read_job(query_id)
