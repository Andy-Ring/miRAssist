from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from sqlalchemy import text

from backend.config import get_jobstore_backend, resolve_job_dir
from backend.db import get_database_engine


JOB_TABLE = "mirassist_jobs"
_INITIALIZED_DATABASES: set[str] = set()


def _to_jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    try:
        import numpy as np  # type: ignore

        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
    except Exception:
        pass
    return str(obj)


def initialize_jobstore() -> None:
    backend = get_jobstore_backend()
    if backend == "filesystem":
        resolve_job_dir().mkdir(parents=True, exist_ok=True)
        return

    engine = get_database_engine()
    if engine is None:
        resolve_job_dir().mkdir(parents=True, exist_ok=True)
        return

    engine_key = str(engine.url)
    if engine_key in _INITIALIZED_DATABASES:
        return

    with engine.begin() as conn:
        conn.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS {JOB_TABLE} (
                    query_id TEXT PRIMARY KEY,
                    payload JSONB NOT NULL,
                    status TEXT,
                    stage TEXT,
                    updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW(),
                    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW()
                )
                """
            )
        )
    _INITIALIZED_DATABASES.add(engine_key)


def job_path(query_id: str) -> Path:
    job_dir = resolve_job_dir()
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir / f"{query_id}.json"


def _is_missing_job(payload: Dict[str, Any]) -> bool:
    return payload == {"status": "unknown"}


def _merge_payload(query_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    safe_payload = _to_jsonable(payload)
    existing = read_job(query_id)
    if _is_missing_job(existing):
        return safe_payload

    merged = dict(existing)
    merged.update(safe_payload)
    return merged


def _read_job_filesystem(query_id: str) -> Dict[str, Any]:
    p = job_path(query_id)
    if not p.exists():
        return {"status": "unknown"}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "running"}


def _write_job_filesystem(query_id: str, payload: Dict[str, Any]) -> None:
    p = job_path(query_id)
    merged = _merge_payload(query_id, payload)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def _read_job_postgres(query_id: str) -> Dict[str, Any]:
    initialize_jobstore()
    engine = get_database_engine()
    if engine is None:
        return _read_job_filesystem(query_id)

    with engine.begin() as conn:
        payload = conn.execute(
            text(f"SELECT payload FROM {JOB_TABLE} WHERE query_id = :query_id"),
            {"query_id": query_id},
        ).scalar_one_or_none()

    if payload is None:
        return {"status": "unknown"}
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        return json.loads(payload)
    return _to_jsonable(payload)


def _write_job_postgres(query_id: str, payload: Dict[str, Any]) -> None:
    initialize_jobstore()
    engine = get_database_engine()
    if engine is None:
        _write_job_filesystem(query_id, payload)
        return

    merged = _merge_payload(query_id, payload)
    conn_payload = json.dumps(merged, ensure_ascii=False)
    status = merged.get("status")
    stage = merged.get("stage")

    with engine.begin() as conn:
        conn.execute(
            text(
                f"""
                INSERT INTO {JOB_TABLE} (
                    query_id,
                    payload,
                    status,
                    stage,
                    updated_at,
                    created_at
                )
                VALUES (
                    :query_id,
                    CAST(:payload AS JSONB),
                    :status,
                    :stage,
                    NOW(),
                    NOW()
                )
                ON CONFLICT (query_id) DO UPDATE
                SET
                    payload = CAST(:payload AS JSONB),
                    status = :status,
                    stage = :stage,
                    updated_at = NOW()
                """
            ),
            {
                "query_id": query_id,
                "payload": conn_payload,
                "status": status,
                "stage": stage,
            },
        )


def read_job(query_id: str) -> Dict[str, Any]:
    if get_jobstore_backend() == "postgres":
        return _read_job_postgres(query_id)
    return _read_job_filesystem(query_id)


def write_job(query_id: str, payload: Dict[str, Any]) -> None:
    if get_jobstore_backend() == "postgres":
        _write_job_postgres(query_id, payload)
        return
    _write_job_filesystem(query_id, payload)


class JobStore:
    """
    Filesystem-backed compatibility wrapper.

    Older code imported `JobStore` directly from this module. The application
    now uses module-level `read_job` / `write_job`, but this class remains
    available for older local tooling.
    """

    def __init__(self, job_dir: str | Path | None = None):
        self.job_dir = Path(job_dir) if job_dir is not None else resolve_job_dir()
        self.job_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, query_id: str) -> Path:
        return self.job_dir / f"{query_id}.json"

    def read(self, query_id: str) -> Dict[str, Any]:
        p = self._path(query_id)
        if not p.exists():
            return {"status": "unknown"}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"status": "running"}

    def write(self, query_id: str, payload: Dict[str, Any]) -> None:
        safe_payload = _to_jsonable(payload)
        existing = self.read(query_id)
        merged = dict(existing) if not _is_missing_job(existing) else {}
        merged.update(safe_payload)

        p = self._path(query_id)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)
