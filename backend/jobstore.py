from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

from sqlalchemy import text

from backend.config import get_jobstore_backend, resolve_job_dir
from backend.db import get_database_engine


JOB_TABLE = "mirassist_jobs"
_INITIALIZED_DATABASES: set[str] = set()


def _sanitize_json_value(obj: Any, path: str = "$") -> Any:
    if obj is None:
        return None

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, bool):
        return obj

    if isinstance(obj, int):
        return obj

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None

    if isinstance(obj, str):
        return obj

    if isinstance(obj, dict):
        return {
            str(k): _sanitize_json_value(v, f"{path}.{k}")
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple, set)):
        return [
            _sanitize_json_value(value, f"{path}[{idx}]")
            for idx, value in enumerate(obj)
        ]

    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore

        if obj is pd.NA or obj is pd.NaT:
            return None

        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()

        if isinstance(obj, pd.DataFrame):
            records = obj.to_dict(orient="records")
            return _sanitize_json_value(records, path)

        if isinstance(obj, pd.Series):
            if obj.index.is_unique:
                return _sanitize_json_value(obj.to_dict(), path)
            return _sanitize_json_value(obj.tolist(), path)

        if isinstance(obj, np.ndarray):
            return _sanitize_json_value(obj.tolist(), path)

        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            native = obj.item()
            if isinstance(native, float) and not math.isfinite(native):
                return None
            return native

        if isinstance(obj, np.datetime64):
            return str(obj)
    except Exception:
        pass

    return str(obj)


def sanitize_json_payload(payload: Any) -> Any:
    return _sanitize_json_value(payload)


def _json_dumps_safe(payload: Any) -> str:
    safe_payload = sanitize_json_payload(payload)
    try:
        return json.dumps(safe_payload, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Job payload is not valid JSON after sanitization: {exc}"
        ) from exc


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
    safe_payload = sanitize_json_payload(payload)
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
    tmp.write_text(_json_dumps_safe(merged), encoding="utf-8")
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
        return sanitize_json_payload(payload)
    if isinstance(payload, str):
        return json.loads(payload)
    return sanitize_json_payload(payload)


def _write_job_postgres(query_id: str, payload: Dict[str, Any]) -> None:
    initialize_jobstore()
    engine = get_database_engine()
    if engine is None:
        _write_job_filesystem(query_id, payload)
        return

    merged = _merge_payload(query_id, payload)
    conn_payload = _json_dumps_safe(merged)
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
        safe_payload = sanitize_json_payload(payload)
        existing = self.read(query_id)
        merged = dict(existing) if not _is_missing_job(existing) else {}
        merged.update(safe_payload)

        p = self._path(query_id)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(_json_dumps_safe(merged), encoding="utf-8")
        tmp.replace(p)
