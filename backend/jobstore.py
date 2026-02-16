import json
import os
from pathlib import Path
from typing import Any, Dict


JOB_DIR = Path(os.environ.get("MIRASSIST_JOB_DIR", "runs/jobs"))


def _to_jsonable(obj: Any) -> Any:
    """Best-effort conversion of unknown objects to JSON-serializable types."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    # pandas / numpy scalars etc.
    try:
        import numpy as np  # type: ignore
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
    except Exception:
        pass
    # fallback
    return str(obj)


def job_path(query_id: str) -> Path:
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    return JOB_DIR / f"{query_id}.json"


def read_job(query_id: str) -> Dict[str, Any]:
    p = job_path(query_id)
    if not p.exists():
        return {"status": "unknown"}
    # If file is mid-write, json could be invalid; treat as running
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {"status": "running"}


def write_job(query_id: str, payload: Dict[str, Any]) -> None:
    p = job_path(query_id)
    safe_payload = _to_jsonable(payload)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(safe_payload, ensure_ascii=False))
    tmp.replace(p)  # atomic write


# ---------------------------------------------------------------------------
# Backward-compatible wrapper
# ---------------------------------------------------------------------------

class JobStore:
    """
    Filesystem-backed job store (compat shim).

    Some earlier versions imported `JobStore` from this module.
    Newer code uses `read_job`/`write_job`.
    This wrapper prevents import errors while keeping behavior consistent.
    """

    def __init__(self, job_dir: str | Path = JOB_DIR):
        self.job_dir = Path(job_dir)
        self.job_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, query_id: str) -> Path:
        return self.job_dir / f"{query_id}.json"

    def read(self, query_id: str) -> Dict[str, Any]:
        p = self._path(query_id)
        if not p.exists():
            return {"status": "unknown"}
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return {"status": "running"}

    def write(self, query_id: str, payload: Dict[str, Any]) -> None:
        p = self._path(query_id)
        safe_payload = _to_jsonable(payload)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(safe_payload, ensure_ascii=False))
        tmp.replace(p)
