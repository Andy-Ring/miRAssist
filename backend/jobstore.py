from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None  # type: ignore


# Default location for job JSONs
DEFAULT_JOB_DIR = Path("/tmp/mirassist_jobs")
DEFAULT_JOB_DIR.mkdir(parents=True, exist_ok=True)


def _is_bad_float(x: float) -> bool:
    return (x is None) or (not math.isfinite(x))


def _to_jsonable(x: Any) -> Any:
    """
    Recursively convert non-JSON-safe types into plain Python and
    replace NaN/Inf with None (Starlette disallows NaN/Inf).
    """
    if x is None:
        return None

    # numpy scalars
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        x = x.item()

    # floats (catch NaN/Inf)
    if isinstance(x, float):
        return None if _is_bad_float(x) else x

    # numpy arrays
    if isinstance(x, np.ndarray):
        return _to_jsonable(x.tolist())

    # pandas
    if pd is not None:
        if isinstance(x, pd.DataFrame):
            return _to_jsonable(x.to_dict(orient="records"))
        if isinstance(x, pd.Series):
            return _to_jsonable(x.tolist())

    # dict
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}

    # list/tuple/set
    if isinstance(x, (list, tuple, set)):
        return [_to_jsonable(v) for v in x]

    # Path
    if isinstance(x, Path):
        return str(x)

    return x


class JobStore:
    """
    Backward/forward compatible job store.

    - Some versions of backend/app.py expect:
        from backend.jobstore import JobStore
        store = JobStore(...)
        store.write(query_id, payload)
        store.read(query_id)

    - Other versions use read_job/write_job functions directly.
    This file supports both.
    """

    def __init__(self, job_dir: str | Path = DEFAULT_JOB_DIR):
        self.job_dir = Path(job_dir)
        self.job_dir.mkdir(parents=True, exist_ok=True)

    def job_path(self, query_id: str) -> Path:
        return self.job_dir / f"{query_id}.json"

    def exists(self, query_id: str) -> bool:
        return self.job_path(query_id).exists()

    def read(self, query_id: str) -> Dict[str, Any]:
        p = self.job_path(query_id)
        if not p.exists():
            return {"status": "unknown"}
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            # can happen if read during atomic swap window or partial write
            return {"status": "running"}

    def write(self, query_id: str, payload: Dict[str, Any]) -> None:
        p = self.job_path(query_id)
        safe_payload = _to_jsonable(payload)

        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(safe_payload, ensure_ascii=False))
        tmp.replace(p)  # atomic write


# -------------------------------------------------------------------
# Function API (keep for compatibility with other backend versions)
# -------------------------------------------------------------------

def job_path(query_id: str) -> Path:
    return DEFAULT_JOB_DIR / f"{query_id}.json"


def read_job(query_id: str) -> Dict[str, Any]:
    return JobStore(DEFAULT_JOB_DIR).read(query_id)


def write_job(query_id: str, payload: Dict[str, Any]) -> None:
    JobStore(DEFAULT_JOB_DIR).write(query_id, payload)
