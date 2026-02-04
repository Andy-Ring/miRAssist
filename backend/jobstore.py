from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None  # type: ignore

JOB_DIR = Path("/tmp/mirassist_jobs")
JOB_DIR.mkdir(parents=True, exist_ok=True)


def job_path(query_id: str) -> Path:
    return JOB_DIR / f"{query_id}.json"


def _is_bad_float(x: float) -> bool:
    return (x is None) or (not math.isfinite(x))


def _to_jsonable(x: Any) -> Any:
    """
    Recursively convert non-JSON-safe types into plain Python and
    replace NaN/Inf with None (Starlette disallows NaN/Inf).
    """
    # None
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


def read_job(query_id: str) -> Dict[str, Any]:
    p = job_path(query_id)
    if not p.exists():
        return {"status": "unknown"}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        # can happen if read during atomic swap window or partial write
        return {"status": "running"}


def write_job(query_id: str, payload: Dict[str, Any]) -> None:
    p = job_path(query_id)
    safe_payload = _to_jsonable(payload)

    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(safe_payload, ensure_ascii=False))
    tmp.replace(p)  # atomic write
