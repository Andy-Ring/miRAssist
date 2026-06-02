from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.jobstore import read_job, write_job


def main() -> int:
    if not os.environ.get("DATABASE_URL"):
        print("DATABASE_URL is not set. Skipping postgres jobstore smoke test.")
        return 0

    os.environ["JOBSTORE_BACKEND"] = "postgres"
    query_id = f"smoke-{uuid.uuid4().hex[:12]}"

    write_job(
        query_id,
        {
            "status": "queued",
            "stage": "queued",
            "source": "smoke",
            "x": float("nan"),
            "y": float("inf"),
            "rows": [{"ts_best_contextpp": float("nan")}],
            "np_float": np.float64(np.nan),
        },
    )
    write_job(query_id, {"status": "done", "answer": {"summary": "ok"}})
    payload = read_job(query_id)

    if payload.get("status") != "done":
        print("Postgres jobstore smoke test failed: status mismatch.")
        return 1

    if payload.get("x") is not None:
        print("Postgres jobstore smoke test failed: x should be null.")
        return 1
    if payload.get("y") is not None:
        print("Postgres jobstore smoke test failed: y should be null.")
        return 1
    if payload.get("rows", [{}])[0].get("ts_best_contextpp") is not None:
        print("Postgres jobstore smoke test failed: shortlist NaN should be null.")
        return 1
    if payload.get("np_float") is not None:
        print("Postgres jobstore smoke test failed: np_float should be null.")
        return 1

    print("Postgres jobstore smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
