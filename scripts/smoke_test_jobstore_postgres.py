from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

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

    write_job(query_id, {"status": "queued", "stage": "queued", "source": "smoke"})
    write_job(query_id, {"status": "done", "answer": {"summary": "ok"}})
    payload = read_job(query_id)

    if payload.get("status") != "done":
        print("Postgres jobstore smoke test failed: status mismatch.")
        return 1

    print("Postgres jobstore smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
