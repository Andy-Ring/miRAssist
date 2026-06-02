from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    import app  # noqa: F401
    from backend import jobstore  # noqa: F401
    from backend import llm_backend  # noqa: F401
    from backend.worker import run_query_job  # noqa: F401

    print("Direct mode import smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
