from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    import backend.config  # noqa: F401
    import backend.worker  # noqa: F401
    import frontend.app  # noqa: F401

    print("Import path smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
