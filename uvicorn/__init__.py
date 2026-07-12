from __future__ import annotations

import importlib.machinery
import os
from pathlib import Path
import sys


def _load_real_uvicorn() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    search_paths: list[str] = []
    for entry in sys.path:
        resolved = Path(entry or ".").resolve()
        if resolved == repo_root:
            continue
        search_paths.append(entry)

    spec = importlib.machinery.PathFinder.find_spec(__name__, search_paths)
    if spec is None or spec.loader is None or spec.origin == __file__:
        raise ImportError("Unable to locate the environment uvicorn package.")

    module = sys.modules[__name__]
    module.__file__ = spec.origin
    module.__loader__ = spec.loader
    module.__package__ = __name__
    module.__spec__ = spec
    if spec.submodule_search_locations is not None:
        module.__path__ = list(spec.submodule_search_locations)
    spec.loader.exec_module(module)

    real_version = getattr(module, "__version__", None)
    if real_version is not None:
        module.__miassist_real_uvicorn_version__ = real_version
        # Streamlit switches to uvicorn's newer websockets-sansio backend at 0.44+.
        # Posit Connect Cloud is rejecting that websocket upgrade for this app.
        module.__version__ = "0.43.0"
        if os.getenv("MIRASSIST_DEBUG_UI"):
            print(
                f"[miRAssist] uvicorn shim forcing legacy websocket selection "
                f"(real={real_version}, reported={module.__version__})",
                file=sys.stderr,
                flush=True,
            )


_load_real_uvicorn()
