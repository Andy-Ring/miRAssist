from __future__ import annotations

import importlib
import os
import platform
import sys
import traceback
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def _masked_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        return "<unset>"
    if len(value) <= 8:
        return "<set>"
    return f"<set len={len(value)}>"


def _import_report(module_name: str) -> None:
    print(f"\n[module] {module_name}")
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "<no __version__>")
        print(f"status=ok version={version}")
        real_version = getattr(module, "__miassist_real_uvicorn_version__", None)
        if real_version:
            print(f"real_version={real_version}")
        module_file = getattr(module, "__file__", None)
        if module_file:
            print(f"path={module_file}")
    except Exception as exc:
        print(f"status=error type={type(exc).__name__} message={exc}")
        print(traceback.format_exc())


def _print_streamlit_server_report() -> None:
    print("\n[module] streamlit.web.server.starlette.starlette_server")
    try:
        from streamlit.web.server.starlette import starlette_server

        protocol = starlette_server._get_websocket_protocol()
        print(f"status=ok websocket_protocol={protocol}")
        print(f"path={starlette_server.__file__}")
    except Exception as exc:
        print(f"status=error type={type(exc).__name__} message={exc}")
        print(traceback.format_exc())


def main() -> int:
    _print_header("Runtime")
    print(f"python={sys.version}")
    print(f"executable={sys.executable}")
    print(f"platform={platform.platform()}")
    print(f"cwd={Path.cwd()}")
    print(f"repo_root={REPO_ROOT}")

    _print_header("Environment")
    for name in [
        "DATABASE_URL",
        "JOBSTORE_BACKEND",
        "EVIDENCE_BACKEND",
        "EVIDENCE_TABLE",
        "OPENAI_API_KEY",
        "MIRASSIST_LLM_BACKEND",
        "MIRASSIST_EVIDENCE",
        "MIRASSIST_EVIDENCE_PATH",
        "MIRASSIST_DEBUG_UI",
        "MIRASSIST_DEBUG_STAGE",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        value = _masked_env(name) if name in {"DATABASE_URL", "OPENAI_API_KEY"} else (os.getenv(name) or "<unset>")
        print(f"{name}={value}")

    _print_header("sys.path")
    for idx, entry in enumerate(sys.path[:10]):
        print(f"{idx}: {entry}")

    _print_header("Package Imports")
    for module_name in [
        "streamlit",
        "uvicorn",
        "pandas",
        "numpy",
        "pyarrow",
        "sqlalchemy",
        "psycopg",
        "openai",
        "requests",
        "altair",
        "xgboost",
        "duckdb",
        "sklearn",
    ]:
        _import_report(module_name)
    _print_streamlit_server_report()

    _print_header("App Imports")
    for module_name in [
        "backend.config",
        "backend.db",
        "backend.jobstore",
        "backend.worker",
        "frontend.help_content",
        "frontend.app",
        "app",
    ]:
        _import_report(module_name)

    _print_header("Filesystem Checks")
    for path in [
        REPO_ROOT / "app.py",
        REPO_ROOT / "frontend" / "app.py",
        REPO_ROOT / "frontend" / "assets" / "miRAssist_logo.png",
        REPO_ROOT / "requirements.txt",
    ]:
        print(f"{path}: {'exists' if path.exists() else 'missing'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
