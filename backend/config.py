from __future__ import annotations

import os
from pathlib import Path


PROJECT_NAME = "miRAssist"
VERSION = os.getenv("MIRASSIST_VERSION", "0.2.0")

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

DEFAULT_EVIDENCE_CANDIDATES = (
    PROCESSED_DIR / "evidence_interactions.parquet",
    PROCESSED_DIR / "evidence_pairs_tcga.parquet",
)


def _first_nonempty(*values: str | None) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _load_local_dotenv() -> None:
    dotenv_path = ROOT_DIR / ".env"
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        os.environ.setdefault(key, value)


_load_local_dotenv()


def resolve_backend_url(default: str = "http://127.0.0.1:7861") -> str:
    return _first_nonempty(
        os.getenv("BACKEND_URL"),
        os.getenv("MIRASSIST_BACKEND_URL"),
        default,
    ) or default


def get_app_mode() -> str:
    mode = (os.getenv("MIRASSIST_APP_MODE", "direct") or "direct").strip().lower()
    return mode if mode in {"direct", "api"} else "direct"


def get_database_url() -> str | None:
    return _first_nonempty(os.getenv("DATABASE_URL"))


def database_configured() -> bool:
    return get_database_url() is not None


def get_jobstore_backend() -> str:
    requested = (os.getenv("JOBSTORE_BACKEND", "filesystem") or "filesystem").strip().lower()
    if requested not in {"filesystem", "postgres"}:
        requested = "filesystem"
    if requested == "postgres" and not database_configured():
        return "filesystem"
    return requested


def get_worker_mode() -> str:
    mode = (os.getenv("WORKER_MODE", "subprocess") or "subprocess").strip().lower()
    return mode if mode in {"subprocess", "inline"} else "subprocess"


def get_evidence_backend() -> str:
    requested = (os.getenv("EVIDENCE_BACKEND", "parquet") or "parquet").strip().lower()
    if requested not in {"parquet", "postgres"}:
        requested = "parquet"
    if requested == "postgres" and not database_configured():
        return "parquet"
    return requested


def get_evidence_table() -> str:
    return (os.getenv("EVIDENCE_TABLE", "mirassist_evidence_pairs") or "mirassist_evidence_pairs").strip()


def resolve_evidence_path(explicit: str | None = None) -> Path:
    candidate = _first_nonempty(
        explicit,
        os.getenv("MIRASSIST_EVIDENCE"),
        os.getenv("MIRASSIST_EVIDENCE_PATH"),
    )
    if candidate:
        return Path(candidate).expanduser().resolve()

    base = _first_nonempty(os.getenv("MIASSIST_BASE"))
    if base:
        base_path = Path(base).expanduser().resolve()
        for name in ("evidence_interactions.parquet", "evidence_pairs_tcga.parquet"):
            candidate_path = base_path / "data" / "processed" / name
            if candidate_path.exists():
                return candidate_path.resolve()

    for path in DEFAULT_EVIDENCE_CANDIDATES:
        if path.exists():
            return path.resolve()

    return DEFAULT_EVIDENCE_CANDIDATES[0].resolve()


def resolve_job_dir() -> Path:
    return Path(os.getenv("MIRASSIST_JOB_DIR", "runs/jobs")).expanduser().resolve()


def resolve_logs_dir() -> Path:
    if os.getenv("MIASSIST_LOG_DIR"):
        return Path(os.environ["MIASSIST_LOG_DIR"]).expanduser().resolve()

    if os.getenv("MIASSIST_BASE"):
        return (Path(os.environ["MIASSIST_BASE"]).expanduser().resolve() / "logs").resolve()

    return (ROOT_DIR / "logs").resolve()


def get_llm_backend() -> str:
    return (os.getenv("MIRASSIST_LLM_BACKEND", "transformers") or "transformers").strip().lower()


def get_model_name() -> str:
    return os.getenv("MIRASSIST_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")


def get_openai_api_key() -> str | None:
    return _first_nonempty(os.getenv("OPENAI_API_KEY"))


def openai_configured() -> bool:
    return get_openai_api_key() is not None


def get_openai_base_url() -> str | None:
    return _first_nonempty(os.getenv("MIRASSIST_OPENAI_BASE_URL"))


def get_planner_model() -> str:
    return os.getenv("MIRASSIST_PLANNER_MODEL", "gpt-5.4-nano")


def get_synth_model() -> str:
    return os.getenv("MIRASSIST_SYNTH_MODEL", "gpt-5.4-mini")


def get_openai_timeout() -> float:
    return float(os.getenv("MIRASSIST_OPENAI_TIMEOUT", "600"))


def get_planner_temperature() -> float:
    return float(os.getenv("MIRASSIST_OPENAI_TEMPERATURE_PLANNER", "0"))


def get_synth_temperature() -> float:
    return float(os.getenv("MIRASSIST_OPENAI_TEMPERATURE_SYNTH", "0.2"))


def get_synth_max_tokens() -> int:
    return int(os.getenv("MIRASSIST_SYNTH_MAX_TOKENS", "2500"))


def get_debug_deep() -> bool:
    return (os.getenv("MIRASSIST_DEBUG_DEEP", "0") or "0").strip() == "1"


def get_debug_ui() -> bool:
    return (os.getenv("MIRASSIST_DEBUG_UI", "0") or "0").strip() == "1" or get_debug_deep()


def get_default_k() -> int:
    return int(os.getenv("MIRASSIST_DEFAULT_K", "5"))


def get_vllm_http_url() -> str | None:
    return _first_nonempty(os.getenv("MIRASSIST_VLLM_HTTP_URL"))


def get_vllm_url() -> str | None:
    return _first_nonempty(os.getenv("MIRASSIST_VLLM_URL"))


def get_vllm_api_key() -> str | None:
    return _first_nonempty(os.getenv("MIRASSIST_VLLM_API_KEY"))


MODEL_NAME = get_model_name()
PLANNER_MODEL = get_planner_model()
SYNTH_MODEL = get_synth_model()
MAX_PROMPT_TOKENS = int(os.getenv("MIRASSIST_MAX_PROMPT_TOKENS", "6500"))
DEFAULT_K = get_default_k()
DEFAULT_MIN_SUPPORT = int(os.getenv("MIRASSIST_DEFAULT_MIN_SUPPORT", "1"))
EVIDENCE_PATH = resolve_evidence_path()
