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


def _env_flag(name: str, default: str = "0") -> bool:
    value = _first_nonempty(os.getenv(name), default) or default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return bool(normalized)


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


# ---------------------------------------------------------------------------
# Skill settings (committed credentials for zero-config distribution)
# ---------------------------------------------------------------------------
# The Supabase project URL and the *publishable* anon key are safe to commit
# when the evidence table is protected by a read-only Row-Level-Security policy.
# The skill author fills these into skill_settings.json once; end users set
# nothing. Environment variables always override the file.
import json as _json

_SKILL_SETTINGS_CACHE: dict | None = None


def _load_skill_settings() -> dict:
    global _SKILL_SETTINGS_CACHE
    if _SKILL_SETTINGS_CACHE is not None:
        return _SKILL_SETTINGS_CACHE
    settings: dict = {}
    # config.py lives in scripts/backend/, so ROOT_DIR (parents[1]) is scripts/.
    path = ROOT_DIR / "skill_settings.json"
    try:
        if path.exists():
            settings = _json.loads(path.read_text(encoding="utf-8")) or {}
    except Exception:
        settings = {}
    _SKILL_SETTINGS_CACHE = settings
    return settings


def _setting(env_name: str, key: str, default: str = "") -> str | None:
    return _first_nonempty(os.getenv(env_name), _load_skill_settings().get(key), default)


def get_supabase_url() -> str | None:
    return _setting("MIRASSIST_SUPABASE_URL", "supabase_url")


def get_supabase_anon_key() -> str | None:
    return _setting("MIRASSIST_SUPABASE_ANON_KEY", "supabase_anon_key")


def get_evidence_parquet_url() -> str | None:
    return _setting("MIRASSIST_EVIDENCE_URL", "evidence_parquet_url")


def supabase_rest_configured() -> bool:
    url = get_supabase_url()
    key = get_supabase_anon_key()
    if not url or not key:
        return False
    # Guard against the unfilled placeholder shipping in skill_settings.json.
    return "PASTE_YOUR" not in str(key)


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


def get_evidence_backend() -> str:
    # The Claude skill defaults to a GitHub-hosted evidence snapshot: github.com is
    # on the sandbox network allowlist, while the live Supabase host usually is not.
    requested = (os.getenv("EVIDENCE_BACKEND", "github") or "github").strip().lower()
    if requested in {"github", "snapshot", "github_snapshot"}:
        return "github"
    if requested in {"rest", "supabase", "supabase_rest"}:
        return "rest"
    if requested not in {"parquet", "postgres", "rest", "github"}:
        return "github"
    if requested == "postgres" and not database_configured():
        return "github"
    return requested


def get_evidence_table() -> str:
    table_name = (_setting("EVIDENCE_TABLE", "evidence_table", "public.mirassist_evidence_pairs")
                  or "public.mirassist_evidence_pairs").strip()
    if "." not in table_name:
        table_name = f"public.{table_name}"
    return table_name


def get_evidence_table_bare() -> str:
    """Table name without schema prefix, as PostgREST expects."""
    name = get_evidence_table()
    return name.split(".", 1)[1] if "." in name else name


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
    return int(os.getenv("MIRASSIST_DEFAULT_K", "10"))


def get_default_result_count() -> int:
    return int(os.getenv("MIRASSIST_DEFAULT_RESULT_COUNT", "5"))


def get_db_candidate_limit() -> int:
    return int(os.getenv("MIRASSIST_DB_CANDIDATE_LIMIT", "1000"))


def get_debug_max_rows() -> int:
    return int(os.getenv("MIRASSIST_DEBUG_MAX_ROWS", "100"))


def get_use_structure_in_score() -> bool:
    return _env_flag("MIRASSIST_USE_STRUCTURE_IN_SCORE", default="0")


def get_use_learned_score() -> bool:
    return _env_flag("MIRASSIST_USE_LEARNED_SCORE", default="1")


def get_learned_score_column() -> str:
    return (os.getenv("MIRASSIST_LEARNED_SCORE_COLUMN", "learned_score_xgb_raw_v1") or "learned_score_xgb_raw_v1").strip()


def get_default_mirna_arm() -> str:
    value = (os.getenv("MIRASSIST_DEFAULT_MIRNA_ARM", "5p") or "5p").strip().lower()
    return value if value in {"5p", "3p", "both"} else "5p"


def get_eval_mode() -> bool:
    return (os.getenv("MIRASSIST_EVAL_MODE", "0") or "0").strip() == "1"


def get_disable_synthesis() -> bool:
    return (os.getenv("MIRASSIST_DISABLE_SYNTHESIS", "0") or "0").strip() == "1"


def use_mirtarbase_evidence() -> bool:
    explicit = _first_nonempty(os.getenv("MIRASSIST_USE_MIRTARBASE_EVIDENCE"))
    if explicit is not None:
        return explicit.strip().lower() not in {"0", "false", "no", "off"}
    return not get_eval_mode()


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
DEFAULT_RESULT_COUNT = get_default_result_count()
DEFAULT_MIN_SUPPORT = int(os.getenv("MIRASSIST_DEFAULT_MIN_SUPPORT", "1"))
EVIDENCE_PATH = resolve_evidence_path()
