from __future__ import annotations

import re
from typing import Any, Optional, Tuple

from sqlalchemy import create_engine

from backend.config import get_database_url


_ENGINE_CACHE: dict[str, Any] = {}
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_\.]*$")


def normalize_database_url(database_url: str) -> str:
    url = (database_url or "").strip()
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://") :]
    if url.startswith("postgresql://") and not url.startswith("postgresql+psycopg://"):
        return "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


def _build_engine_kwargs(resolved_url: str) -> dict:
    kwargs = {
        "future": True,
        "pool_pre_ping": True,
    }
    if resolved_url.startswith("postgresql+psycopg://"):
        # Supabase commonly uses a PgBouncer-style pooler. Psycopg's docs recommend
        # disabling automatic prepared statements in that setup unless the pooler
        # explicitly supports them end-to-end.
        kwargs["connect_args"] = {"prepare_threshold": None}
    return kwargs


def get_database_engine(database_url: Optional[str] = None) -> Optional[Any]:
    resolved_url = normalize_database_url(database_url or get_database_url() or "")
    if not resolved_url:
        return None

    if resolved_url not in _ENGINE_CACHE:
        _ENGINE_CACHE[resolved_url] = create_engine(resolved_url, **_build_engine_kwargs(resolved_url))
    return _ENGINE_CACHE[resolved_url]


def check_database_connectivity() -> Tuple[str, Optional[str]]:
    from sqlalchemy import text

    engine = get_database_engine()
    if engine is None:
        return "not_configured", None

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return "ok", None
    except Exception as exc:
        return "error", str(exc)


def is_safe_identifier(name: str) -> bool:
    return bool(_IDENTIFIER_RE.fullmatch((name or "").strip()))


def quote_identifier(name: str) -> str:
    if not is_safe_identifier(name):
        raise ValueError(f"Unsafe SQL identifier: {name!r}")
    return ".".join(f'"{part}"' for part in name.split("."))
