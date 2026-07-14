"""
Evidence snapshot bootstrap for the miRAssist Claude skill.

The Cowork sandbox blocks arbitrary network egress, but github.com is on the
default allowlist. So instead of querying Supabase live, the skill downloads a
one-time evidence snapshot (a parquet exported from the Supabase table, learned
XGBoost scores and precomputed percentiles included) from the project's GitHub
Releases, caches it locally, and reads from it thereafter.

Configure the release asset URL via `evidence_parquet_url` in skill_settings.json
(or the MIRASSIST_EVIDENCE_URL environment variable).
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from backend.config import get_evidence_parquet_url

_DOWNLOAD_TIMEOUT = 120


def _cache_dir() -> Path:
    base = os.getenv("MIRASSIST_CACHE_DIR")
    if base:
        path = Path(base).expanduser()
    else:
        xdg = os.getenv("XDG_CACHE_HOME")
        root = Path(xdg).expanduser() if xdg else (Path.home() / ".cache")
        path = root / "mirassist"
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception:
        # Fall back to a temp dir if home/cache is not writable.
        tmp = Path(tempfile.gettempdir()) / "mirassist_cache"
        tmp.mkdir(parents=True, exist_ok=True)
        return tmp


def _cache_path_for(url: str) -> Path:
    # Version the cache by URL so bumping the release tag re-downloads cleanly.
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    lower_url = url.split("?", 1)[0].lower()
    suffix = ".csv.gz" if lower_url.endswith(".csv.gz") else (".csv" if lower_url.endswith(".csv") else ".parquet")
    return _cache_dir() / f"evidence_{digest}{suffix}"


def _download(url: str, dest: Path) -> None:
    import requests

    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".part", dir=str(dest.parent))
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        with requests.get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT) as resp:
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"Failed to download evidence snapshot ({resp.status_code}) from {url}. "
                    "Check that the GitHub Release asset exists and is public."
                )
            with open(tmp_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        fh.write(chunk)
        if tmp_path.stat().st_size == 0:
            raise RuntimeError(f"Downloaded evidence snapshot from {url} was empty.")
        shutil.move(str(tmp_path), str(dest))
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def ensure_evidence_parquet() -> str:
    """Return a local path to the evidence snapshot, downloading + caching if needed.

    Resolution order:
      1) MIRASSIST_EVIDENCE env pointing at an existing file (offline / dev override).
      2) cached download of `evidence_parquet_url`.
    """
    explicit = os.getenv("MIRASSIST_EVIDENCE")
    if explicit and Path(explicit).expanduser().exists():
        return str(Path(explicit).expanduser())

    url = get_evidence_parquet_url()
    if not url:
        raise RuntimeError(
            "No evidence snapshot is configured. Set 'evidence_parquet_url' in "
            "skill_settings.json to a GitHub Release asset URL, or set MIRASSIST_EVIDENCE "
            "to a local parquet path."
        )

    cache = _cache_path_for(url)
    if cache.exists() and cache.stat().st_size > 0:
        return str(cache)

    _download(url, cache)
    return str(cache)
