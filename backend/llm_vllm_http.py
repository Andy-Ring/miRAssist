# backend/llm_vllm_http.py
from __future__ import annotations

import os
import requests

from backend.config import MODEL_NAME

# Example:
# export MIRASSIST_VLLM_URL="http://127.0.0.1:8000"
# export MIRASSIST_VLLM_API_KEY=""   (optional)
DEFAULT_VLLM_URL = "http://127.0.0.1:8000"


def _vllm_url() -> str:
    return os.getenv("MIRASSIST_VLLM_URL", DEFAULT_VLLM_URL).rstrip("/")


def _api_key() -> str:
    return os.getenv("MIRASSIST_VLLM_API_KEY", "")


def chat(
    system: str,
    user: str,
    *,
    max_new_tokens: int = 600,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> str:
    url = f"{_vllm_url()}/v1/chat/completions"

    headers = {"Content-Type": "application/json"}
    key = _api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_new_tokens),
    }

    r = requests.post(url, headers=headers, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()

    # OpenAI-style response
    return data["choices"][0]["message"]["content"].strip()
