# backend/llm_backend.py
from __future__ import annotations

import os
from typing import Optional

# Backends:
# - transformers: in-process HF transformers (best for Colab dev)
# - vllm_http: call a vLLM OpenAI-compatible server over HTTP (best for rented GPU prod)
DEFAULT_BACKEND = "transformers"


def get_backend_name() -> str:
    return os.getenv("MIRASSIST_LLM_BACKEND", DEFAULT_BACKEND).strip().lower()


def chat(
    system: str,
    user: str,
    *,
    max_new_tokens: int = 600,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> str:
    backend = get_backend_name()

    if backend == "transformers":
        from backend.llm_transformers import chat as _chat
        return _chat(
            system=system,
            user=user,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    if backend == "vllm_http":
        from backend.llm_vllm_http import chat as _chat
        return _chat(
            system=system,
            user=user,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    raise ValueError(
        f"Unknown MIRASSIST_LLM_BACKEND='{backend}'. "
        f"Expected 'transformers' or 'vllm_http'."
    )
