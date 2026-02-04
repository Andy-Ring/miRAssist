# backend/synthesizer.py
from __future__ import annotations

from typing import Dict, Any

from backend.llm_backend import chat
from backend.prompting import SYSTEM_PROMPT


def run_synthesizer(bundle: Dict[str, Any], model: str = None) -> Dict[str, Any]:
    # model is ignored here; MODEL_NAME is configured in backend/config.py.
    out = chat(
        system=bundle.get("system_prompt", SYSTEM_PROMPT),
        user=bundle["user_prompt"],
        max_new_tokens=1200,
        temperature=0.2,
        top_p=0.95,
    )

    return {
        "raw_text": out,
        "summary": out,
        "suggested_experiments": [],
    }

