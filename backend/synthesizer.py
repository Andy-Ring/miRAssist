# backend/synthesizer.py
from __future__ import annotations

from typing import Dict, Any

from backend.config import get_synth_max_tokens, get_synth_model, get_synth_temperature
from backend.llm_backend import chat
from backend.prompting import SYSTEM_PROMPT


def run_synthesizer(bundle: Dict[str, Any], model: str = None) -> Dict[str, Any]:
    selected_model = model or get_synth_model()
    print("[miRAssist] synthesizer starting", flush=True)
    out = chat(
        system=bundle.get("system_prompt", SYSTEM_PROMPT),
        user=bundle["user_prompt"],
        model=selected_model,
        max_new_tokens=get_synth_max_tokens(),
        temperature=get_synth_temperature(),
        top_p=0.95,
    )
    print("[miRAssist] synthesizer received text", flush=True)

    return {
        "raw_text": out,
        "summary": out,
        "suggested_experiments": [],
    }
