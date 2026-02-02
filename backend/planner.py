# backend/planner.py
from __future__ import annotations

import json
from typing import Dict, Any

from backend.llm_backend import chat
from copilot.planner import (
    PLANNER_SYSTEM_PROMPT,
    make_planner_prompt,
    validate_queryspec,
)


def _json_from_text(s: str) -> Dict[str, Any]:
    s = s.strip()

    # Handle fenced blocks
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 2:
            lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            s = "\n".join(lines).strip()

    # Extract first JSON object if there is extra chatter
    i = s.find("{")
    j = s.rfind("}")
    if i >= 0 and j >= 0 and j > i:
        s = s[i : j + 1]

    return json.loads(s)


def plan_query(question: str) -> Dict[str, Any]:
    question = (question or "").strip()
    if not question:
        raise ValueError("Question is empty.")

    user_prompt = make_planner_prompt(question)

    out = chat(
        system=PLANNER_SYSTEM_PROMPT,
        user=user_prompt,
        max_new_tokens=512,
        temperature=0.0,
        top_p=1.0,
    )

    try:
        qs = _json_from_text(out)
    except Exception:
        raise RuntimeError(f"Planner did not return valid JSON. Raw output:\n{out}")

    qs = validate_queryspec(qs)

    # Defensive defaults for downstream
    qs["original_question"] = question
    qs.setdefault("filters", {})
    qs.setdefault("phenotype_keywords", [])
    qs.setdefault("pathway_keywords", [])
    qs.setdefault("needs_clarification", [])
    qs.setdefault("k", 50)
    qs.setdefault("novel", False)

    return qs


def run_planner(question: str) -> Dict[str, Any]:
    return plan_query(question)
