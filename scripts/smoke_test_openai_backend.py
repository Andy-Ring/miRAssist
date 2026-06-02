from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.config import get_planner_model, get_synth_model
from backend.llm_backend import chat


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.")
        return 1

    planner_model = get_planner_model()
    synth_model = get_synth_model()

    planner_output = chat(
        system="Return only compact JSON.",
        user='Return {"ok": true, "tool": "planner"} and nothing else.',
        model=planner_model,
        max_new_tokens=80,
        temperature=0.0,
        top_p=1.0,
    )

    try:
        parsed = json.loads(planner_output)
    except Exception as exc:
        print(f"Planner smoke test failed to parse JSON: {exc}")
        return 1

    if parsed.get("ok") is not True:
        print("Planner smoke test returned unexpected JSON.")
        return 1

    synth_output = chat(
        system="You are a concise scientific assistant.",
        user="Summarize this in one sentence: miR-21 may regulate PTEN in colon cancer.",
        model=synth_model,
        max_new_tokens=80,
        temperature=0.2,
        top_p=0.95,
    )

    if not synth_output.strip():
        print("Synthesizer smoke test returned empty output.")
        return 1

    print(f"Planner model OK: {planner_model}")
    print(f"Synth model OK: {synth_model}")
    print("OpenAI backend smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
