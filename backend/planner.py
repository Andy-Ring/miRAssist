#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict

from vllm import LLM, SamplingParams

from copilot.planner import (
    PLANNER_SYSTEM_PROMPT,
    make_planner_prompt,
    validate_queryspec,
)

# NOTE: If you want miRAssist branding inside the planner system prompt,
# update PLANNER_SYSTEM_PROMPT in copilot/planner.py. This file just uses it.


def _json_from_text(s: str):
    s = s.strip()

    # Handle ```json ... ``` or ``` ... ```
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 2:
            lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            s = "\n".join(lines).strip()

    # attempt to locate first/last braces
    i = s.find("{")
    j = s.rfind("}")
    if i >= 0 and j >= 0 and j > i:
        s = s[i : j + 1]

    return json.loads(s)


def plan_query(
    question: str,
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    max_model_len: int = 8192,
) -> Dict:
    """
    Core planner function: question -> validated QuerySpec dict.
    Suitable for FastAPI usage.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("Question is empty.")

    llm = LLM(
        model=model,
        dtype="bfloat16",
        max_model_len=max_model_len,
        disable_log_stats=True,
    )

    user_prompt = make_planner_prompt(question)

    # planner should be deterministic
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=512)

    # Flatten to llama3 chat template manually
    flat_prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{PLANNER_SYSTEM_PROMPT}\n"
        f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{user_prompt}\n"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )

    out = llm.generate([flat_prompt], sampling)[0].outputs[0].text

    try:
        qs = _json_from_text(out)
    except Exception:
        raise RuntimeError(f"Planner did not return valid JSON. Raw output:\n{out}")

    qs = validate_queryspec(qs)

    # Guarantee presence + correctness for downstream usage
    qs["original_question"] = question

    # Ensure nested keys exist (defensive)
    qs.setdefault("filters", {})
    qs.setdefault("phenotype_keywords", [])
    qs.setdefault("pathway_keywords", [])
    qs.setdefault("needs_clarification", [])
    qs.setdefault("k", 50)
    qs.setdefault("novel", False)

    return qs


def run_planner(question: str) -> Dict:
    """
    Thin wrapper for FastAPI.
    """
    return plan_query(question)


def _read_question(question_file: str) -> str:
    p = Path(question_file)
    if not p.exists():
        raise FileNotFoundError(f"Question file not found: {p}")
    q = p.read_text().strip()
    if not q:
        raise ValueError("Question file is empty.")
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--max_model_len", type=int, default=8192)
    args = ap.parse_args()

    question = _read_question(args.question_file)

    qs = plan_query(
        question=question,
        model=args.model,
        max_model_len=args.max_model_len,
    )

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(qs, indent=2))
    print(f"[OK] Wrote QuerySpec → {outp}")


if __name__ == "__main__":
    main()
