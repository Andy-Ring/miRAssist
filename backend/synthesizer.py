#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def answer_from_bundle(
    bundle: Dict[str, Any],
    model: str,
    max_tokens: int = 1200,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_model_len: int = 8192,
) -> Dict[str, Any]:
    system_prompt = bundle["system_prompt"]
    user_prompt = bundle["user_prompt"]

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    llm = LLM(
        model=model,
        dtype="bfloat16",
        max_model_len=max_model_len,
    )

    sampling = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=["<|eot_id|>", "<|endoftext|>", "</s>"],
    )

    outputs = llm.generate([prompt], sampling)
    text = outputs[0].outputs[0].text
    text = text.replace("Read more Read less", "").strip()

    # v0.1: return minimally-structured answer
    # (Later we can parse out ranked items / experiments deterministically.)
    return {
        "raw_text": text,
        "summary": text,
        "suggested_experiments": [],
    }


def run_synthesizer(bundle: dict, model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct") -> Dict[str, Any]:
    return answer_from_bundle(bundle=bundle, model=model)


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max_tokens", type=int, default=1200)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bundle = json.loads(Path(args.bundle).read_text())
    ans = answer_from_bundle(
        bundle=bundle,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(ans["raw_text"])
        print(f"[OK] Wrote answer → {outp}")
    else:
        print(ans["raw_text"])


if __name__ == "__main__":
    run()

