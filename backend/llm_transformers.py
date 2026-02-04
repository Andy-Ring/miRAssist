# backend/llm_transformers.py
from __future__ import annotations

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from backend.config import MODEL_NAME

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        _model.eval()
    return _tokenizer, _model


def chat(
    system: str,
    user: str,
    *,
    max_new_tokens: int = 600,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> str:
    tok, mdl = _load()

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(mdl.device) for k, v in inputs.items()}

    prompt_len = inputs["input_ids"].shape[1]
    do_sample = temperature > 0

    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=do_sample,
            temperature=float(temperature) if do_sample else None,
            top_p=float(top_p) if do_sample else None,
            eos_token_id=tok.eos_token_id,
        )

    # Slice off the prompt tokens so we return ONLY newly generated tokens
    gen_ids = out[0][prompt_len:]
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()
    return text
