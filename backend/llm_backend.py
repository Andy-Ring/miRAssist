import os
from typing import Optional

import requests


def _chat_transformers(system: str, user: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    # Lazy import so environments without transformers still work
    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
    import torch  # type: ignore

    model_name = os.environ.get("MIRASSIST_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system}\n"
        f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{user}\n"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )

    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=(temperature > 0),
            temperature=float(temperature),
            top_p=float(top_p),
        )
    text = tok.decode(out[0], skip_special_tokens=False)
    # naive strip: return after assistant header
    idx = text.rfind("<|start_header_id|>assistant<|end_header_id|>")
    if idx >= 0:
        text = text[idx:].split("\n", 1)[-1]
    return text.strip()


def _chat_vllm_http(system: str, user: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    url = os.environ.get("MIRASSIST_VLLM_HTTP_URL", "").rstrip("/")
    if not url:
        raise RuntimeError("MIRASSIST_VLLM_HTTP_URL is not set for vllm_http backend.")

    payload = {
        "system": system,
        "user": user,
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
    r = requests.post(f"{url}/chat", json=payload, timeout=600)
    r.raise_for_status()
    obj = r.json()
    return (obj.get("text") or "").strip()


def chat(
    *,
    system: str,
    user: str,
    max_new_tokens: int = 600,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> str:
    backend = (os.environ.get("MIRASSIST_LLM_BACKEND", "transformers") or "").strip().lower()
    if backend == "transformers":
        return _chat_transformers(system, user, max_new_tokens, temperature, top_p)
    if backend == "vllm_http":
        return _chat_vllm_http(system, user, max_new_tokens, temperature, top_p)

    raise ValueError(
        f"Unknown MIRASSIST_LLM_BACKEND='{backend}'. Expected 'transformers' or 'vllm_http'."
    )


def generate_answer(
    *,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    bundle: Optional[dict] = None,
    max_new_tokens: int = 600,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> str:
    """
    Backward-compatible entrypoint.

    Some older versions of the backend imported `generate_answer` from
    `backend.llm_backend`. Newer code calls the synthesizer directly, but
    keeping this helper prevents import errors.

    Provide either:
      - `bundle={"system_prompt":..., "user_prompt":...}`
      - or `system_prompt=...` and `user_prompt=...`
    """
    if bundle is not None:
        if system_prompt is None:
            system_prompt = bundle.get("system_prompt")
        if user_prompt is None:
            user_prompt = bundle.get("user_prompt")

    if not system_prompt or not user_prompt:
        raise ValueError("generate_answer requires system_prompt and user_prompt (or a bundle containing them).")

    return chat(
        system=system_prompt,
        user=user_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
