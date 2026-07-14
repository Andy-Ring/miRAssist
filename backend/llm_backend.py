import json
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import requests

from backend.config import (
    get_llm_backend,
    get_model_name,
    get_openai_api_key,
    get_openai_base_url,
    get_openai_timeout,
    get_vllm_http_url,
)


def _chat_transformers(
    system: str,
    user: str,
    model: Optional[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    # Lazy import so environments without transformers still work
    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
    import torch  # type: ignore

    model_name = model or get_model_name()
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


def _chat_vllm_http(
    system: str,
    user: str,
    model: Optional[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    url = (get_vllm_http_url() or "").rstrip("/")
    if not url:
        raise RuntimeError("MIRASSIST_VLLM_HTTP_URL is not set for vllm_http backend.")

    payload = {
        "system": system,
        "user": user,
        "model": model or get_model_name(),
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
    r = requests.post(f"{url}/chat", json=payload, timeout=600)
    r.raise_for_status()
    obj = r.json()
    return (obj.get("text") or "").strip()


def _chat_openai(
    system: str,
    user: str,
    model: Optional[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set for openai backend.")

    base_url = get_openai_base_url()
    if base_url:
        endpoint = base_url.rstrip("/") + "/chat/completions"
    else:
        endpoint = "https://api.openai.com/v1/chat/completions"

    payload = {
        "model": model or get_model_name(),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_completion_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        # Avoid hosted-runtime native gzip/decompression paths after HTTP 200.
        "Accept-Encoding": "identity",
    }

    print("[miRAssist] OpenAI request starting via urllib", flush=True)
    try:
        request = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=float(get_openai_timeout())) as response:
            status_code = int(response.status)
            print(f"[miRAssist] OpenAI request finished with status {status_code}", flush=True)
            raw_body = response.read()
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise RuntimeError(f"OpenAI chat completion failed: HTTP {exc.code}: {body[:1000]}") from exc
    except URLError as exc:
        raise RuntimeError(f"OpenAI chat completion failed: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"OpenAI chat completion failed: {exc}") from exc

    try:
        print(f"[miRAssist] OpenAI response body read: {len(raw_body)} bytes", flush=True)
        completion = json.loads(raw_body.decode("utf-8"))
        print("[miRAssist] OpenAI response JSON parsed", flush=True)
    except Exception as exc:
        raise RuntimeError(f"OpenAI chat completion returned invalid JSON: {exc}") from exc

    choices = completion.get("choices") or []
    choice = choices[0] if choices else None
    if not isinstance(choice, dict):
        raise RuntimeError("OpenAI chat completion returned no choices.")

    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        text = content.strip()
        print(f"[miRAssist] OpenAI message content extracted: {len(text)} chars", flush=True)
        return text
    if isinstance(content, list):
        parts = []
        for item in content:
            text = item.get("text") if isinstance(item, dict) else None
            if text:
                parts.append(text)
        joined = "".join(parts).strip()
        if joined:
            print(f"[miRAssist] OpenAI message content extracted: {len(joined)} chars", flush=True)
            return joined

    raise RuntimeError("OpenAI chat completion returned an empty message content.")


def chat(
    *,
    system: str,
    user: str,
    model: Optional[str] = None,
    max_new_tokens: int = 600,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> str:
    backend = get_llm_backend()
    if backend == "transformers":
        return _chat_transformers(system, user, model, max_new_tokens, temperature, top_p)
    if backend == "vllm_http":
        return _chat_vllm_http(system, user, model, max_new_tokens, temperature, top_p)
    if backend == "openai":
        return _chat_openai(system, user, model, max_new_tokens, temperature, top_p)

    raise ValueError(
        f"Unknown MIRASSIST_LLM_BACKEND='{backend}'. Expected 'transformers', 'vllm_http', or 'openai'."
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
        model=None,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
