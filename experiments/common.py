"""Shared infra: .env loading, Groq client with disk cache + cost cap, and a
local-CPU generation fallback so a rate limit never blocks the sprint.

Never load raw generations into the working context; scripts aggregate to summaries.
"""

import hashlib
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = ROOT / "results" / "cache"
RAW = ROOT / "results" / "raw"
SUMMARIES = ROOT / "results" / "summaries"
for d in (DATA, CACHE, RAW, SUMMARIES):
    d.mkdir(parents=True, exist_ok=True)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
BUDGET_CAP = 4.0  # hard $ cap

# $ per 1M tokens (input, output), approximate, for budget tracking only
PRICES = {
    "llama-3.1-8b-instant": (0.05, 0.08),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "qwen/qwen3-32b": (0.29, 0.59),
    "openai/gpt-oss-20b": (0.10, 0.50),
    "openai/gpt-oss-120b": (0.15, 0.75),
}
SPEND_FILE = RAW / "spend.json"


def atomic_write(path, text):
    import os
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def load_env():
    env = ROOT / ".env"
    out = {}
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"')
    return out


def api_key():
    import os
    key = os.environ.get("GROQ_API_KEY") or load_env().get("GROQ_API_KEY")
    if not key:
        raise SystemExit("GROQ_API_KEY not found in environment or .env")
    return key


def track_spend(model, usage):
    spend = json.loads(SPEND_FILE.read_text()) if SPEND_FILE.exists() else {"total_usd": 0.0, "by_model": {}}
    pin, pout = PRICES.get(model, (1.0, 3.0))
    cost = usage.get("prompt_tokens", 0) / 1e6 * pin + usage.get("completion_tokens", 0) / 1e6 * pout
    spend["total_usd"] += cost
    spend["by_model"][model] = spend["by_model"].get(model, 0.0) + cost
    atomic_write(SPEND_FILE, json.dumps(spend, indent=1))
    return spend["total_usd"]


def cache_key(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class DailyCapError(Exception):
    """Groq daily quota hit, caller should switch to local generation (pivot rung A)."""


def chat(payload, budget_cap=BUDGET_CAP, max_retries=4):
    """Cached Groq call. payload = full request body. Returns response JSON.
    Raises DailyCapError on a long retry-after so the caller can fall back to local."""
    key = cache_key(payload)
    cf = CACHE / f"groq_{key}.json"
    if cf.exists():
        return json.loads(cf.read_text(encoding="utf-8"))

    spend = json.loads(SPEND_FILE.read_text())["total_usd"] if SPEND_FILE.exists() else 0.0
    if spend >= budget_cap:
        raise SystemExit(f"BUDGET CAP HIT: ${spend:.2f} >= ${budget_cap}")

    headers = {"Authorization": f"Bearer {api_key()}"}
    for attempt in range(max_retries):
        r = requests.post(GROQ_URL, json=payload, headers=headers, timeout=120)
        if r.status_code == 200:
            data = r.json()
            track_spend(payload["model"], data.get("usage", {}))
            atomic_write(cf, json.dumps(data))
            return data
        if r.status_code in (429, 500, 502, 503):
            retry_after = float(r.headers.get("retry-after") or 0)
            if retry_after > 300:
                raise DailyCapError(f"{payload['model']}: retry-after={retry_after:.0f}s")
            wait = max(min(2 ** attempt * 2, 45), retry_after)
            print(f"  [{payload['model']}] {r.status_code}, waiting {wait:.0f}s "
                  f"({attempt+1}/{max_retries})", flush=True)
            time.sleep(wait)
            continue
        raise RuntimeError(f"Groq error {r.status_code}: {r.text[:300]}")
    raise DailyCapError(f"{payload['model']}: exhausted retries")


# ---- Local CPU fallback (pivot rung A): unlimited, token-light, cached ----
_LOCAL = {}


def local_generate(prompt, model_id="Qwen/Qwen2.5-1.5B-Instruct", max_new_tokens=512, seed=0):
    """Generate on CPU with a small model. Cached by (model,prompt,seed).
    Lazy-imports transformers so importing common.py stays cheap."""
    key = cache_key({"m": model_id, "p": prompt, "s": seed, "n": max_new_tokens})
    cf = CACHE / f"local_{key}.json"
    if cf.exists():
        return json.loads(cf.read_text(encoding="utf-8"))["text"]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if model_id not in _LOCAL:
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        mdl.eval()
        _LOCAL[model_id] = (tok, mdl)
    tok, mdl = _LOCAL[model_id]
    msgs = [{"role": "user", "content": prompt}]
    text_in = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text_in, return_tensors="pt")
    torch.manual_seed(seed)
    with torch.no_grad():
        out = mdl.generate(**ids, max_new_tokens=max_new_tokens, do_sample=True,
                           temperature=0.8, top_p=0.95, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    atomic_write(cf, json.dumps({"text": text}))
    return text


def groq_generate(prompt, model, temperature=0.8, max_tokens=1024):
    """Convenience: one-shot Groq generation, returns assistant text. Cached via chat()."""
    payload = {"model": model, "temperature": temperature, "max_tokens": max_tokens,
               "messages": [{"role": "user", "content": prompt}]}
    data = chat(payload)
    return data["choices"][0]["message"]["content"]


# ---- OpenAI-compatible endpoint (OpenRouter / Gemini-OpenAI / etc.) ----
# Chosen for the capability axis: one key, many model families/sizes, quota
# fully separate from the stalled Groq job. Set OPENROUTER_API_KEY in .env.
ENDPOINTS = {
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY"),
    # Gemini exposes an OpenAI-compatible shim; swap in if using Gemini instead.
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "GEMINI_API_KEY"),
}


def _key_for(env_name):
    import os
    key = os.environ.get(env_name) or load_env().get(env_name)
    if not key:
        raise SystemExit(f"{env_name} not found in environment or .env, add it to Research 5/.env")
    return key


def oai_generate(prompt, model, provider="openrouter", temperature=0.8, max_tokens=1024,
                 max_retries=5):
    """Cached one-shot generation against an OpenAI-compatible endpoint.
    Returns assistant text. Non-Groq -> no cost tracking (free tier)."""
    url, env_name = ENDPOINTS[provider]
    payload = {"model": model, "temperature": temperature, "max_tokens": max_tokens,
               "messages": [{"role": "user", "content": prompt}]}
    ck = cache_key({"provider": provider, **payload})
    cf = CACHE / f"{provider}_{ck}.json"
    if cf.exists():
        cached = json.loads(cf.read_text(encoding="utf-8"))
        if "choices" in cached:                     # only trust well-formed cached bodies
            return cached["choices"][0]["message"]["content"]
        cf.unlink()                                 # drop a stale error envelope and refetch

    headers = {"Authorization": f"Bearer {_key_for(env_name)}"}
    for attempt in range(max_retries):
        r = requests.post(url, json=payload, headers=headers, timeout=120)
        if r.status_code == 200:
            data = r.json()
            if "choices" not in data:               # provider returned an error body with HTTP 200
                emsg = json.dumps(data.get("error", data))[:200]
                if attempt < max_retries - 1:
                    time.sleep(min(2 ** attempt * 2, 20)); continue
                raise RuntimeError(f"{provider}/{model} 200-but-no-choices: {emsg}")
            atomic_write(cf, json.dumps(data))       # cache only well-formed responses
            return data["choices"][0]["message"]["content"]
        if r.status_code == 429:
            # free-tier per-minute limit; short backoff (daily cap surfaces as persistent 429)
            wait = min(2 ** attempt * 3, 60)
            print(f"  [{provider}/{model}] 429, waiting {wait}s ({attempt+1}/{max_retries})", flush=True)
            time.sleep(wait)
            continue
        if r.status_code in (500, 502, 503):
            time.sleep(min(2 ** attempt * 2, 30))
            continue
        raise RuntimeError(f"{provider} error {r.status_code}: {r.text[:300]}")
    raise DailyCapError(f"{provider}/{model}: exhausted retries (likely daily free cap)")
