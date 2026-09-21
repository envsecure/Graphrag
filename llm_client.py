"""
llm_client.py — one LLM client for the whole pipeline, backend-agnostic.

Backends (choose with LLM_BACKEND env var, else auto-detect):
  "openai"  -> OpenAI-compatible endpoint (OPENAI_BASE_URL + OPENAI_MODEL)
               e.g. llama.cpp / vLLM / LM Studio servers
  "gemini"  -> Google Gemini REST API (GEMINI_API_KEY or .env GEMINI=)
  "ollama"  -> Ollama (OLLAMA_BASE_URL + OLLAMA_MODEL)

Auto-detect order when LLM_BACKEND is unset:
  gemini (if key present) -> openai (if OPENAI_BASE_URL set) -> ollama

The .env file is read at RUNTIME by this module. Keys are never printed.
Model overrides: GEMINI_MODEL (default gemini-3.6-flash), OPENAI_MODEL,
OLLAMA_MODEL.

Usage:
    from llm_client import chat_json, backend_name

    result = chat_json(system_prompt, user_prompt)   # -> parsed dict/list
    # raises LLMError on failure after retries
"""

import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path


class LLMError(Exception):
    pass


# ---------------------------------------------------------------
# .env loading (runtime, minimal, no dependency on python-dotenv)
# ---------------------------------------------------------------

def _load_dotenv(path: str = ".env") -> None:
    """Load KEY=VALUE lines into os.environ (without printing anything).
    Existing environment variables take precedence over .env values."""
    p = Path(path)
    if not p.exists():
        return
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        pass  # never crash the pipeline over .env formatting


_load_dotenv()

# ---------------------------------------------------------------
# Backend detection
# Accepts GEMINI_API_KEY (preferred) or GEMINI=<key> alias in .env
# ---------------------------------------------------------------

GEMINI_API_KEY = (os.environ.get("GEMINI_API_KEY")
                  or os.environ.get("GEMINI")
                  or "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "https://twitter-polygraph-vowed.ngrok-free.dev")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b-instruct")

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "")

_SELECTED = os.environ.get("LLM_BACKEND", "").strip().lower()
if _SELECTED == "openai":
    USE_OPENAI, USE_GEMINI = True, False
elif _SELECTED == "gemini":
    USE_OPENAI, USE_GEMINI = False, bool(GEMINI_API_KEY)
elif _SELECTED == "ollama":
    USE_OPENAI, USE_GEMINI = False, False
else:
    # auto-detect: gemini -> openai -> ollama
    if GEMINI_API_KEY:
        USE_OPENAI, USE_GEMINI = False, True
    elif OPENAI_BASE_URL:
        USE_OPENAI, USE_GEMINI = True, False
    else:
        USE_OPENAI, USE_GEMINI = False, False

_GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
) if USE_GEMINI else ""


def backend_name() -> str:
    if USE_OPENAI:
        return f"openai:{OPENAI_MODEL or '?'} @ {OPENAI_BASE_URL}"
    if USE_GEMINI:
        return f"gemini:{GEMINI_MODEL}"
    return f"ollama:{OLLAMA_MODEL}"


# ---------------------------------------------------------------
# Backend calls (each returns the raw text response)
# ---------------------------------------------------------------

def _gemini_call(system: str, user: str, timeout: int) -> str:
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",   # Gemini's JSON mode
            "maxOutputTokens": 2048,
        },
    }
    req = urllib.request.Request(
        _GEMINI_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read().decode())
    try:
        return resp["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        # safety-block or empty candidate — surface a readable error
        reason = resp.get("promptFeedback", {}).get("blockReason", "empty response")
        raise LLMError(f"gemini returned no content ({reason})") from e


def _openai_call(system: str, user: str, timeout: int) -> str:
    """OpenAI-compatible /v1/chat/completions (llama.cpp, vLLM, LM Studio...).
    max_tokens is generous because thinking models (Qwen3) spend tokens on
    reasoning_content before the actual answer — a tight cap returns EMPTY
    content, which we raise on so chat_json retries."""
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": 800,
    }
    req = urllib.request.Request(
        OPENAI_BASE_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read().decode())
    content = resp["choices"][0]["message"].get("content", "")
    if not content.strip():
        raise LLMError("openai backend returned empty content (thinking tokens ate the cap?)")
    return content


def _ollama_call(system: str, user: str, timeout: int) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 600},
    }
    req = urllib.request.Request(
        OLLAMA_BASE_URL.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "ngrok-skip-browser-warning": "true"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read().decode())
    return resp["message"]["content"]


# ---------------------------------------------------------------
# Public API: one JSON chat call, retries, backend-agnostic
# ---------------------------------------------------------------

def chat_json(system: str, user: str, retries: int = 3,
              timeout: int = 120) -> dict | list:
    """Send system+user prompt, get back a parsed JSON object.
    Retries on transient errors; JSON repair retry with lower output cap
    (mirrors the truncation handling from build_graph.py)."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            if USE_OPENAI:
                raw = _openai_call(system, user, timeout)
            elif USE_GEMINI:
                raw = _gemini_call(system, user, timeout)
            else:
                raw = _ollama_call(system, user, timeout)
            return json.loads(raw)
        except json.JSONDecodeError as e:
            last_err = f"bad JSON from {backend_name()}: {str(e)[:100]}"
            # deterministic-ish models repeat the same output; one repair
            # retry with a tighter cap, then give up
            if attempt < retries:
                time.sleep(1)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode(errors="replace")[:200]
            except Exception:
                pass
            last_err = f"HTTP {e.code}: {body}"
            # 429 = rate limit: back off longer; 4xx others: don't bother retrying
            if e.code == 429 and attempt < retries:
                time.sleep(5 * attempt)
            elif e.code < 500:
                break
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:120]}"
            if attempt < retries:
                time.sleep(2 * attempt)   # 2s, 4s backoff
    raise LLMError(f"LLM call failed after {retries} tries: {last_err}")


if __name__ == "__main__":
    # quick self-test:  python llm_client.py
    print(f"backend: {backend_name()}")
    out = chat_json(
        "You are a JSON-only API.",
        'Extract one triple from: "Satya Nadella is the CEO of Microsoft." '
        'Return {"head":..., "relation":..., "tail":...}')
    print("self-test result:", out)
