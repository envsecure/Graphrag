"""
test_llm.py — Verify the remote Ollama endpoint works before using it in the pipeline.

Tests, in order:
  1. SERVER UP      : GET /api/tags (list models)
  2. BASIC GEN      : POST /api/generate  ("Say OK")
  3. JSON OUTPUT    : POST /api/generate  (forced JSON — needed for extraction)
  4. CHAT FORMAT    : POST /api/chat      (what we'll use for extraction prompts)

Run:  python test_llm.py
Exit code 0 = all good, 1 = something failed.
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error

# --- Config -------------------------------------------------------------
# Override with env var:  set OLLAMA_BASE_URL=https://your-new-tunnel.ngrok-free.dev
BASE_URL = os.environ.get("OLLAMA_BASE_URL", "https://twitter-polygraph-vowed.ngrok-free.dev")
TIMEOUT = 180          # seconds — local LLMs can be slow on first load
EXPECTED_JSON = False  # set True later if you want to REQUIRE JSON mode

# ngrok's free tier shows a browser warning page on the first request;
# this header bypasses it for API clients.
HEADERS = {
    "Content-Type": "application/json",
    "ngrok-skip-browser-warning": "true",
}


def request(path: str, payload: dict | None = None, timeout: int = TIMEOUT):
    """POST (or GET if payload is None) to the Ollama endpoint. Returns (ok, data)."""
    url = BASE_URL.rstrip("/") + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return True, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        return False, {"error": f"HTTP {e.code}: {body}"}
    except Exception as e:
        return False, {"error": f"{type(e).__name__}: {e}"}


results = []


def check(name: str, ok: bool, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    results.append(ok)
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))


# --- Test 1: server up + list models ------------------------------------
print(f"Testing Ollama endpoint: {BASE_URL}\n")

ok, tags = request("/api/tags", timeout=30)
models = []
if ok:
    models = [m["name"] for m in tags.get("models", [])]
    check("1. server up, /api/tags reachable", True, f"{len(models)} model(s) found")
    for m in models:
        print(f"        - {m}")
else:
    check("1. server up, /api/tags reachable", False, tags.get("error", ""))

if not models:
    print("\nNo models available — cannot continue tests.")
    sys.exit(1)

# Prefer explicitly chosen model (OLLAMA_MODEL), else first available
model = os.environ.get("OLLAMA_MODEL") or models[0]
if model not in models:
    print(f"\nModel '{model}' not found on server. Available: {models}")
    sys.exit(1)
print(f"\nUsing model: {model}\n")

# --- Test 2: basic generation -------------------------------------------
t0 = time.time()
ok, gen = request("/api/generate", {"model": model, "prompt": "Reply with exactly: OK", "stream": False})
elapsed = time.time() - t0
if ok:
    text = gen.get("response", "").strip()
    check("2. basic generation", len(text) > 0, f'"{text[:60]}" ({elapsed:.1f}s, {gen.get("eval_count", "?")} tokens)')
else:
    check("2. basic generation", False, gen.get("error", ""))

# --- Test 3: JSON structured output (needed for extraction) --------------
json_prompt = (
    "Extract the relationship in this sentence as JSON with keys "
    '"head", "relation", "tail". Sentence: "Satya Nadella is the CEO of Microsoft." '
    'Reply with ONLY the JSON object, no other text.'
)
t0 = time.time()
ok, gen = request("/api/generate", {
    "model": model,
    "prompt": json_prompt,
    "stream": False,
    "format": "json",   # Ollama's JSON mode — guarantees valid JSON
})
elapsed = time.time() - t0
json_ok = False
if ok:
    try:
        parsed = json.loads(gen.get("response", "{}"))
        json_ok = all(k in parsed for k in ("head", "relation", "tail"))
        detail = json.dumps(parsed)
    except Exception as e:
        detail = f"invalid JSON: {e}"
    check("3. JSON structured output", json_ok, f"{detail[:100]} ({elapsed:.1f}s)")
else:
    check("3. JSON structured output", False, gen.get("error", ""))

# --- Test 4: chat format (what extraction will use) ----------------------
t0 = time.time()
ok, chat = request("/api/chat", {
    "model": model,
    "messages": [
        {"role": "system", "content": "You are a precise extraction engine. Output only JSON."},
        {"role": "user", "content": json_prompt},
    ],
    "stream": False,
    "format": "json",
})
elapsed = time.time() - t0
if ok:
    text = chat.get("message", {}).get("content", "").strip()
    try:
        parsed = json.loads(text)
        chat_ok = isinstance(parsed, dict) and len(parsed) > 0
    except Exception:
        chat_ok, parsed = False, text
    check("4. chat endpoint + JSON", chat_ok, f"{str(parsed)[:100]} ({elapsed:.1f}s)")
else:
    check("4. chat endpoint + JSON", False, chat.get("error", ""))

# --- Summary --------------------------------------------------------------
passed = sum(results)
print(f"\n{'='*50}")
print(f"{passed}/{len(results)} tests passed")
if passed == len(results):
    print("Endpoint is ready. Next step: wire it into the extraction pipeline.")
    sys.exit(0)
print("Endpoint has problems — fix the FAIL lines above before continuing.")
sys.exit(1)
