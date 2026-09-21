"""
build_graph.py — Extract triples from all article chunks via the Ollama LLM
and build the knowledge graph.

Pipeline (theory Stage 2-5):
  chunk each article -> LLM extraction call -> validate/normalize
  -> merge into Graph -> checkpoint to JSONL (crash-safe, resumable)
  -> save graph.json + print stats

Usage:
  python build_graph.py --limit 10     # smoke test: 10 chunks
  python build_graph.py                # full run (resumes automatically)
  python build_graph.py --articles 3   # only first 3 articles
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from graph import Graph

# --- Config --------------------------------------------------------------
# LLM backend comes from llm_client (Gemini if GEMINI_API_KEY/.env GEMINI
# is set, otherwise Ollama via OLLAMA_BASE_URL / OLLAMA_MODEL).
TIMEOUT = 300
HEADERS = {"Content-Type": "application/json", "ngrok-skip-browser-warning": "true"}

DATA_DIR = Path("data")
ARTICLES_DIR = DATA_DIR / "articles"
TRIPLES_PATH = DATA_DIR / "triples.jsonl"
FAILED_PATH = DATA_DIR / "failed_chunks.json"
GRAPH_PATH = DATA_DIR / "graph.json"

MAX_TRIPLES = 20

# --- Import chunker from data.py ------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from data import chunk_text  # noqa: E402

# --- Extraction prompt -----------------------------------------------------
SYSTEM_PROMPT = (
    "You are a precise entity and relationship extraction engine. "
    "You respond with ONLY valid JSON, no commentary."
)

USER_PROMPT = """Extract entities and relationships from the text below.

Rules:
1. Entity names: use the most complete form ("Tesla, Inc.", "Elon Musk", not "the company").
2. Relations: UPPER_SNAKE_CASE (CEO_OF, FOUNDED_BY, LOCATED_IN, PARTNER_WITH, ACQUIRED, SUBSIDIARY_OF, COMPETES_WITH, EMPLOYS, INVESTED_IN, MANUFACTURES, LOCATED_IN).
3. Extract ONLY facts explicitly stated in the text. Never invent.
4. Entity types: PERSON, ORGANIZATION, LOCATION, PRODUCT, EVENT, MISC.
5. Max {max_t} triples. Skip trivial facts.

Return exactly this JSON shape:
{{"entities": [{{"name": "...", "type": "PERSON"}}],
  "triples": [{{"head": "...", "relation": "...", "tail": "..."}}]}}

TEXT:
{chunk}"""


# ---------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------

def extract_chunk(text: str, retries: int = 3) -> dict | None:
    """Send one chunk to the LLM (Gemini or Ollama via llm_client).
    Returns parsed JSON dict or None after all retries fail."""
    from llm_client import chat_json, LLMError
    prompt = USER_PROMPT.format(max_t=MAX_TRIPLES, chunk=text)
    try:
        return chat_json(SYSTEM_PROMPT, prompt, retries=retries)
    except LLMError as e:
        print(f"      extraction failed: {str(e)[:140]}")
        return None


# ---------------------------------------------------------------
# Validation (LLM output is UNTRUSTED)
# ---------------------------------------------------------------

def valid_triple(t: dict) -> bool:
    return (
        isinstance(t, dict)
        and all(k in t for k in ("head", "relation", "tail"))
        and all(isinstance(t[k], str) and t[k].strip() for k in ("head", "relation", "tail"))
    )


def load_all_chunks(only_articles: int | None = None) -> list[dict]:
    """Read index.json -> chunk every article -> [{chunk_id, doc, text}]"""
    index = json.loads((DATA_DIR / "index.json").read_text(encoding="utf-8"))
    chunks = []
    for fname, meta in index.items():
        if fname == "_meta":
            continue
        path = ARTICLES_DIR / fname
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for i, c in enumerate(chunk_text(text)):
            chunks.append({"chunk_id": f"{fname[:-4]}::{i}", "doc": meta["title"], "text": c})
    if only_articles:
        chunks = chunks[: only_articles * 15]  # rough per-article chunk estimate
    return chunks


# ---------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="process only first N chunks (smoke test)")
    ap.add_argument("--articles", type=int, default=0, help="process only first N articles")
    args = ap.parse_args()

    chunks = load_all_chunks(only_articles=args.articles or None)
    if args.limit:
        chunks = chunks[: args.limit]

    # Resume support: skip chunks already in the checkpoint file
    done: set[str] = set()
    if TRIPLES_PATH.exists():
        with open(TRIPLES_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["chunk_id"])
                except Exception:
                    pass
    # Chunks that permanently failed before: skip them (delete the file to retry)
    failed_before: set[str] = set()
    if FAILED_PATH.exists():
        try:
            failed_before = set(json.loads(FAILED_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass

    todo = [c for c in chunks if c["chunk_id"] not in done and c["chunk_id"] not in failed_before]
    print(f"chunks: {len(chunks)} total, {len(done)} done, {len(failed_before)} permanently failed, {len(todo)} to do")

    # Build graph from checkpoint so far (or start fresh)
    G = Graph()
    if TRIPLES_PATH.exists():
        with open(TRIPLES_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    for t in rec["triples"]:
                        G.add_triple(t["head"], t["relation"], t["tail"],
                                     source_chunk=rec["chunk_id"])
                except Exception:
                    pass

    new_calls = 0
    failed = 0
    permanently_failed: set[str] = set()
    t0 = time.time()
    with open(TRIPLES_PATH, "a", encoding="utf-8") as out:
        for n, chunk in enumerate(todo, start=1):
            result = extract_chunk(chunk["text"])
            if result is None:
                failed += 1
                permanently_failed.add(chunk["chunk_id"])
                continue

            triples = [t for t in result.get("triples", []) if valid_triple(t)][:MAX_TRIPLES]
            entities = result.get("entities", [])
            etype = {}
            if isinstance(entities, list):
                etype = {e.get("name", ""): e.get("type", "MISC")
                         for e in entities if isinstance(e, dict) and e.get("name")}

            # append checkpoint FIRST (crash-safe), then update graph
            out.write(json.dumps({
                "chunk_id": chunk["chunk_id"],
                "doc": chunk["doc"],
                "triples": triples,
                "entities": entities,
            }) + "\n")
            out.flush()

            for t in triples:
                G.add_triple(t["head"], t["relation"], t["tail"],
                             head_type=etype.get(t["head"], "MISC"),
                             tail_type=etype.get(t["tail"], "MISC"),
                             source_chunk=chunk["chunk_id"])

            new_calls += 1
            if n % 5 == 0 or n == len(todo):
                rate = new_calls / max(time.time() - t0, 1)
                eta_min = (len(todo) - n) / max(rate, 0.01) / 60
                print(f"  [{n}/{len(todo)}] chunks done | "
                      f"{len(G.nodes)} nodes, {len(G.edges)} edges | "
                      f"{rate:.1f} chunks/s | ETA {eta_min:.0f} min")

    if permanently_failed:
        all_failed = failed_before | permanently_failed
        FAILED_PATH.write_text(json.dumps(sorted(all_failed), indent=1), encoding="utf-8")

    G.save(GRAPH_PATH)
    print(f"\n{'='*60}")
    print(f"Saved {GRAPH_PATH}   (checkpoint: {TRIPLES_PATH})")
    print(f"This run: {new_calls} new chunks, {failed} failed")
    print(f"\nGRAPH STATS:\n{G.stats()}")


if __name__ == "__main__":
    main()
