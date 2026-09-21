<<<<<<< HEAD
# GraphRAG — A Learning Project, Built From Scratch

A complete **GraphRAG pipeline** over real business/economy Wikipedia data:
raw text → LLM-extracted knowledge graph → LLM-cleaned entities →
communities → interactive visualization → **question answering with
citations** (local search + global search).

Built as a learning project — every layer is small, readable, and
understandable. No vector database, no heavy frameworks; the point is to
see every moving part.

---

## What this does (30-second version)

```
49 Wikipedia articles (business + global economy)
        │  data.py            fetch, clean, chunk (~1,200 chunks, provenance IDs)
        ▼
build_graph.py           LLM extracts (head, relation, tail) triples per chunk
        │                 resumable checkpoint: data/triples.jsonl
        ▼
graph.py                 validation gate + normalization + edge aggregation
        │                 graph.json: nodes/edges with mention counts + source chunks
        ▼
merge_entities.py        entity resolution: blocking → fuzzy nominates → LLM decides
        │                 ("Tesla, Inc." + "Tesla" + "Tesla Motors, Inc." → one node)
        ▼
communities.py           label-propagation clustering + LLM summary per community
        │                 communities.json (enables GLOBAL search)
        ▼
query_graph.py           ASK QUESTIONS:
        │                 local  = link entities → k-hop walk → cited answer
        │                 global = map-reduce over community summaries
        ▼
visualize.py             interactive graph.html (zoom/pan/drag, communities, tooltips)
```

Example of what it answers:

```
Q: How is Elon Musk connected to SolarCity?
A: Elon Musk is connected to SolarCity through the acquisition of SolarCity
   by Tesla. Tesla acquired SolarCity in 2016, and Musk was involved in the
   acquisition. [Tesla_Inc::9] [Tesla_Inc::15]
   ↑ real chunk citations back to source text
```

No single chunk contains that answer — the graph found the connection.

---

## Quick start

```bash
# 0. configure an LLM backend in .env (see below)

# 1. fetch the corpus (free, no API key)
python data.py

# 2. sanity-check the LLM backend
python llm_client.py

# 3. extract triples + build the graph (long; resumable — rerun anytime)
python build_graph.py

# 4. entity dedup (LLM-decided; seconds)
python merge_entities.py

# 5. communities + summaries (one LLM call per community)
python communities.py

# 6. visualize
python visualize.py && start graph.html     # Windows
# python visualize.py && open graph.html    # macOS

# 7. ASK QUESTIONS
python query_graph.py "Who is the CEO of Tesla?"
python query_graph.py "How is Elon Musk connected to SolarCity?"
python query_graph.py "What are the main themes of this dataset?"
```

## LLM backend configuration (`.env`)

One client (`llm_client.py`) serves the whole pipeline. Pick a backend by
setting `LLM_BACKEND` in `.env` (keys are never printed or logged):

```ini
# Option A: OpenAI-compatible server (llama.cpp / vLLM / LM Studio / Ollama-openai)
LLM_BACKEND=openai
OPENAI_BASE_URL=https://your-server/v1
OPENAI_MODEL=unsloth/Qwen3-1.7B-GGUF:Q4_K_M

# Option B: Google Gemini
LLM_BACKEND=gemini
GEMINI=your_api_key_here
GEMINI_MODEL=gemini-3.6-flash        # optional override

# Option C: native Ollama endpoint
LLM_BACKEND=ollama
OLLAMA_BASE_URL=https://your-tunnel.ngrok-free.dev
OLLAMA_MODEL=qwen2.5:3b-instruct
```

If `LLM_BACKEND` is unset: auto-detect → Gemini (if key present) →
OpenAI-compatible (if `OPENAI_BASE_URL` set) → Ollama.

---

## The files

| File | Role |
|---|---|
| `data.py` | fetch 49 Wikipedia articles, clean, chunk (~1,200 chunks, provenance IDs) |
| `llm_client.py` | backend-agnostic LLM client: Gemini / OpenAI-compatible / Ollama, retries, JSON repair |
| `test_llm.py` | 4-point backend health check (server, generation, JSON mode, chat) |
| `graph.py` | `Graph` class: validation gate, normalization, edge aggregation, type backfill, save/load |
| `build_graph.py` | extraction pipeline: chunk → LLM → validate → merge → **resumable checkpoint** |
| `merge_entities.py` | entity resolution: blocking → fuzzy → **LLM is the sole merge authority** → audit log |
| `communities.py` | weighted label-propagation clustering + LLM community summaries |
| `query_graph.py` | the query layer: router + local search (cited) + global search (map-reduce) |
| `visualize.py` | interactive HTML visualization (community colors, bridges, tooltips) |
| `graphrag_simple.py` | the original teaching skeleton (kept for reference) |

### Deeper documentation

- **[docs/architecture.md](docs/architecture.md)** — the whole architecture as
  theory: diagrams of every stage, the ideas behind them, and why each design
  decision exists. Readable without reading the code.
- **[docs/graph_making.md](docs/graph_making.md)** — how the graph is built:
  every pipeline stage, design decisions, real examples, mermaid diagrams.
- **[docs/solution.md](docs/solution.md)** — the entity-merging problem
  ("Tesla, Inc. / Tesla / Tesla Motors, Inc.") and its full solution story,
  including the three live failures that shaped the final design.

---

## Design principles (learned the hard way)

1. **LLM output is a suggestion, not data.** Every stage validates,
   normalizes, caps, and logs before trusting anything.
2. **Cheap layers nominate, expensive layers decide.** Blocking + fuzzy
   matching find candidate duplicates for free; the LLM only judges those.
3. **Give the judge evidence, not just names.** Types, mention counts and
   graph edges turned a sycophantic string-matcher into a reliable
   adjudicator.
4. **The graph stores structure; text stays retrievable.** Every edge
   carries its `source_chunks` — answers are citable, never black-box.
5. **Mention counts are free confidence scores.** Real relationships get
   re-extracted across articles; junk appears once.
6. **Checkpoint everything expensive.** Extraction is paid once
   (`triples.jsonl`); every downstream stage rebuilds in seconds for free.
7. **Over-merging is worse than under-merging.** Two fragments of Tesla
   cost edge weight; one false merge poisons traversals.

---

## Known limitations (honest list)

- **No embeddings.** Pure-structural v1: linking is name-based (exact →
  normalized → substring). Semantic linking + vector fallback over chunks
  is the natural v2 (see the "embeddings" discussion in the project history).
- **Extraction noise.** Small models emit inverted triples
  (`SUBSIDIARY_OF → Elon Musk`) and vague entities; the validation gate and
  mention-count filter remove most, not all.
- **Extraction cost.** ~1,200 chunks × 1 LLM call at ~50 s/call on a small
  local model ≈ 16 h (resumable). Flip `.env` to Gemini for ~10× faster.
- **Communities need scale.** Modularity was Q ≈ 0.04 on the one-article
  sample — one topic, no communities. They emerge with the full corpus.
- **Single-file storage.** JSON, not a graph database (Neo4j/Kuzu) — fine
  at this scale, the swap is mechanical.

---

## Requirements

- Python 3.10+ (uses `X | Y` type hints)
- **Zero pip dependencies** — stdlib only (`urllib`, `json`, `pathlib`...)
- An LLM reachable via any of: OpenAI-compatible endpoint, Gemini API key,
  or Ollama
=======
# Graphrag
>>>>>>> f2fd66f7ee5d335df62d52669ca41aa68e64c657
