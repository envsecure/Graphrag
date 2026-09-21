"""
communities.py — Community detection + LLM community summaries.

This is the last missing INDEXING step: it enables GLOBAL search
("What are the main themes of this dataset?") by compressing the graph
into a handful of pre-digested summaries.

Pipeline:
  1. LOAD      graph.json (run merge_entities.py FIRST — merging before
               clustering, otherwise fragmented entities split clusters)
  2. CLUSTER   weighted label propagation (~Louvain-lite, zero deps):
               every node gets its own label; each node repeatedly adopts
               the most common/heaviest label among its neighbors;
               labels stabilize -> communities
  3. SUMMARIZE one LLM call per community: entities + triples -> summary
  4. SAVE      communities.json (clusters, summaries, stats)

Usage:
  python communities.py --no-llm        # clustering only, no LLM calls
  python communities.py                 # cluster + LLM summaries
  python communities.py --min-size 4    # ignore tiny communities
"""

import argparse
import json
import os
import random
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from graph import Graph

DATA_DIR = Path("data")
GRAPH_PATH = DATA_DIR / "graph.json"
COMMUNITIES_PATH = DATA_DIR / "communities.json"

# LLM endpoint (same env vars as build_graph.py / merge_entities.py)
BASE_URL = os.environ.get("OLLAMA_BASE_URL", "https://twitter-polygraph-vowed.ngrok-free.dev")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b-instruct")
HEADERS = {"Content-Type": "application/json", "ngrok-skip-browser-warning": "true"}


# ---------------------------------------------------------------
# 1. Graph -> weighted undirected view (mention counts = weights)
# ---------------------------------------------------------------

def build_adjacency(G: Graph) -> dict[str, list[tuple[str, float]]]:
    """node -> [(neighbor, weight), ...] treating the graph as undirected.
    Parallel edges in opposite directions aggregate into one weight.
    Weights are LOG-SCALED (1 + ln(mentions)): otherwise a 62-mention hub
    like Tesla floods every neighbor vote and swallows the whole graph
    into one community (the classic hub-swallowing failure mode)."""
    import math
    adj: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for (h, _rel, t), e in G.edges.items():
        w = 1.0 + math.log(e["mentions"])
        adj[h][t] += w
        adj[t][h] += w
    return {n: list(nbrs.items()) for n, nbrs in adj.items()}


# ---------------------------------------------------------------
# 2. Weighted label propagation
# ---------------------------------------------------------------

def label_propagation(adj: dict[str, list[tuple[str, float]]],
                      seed: int = 42, max_iter: int = 20) -> dict[str, int]:
    """Every node starts with its own label; each node repeatedly adopts
    the heaviest label among its neighbors. Returns node -> community id."""
    rng = random.Random(seed)
    labels: dict[str, int] = {n: i for i, n in enumerate(adj)}

    for iteration in range(max_iter):
        changed = 0
        # random order each round (fixed seed = reproducible)
        order = list(adj)
        rng.shuffle(order)
        for node in order:
            nbrs = adj[node]
            if not nbrs:
                continue
            votes: Counter = Counter()
            for nbr, w in nbrs:
                votes[labels[nbr]] += w
            best_label, best_weight = max(votes.items(), key=lambda kv: (kv[1], -kv[0]))
            current_weight = votes.get(labels[node], 0)
            # move only if a NEIGHBOR label strictly beats your own
            if best_label != labels[node] and best_weight > current_weight:
                labels[node] = best_label
                changed += 1
        if changed == 0:
            break
    print(f"label propagation converged after {iteration + 1} iterations ({changed} changes in last round)")
    return labels


# ---------------------------------------------------------------
# 3. LLM community summary
# ---------------------------------------------------------------

SUMMARY_SYSTEM = ("You are a precise analyst who summarizes knowledge-graph "
                  "communities. You reply with ONLY valid JSON.")

SUMMARY_TEMPLATE = """Below are the entities and relationships of one community in a knowledge graph.

TASK: Write a concise summary of this community (3-5 sentences): what it is
about, who the key entities are, and how they relate. State only facts
supported by the listed relationships.

ENTITIES:
{entities}

RELATIONSHIPS:
{relations}

Return ONLY: {{"summary": "..."}}"""


def llm_summarize_community(entities: list[str], relations: list[str]) -> str | None:
    """One LLM call per community via the shared client (Gemini or Ollama)."""
    from llm_client import chat_json, LLMError
    entity_lines = "\n".join(f"  - {e}" for e in entities[:40])
    relation_lines = "\n".join(f"  - {r}" for r in relations[:40])
    try:
        result = chat_json(
            SUMMARY_SYSTEM,
            SUMMARY_TEMPLATE.format(entities=entity_lines, relations=relation_lines))
        return (result.get("summary", "") or "").strip() if isinstance(result, dict) else None
    except LLMError as e:
        print(f"      summarize failed: {str(e)[:100]}")
        return None


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="cluster only, skip summaries")
    ap.add_argument("--min-size", type=int, default=3,
                    help="communities smaller than this get no summary (default 3)")
    args = ap.parse_args()

    if not GRAPH_PATH.exists():
        print("No data/graph.json — run build_graph.py (and merge_entities.py) first.")
        raise SystemExit(1)

    G = Graph.load(GRAPH_PATH)
    print(f"loaded graph: {len(G.nodes)} nodes, {len(G.edges)} edges")

    # ---- cluster ----
    adj = build_adjacency(G)
    labels = label_propagation(adj)

    communities: dict[int, list[str]] = defaultdict(list)
    for node, label in labels.items():
        communities[label].append(node)

    # sort members by mentions (most important first); relabel by size order
    ordered = sorted(communities.values(),
                     key=lambda members: -sum(G.nodes[m]["mentions"] for m in members))
    print(f"\nfound {len(ordered)} communities:")
    for cid, members in enumerate(ordered):
        names = ", ".join(G.nodes[m]["display"] for m in members[:8])
        more = f" (+{len(members)-8} more)" if len(members) > 8 else ""
        print(f"  C{cid} [{len(members)} nodes] {names}{more}")

    # ---- summarize (LLM) ----
    out = {"model": MODEL, "generated_with_llm": not args.no_llm, "communities": []}
    for cid, members in enumerate(ordered):
        community = {"id": cid, "size": len(members),
                     "entities": [{"key": m, "display": G.nodes[m]["display"],
                                   "type": G.nodes[m]["type"],
                                   "mentions": G.nodes[m]["mentions"]}
                                  for m in members]}

        if len(members) >= args.min_size and not args.no_llm:
            member_set = set(members)
            relations = []
            for (h, rel, t), e in G.edges.items():
                if h in member_set or t in member_set:
                    hd, td = G.nodes[h]["display"], G.nodes[t]["display"]
                    cross = "" if (h in member_set and t in member_set) else "  (border)"
                    relations.append(f"{hd} --[{rel} x{e['mentions']}]--> {td}{cross}")
            print(f"\nsummarizing C{cid} ({len(members)} nodes, {len(relations)} relations)...")
            summary = llm_summarize_community(
                [G.nodes[m]["display"] for m in members], relations)
            if summary:
                print(f"  -> {summary[:100]}...")
            community["summary"] = summary or "(summary failed)"
        elif len(members) < args.min_size:
            community["summary"] = None   # too small to be worth a call
        else:
            community["summary"] = None   # --no-llm mode

        out["communities"].append(community)

    out["_meta"] = {
        "nodes": len(G.nodes), "edges": len(G.edges),
        "num_communities": len(ordered),
        "largest": max(len(c) for c in ordered),
        "algorithm": "weighted label propagation (seed=42)",
    }
    COMMUNITIES_PATH.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nsaved {COMMUNITIES_PATH}")
    print("Global search can now answer: 'What are the main themes?'")


if __name__ == "__main__":
    main()
