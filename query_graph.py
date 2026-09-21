"""
query_graph.py — THE QUERY LAYER. Talk to the knowledge graph.

Two search modes (GraphRAG's two fundamental question types):

  LOCAL  — "How is OpenAI connected to Microsoft?"
           entity linking -> k-hop traversal -> evidence subgraph
           -> source chunks -> LLM composes a CITED answer.

  GLOBAL — "What are the main themes of this dataset?"
           MAP: rate each community summary vs the question (parallel)
           -> keep top-rated -> REDUCE: synthesize a final answer.
           Requires data/communities.json (run communities.py first).

  AUTO   — one cheap LLM call reads the question and picks the mode.

Usage:
  python query_graph.py "How is OpenAI connected to Microsoft?"
  python query_graph.py "What are the main themes?" --mode global
  python query_graph.py "Who founded Tesla?" --mode local
  python query_graph.py "..." --mode auto            (default)
  python query_graph.py "..." --hops 1               (smaller evidence)
"""

import argparse
import json
import re
import sys
from pathlib import Path

from graph import Graph, normalize_entity

DATA_DIR = Path("data")
GRAPH_PATH = DATA_DIR / "graph.json"
COMMUNITIES_PATH = DATA_DIR / "communities.json"

from llm_client import chat_json, LLMError, backend_name  # noqa: E402


# ================================================================
# 1. QUESTION ANALYSIS
# ================================================================

def route_question(question: str) -> str:
    """AUTO mode: one cheap LLM call decides local vs global."""
    system = "You are a search router. Reply with ONLY JSON."
    user = (
        'A knowledge-graph search system has two modes:\n'
        '- "local": for questions about SPECIFIC named entities and their '
        'relations (people, companies, products, places, events)\n'
        '- "global": for broad questions about the WHOLE dataset (themes, '
        'topics, patterns, summaries, overviews)\n\n'
        f'Question: "{question}"\n\n'
        'Reply ONLY: {"mode": "local"} or {"mode": "global"}'
    )
    try:
        result = chat_json(system, user, retries=2)
        mode = result.get("mode", "local")
        return mode if mode in ("local", "global") else "local"
    except LLMError:
        return "local"   # safe default: local is cheaper


def extract_question_entities(question: str) -> list[str]:
    """LLM call: which entities from the question exist in a knowledge graph?"""
    system = ("You are a precise entity-linking engine. You reply with ONLY "
              "valid JSON.")
    user = (
        'Extract the named entities (people, organizations, products, '
        'places, events) mentioned in this question. Only proper names — '
        'not generic concepts.\n\n'
        f'Question: "{question}"\n\n'
        'Reply ONLY: {"entities": ["Name1", "Name2"]}'
    )
    try:
        result = chat_json(system, user, retries=2)
        return [e for e in result.get("entities", []) if isinstance(e, str)][:5]
    except LLMError:
        return []


# ================================================================
# 2. LOCAL SEARCH
# ================================================================

def link_entity(G: Graph, mention: str) -> str | None:
    """Question mention -> graph node key. Exact, then normalized, then
    substring (prefer highest-mention candidate)."""
    nk = normalize_entity(mention)
    if nk in G.nodes:
        return nk
    candidates = [k for k in G.nodes if nk and (nk in k or k in nk)]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return max(candidates, key=lambda k: G.nodes[k]["mentions"])
    return None


def local_search(G: Graph, seeds: list[str], hops: int = 2,
                 max_facts: int = 40) -> dict:
    """k-hop traversal from each seed -> evidence facts + source chunks.
    This is the graph's core value: the evidence CONTAINS the bridge if
    one exists within k hops of any seed."""
    facts: list[tuple[str, str, str, list]] = []   # (head, rel, tail, chunks)
    evidence_chunks: set[str] = set()

    frontier: set[str] = set()
    for mention in seeds:
        key = link_entity(G, mention)
        if key:
            frontier.add(key)
        else:
            print(f"  [link] '{mention}' -> not in graph")

    # hop 1: all edges touching each seed, both directions
    seen_edges: set[tuple] = set()
    for key in frontier:
        for (h, rel, t), e in G.edges.items():
            if h == key or t == key:
                if (h, rel, t) not in seen_edges:
                    seen_edges.add((h, rel, t))
                    facts.append((h, rel, t, e["source_chunks"]))
                    evidence_chunks.update(e["source_chunks"])

    # hop 2: edges of direct neighbors
    if hops >= 2:
        neighbors: set[str] = set()
        for h, rel, t in seen_edges:
            neighbors.update((h, t))
        neighbors -= frontier
        for nbr in neighbors:
            for (h, rel, t), e in G.edges.items():
                if (h == nbr or t == nbr) and (h, rel, t) not in seen_edges:
                    seen_edges.add((h, rel, t))
                    facts.append((h, rel, t, e["source_chunks"]))
                    evidence_chunks.update(e["source_chunks"])

    # rank: facts touching a seed first, then by source count
    seed_set = frontier

    def rank(f):
        h, rel, t, chunks = f
        touches = (h in seed_set) + (t in seed_set)
        return (-touches, -len(chunks))

    facts.sort(key=rank)
    facts = facts[:max_facts]
    return {"seeds": sorted(seed_set), "facts": facts,
            "chunks": sorted(evidence_chunks)}


def load_chunk_texts(chunk_ids: list[str]) -> dict[str, str]:
    """chunk_id ('Tesla_Inc::3') -> its text, from data/articles + index order.
    Rechunks deterministically exactly like build_graph.py did, so the ID
    maps back to the same text."""
    from data import chunk_text
    index = json.loads((DATA_DIR / "index.json").read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    wanted = {cid.split("::")[0] for cid in chunk_ids}
    for fname, meta in index.items():
        if fname == "_meta" or fname[:-4] not in wanted:
            continue
        text = (DATA_DIR / "articles" / fname).read_text(encoding="utf-8")
        for i, c in enumerate(chunk_text(text)):
            cid = f"{fname[:-4]}::{i}"
            if cid in set(chunk_ids):
                out[cid] = c
    return out


def local_answer(question: str, search: dict, chunk_texts: dict[str, str]) -> str:
    """Final LLM call: question + evidence triples + source chunks -> answer."""
    triple_lines = []
    for h, rel, t, _ in search["facts"]:
        triple_lines.append(f"  {h} --[{rel}]--> {t}")
    chunk_lines = []
    for cid in search["chunks"]:
        if cid in chunk_texts:
            chunk_lines.append(f"  [{cid}] {chunk_texts[cid][:500]}")

    system = ("You are a precise analyst answering questions from knowledge-graph "
              "evidence. You reply with ONLY valid JSON.")
    user = (
        "Answer the question using ONLY the evidence below.\n"
        "Rules:\n"
        "1. Cite chunk IDs (like [Tesla_Inc::3]) for every claim.\n"
        "2. If the evidence does not contain the answer, say exactly that.\n"
        "3. Be concise.\n\n"
        f"QUESTION: {question}\n\n"
        f"GRAPH FACTS:\n" + ("\n".join(triple_lines) or "  (none)") + "\n\n"
        f"SOURCE CHUNKS:\n" + ("\n".join(chunk_lines) or "  (none)") + "\n\n"
        'Reply ONLY: {"answer": "...", "chunks_used": ["chunk_id", ...]}'
    )
    result = chat_json(system, user)
    return result.get("answer", "(no answer returned)")


# ================================================================
# 3. GLOBAL SEARCH (map-reduce over community summaries)
# ================================================================

def global_answer(question: str) -> str:
    if not COMMUNITIES_PATH.exists():
        raise SystemExit("No data/communities.json — run communities.py first.")
    data = json.loads(COMMUNITIES_PATH.read_text(encoding="utf-8"))
    communities = data["communities"]
    with_summaries = [c for c in communities if c.get("summary")]
    print(f"  [global] rating {len(with_summaries)} community summaries...")

    # ---- MAP: one rating call per community (each sees ONLY its summary) ----
    scored = []
    for c in with_summaries:
        names = ", ".join(e["display"] for e in c["entities"][:10])
        system = "You are a relevance rater. Reply with ONLY valid JSON."
        user = (
            "Rate how relevant this community summary is to the question, "
            "0 (irrelevant) to 100 (highly relevant), and pull out the key "
            "points that bear on it.\n\n"
            f"QUESTION: {question}\n\n"
            f"COMMUNITY (entities: {names}):\n{c['summary']}\n\n"
            'Reply ONLY: {"score": 0-100, "points": ["...", "..."]}'
        )
        try:
            r = chat_json(system, user, retries=2)
            score = max(0, min(100, int(r.get("score", 0))))
            scored.append((score, c, r.get("points", [])))
            print(f"    C{c['id']}: score {score}")
        except LLMError as e:
            print(f"    C{c['id']}: rating failed ({str(e)[:60]})")

    if not scored:
        return "(no community summaries could be rated — is the LLM backend up?)"

    scored.sort(key=lambda s: -s[0])
    top = scored[:5]   # keep only the best communities for the reduce call

    # ---- REDUCE: one synthesis call over the top-rated points ----
    evidence = ""
    for score, c, points in top:
        if score < 20:
            continue
        names = ", ".join(e["display"] for e in c["entities"][:8])
        pts = "\n".join(f"  - {p}" for p in points[:5] if isinstance(p, str))
        evidence += (f"\n[Community {c['id']} | relevance {score}/100 | "
                     f"entities: {names}]\n{c['summary']}\nKey points:\n{pts}\n")

    system = ("You are a synthesizing analyst. You reply with ONLY valid JSON.")
    user = (
        "Answer the question by synthesizing the community summaries below "
        "(from a knowledge graph over business/economy articles).\n"
        "Cite communities as [C0], [C2] etc. If they don't answer the "
        "question, say so.\n\n"
        f"QUESTION: {question}\n\n"
        f"COMMUNITY EVIDENCE:\n{evidence}\n\n"
        'Reply ONLY: {"answer": "...", "communities_used": [0, 2]}'
    )
    result = chat_json(system, user)
    return result.get("answer", "(no answer returned)")


# ================================================================
# MAIN
# ================================================================

def main():
    ap = argparse.ArgumentParser(description="Ask the knowledge graph.")
    ap.add_argument("question", nargs="+", help="your question")
    ap.add_argument("--mode", choices=["auto", "local", "global"], default="auto")
    ap.add_argument("--hops", type=int, default=2, help="traversal depth (local)")
    args = ap.parse_args()
    question = " ".join(args.question)

    print(f"Q: {question}")
    mode = args.mode
    if mode == "auto":
        mode = route_question(question)
    print(f"mode: {mode}   backend: {backend_name()}\n")

    if mode == "global":
        answer = global_answer(question)
        print("\n=== ANSWER (global search) ===")
        print(answer)
        return

    # ---- local search ----
    if not GRAPH_PATH.exists():
        raise SystemExit("No data/graph.json — run build_graph.py first.")
    G = Graph.load(GRAPH_PATH)
    print(f"graph: {len(G.nodes)} nodes, {len(G.edges)} edges")

    entities = extract_question_entities(question)
    print(f"question entities: {entities or '(none found — falling back to full question words)'}")
    if not entities:
        entities = [question]   # let substring linking try its best

    search = local_search(G, entities, hops=args.hops)
    if not search["seeds"]:
        print("No question entity matched any graph node. Try --mode global, "
              "or check entity names in the graph (python visualize.py).")
        return
    print(f"seed nodes: {search['seeds']}")
    print(f"evidence: {len(search['facts'])} facts, {len(search['chunks'])} source chunks\n")

    chunk_texts = load_chunk_texts(search["chunks"])
    answer = local_answer(question, search, chunk_texts)

    print("=== EVIDENCE SUBGRAPH ===")
    for h, rel, t, _ in search["facts"][:15]:
        print(f"  {h} --[{rel}]--> {t}")
    print("\n=== ANSWER (local search) ===")
    print(answer)


if __name__ == "__main__":
    main()
