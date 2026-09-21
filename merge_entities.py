"""
merge_entities.py — LLM-decided entity resolution.

Flow (the funnel):
  1. BLOCKING    bucket entity keys by shared token      -> candidate pairs (free)
  2. FUZZY       Jaro-Winkler + overlap >= --threshold   -> PROBABLE pairs (free)
                 (fuzzy only NOMINATES — it never merges)
  3. CLUSTER     connected components of nominations     -> one batch per cluster
  4. LLM         one batch-grouping call per cluster     -> THE ONLY merge authority
  5. REBUILD     union-find style rewrite via key_map    -> cleaned graph.json
  6. AUDIT       data/merge_log.json — every nomination + LLM verdict

Usage:
  python merge_entities.py --dry-run     # show nominations, no LLM calls
  python merge_entities.py               # nominate + LLM adjudicate + rebuild
  python merge_entities.py --threshold 0.4   # wider nomination net
"""

import json
import os
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

from graph import Graph, normalize_entity, normalize_relation

DATA_DIR = Path("data")
TRIPLES_PATH = DATA_DIR / "triples.jsonl"
GRAPH_PATH = DATA_DIR / "graph.json"
LOG_PATH = DATA_DIR / "merge_log.json"

NOMINATE_THRESHOLD = 0.5    # fuzzy only nominates; LLM decides
MAX_BATCH = 15              # LLM attention degrades on long lists

# LLM endpoint (same env vars as build_graph.py)
BASE_URL = os.environ.get("OLLAMA_BASE_URL", "https://twitter-polygraph-vowed.ngrok-free.dev")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b-instruct")
HEADERS = {"Content-Type": "application/json", "ngrok-skip-browser-warning": "true"}


# ---------------------------------------------------------------
# Fuzzy similarity (nominator, not decider)
# ---------------------------------------------------------------

def jaro_winkler(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if not len1 or not len2:
        return 0.0
    window = max(max(len1, len2) // 2 - 1, 0)
    m1, m2 = [False] * len1, [False] * len2
    matches = 0
    for i in range(len1):
        for j in range(max(0, i - window), min(i + window + 1, len2)):
            if not m2[j] and s1[i] == s2[j]:
                m1[i] = m2[j] = True
                matches += 1
                break
    if not matches:
        return 0.0
    t = k = 0
    for i in range(len1):
        if m1[i]:
            while not m2[k]:
                k += 1
            if s1[i] != s2[k]:
                t += 1
            k += 1
    t //= 2
    jaro = (matches / len1 + matches / len2 + (matches - t) / matches) / 3
    prefix = 0
    for a, b in zip(s1, s2):
        if a != b or prefix == 4:
            break
        prefix += 1
    return jaro + prefix * 0.1 * (1 - jaro)


def token_overlap(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def similarity(k1: str, k2: str) -> float:
    return 0.6 * jaro_winkler(k1, k2) + 0.4 * token_overlap(set(k1.split()), set(k2.split()))


# ---------------------------------------------------------------
# Blocking: bucket by shared token, compare within buckets only
# ---------------------------------------------------------------

def candidate_pairs(keys: list[str]) -> list[tuple[str, str]]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        for tok in set(k.split()):
            buckets[tok].append(k)
    seen, pairs = set(), []
    for bucket in buckets.values():
        if len(bucket) < 2 or len(bucket) > 60:   # tiny=useless, huge=explosion
            continue
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                a, b = bucket[i], bucket[j]
                if a != b and (a, b) not in seen:
                    seen.add((a, b))
                    pairs.append((a, b))
    return pairs


# ---------------------------------------------------------------
# Nomination guards (cheap filters — they only block NOMINATIONS,
# the LLM still makes every actual merge decision)
# ---------------------------------------------------------------

_PRODUCT_EDGES = {"MANUFACTURES", "PRODUCES", "MAKES", "SHIPS", "SELLS",
                  "OFFERS", "LAUNCHED", "DEVELOPS", "DESIGNS", "CREATED"}


def product_edge_between(G: Graph, a: str, b: str) -> bool:
    """An explicit maker/made edge between the pair = provably different things."""
    return any({h, t} == {a, b} and rel in _PRODUCT_EDGES for h, rel, t in G.edges)


# ---------------------------------------------------------------
# Union-find (used to cluster nominations into LLM batches)
# ---------------------------------------------------------------

class UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


# ---------------------------------------------------------------
# THE LLM BATCH-GROUPING CALL — the only merging authority
# ---------------------------------------------------------------

SYSTEM_PROMPT = ("You are a precise entity-resolution engine. You group name "
                 "variants that refer to the SAME real-world entity. "
                 "You reply with ONLY valid JSON.")

USER_TEMPLATE = """Below are entities from one knowledge graph, flagged as POSSIBLE duplicates of each other. For each entity you get its name, type, mention count, and its actual graph edges.

TASK: Group together the entities that refer to the SAME real-world entity.

Rules:
1. Merge legal-form variants ("Tesla" / "Tesla Inc"), historical names ("Tesla Motors" is the former name of "Tesla, Inc"), and unambiguous short forms.
2. Do NOT merge different things that merely share a brand word. Judge by the EDGES: a company has a CEO, locations, acquisitions; a PRODUCT is manufactured/launched by a company; a DIVISION is a part of a parent. If one entity has edges like 'MANUFACTURES X' or 'LAUNCHED X' and the other IS that company, they are DIFFERENT.
3. Two entities with DIFFERENT location/factory/country names are different even if the rest of the name matches ("Gigafactory Texas" is NOT "Gigafactory Mexico").
4. If unsure, do NOT group. Missing a merge is a small error; a wrong merge is a big error.
5. Entities belonging to no group must NOT appear in the output.
6. Each entity may appear in AT MOST ONE group; put the best-known name FIRST (it becomes canonical).

Return ONLY this JSON shape (groups are lists of the exact entity names given):
{{"groups": [["canonical_name", "variant2"], ...]}}

EXAMPLE:
Entities:
  A: "apple inc" (ORGANIZATION, 50 mentions) | edges: EMPLOYS tim cook; LOCATED_IN california
  B: "apple" (ORGANIZATION, 10 mentions) | edges: EMPLOYS tim cook
  C: "iphone" (PRODUCT, 8 mentions) | edges: MANUFACTURED_BY apple inc
  D: "tim cook" (PERSON, 12 mentions) | edges: CEO_OF apple inc
Output: {{"groups": [["apple inc", "apple"]]}}

NOW GROUP THESE:
Entities:
{entities}
Output:"""


def llm_group_batch(context_block: str, batch_displays: list[str]) -> list[list[str]]:
    """One LLM call for one nomination cluster, with graph evidence per entity.
    Returns validated groups of display names, or [] on failure.
    Uses the shared llm_client (Gemini when GEMINI_API_KEY/.env GEMINI is set,
    otherwise the Ollama endpoint)."""
    from llm_client import chat_json, LLMError
    try:
        result = chat_json(SYSTEM_PROMPT, USER_TEMPLATE.format(entities=context_block))
    except LLMError as e:
        print(f"      LLM call failed: {str(e)[:100]}")
        return []
    raw_groups = result.get("groups", []) if isinstance(result, dict) else []

    # ---- UNTRUSTED OUTPUT: validate ----
    valid = set(batch_displays)
    claimed: set[str] = set()
    groups = []
    for g in raw_groups:
        if not isinstance(g, list):
            continue
        members = [n for n in g if isinstance(n, str) and n in valid and n not in claimed]
        if len(members) < 2:            # singleton group = no-op
            continue
        for m in members:
            claimed.add(m)
        groups.append(members)          # members[0] = LLM's canonical choice
    return groups


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=NOMINATE_THRESHOLD,
                    help="fuzzy nomination threshold (LLM still decides)")
    ap.add_argument("--dry-run", action="store_true",
                    help="show nominations only, no LLM calls, no rebuild")
    args = ap.parse_args()

    if not TRIPLES_PATH.exists():
        sys.exit("No data/triples.jsonl — run build_graph.py first.")

    # ---- 1. raw graph from checkpoint ----
    G = Graph()
    with open(TRIPLES_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            for t in rec["triples"]:
                G.add_triple(t["head"], t["relation"], t["tail"], source_chunk=rec["chunk_id"])
    G.backfill_types()
    print(f"raw graph: {len(G.nodes)} nodes, {len(G.edges)} edges")

    # ---- 2. blocking ----
    pairs = candidate_pairs(list(G.nodes))
    print(f"blocking: {len(pairs)} candidate pairs")

    # ---- 3. fuzzy nomination (NO merging here) ----
    nominations, rejected = [], []
    for a, b in pairs:
        if " and " in a or " and " in b:
            continue                              # compound artifact: not a merge case
        if product_edge_between(G, a, b):
            continue                              # provably different: maker vs made
        score = similarity(a, b)
        if score >= args.threshold:
            nominations.append((a, b, score))
        else:
            rejected.append((a, b, score))

    print(f"fuzzy nominated: {len(nominations)} probable pairs (threshold {args.threshold})")

    # ---- 4. cluster nominations -> LLM batches ----
    uf = UnionFind()
    for a, b, _s in nominations:
        uf.union(a, b)
    clusters: dict[str, list[str]] = defaultdict(list)
    for k in G.nodes:
        if any(k in (a, b) for a, b, _ in nominations):
            clusters[uf.find(k)].append(k)

    print(f"nomination clusters: {len(clusters)}")
    for root, members in clusters.items():
        names = ", ".join(G.nodes[m]["display"] for m in members)
        print(f"  [{len(members)}] {names}")

    if args.dry_run:
        print("\ndry run — no LLM calls, graph unchanged")
        return

    # ---- 5. LLM decides (one batch call per cluster) ----
    key_map: dict[str, str] = {}          # loser key -> canonical key
    canon_displays: dict[str, str] = {}   # canonical key -> preferred display name
    audit = []
    def entity_context(key: str) -> str:
        """Graph-evidence block for one entity: name, type, mentions, edges."""
        n = G.nodes[key]
        lines = []
        count = 0
        for (h, rel, t), e in G.edges.items():
            if count >= 5:
                break
            if h == key:
                other = G.nodes.get(t, {}).get("display", t)
                lines.append(f"{rel} -> {other}")
                count += 1
            elif t == key:
                other = G.nodes.get(h, {}).get("display", h)
                lines.append(f"<- {rel} from {other}")
                count += 1
        edges = "; ".join(lines) if lines else "(no edges)"
        return f'  "{n["display"]}" (type={n["type"]}, mentions={n["mentions"]}) | edges: {edges}'

    for root, members in clusters.items():
        members = sorted(members, key=lambda m: -G.nodes[m]["mentions"])
        # batches if a cluster is huge (rare)
        for i in range(0, len(members), MAX_BATCH):
            batch = members[i:i + MAX_BATCH]
            displays = [G.nodes[m]["display"] for m in batch]
            display_to_key = {G.nodes[m]["display"]: m for m in batch}
            print(f"\nLLM judging batch: {displays}")
            for m in batch:
                print(entity_context(m))
            context_block = "\n".join(entity_context(m) for m in batch)
            groups = llm_group_batch(context_block, displays)
            if not groups:
                audit.append({"batch": displays, "verdict": "no groups / call failed"})
                continue
            for g in groups:
                # We pick the canonical OURSELVES (most mentions, then longest
                # name) — don't trust the LLM's ordering for naming.
                members_keys = [display_to_key[d] for d in g if d in display_to_key]
                if len(members_keys) < 2:
                    audit.append({"group": g, "verdict": "rejected (unknown members)"})
                    continue
                canon_key = max(members_keys,
                                key=lambda k: (G.nodes[k]["mentions"], len(G.nodes[k]["display"])))
                canon_display = G.nodes[canon_key]["display"]
                absorbed = [G.nodes[k]["display"] for k in members_keys if k != canon_key]
                for k in members_keys:
                    if k != canon_key:
                        key_map[k] = canon_key
                canon_displays[canon_key] = canon_display
                audit.append({"canonical": canon_display, "absorbed": absorbed,
                              "verdict": "merged"})
                print(f"   MERGE: {canon_display} <- {absorbed}")

    if not key_map:
        print("\nLLM approved no merges. Graph unchanged.")
        LOG_PATH.write_text(json.dumps(
            {"threshold": args.threshold, "nominations": [
                {"a": G.nodes[a]["display"], "b": G.nodes[b]["display"],
                 "score": round(s, 3)} for a, b, s in nominations],
             "audit": audit}, indent=2), encoding="utf-8")
        return

    # ---- 6. rebuild with LLM-approved merges ----
    G2 = Graph()
    with open(TRIPLES_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            for t in rec["triples"]:
                hk = normalize_entity(t["head"])
                tk = normalize_entity(t["tail"])
                hk, tk = key_map.get(hk, hk), key_map.get(tk, tk)
                G2.add_edge_keys(hk, normalize_relation(t["relation"]), tk,
                                 head_display=t["head"], tail_display=t["tail"],
                                 source_chunk=rec["chunk_id"])
    G2.backfill_types()
    # Restore the chosen canonical display names (add_edge_keys prefers the
    # LONGER name, which would make the hub display as "Tesla Motors, Inc.")
    for k, d in canon_displays.items():
        if k in G2.nodes:
            G2.nodes[k]["display"] = d
    G2.save(GRAPH_PATH)

    LOG_PATH.write_text(json.dumps(
        {"threshold": args.threshold,
         "nominations": [{"a": G.nodes[a]["display"], "b": G.nodes[b]["display"],
                          "score": round(s, 3)} for a, b, s in nominations],
         "audit": audit}, indent=2), encoding="utf-8")

    print(f"\ncleaned graph: {len(G2.nodes)} nodes, {len(G2.edges)} edges -> {GRAPH_PATH}")
    print(f"audit log -> {LOG_PATH}")
    print(f"\nSTATS AFTER LLM MERGE:\n{G2.stats()}")


if __name__ == "__main__":
    main()
