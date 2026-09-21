"""
graph.py — Property-graph store for the real GraphRAG pipeline.

Same concepts as graphrag_simple.py's Graph class, upgraded for real data:
  - entity normalization + aliases   ("Tesla, Inc." / "tesla inc" -> one node)
  - edge aggregation with MENTION COUNTS (12 chunks saying Musk->CEO_OF->Tesla
    = one strong edge, not 12 duplicates)
  - full provenance: every edge keeps the list of source chunk IDs
  - save/load to data/graph.json

Node shape:
  nodes[key] = {display, type, mentions, aliases:[...]}
Edge shape:
  edges[(head_key, relation, tail_key)] = {head, relation, tail, mentions, source_chunks:[...]}
"""

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------
# Normalization — the "entity resolution lite" from the theory.
# Not perfect ("Musk" vs "Elon Musk" needs fuzzy matching later),
# but free, deterministic, and catches most duplicates.
# ---------------------------------------------------------------

_STOP_PREFIX = {"the", "a", "an"}

# Corporate-suffix whitelist (LEVEL 1 entity resolution). Stripped from the
# MERGE KEY only — display names keep their full form ("Tesla, Inc.").
# Deliberately conservative: 'group'/'holdings'/'motors' carry meaning and
# stay ("Volkswagen Group", "General Motors").
_CORP_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "llc", "ltd", "limited", "plc", "gmbh", "ag", "nv", "bv", "sa",
}


def normalize_entity(name: str) -> str:
    """Canonical merge key for an entity mention."""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))  # strip accents
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)          # drop punctuation: "tesla, inc." -> "tesla  inc"
    s = re.sub(r"\s+", " ", s).strip()
    parts = [p for p in s.split() if p not in _STOP_PREFIX]
    # Strip trailing corporate suffixes, repeatedly ("apple inc llc" -> "apple")
    while len(parts) > 1 and parts[-1] in _CORP_SUFFIXES:
        parts = parts[:-1]
    return " ".join(parts)


def normalize_relation(rel: str) -> str:
    """'is CEO of' -> 'IS_CEO_OF'"""
    s = re.sub(r"[^\w\s]", " ", rel)
    s = re.sub(r"\s+", "_", s.strip()).upper()
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "RELATED_TO"


VALID_TYPES = {"PERSON", "ORGANIZATION", "LOCATION", "PRODUCT", "EVENT", "MISC"}

# Vague/generic mentions small models love to emit ("other automakers",
# "various countries") — not real entities, so reject them at the gate.
_VAGUE_RE = re.compile(
    r"^(other|others|various|many|several|some|multiple|numerous|certain|"
    r"company|companies|country|countries|people|customers|users|automakers|"
    r"manufacturer|manufacturers|investors|government|authorities|experts)\b"
)


# ---------------------------------------------------------------
# Graph
# ---------------------------------------------------------------

class Graph:
    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: dict[tuple, dict] = {}

    # ---- building ----

    def add_edge_keys(self, hk: str, rel: str, tk: str,
                      head_display: str = "", tail_display: str = "",
                      head_type: str = "MISC", tail_type: str = "MISC",
                      source_chunk: str = ""):
        # Token-count gate: real entity names are rarely > 7 tokens.
        # Kills LLM artifacts like "tesla inc series b venture capital
        # funding round of 13 million in february 2005".
        if len(hk.split()) > 7 or len(tk.split()) > 7:
            return False
        """Low-level: add an edge using PRE-NORMALIZED entity keys.
        Used by add_triple and by the merge post-pass (which maps keys
        before insertion)."""
        for key, display, etype in ((hk, head_display or hk, head_type),
                                    (tk, tail_display or tk, tail_type)):
            node = self.nodes.get(key)
            if node is None:
                self.nodes[key] = {"display": display, "type": etype,
                                   "mentions": 1, "aliases": {display}}
            else:
                node["mentions"] += 1
                node["aliases"].add(display)
                if etype in VALID_TYPES and etype != "MISC":
                    node["type"] = etype
                if len(display) > len(node["display"]):
                    node["display"] = display   # prefer the longer/more complete name

        ekey = (hk, rel, tk)
        edge = self.edges.get(ekey)
        if edge is None:
            self.edges[ekey] = {"head": hk, "relation": rel, "tail": tk,
                                "mentions": 1, "source_chunks": [source_chunk]}
        else:
            edge["mentions"] += 1
            if source_chunk not in edge["source_chunks"]:
                edge["source_chunks"].append(source_chunk)
        return True

    def add_triple(self, head: str, relation: str, tail: str,
                   head_type: str = "MISC", tail_type: str = "MISC",
                   source_chunk: str = ""):
        """Validate + normalize + merge one extraction result."""
        head, tail = head.strip(), tail.strip()
        if not head or not tail or head == tail:
            return False
        if len(head) > 100 or len(tail) > 100 or len(relation) > 60:
            return False
        if _VAGUE_RE.match(head.lower()) or _VAGUE_RE.match(tail.lower()):
            return False

        hk, tk = normalize_entity(head), normalize_entity(tail)
        if not hk or not tk or hk == tk:
            return False
        rel = normalize_relation(relation)
        if rel in {"IN", "OF", "THE", "IS", "A", "AND"}:   # junk relations
            return False

        # Compound-mention split: "Martin Eberhard and Marc Tarpenning" is
        # TWO people jammed into one node by the LLM. Replace it with one
        # edge per part (only the parts — the compound itself is dropped).
        def _parts(name: str) -> list[str]:
            if " and " in name:
                ps = [p.strip() for p in name.split(" and ") if len(p.strip()) > 2]
                if len(ps) >= 2:
                    return ps
            return [name]

        added = False
        for h_disp in _parts(head):
            for t_disp in _parts(tail):
                if normalize_entity(h_disp) == normalize_entity(t_disp):
                    continue
                added = self.add_edge_keys(
                    normalize_entity(h_disp), rel, normalize_entity(t_disp),
                    head_display=h_disp, tail_display=t_disp,
                    head_type=head_type, tail_type=tail_type,
                    source_chunk=source_chunk) or added
        return added

    # ---- querying helpers ----

    def resolve(self, mention: str) -> str | None:
        """Question entity -> node key. Exact display/alias, then substring match."""
        nk = normalize_entity(mention)
        if nk in self.nodes:
            return nk
        # substring fallback (small graphs only; v1 convenience)
        cands = [k for k in self.nodes if nk and (nk in k or k in nk)]
        if len(cands) == 1:
            return cands[0]
        if cands:  # prefer highest-mention candidate
            return max(cands, key=lambda k: self.nodes[k]["mentions"])
        return None

    def stats(self) -> str:
        degrees = Counter()
        for (h, _, t), e in self.edges.items():
            degrees[h] += 1
            degrees[t] += 1
        top_hubs = ", ".join(f"{self.nodes[k]['display']} ({d})" for k, d in degrees.most_common(8))
        rel_hist = Counter(e["relation"] for e in self.edges.values()).most_common(10)
        rel_str = ", ".join(f"{r} x{c}" for r, c in rel_hist)
        n_edges = len(self.edges)
        avg_deg = round(sum(degrees.values()) / max(len(self.nodes), 1), 2)
        return (f"Nodes: {len(self.nodes)}   Edges: {n_edges}   Avg degree: {avg_deg}\n"
                f"Top hubs: {top_hubs}\n"
                f"Relations: {rel_str}")

    # ---- persistence ----

    def backfill_types(self):
        """Small models often emit type MISC for everything. For those nodes,
        infer type by VOTING over their edge relations:
          tail of LOCATED_IN/HEADQUARTERED_IN        -> LOCATION
          head of CEO_OF/*_SCIENTIST_OF              -> PERSON
          tail of CEO_OF/FOUNDED_BY/SUBSIDIARY_OF... -> ORGANIZATION
          head of SUBSIDIARY_OF/COMPETES_WITH/...    -> ORGANIZATION
        First concrete vote wins; still-MISC nodes keep MISC."""
        votes: dict[str, str] = {}
        for (h, rel, t), e in self.edges.items():
            if e["mentions"] < 2:
                continue   # one-off edges are where LLM inversions live; only repeated edges vote
            if rel in {"LOCATED_IN", "HEADQUARTERED_IN", "BASED_IN"}:
                votes.setdefault(t, "LOCATION")
            if rel in {"CEO_OF", "CHIEF_SCIENTIST_OF", "CTO_OF", "CFO_OF"}:
                votes.setdefault(h, "PERSON")
                votes.setdefault(t, "ORGANIZATION")
            # NOTE: deliberately NO vote from FOUNDED_BY / ACQUIRED_BY tails.
            # Small models frequently emit these INVERTED ((founder,
            # FOUNDED_BY, company) instead of (company, FOUNDED_BY,
            # founder)), which typed real companies as PERSON and blocked
            # legit merges ("tesla motors" became a "PERSON").
            # Strong ORG signals only. Deliberately EXCLUDES COMPETES_WITH /
            # PARTNER_WITH: products compete and partner too, so those votes
            # mis-upgrade PRODUCT nodes to ORGANIZATION (caused a real bug:
            # "Tesla Model 3" being absorbed into "Tesla, Inc.").
            if rel in {"SUBSIDIARY_OF", "ACQUIRED", "INVESTED_IN",
                       "EMPLOYS", "MANUFACTURES", "HEADQUARTERED_IN"}:
                votes.setdefault(h, "ORGANIZATION")
        for key, node in self.nodes.items():
            if node.get("type", "MISC") in ("", "MISC") and key in votes:
                node["type"] = votes[key]

    def save(self, path: str | Path):
        self.backfill_types()
        data = {
            "nodes": {k: {**v, "aliases": sorted(v["aliases"])} for k, v in self.nodes.items()},
            "edges": [{"head": h, "relation": r, "tail": t,
                       "mentions": e["mentions"], "source_chunks": e["source_chunks"]}
                      for (h, r, t), e in self.edges.items()],
        }
        Path(path).write_text(json.dumps(data, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Graph":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        g = cls()
        for k, v in data["nodes"].items():
            v["aliases"] = set(v["aliases"])
            g.nodes[k] = v
        for e in data["edges"]:
            g.edges[(e["head"], e["relation"], e["tail"])] = e
        return g
