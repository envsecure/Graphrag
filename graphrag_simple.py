"""
Simplest possible GraphRAG — ONE file, ZERO dependencies, no API key.
Run:  python graphrag_simple.py

PIPELINE (maps to the theory):
  1. CHUNK        : split document into chunks
  2. EXTRACT      : text -> (head, relation, tail) triples   [LLM step, mocked here]
  3. BUILD GRAPH  : store nodes/edges, merge repeated mentions
  4. (COMMUNITIES : skipped — see notes at the bottom)
  5. LOCAL SEARCH : find seed node -> walk k hops -> collect evidence

In a real system, step 2 is an LLM call and step 5 ends by feeding
the evidence into an LLM. Everything else is exactly what real
GraphRAG systems do — you are looking at the actual skeleton.
"""

# ================================================================
# STEP 1 — CHUNKING
# Real GraphRAG chunks long docs into ~300-800 token pieces.
# Here each sentence is one chunk so you can SEE the chunk IDs.
# ================================================================

DOCUMENT = [
    "Sarah Chen is the CEO of Helios Energy, a startup based in Austin.",
    "Helios Energy signed a $2M supply contract with Vertex Manufacturing in March 2025.",
    "Vertex Manufacturing, headquartered in Detroit, is a subsidiary of Titan Industrial.",
    "Sarah previously worked at Titan Industrial as an engineer.",
    "Dr. Patel, chief scientist at Helios, co-authored a battery paper with Dr. Lopez, who now leads research at Vertex.",
]

chunks = {f"chunk_{i}": text for i, text in enumerate(DOCUMENT, start=1)}


# ================================================================
# STEP 2 — EXTRACTION (text -> triples)   <<< THE ONLY LLM STEP >>>
#
# A real system sends each chunk to an LLM with a prompt like:
#   "Extract entities and relationships as JSON triples
#    [{head, relation, tail}] from this text: ..."
#
# We MOCK it with the hand-written triples from our Step-2 exercise
# so this runs offline, free, with no API key. Swap this function
# for a real LLM call later and NOTHING else changes.
# ================================================================

MOCK_EXTRACTION = {
    "chunk_1": [
        ("Sarah Chen", "CEO_OF", "Helios Energy"),
        ("Helios Energy", "LOCATED_IN", "Austin"),
    ],
    "chunk_2": [
        ("Helios Energy", "SIGNED_CONTRACT_WITH", "Vertex Manufacturing"),
    ],
    "chunk_3": [
        ("Vertex Manufacturing", "SUBSIDIARY_OF", "Titan Industrial"),
        ("Vertex Manufacturing", "HEADQUARTERED_IN", "Detroit"),
    ],
    "chunk_4": [
        ("Sarah Chen", "FORMER_EMPLOYEE_OF", "Titan Industrial"),
    ],
    "chunk_5": [
        ("Dr. Patel", "CHIEF_SCIENTIST_OF", "Helios Energy"),
        ("Dr. Patel", "CO_AUTHORED_WITH", "Dr. Lopez"),
        ("Dr. Lopez", "LEADS_RESEARCH_AT", "Vertex Manufacturing"),
    ],
}


def extract_triples(chunk_id: str) -> list[tuple[str, str, str]]:
    """Mock LLM extraction. In production: return llm_extraction_prompt(chunks[chunk_id])."""
    return MOCK_EXTRACTION[chunk_id]


# ================================================================
# STEP 3 — GRAPH STORE (+ entity merging)
# A graph is just: {node -> properties} and {node -> [(neighbor, edge_props)]}
# networkx / Neo4j are fancy versions of exactly this.
#
# KEY THEORY MADE VISIBLE:
#  - When the same entity appears in a second chunk, add_edge finds
#    the EXISTING node and just attaches a new edge -> this is
#    entity merging, and it's what connects the graph together.
#  - Every edge stores source_chunk -> answers stay citable.
# ================================================================

class Graph:
    """Tiny property graph: nodes and directed edges with properties."""

    def __init__(self):
        self.nodes: dict[str, dict] = {}                 # node name -> properties
        self.out_edges: dict[str, list[tuple]] = {}      # node -> [(dst, props), ...]

    def add_edge(self, src, relation, dst, source_chunk):
        for n in (src, dst):
            self.nodes.setdefault(n, {"type": "UNKNOWN"})
        self.out_edges.setdefault(src, []).append(
            (dst, {"relation": relation, "source_chunk": source_chunk})
        )

    def out(self, node):
        """All edges leaving this node."""
        return self.out_edges.get(node, [])

    def incoming(self, node):
        """All edges pointing AT this node (reverse index built on the fly)."""
        result = []
        for src, edges in self.out_edges.items():
            for dst, props in edges:
                if dst == node:
                    result.append((src, props))
        return result

    def all_nodes(self):
        return list(self.nodes)

    def all_edges(self):
        result = []
        for src, edges in self.out_edges.items():
            for dst, props in edges:
                result.append((src, props["relation"], dst, props))
        return result


G = Graph()

for chunk_id in chunks:
    for head, relation, tail in extract_triples(chunk_id):
        G.add_edge(head, relation, tail, source_chunk=chunk_id)


# ================================================================
# STEP 5 — LOCAL SEARCH (query time)
# Theory: entity linking (find the node) -> k-hop traversal
# (walk the edges) -> evidence subgraph -> original chunks -> LLM.
# ================================================================

def local_search(entity: str, hops: int = 2) -> dict:
    """Find an entity node and walk `hops` hops to build the evidence subgraph."""
    if entity not in G.nodes:
        return {"error": f"'{entity}' not in graph. Known entities: {G.all_nodes()}"}

    facts, evidence_chunks = [], set()

    # --- Hop 1: direct edges in/out of the entity ---
    for dst, props in G.out(entity):
        facts.append((entity, props["relation"], dst))
        evidence_chunks.add(props["source_chunk"])
    for src, props in G.incoming(entity):
        facts.append((src, props["relation"], entity))
        evidence_chunks.add(props["source_chunk"])

    # --- Hop 2: follow edges of the direct neighbors ---
    if hops >= 2:
        neighbors = [dst for dst, _ in G.out(entity)] + [src for src, _ in G.incoming(entity)]
        for nbr in set(neighbors):
            if nbr == entity:
                continue
            for dst, props in G.out(nbr):
                if dst != entity:
                    facts.append((nbr, props["relation"], dst))
                    evidence_chunks.add(props["source_chunk"])

    return {"entity": entity, "facts": facts, "source_chunks": sorted(evidence_chunks)}


def answer(entity: str, hops: int = 2) -> str:
    """
    Pretend-LLM step: a real GraphRAG would now send [question + evidence
    + original chunk texts] to the LLM to compose the final answer.
    Here we just print the evidence so you can SEE what the LLM would get.
    """
    result = local_search(entity, hops)
    if "error" in result:
        return result["error"]

    lines = [f"\nEvidence subgraph around '{entity}' ({hops}-hop traversal):"]
    for h, r, t in result["facts"]:
        lines.append(f"   {h} --[{r}]--> {t}")

    lines.append("\nOriginal chunks the LLM would receive as context:")
    for c in result["source_chunks"]:
        lines.append(f"   [{c}] {chunks[c]}")
    return "\n".join(lines)


# ================================================================
# DEMO
# ================================================================

if __name__ == "__main__":
    print("=== 1. THE KNOWLEDGE GRAPH ===")
    print(f"Nodes ({len(G.all_nodes())}): {G.all_nodes()}\n")
    print(f"Edges ({len(G.all_edges())}):")
    for src, rel, dst, props in G.all_edges():
        print(f"   {src} --[{rel}]--> {dst}   (from {props['source_chunk']})")

    print("\n=== 2. MULTI-HOP QUESTION ===")
    print("Q: What connects Helios Energy's leadership to Vertex Manufacturing?")
    print("(No single chunk answers this — only the GRAPH structure does)\n")
    print(answer("Helios Energy", hops=2))

    print("\n=== 3. WHAT A REAL LLM WOULD NOW WRITE ===")
    print(
        "\"Helios Energy's leadership is connected to Vertex Manufacturing in "
        "three ways: (1) a $2M supply contract (chunk_2); (2) chief scientist "
        "Dr. Patel co-authored battery research with Dr. Lopez, who now leads "
        "research at Vertex (chunk_5); (3) CEO Sarah Chen previously worked at "
        "Titan Industrial, Vertex's parent company (chunk_4).\""
    )
