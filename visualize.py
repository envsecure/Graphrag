"""
visualize.py — Render the knowledge graph as an INTERACTIVE web page.

  mouse wheel  = zoom in/out
  click+drag canvas = pan
  click+drag node   = move a node (physics re-settles the graph)
  hover node/edge   = tooltip with details

No Python packages needed — generates a self-contained graph.html that
loads vis-network from a CDN in the browser.

Usage:
  python visualize.py                      # render everything
  python visualize.py --min-mentions 2     # only edges seen in 2+ chunks
  python visualize.py --max-nodes 300      # cap node count (big graphs)
"""

import argparse
import json
from pathlib import Path
from urllib.parse import quote

GRAPH_PATH = Path("data/graph.json")
COMMUNITIES_PATH = Path("data/communities.json")
OUT_PATH = Path("graph.html")

# Color per entity type (vis-network group colors)
TYPE_COLORS = {
    "PERSON":       "#e67e22",  # orange
    "ORGANIZATION": "#3498db",  # blue
    "LOCATION":     "#2ecc71",  # green
    "PRODUCT":      "#9b59b6",  # purple
    "EVENT":        "#e74c3c",  # red
    "MISC":         "#95a5a6",  # gray
}


# Community palette (cycles if there are more communities than colors)
COMMUNITY_COLORS = ["#3498db", "#e67e22", "#2ecc71", "#e74c3c", "#9b59b6",
                    "#1abc9c", "#f39c12", "#34495e", "#16a085", "#d35400"]


def load_communities():
    """data/communities.json -> {node_key: community_id}, or {} if missing."""
    cpath = Path("data/communities.json")
    if not cpath.exists():
        return {}
    data = json.loads(cpath.read_text(encoding="utf-8"))
    return {e["key"]: c["id"] for c in data["communities"] for e in c["entities"]}


def build_vis_data(graph: dict, min_mentions: int, max_nodes: int,
                   color_by: str = "community"):
    """Convert graph.json -> {nodes: [...], edges: [...]} for vis-network.
    color_by: "community" (cluster colors + legend) or "type" (entity type)."""

    communities = load_communities() if color_by == "community" else {}

    # ---- filter edges first (drops their low-mention endpoints too) ----
    edges = [e for e in graph["edges"] if e["mentions"] >= min_mentions]

    # ---- keep only the max_nodes most-connected nodes ----
    degree: dict[str, int] = {}
    for e in edges:
        degree[e["head"]] = degree.get(e["head"], 0) + 1
        degree[e["tail"]] = degree.get(e["tail"], 0) + 1
    if max_nodes and len(degree) > max_nodes:
        keep = {k for k, _ in sorted(degree.items(), key=lambda kv: -kv[1])[:max_nodes]}
        edges = [e for e in edges if e["head"] in keep and e["tail"] in keep]
    else:
        keep = set(degree)

    # ---- nodes ----
    nodes = []
    for key, n in graph["nodes"].items():
        if key not in keep:
            continue
        mentions = n.get("mentions", 1)
        etype = n.get("type", "MISC")
        aliases = ", ".join(n.get("aliases", [])[:5])
        cid = communities.get(key)
        tooltip = (
            f"<b>{n['display']}</b><br>"
            f"type: {etype}<br>"
            f"mentions: {mentions}<br>"
            + (f"community: C{cid}<br>" if cid is not None else "")
            + f"<small>aliases: {aliases}</small>"
        )
        vis_group = (f"C{cid}" if cid is not None else "Cnone") \
            if color_by == "community" else etype
        nodes.append({
            "id": key,
            "label": n["display"],
            "title": tooltip,                      # vis shows on hover
            "group": vis_group,
            "size": 6 + min(mentions, 40) * 0.7,   # scale node by importance
        })

    # ---- edges ----
    vis_edges = []
    for e in edges:
        m = e["mentions"]
        chunks = ", ".join(e["source_chunks"][:4])
        more = f" (+{len(e['source_chunks'])-4} more)" if len(e["source_chunks"]) > 4 else ""
        # border edges (different communities) get dashed style in community mode
        dashed = (color_by == "community"
                  and communities.get(e["head"]) is not None
                  and communities.get(e["tail"]) is not None
                  and communities[e["head"]] != communities[e["tail"]])
        tooltip = (
            f"<b>{e['relation']}</b><br>"
            f"mentions: {m}<br>"
            + ("<i>community bridge</i><br>" if dashed else "")
            + f"<small>sources: {chunks}{more}</small>"
        )
        vis_edges.append({
            "from": e["head"],
            "to": e["tail"],
            "label": e["relation"],
            "title": tooltip,
            "dashes": dashed,
            "width": 1 + min(m, 8) * 0.6,          # scale edge by strength
            "font": {"size": 9, "color": "#666"},
        })

    return nodes, vis_edges


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Knowledge Graph</title>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    html, body, #mynetwork { height: 100%; width: 100%; margin: 0; padding: 0; }
    #legend {
      position: absolute; top: 10px; left: 10px; z-index: 10;
      background: rgba(255,255,255,0.92); padding: 8px 12px; border-radius: 6px;
      font-family: sans-serif; font-size: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.3);
      max-width: 340px; max-height: 80vh; overflow-y: auto;
    }
    .swatch { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; }
    .csummary { color:#555; font-size:11px; margin-left:15px; }
  </style>
</head>
<body>
<div id="legend">
  <b>__LEGEND__</b><br>
  __SWATCHES__
  <br><small>wheel=zoom · drag=pan · drag node=move · hover=details · dashed edge = community bridge</small>
</div>
<div id="mynetwork"></div>
<script>
  const data = __DATA__;

  const container = document.getElementById("mynetwork");
  const options = {
    groups: __GROUPS__,
    edges: {
      arrows: { to: { enabled: true, scaleFactor: 0.5 } },
      smooth: { type: "continuous" },
      font: { strokeWidth: 0 },
      labelHighlightBold: true,
    },
    interaction: {
      hover: true,
      tooltipDelay: 120,
      navigationButtons: true,   // little arrows for panning too
      keyboard: true,            // arrow keys pan, +/- zoom
    },
    physics: {
      solver: "barnesHut",
      // repel and long springs so distinct communities visibly separate
      barnesHut: { gravitationalConstant: -12000, springLength: 160, springConstant: 0.03 },
      stabilization: { iterations: 300 },
    },
  };
  const network = new vis.Network(container, data, options);
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-mentions", type=int, default=1)
    ap.add_argument("--max-nodes", type=int, default=400)
    ap.add_argument("--color-by", choices=["community", "type"], default="community",
                    help="color nodes by detected community or by entity type")
    args = ap.parse_args()

    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    nodes, edges = build_vis_data(graph, args.min_mentions, args.max_nodes,
                                  color_by=args.color_by)

    swatches = ""
    if args.color_by == "community":
        # group colors by community id + a summary line from communities.json
        comm_ids = sorted({n["group"] for n in nodes},
                          key=lambda g: int(g[1:]) if g[1:].isdigit() else 999)
        groups = {}
        legend_lines = []
        summaries = {}
        if COMMUNITIES_PATH.exists():
            cdata = json.loads(COMMUNITIES_PATH.read_text(encoding="utf-8"))
            for c in cdata["communities"]:
                top = ", ".join(e["display"] for e in c["entities"][:4])
                s = (c.get("summary") or "")[:90]
                summaries[f"C{c['id']}"] = (top, s)
        for i, gid in enumerate(comm_ids):
            color = COMMUNITY_COLORS[i % len(COMMUNITY_COLORS)]
            groups[gid] = {"color": color}
            top, s = summaries.get(gid, ("", ""))
            legend_lines.append(
                f'<span class="swatch" style="background:{color}"></span>{gid}'
                f'<span class="csummary">{top}{" — " + s if s else ""}</span><br>')
        swatches = "".join(legend_lines) or "<i>(no communities found — run communities.py)</i>"
        legend_title = "Communities"
    else:
        groups = {t: {"color": c} for t, c in TYPE_COLORS.items()}
        swatches = "".join(
            f'<span class="swatch" style="background:{c}"></span>{t} &nbsp; '
            for t, c in TYPE_COLORS.items() if t != "MISC"
        ) + '<span class="swatch" style="background:#95a5a6"></span>MISC'
        legend_title = "Entity types"

    html = (HTML_TEMPLATE
            .replace("__DATA__", json.dumps({"nodes": nodes, "edges": edges}))
            .replace("__GROUPS__", json.dumps(groups))
            .replace("__SWATCHES__", swatches)
            .replace("__LEGEND__", legend_title))

    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"Rendered {len(nodes)} nodes, {len(edges)} edges -> {OUT_PATH}")
    print(f"color mode: {args.color_by}")
    print(f"Open it:  start {OUT_PATH}   (or double-click the file)")


if __name__ == "__main__":
    main()
