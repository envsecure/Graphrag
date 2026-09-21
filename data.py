"""
data.py — Fetch real business & global-economy articles from Wikipedia.

Wikipedia API needs NO key, so this runs immediately. Your LLM key
is only needed later, for the EXTRACTION step (building the graph).

Run:  python data.py
Out:  data/articles/*.txt   (one clean text file per article)
      data/index.json       (metadata + per-article chunk counts)
"""

import json
import re
import time
import urllib.request
import urllib.parse
from pathlib import Path

DATA_DIR = Path("data")
ARTICLES_DIR = DATA_DIR / "articles"
INDEX_PATH = DATA_DIR / "index.json"

# --- Business / global economy related articles -------------------------
# (all real Wikipedia articles, chosen for rich entity relationships:
#  companies, people, deals, countries, products)
ARTICLES = [
    # -- Companies & people --
    "Tesla, Inc.",
    "Elon Musk",
    "Apple Inc.",
    "Tim Cook",
    "Microsoft",
    "Satya Nadella",
    "Amazon (company)",
    "Jeff Bezos",
    "Nvidia",
    "Jensen Huang",
    "OpenAI",
    "Sam Altman",
    "Microsoft Copilot",
    "Anthropic",
    "Amazon Web Services",
    "SpaceX",
    "Starlink",
    "Alphabet Inc.",
    "Sundar Pichai",
    "Samsung Electronics",
    "TSMC",
    "Intel",
    "Foxconn",
    # -- Deals / competition / geopolitics --
    "Stargate LLC",
    "CHIPS and Science Act",
    "United States–China trade war",
    "OPEC",
    "Saudi Aramco",
    "European Union",
    "World Trade Organization",
    "International Monetary Fund",
    "World Bank",
    "Berkshire Hathaway",
    "Warren Buffett",
    "Reliance Industries",
    "Mukesh Ambani",
    "Jio",
    "Alibaba Group",
    "Jack Ma",
    "Tencent",
    "Huawei",
    "ByteDance",
    "TikTok",
    "Volkswagen Group",
    "Toyota",
    "BYD Auto",
    "Shell plc",
    "BP",
    "ExxonMobil",
]

HEADERS = {
    "User-Agent": "GraphRAG-Learning-Demo/1.0 (educational project; contact: student@example.com)"
}


def fetch_wikititle(title: str) -> str | None:
    """Download the plain-text extract of a Wikipedia article via the public API."""
    url = (
        "https://en.wikipedia.org/w/api.php"
        "?action=query&format=json&prop=extracts&explaintext=1"
        "&redirects=1&titles=" + urllib.parse.quote(title)
    )
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        payload = json.loads(r.read().decode())

    pages = payload["query"]["pages"]
    for _, page in pages.items():
        if "extract" not in page or not page["extract"].strip():
            return None
        return page["extract"]
    return None


def clean_text(text: str) -> str:
    """Light cleanup: drop section headers' stray equals signs and excess newlines."""
    text = re.sub(r"^==+\s*(.+?)\s*==+\s*$", r"\1:", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, target_size: int = 1000, overlap: int = 150) -> list[str]:
    """
    Real chunking (STEP 1 of GraphRAG): paragraph-aware sliding window.
    ~1000 chars per chunk with 150 char overlap so no fact is cut in half.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, current = [], ""

    for para in paragraphs:
        # If a single paragraph is huge, split it into sentences
        while len(para) > target_size * 2:
            cut = para.rfind(". ", 0, target_size)
            cut = cut + 1 if cut != -1 else target_size
            piece, para = para[:cut].strip(), para[cut:].strip()
            if piece:
                chunks.append(piece)
        # Try to pack paragraphs into the current chunk
        if len(current) + len(para) + 1 <= target_size:
            current = (current + "\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            current = para

    if current:
        chunks.append(current)

    # Add overlap: start each chunk (after the first) with the tail of the previous one
    overlapped = []
    for i, c in enumerate(chunks):
        if i > 0 and overlap > 0:
            tail = chunks[i - 1][-overlap:]
            c = tail + " " + c
        overlapped.append(c)
    return overlapped


def main():
    ARTICLES_DIR.mkdir(parents=True, exist_ok=True)

    index = {}
    ok, failed = 0, []

    for i, title in enumerate(ARTICLES, start=1):
        print(f"[{i}/{len(ARTICLES)}] fetching: {title} ...", flush=True)
        try:
            raw = fetch_wikititle(title)
        except Exception as e:
            print(f"    FAILED: {e}")
            failed.append({"title": title, "reason": f"network error: {e}"})
            continue

        if not raw:
            print("    SKIPPED: empty extract")
            failed.append({"title": title, "reason": "empty or missing article"})
            continue

        text = clean_text(raw)
        # Truncate very long articles: we want breadth, not one giant doc
        text = text[:40_000]

        # filesystem-safe filename
        fname = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_") + ".txt"
        (ARTICLES_DIR / fname).write_text(text, encoding="utf-8")

        chunks = chunk_text(text)
        index[fname] = {
            "title": title,
            "chars": len(text),
            "num_chunks": len(chunks),
            "chunks_preview": chunks[0][:120] + "..." if chunks else "",
        }
        ok += 1
        print(f"    saved {fname} ({len(text):,} chars, {len(chunks)} chunks)")
        time.sleep(0.2)  # be polite to Wikipedia

    index["_meta"] = {
        "source": "en.wikipedia.org",
        "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "articles_ok": ok,
        "articles_failed": failed,
        "chunking": "paragraph-aware, ~1000 chars, 150 char overlap",
        "note": "One .txt per article in data/articles/. Extraction (next step) reads index.json.",
    }

    INDEX_PATH.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"\nDone: {ok} articles saved to {ARTICLES_DIR}/")
    print(f"Index: {INDEX_PATH}")
    if failed:
        print(f"Failed/skipped {len(failed)}: " + ", ".join(f["title"] for f in failed))


if __name__ == "__main__":
    main()
