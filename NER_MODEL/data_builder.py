"""
data_builder.py — silver labels -> span dataset (docs/NER_model.md step E2).

stdlib only (no torch): reads data/triples.jsonl, finds every gold entity
mention in its chunk's text, writes char-span rows + stats into build/.

Why char spans: the tokenizer lives on the torch side; a span is the durable
label format (§5.2). Mentions that cannot be found are DROPPED and COUNTED —
the drop rate is the hard ceiling of this route (§5.2).

Split BY DOCUMENT (§5.4): data.py's chunker overlaps neighbouring chunks by
150 chars, so a chunk-level split would leak text across the boundary.

Run:  python data_builder.py                # build the real dataset
      python data_builder.py --self-test     # asserts only, temp dir
"""
from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path

import config                            # also puts the repo root on sys.path
from data import chunk_text


def find_spans(text: str, name: str, words: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Locate one gold mention: exact substring first, then a word-window
    whose normalized form matches ("Tesla, Inc." vs text "Tesla Inc")."""
    if not name:
        return []
    hits, i = [], text.find(name)
    while i != -1:
        hits.append((i, i + len(name)))
        i = text.find(name, i + 1)
    if hits:
        return hits
    n = len(name.split())
    if n == 0:
        return []
    key = config.normalize_entity(name)
    for a in range(len(words) - n + 1):
        s, e = words[a][0], words[a + n - 1][1]
        if config.normalize_entity(text[s:e]) == key:
            return [(s, e)]
    return []


def annotate(text: str, entities: list) -> tuple[list[dict], int]:
    """(char spans, dropped_count). Overlap resolution (longest wins, §3.4)
    happens later, on word labels in dataset.py."""
    words = config.word_offsets(text)
    spans: list[dict] = []
    dropped = 0
    for ent in entities:
        name = (ent or {}).get("name") or ""
        hits = find_spans(text, name, words) if name else []
        if not hits:
            dropped += 1
            continue
        typ = config.clean_type(ent.get("type"))
        spans += [{"start": s, "end": e, "type": typ} for s, e in hits]
    return spans, dropped


def chunk_lookup(articles_dir):
    """chunk_id 'Tesla_Inc::7' -> its text, re-chunked with data.chunk_text
    (the same function build_graph.py used, so ids map to the same text)."""
    cache: dict[str, list[str] | None] = {}

    def get(chunk_id: str) -> str | None:
        if not chunk_id or "::" not in chunk_id:
            return None
        art, idx = chunk_id.rsplit("::", 1)
        if art not in cache:
            p = Path(articles_dir) / f"{art}.txt"
            cache[art] = chunk_text(p.read_text(encoding="utf-8")) if p.exists() else None
        chunks = cache[art]
        if chunks is None:
            return None
        try:
            return chunks[int(idx)]
        except (ValueError, IndexError):
            return None

    return get


def build(silver_path, articles_dir, out_dir, seed: int = 0,
          test_ratio: float = 0.1, dev_ratio: float = 0.1) -> dict:
    get_chunk = chunk_lookup(articles_dir)
    rows: list[dict] = []
    missing = dropped = mention_total = 0
    by_type: dict[str, int] = {}
    rel_counts: dict[str, int] = {}

    with open(silver_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = rec.get("chunk_id")
            text = get_chunk(cid)
            if text is None:
                missing += 1
                continue
            ents = rec.get("entities") or []
            if not isinstance(ents, list):
                ents = []
            spans, drop = annotate(text, ents)
            mention_total += len(ents)
            dropped += drop
            for ent in ents:
                if isinstance(ent, dict) and ent.get("name"):
                    t = config.clean_type(ent.get("type"))
                    by_type[t] = by_type.get(t, 0) + 1
            for tri in rec.get("triples") or []:
                if isinstance(tri, dict) and tri.get("relation"):
                    rel = config.canonical_relation(tri["relation"])
                    rel_counts[rel] = rel_counts.get(rel, 0) + 1
            rows.append({"chunk_id": cid,
                         "doc": rec.get("doc") or (cid.split("::")[0] if cid else ""),
                         "text": text, "spans": spans})

    if not rows:
        raise SystemExit(f"no usable rows from {silver_path} — run build_graph.py first")

    # ---- split by DOCUMENT (§5.4), deterministically ----
    docs = sorted({r["chunk_id"].split("::")[0] for r in rows})
    if len(docs) < 3:
        raise SystemExit("need >= 3 documents to split train/dev/test")
    rng = random.Random(seed)
    rng.shuffle(docs)
    n_test = max(1, round(len(docs) * test_ratio))
    n_dev = max(1, round(len(docs) * dev_ratio))
    if n_test + n_dev >= len(docs):
        n_test = n_dev = 1
    test_docs = set(docs[:n_test])
    dev_docs = set(docs[n_test:n_test + n_dev])
    train_docs = set(docs) - test_docs - dev_docs
    splits = {"train": sorted(train_docs), "dev": sorted(dev_docs),
              "test": sorted(test_docs)}

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    handles = {k: open(config.build_file(f"ner_{k}", out), "w", encoding="utf-8")
               for k in ("train", "dev", "test")}
    for r in rows:
        d = r["chunk_id"].split("::")[0]
        k = "test" if d in test_docs else "dev" if d in dev_docs else "train"
        handles[k].write(json.dumps(r) + "\n")
    for f in handles.values():
        f.close()

    stats = {
        "rows": len(rows),
        "missing_chunk": missing,
        "mentions": mention_total,
        "found": mention_total - dropped,
        "dropped": dropped,
        "drop_rate": round(dropped / mention_total, 4) if mention_total else 0.0,
        "by_type": by_type,
        "documents": len(docs),
        "splits": {k: len(v) for k, v in splits.items()},
        "relations": dict(sorted(rel_counts.items(), key=lambda kv: -kv[1])),
        "unsupported_relations": {k: v for k, v in rel_counts.items()
                                  if not config.is_trainable_relation(k)},
        "seed": seed,
    }
    config.build_file("splits", out).write_text(json.dumps(splits, indent=1), encoding="utf-8")
    config.build_file("build_stats", out).write_text(json.dumps(stats, indent=1), encoding="utf-8")
    config.build_file("label_maps", out).write_text(
        json.dumps({"bio_labels": config.BIO_LABELS,
                    "ignore_index": config.IGNORE_INDEX}, indent=1), encoding="utf-8")
    return stats


def _self_test() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        art = td / "articles"
        art.mkdir()
        (art / "DocA.txt").write_text(
            "Elon Musk is the CEO of Tesla Inc here.", encoding="utf-8")
        (art / "DocB.txt").write_text("United States matters.", encoding="utf-8")
        (art / "DocC.txt").write_text("Nothing here.", encoding="utf-8")
        silver = td / "triples.jsonl"
        silver.write_text("\n".join(json.dumps(r) for r in [
            {"chunk_id": "DocA::0", "doc": "DocA",
             "triples": [{"head": "Elon Musk", "relation": "IS_CEO_OF",
                          "tail": "Tesla, Inc."}],
             # exact hit + normalized-window hit ("Tesla Inc") + one drop
             "entities": [{"name": "Elon Musk", "type": "PERSON"},
                          {"name": "Tesla, Inc.", "type": "ORGANIZATION"},
                          {"name": "Zorp Industries", "type": "ORGANIZATION"}]},
            {"chunk_id": "DocB::0", "doc": "DocB", "triples": [],
             "entities": [{"name": "United States", "type": "LOCATION"}]},
            {"chunk_id": "DocC::0", "doc": "DocC", "triples": [],
             "entities": []},                       # empty row kept (§5.3)
        ]), encoding="utf-8")

        st = build(silver, art, td / "build", seed=0)
        assert st["rows"] == 3 and st["missing_chunk"] == 0, st
        assert st["dropped"] == 1 and st["drop_rate"] == 0.25, st
        assert st["relations"] == {"CEO_OF": 1}, st          # alias folded
        assert st["unsupported_relations"] == {}, st

        all_rows = [json.loads(l)
                    for k in ("train", "dev", "test")
                    for l in open(config.build_file(f"ner_{k}", td / "build"),
                                  encoding="utf-8")]
        a0 = next(r for r in all_rows if r["chunk_id"] == "DocA::0")
        got = {(a0["text"][s["start"]:s["end"]], s["type"]) for s in a0["spans"]}
        assert got == {("Elon Musk", "PERSON"), ("Tesla Inc", "ORGANIZATION")}, got
        c0 = next(r for r in all_rows if r["chunk_id"] == "DocC::0")
        assert c0["spans"] == [], c0

        sp = json.loads((td / "build" / "splits.json").read_text(encoding="utf-8"))
        for a in ("train", "dev", "test"):
            for b in ("train", "dev", "test"):
                if a < b:
                    assert set(sp[a]).isdisjoint(sp[b]), sp
        assert set(sp["train"]) | set(sp["dev"]) | set(sp["test"]) == {"DocA", "DocB", "DocC"}
        assert len(sp["train"]) >= 1 and len(sp["dev"]) >= 1 and len(sp["test"]) >= 1

        lm = json.loads((td / "build" / "label_maps.json").read_text(encoding="utf-8"))
        assert len(lm["bio_labels"]) == 13 and lm["ignore_index"] == -100, lm
        assert (td / "build" / "build_stats.json").exists()
    print("data_builder self-test OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="silver triples.jsonl -> BIO span dataset")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--dev-ratio", type=float, default=0.1)
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        stats = build(config.TRIPLES_PATH, config.ARTICLES_DIR, config.BUILD_DIR,
                      a.seed, a.test_ratio, a.dev_ratio)
        print(json.dumps({k: stats[k] for k in
                          ("rows", "documents", "splits", "mentions", "dropped",
                           "drop_rate", "missing_chunk")}, indent=1))
        print(f"-> {config.BUILD_DIR}")
