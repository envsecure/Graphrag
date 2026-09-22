"""
hf_builder.py — test the NER head on a HuggingFace dataset (CoNLL-2003).

Converts word-tag sentences into the SAME build/ner_*.jsonl row format that
data_builder.py writes, so dataset.py / train_ner.py / predict.py are
untouched: testing a public dataset, and switching back to this repo's
silver data, is just running the other builder.

CoNLL tags PER/LOC/ORG/MISC -> this project's six types (§2.4); anything
unknown buckets to MISC (config.clean_type). Official train/dev/test splits
are used as-is — no shuffle needed, they never share a document.

Run:  python hf_builder.py --self-test    # pure conversion asserts, no network
      python hf_builder.py                # downloads conll2003 (~4 MB)
      python train_ner.py --epochs 1      # same command as always

Back to the repo's silver data:  python data_builder.py
Needs: pip install datasets   (NER_MODEL/requirements.txt)
"""
from __future__ import annotations

import argparse
import json

import config

# CoNLL-style type names -> graph.py ENTITY_TYPES (then clean_type coercion)
HF_TYPE_MAP = {
    "PER": "PERSON", "PERSON": "PERSON",
    "ORG": "ORGANIZATION", "ORGANISATION": "ORGANIZATION",
    "LOC": "LOCATION", "GPE": "LOCATION",
    "MISC": "MISC",
}


def tag_to_span_type(tag: str) -> str:
    """'B-PER' -> 'PERSON'; unknown types bucket to MISC."""
    etype = tag.split("-", 1)[1] if "-" in tag else tag
    return config.clean_type(HF_TYPE_MAP.get(etype.upper(), etype))


def sentence_to_row(tokens: list[str], tags: list[str], chunk_id: str) -> dict:
    """CoNLL word tags -> char-span row (the format data_builder writes)."""
    text = " ".join(tokens)
    offsets = []
    p = 0
    for tok in tokens:
        offsets.append((p, p + len(tok)))
        p += len(tok) + 1

    def close(i: int) -> dict:
        return {"start": offsets[start][0], "end": offsets[i][1], "type": etype}

    spans: list[dict] = []
    start = None          # index of the open span's first token
    etype = None
    end_tok = None
    for i, tag in enumerate(tags):
        if tag.startswith(("B-", "I-")):
            kind, typ = tag[0], tag_to_span_type(tag)
        else:
            kind, typ = "O", None
        if kind == "B" or (start is not None and typ != etype):
            if start is not None:
                spans.append(close(end_tok))
            start, etype, end_tok = i, typ, i
        elif kind == "I" and typ == etype:
            end_tok = i
        else:                                  # O closes the open span
            if start is not None:
                spans.append(close(end_tok))
            start = etype = end_tok = None
    if start is not None:
        spans.append(close(end_tok))
    return {"chunk_id": chunk_id, "doc": chunk_id, "text": text, "spans": spans}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="HF token-classification dataset -> build/ner_*.jsonl")
    ap.add_argument("--dataset", default="conll2003")
    ap.add_argument("--max-sentences", type=int, default=0,
                    help="0 = all (smoke test: 200)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        _self_test()
        return

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("pip install datasets   (see requirements.txt)")
    try:
        ds = load_dataset(a.dataset)
    except Exception as e:
        raise SystemExit(f"could not load {a.dataset!r}: {e}")

    tag_names = ds["train"].features["ner_tags"].feature.names
    print(f"source: {a.dataset}  tags: {tag_names}")

    split_map = {"train": "train", "validation": "dev", "test": "test"}
    by_type: dict[str, int] = {}
    rows_written: dict[str, int] = {}

    for hf_split, our_name in split_map.items():
        if hf_split not in ds:
            continue
        out = config.build_file(f"ner_{our_name}")
        n = 0
        with open(out, "w", encoding="utf-8") as fh:
            for i, ex in enumerate(ds[hf_split]):
                if a.max_sentences and i >= a.max_sentences:
                    break
                tags = [tag_names[t] for t in ex["ner_tags"]]
                row = sentence_to_row(ex["tokens"], tags, f"{our_name}::{i}")
                fh.write(json.dumps(row) + "\n")
                n += 1
                for s in row["spans"]:
                    by_type[s["type"]] = by_type.get(s["type"], 0) + 1
        rows_written[our_name] = n
        print(f"  {our_name}: {n} rows -> {out}")

    config.build_file("splits").write_text(json.dumps(
        {k: [f"{a.dataset}:{k}"] for k in rows_written}, indent=1),
        encoding="utf-8")
    config.build_file("build_stats").write_text(json.dumps(
        {"source": a.dataset, "rows": rows_written, "by_type": by_type},
        indent=1), encoding="utf-8")
    config.build_file("label_maps").write_text(json.dumps(
        {"bio_labels": config.BIO_LABELS, "ignore_index": config.IGNORE_INDEX},
        indent=1), encoding="utf-8")
    print(f"-> {config.BUILD_DIR}  (back to silver: python data_builder.py)")


def _self_test() -> None:
    row = sentence_to_row(
        ["Elon", "Musk", "is", "CEO", "of", "Tesla", "Inc", "."],
        ["B-PER", "I-PER", "O", "O", "O", "B-ORG", "I-ORG", "O"],
        "train::0")
    got = {(row["text"][s["start"]:s["end"]], s["type"]) for s in row["spans"]}
    assert got == {("Elon Musk", "PERSON"), ("Tesla Inc", "ORGANIZATION")}, got

    row2 = sentence_to_row(["A", "B", "C"], ["B-MISC", "I-MISC", "B-MISC"], "x")
    assert len(row2["spans"]) == 2, row2          # adjacent same-type split

    assert tag_to_span_type("B-LOC") == "LOCATION"
    assert tag_to_span_type("I-PER") == "PERSON"
    assert tag_to_span_type("B-TADPOLE") == "MISC"   # unknown bucket
    print("hf_builder self-test OK")


if __name__ == "__main__":
    main()
