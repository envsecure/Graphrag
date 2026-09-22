"""
predict.py — tag raw text with a trained run (CPU-friendly; cuda if present).

  python predict.py --model runs/<name> "Elon Musk is the CEO of Tesla."
  python predict.py --model runs/<name> --file chunks.txt
  echo "text" | python predict.py --model runs/<name>

Output rows: <score>  <TYPE>  <surface text>. Char-accurate spans (word
offsets), scored by mean softmax over the words' first sub-tokens.
"""
from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoTokenizer

from config import BIO_ID2LABEL, word_offsets
from model import load_model


def tag(text: str, model, tok, device, max_len: int = 384) -> list[dict]:
    words = word_offsets(text)
    if not words:
        return []
    enc = tok(words, is_split_into_words=True, truncation=True,
              max_length=max_len, return_tensors="pt")
    wids = enc.word_ids(batch_index=0)
    with torch.no_grad():
        logits = model(**{k: v.to(device) for k, v in enc.items()}).logits[0]
    probs = logits.softmax(-1)

    # one label + score per word (first sub-token only; §2.2)
    labs: list[str] = []
    scores: list[float] = []
    seen = None
    for i, w in enumerate(wids):
        if w is None or w == seen:
            continue
        seen = w
        tid = int(logits[i].argmax())
        labs.append(BIO_ID2LABEL[tid])
        scores.append(float(probs[i, tid]))

    # word labels -> char spans (longest run of same type via B/I grammar)
    spans: list[dict] = []
    cur = None
    for k, lab in enumerate(labs):
        if lab == "O":
            if cur:
                spans.append(cur)
                cur = None
            continue
        begin, typ = lab.split("-", 1)
        if begin == "B" or cur is None or cur["type"] != typ:
            if cur:
                spans.append(cur)
            cur = {"start": words[k][0], "end": words[k][1],
                   "type": typ, "_scores": [scores[k]]}
        else:
            cur["end"] = words[k][1]
            cur["_scores"].append(scores[k])
    if cur:
        spans.append(cur)
    for s in spans:
        s["score"] = round(sum(s["_scores"]) / len(s["_scores"]), 3)
        del s["_scores"]
    return spans


def main() -> None:
    ap = argparse.ArgumentParser(description="Tag raw text with the NER model.")
    ap.add_argument("--model", required=True, help="run dir (runs/<name>)")
    ap.add_argument("text", nargs="*", help="text to tag (or --file / stdin)")
    ap.add_argument("--file", help="one text per line")
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "cuda", "cpu"])
    a = ap.parse_args()

    if a.text:
        texts = [" ".join(a.text)]
    elif a.file:
        with open(a.file, encoding="utf-8") as fh:
            texts = [l.rstrip("\n") for l in fh if l.strip()]
    elif not sys.stdin.isatty():
        texts = [sys.stdin.read()]
    else:
        raise SystemExit("give text, --file, or pipe stdin")

    device = torch.device("cuda" if a.backend in ("auto", "cuda")
                          and torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = load_model(a.model).to(device).eval()
    print(f"device: {device.type}  model: {a.model}")

    for text in texts:
        for s in tag(text, model, tok, device, a.max_len):
            if s["score"] >= a.min_score:
                print(f"{s['score']:.3f}\t{s['type']}\t"
                      f"{text[s['start']:s['end']]}")
        print("---")


if __name__ == "__main__":
    main()
