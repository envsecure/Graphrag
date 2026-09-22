"""
dataset.py — BIO rows -> sub-token batches (docs/NER_model.md §2.2, §3.4).

No torch import: a plain sequence plus the HF collator. Labels land on the
FIRST sub-token of each word; continuations and specials get IGNORE_INDEX
(-100) so `Eberhard` never becomes two entities. Overlapping spans: longest
wins, decided once here on word labels.
"""
from __future__ import annotations

import json

from transformers import DataCollatorForTokenClassification

from config import BIO_LABEL2ID, IGNORE_INDEX, OUTSIDE_ID, word_offsets


def word_labels(words: list[tuple[int, int]], spans: list[dict]) -> list[int]:
    """Char spans -> one BIO id per word. Longest span claims first; a shorter
    span whose first word is already claimed is dropped entirely (§3.4)."""
    labels = [OUTSIDE_ID] * len(words)
    for sp in sorted(spans, key=lambda s: -(s["end"] - s["start"])):
        b = BIO_LABEL2ID.get(f'B-{sp["type"]}')
        i = BIO_LABEL2ID.get(f'I-{sp["type"]}')
        if b is None:
            continue                              # unknown type: leave as O
        idxs = [k for k, (ws, we) in enumerate(words)
                if ws < sp["end"] and we > sp["start"]]
        if not idxs or labels[idxs[0]] != OUTSIDE_ID:
            continue                              # empty, or inside a longer span
        labels[idxs[0]] = b
        for k in idxs[1:]:
            if labels[k] == OUTSIDE_ID:
                labels[k] = i
    return labels


class NerDataset:
    """build/ner_*.jsonl rows -> {input_ids, attention_mask, labels}."""

    def __init__(self, path, tokenizer, max_len: int = 384):
        self.items: list[dict] = []
        self.chunk_ids: list[str] = []
        self.truncated = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                words = word_offsets(row["text"])
                if not words:
                    continue
                labels = word_labels(words, row.get("spans") or [])
                enc = tokenizer(words, is_split_into_words=True,
                                truncation=True, max_length=max_len)
                wids = enc.word_ids()
                kept = {w for w in wids if w is not None}
                if not kept or max(kept) < len(words) - 1:
                    self.truncated += 1
                lab, prev = [], None
                for w in wids:
                    if w is None:
                        lab.append(IGNORE_INDEX)     # [CLS]/[SEP]
                    elif w != prev:
                        lab.append(labels[w])        # first piece of the word
                        prev = w
                    else:
                        lab.append(IGNORE_INDEX)     # continuation piece (§2.2)
                self.items.append({"input_ids": enc["input_ids"],
                                   "attention_mask": enc["attention_mask"],
                                   "labels": lab})
                self.chunk_ids.append(row["chunk_id"])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]


def make_collator(tokenizer):
    return DataCollatorForTokenClassification(tokenizer)
