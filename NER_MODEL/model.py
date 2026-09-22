"""
model.py — pretrained encoder + BIO head (docs/NER_model.md §7.1).

Default bert-base-cased; any HF encoder id is one flag away
(--model microsoft/deberta-v3-base is the documented upgrade).
ignore_mismatched_sizes swaps the pretrained 2-way head for our 13 labels.
"""
from __future__ import annotations

from transformers import AutoModelForTokenClassification

from config import BIO_ID2LABEL, BIO_LABEL2ID, NUM_BIO_LABELS


def build_model(name: str = "bert-base-cased"):
    return AutoModelForTokenClassification.from_pretrained(
        name,
        num_labels=NUM_BIO_LABELS,
        id2label={str(i): lab for i, lab in BIO_ID2LABEL.items()},
        label2id=BIO_LABEL2ID,
        ignore_mismatched_sizes=True,
    )


def load_model(path: str):
    return AutoModelForTokenClassification.from_pretrained(path)
