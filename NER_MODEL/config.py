"""
config.py — the single shared contract for the NER_MODEL package.

Every module in this package imports its paths, label spaces, thresholds and
hyper-parameter defaults from here, so the design written down in
`docs/NER_model.md` has exactly one place where the numbers live.

Torch-free by design. Only the model/training/serving files (`dataset.py`,
`model.py`, `accelerator.py`, `train_ner.py`, `train_relation.py`, `train.py`,
`predict.py`, `export_onnx.py`) may import torch / transformers. The
deterministic layers (`data_builder.py`, `stitcher.py`, `metrics.py`,
`thresholds.py`, `evaluate.py`) are stdlib-only so they keep working on a
machine that has never seen a GPU — the same ethos as the rest of this repo
(README: "zero pip dependencies — stdlib only").

Design references, all from docs/NER_model.md:
  §2.2  sub-token labeling: first piece gets the label, continuations IGNORE
  §2.4  head A: BIO token classification over the six `graph.VALID_TYPES`
  §2.5  head B: pair classification, entities marked *in their own context*
  §4.2  relation vocabulary: measured from data/graph.json, not invented
  §4.3  direction-flipped triples are the most valuable hard negatives
  §4.4  imbalance: down-sample easy negatives to ~3:1, never down-sample hard
  §4.5  per-relation thresholds instead of one global rule
  §5.2  drop rate = the hard ceiling of the silver-label route
  §5.4  split by DOCUMENT, never by chunk (the chunker overlaps text)
  §6.1  hybrid: encoder nominates, LLM decides the escalated chunks
  §7.1  bert-base-cased default; deberta-v3-base is the drop-in upgrade

Contract shape that must never change (build_graph.py's checkpoint row):
  {"chunk_id": "...", "doc": "...", "triples": [{"head","relation","tail"}],
   "entities": [{"name","type"}]}
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Iterator

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------

NER_DIR = Path(__file__).resolve().parent           # graph_rag/NER_MODEL/
REPO_ROOT = NER_DIR.parent                          # graph_rag/

# graph.py is stdlib-only and owns the normalizers this contract folds over;
# put the repo root on sys.path once so every NER script can import it.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from graph import normalize_entity, normalize_relation  # noqa: E402


def word_offsets(text: str) -> list[tuple[int, int]]:
    """Whitespace words as (start, end) char offsets — the shared word
    definition for span search, BIO alignment and prediction."""
    out: list[tuple[int, int]] = []
    pos = 0
    for w in text.split():
        start = text.find(w, pos)
        if start < 0:
            break                                   # cannot happen with split() output
        out.append((start, start + len(w)))
        pos = start + len(w)
    return out

DATA_DIR = REPO_ROOT / "data"
ARTICLES_DIR = DATA_DIR / "articles"
INDEX_PATH = DATA_DIR / "index.json"
TRIPLES_PATH = DATA_DIR / "triples.jsonl"           # the silver-label checkpoint
GRAPH_PATH = DATA_DIR / "graph.json"

BUILD_DIR = NER_DIR / "build"                       # datasets + stats + configs
RUNS_DIR = NER_DIR / "runs"                         # checkpoints per experiment

# Every artifact the pipeline reads or writes, by logical name, so no filename
# is ever hard-coded twice.
BUILD_FILES = {
    "ner_train": "ner_train.jsonl",
    "ner_dev": "ner_dev.jsonl",
    "ner_test": "ner_test.jsonl",
    "pairs_train": "pairs_train.jsonl",
    "pairs_dev": "pairs_dev.jsonl",
    "pairs_test": "pairs_test.jsonl",
    "build_stats": "build_stats.json",       # drop rate + per-relation census
    "splits": "splits.json",                 # which documents went where
    "label_maps": "label_maps.json",         # label -> id, frozen for serving
    "pairs_dev_probs": "pairs_dev_probs.jsonl",   # dev pair probabilities
    "thresholds": "thresholds.json",         # per-relation cuts (§4.5)
    "escalation": "escalation.jsonl",        # hybrid escalation log (§6.1)
    "encoder_output": "triples_encoder.jsonl",
}


def build_file(name: str, build_dir: Path | None = None) -> Path:
    """Absolute path of a named build artifact (see BUILD_FILES)."""
    if name not in BUILD_FILES:
        raise KeyError(f"unknown build artifact {name!r}; known: {sorted(BUILD_FILES)}")
    return Path(build_dir or BUILD_DIR) / BUILD_FILES[name]


def build_paths(build_dir: Path | None = None) -> dict[str, Path]:
    """All build artifacts as {logical_name: absolute Path}."""
    return {name: build_file(name, build_dir) for name in BUILD_FILES}


# ---------------------------------------------------------------
# Entity label space (§2.4) — identical to graph.py's VALID_TYPES
# ---------------------------------------------------------------

ENTITY_TYPES: list[str] = [
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "PRODUCT",
    "EVENT",
    "MISC",
]

# BIO (BIO1, §7.3): "O" is id 0, then B-/I- for every type in ENTITY_TYPES order.
BIO_LABELS: list[str] = ["O"] + [
    f"{prefix}-{etype}" for etype in ENTITY_TYPES for prefix in ("B", "I")
]                                                   # 13 labels
BIO_LABEL2ID: dict[str, int] = {lab: i for i, lab in enumerate(BIO_LABELS)}
BIO_ID2LABEL: dict[int, str] = {i: lab for lab, i in BIO_LABEL2ID.items()}
OUTSIDE_LABEL = "O"
OUTSIDE_ID = BIO_LABEL2ID[OUTSIDE_LABEL]            # 0

# Continuation sub-tokens of a split word (`Eber` + `##hard`) and padding get
# this label: they contribute no loss (§2.2).
IGNORE_INDEX = -100
NUM_BIO_LABELS = len(BIO_LABELS)

# ---------------------------------------------------------------
# Relation label space (§4.2, §7.4) — measured from data/graph.json
# ---------------------------------------------------------------
#
# The doc's table of live edge counts decided what is trainable:
#   PARTNER_WITH 13 | LOCATED_IN 10 | COMPETES_WITH 7 | SUBSIDIARY_OF 6
#   ACQUIRED 5 | EMPLOYS 5 | CEO_OF 4 | FOUNDED_BY 3 | INVESTED_IN 3
#   MANUFACTURES 2 (borderline) | everything else 1 each -> not learnable
#
# A classifier needs roughly >=50 positives per class, so those ten are the
# core set. On top of that, the direction pairs the *prompt* gets wrong (see
# backfill_types() in graph.py: "small models frequently emit these INVERTED")
# get their reverse direction as an explicit class, so pointing an edge the
# right way is supervised instead of hoped for (§2.5, §4.3).
#
# Anything outside this vocabulary is an *escalation* problem, not a training
# problem: the hybrid extractor (Part 6) routes those chunks to the LLM.

NO_RELATION = "no_relation"

CORE_RELATIONS: list[str] = [
    "CEO_OF",           # (person, CEO_OF, organization)
    "FOUNDED_BY",       # (organization, FOUNDED_BY, person/group)
    "LOCATED_IN",       # (thing, LOCATED_IN, place)
    "PARTNER_WITH",
    "COMPETES_WITH",
    "ACQUIRED",         # (acquirer, ACQUIRED, target)
    "SUBSIDIARY_OF",
    "EMPLOYS",
    "INVESTED_IN",
    "MANUFACTURES",
]

REVERSE_RELATIONS: list[str] = [
    "HAS_CEO",          # inverse of CEO_OF
    "FOUNDED",          # inverse of FOUNDED_BY
    "ACQUIRED_BY",      # inverse of ACQUIRED
    "PARENT_OF",        # inverse of SUBSIDIARY_OF
    "EMPLOYED_BY",      # inverse of EMPLOYS
    "MANUFACTURED_BY",  # inverse of MANUFACTURES
]

# id 0 must stay `no_relation`: it is the majority class and the default.
RELATION_LABELS: list[str] = [NO_RELATION] + CORE_RELATIONS + REVERSE_RELATIONS
RELATION_LABEL2ID: dict[str, int] = {lab: i for i, lab in enumerate(RELATION_LABELS)}
RELATION_ID2LABEL: dict[int, str] = {i: lab for lab, i in RELATION_LABEL2ID.items()}
NO_RELATION_ID = RELATION_LABEL2ID[NO_RELATION]     # 0
NUM_RELATION_LABELS = len(RELATION_LABELS)

# Observed in data/graph.json but with a single edge each: <50 positives means
# structurally unlearnable (doc §4.2 last table row). Counted by data_builder as
# "unsupported" and routed to LLM escalation, never trained.
UNSUPPORTED_RELATIONS: set[str] = {
    "FOUNDING_MEMBER_OF", "LOANED_TO", "ISSUED", "RAISED", "OPENED_ON",
    "AUTHORED", "ASSUMED_RESPONSIBILITY_FOR", "FILED_CHARGE_AGAINST",
    "OWNED", "HAS_SUBSIDIARY", "SUPPLIES", "DEVELOPED", "PRODUCES",
    "DIVESTED", "SPUN_OFF", "MEMBER_OF", "HAS_PARTNER", "HAS_EMPLOYEE",
    "INVESTED_BY", "LOCATED_AT",
}

# Surface variants the teacher LLM emits for the same idea, folded onto the
# canonical class before the label is looked up. Keys are already normalized by
# normalize_relation(), so only UPPER_SNAKE_CASE forms are needed.
RELATION_ALIASES: dict[str, str] = {
    "IS_CEO_OF": "CEO_OF",
    "CEO_OF_COMPANY": "CEO_OF",
    "CHIEF_EXECUTIVE_OFFICER_OF": "CEO_OF",
    "FOUNDER_OF": "FOUNDED",
    "CO_FOUNDED": "FOUNDED",
    "WAS_FOUNDED_BY": "FOUNDED_BY",
    "SUBSIDIARY": "SUBSIDIARY_OF",
    "IS_SUBSIDIARY_OF": "SUBSIDIARY_OF",
    "OWNED_BY": "SUBSIDIARY_OF",
    "PARENT": "PARENT_OF",
    "HEADQUARTERED_IN": "LOCATED_IN",
    "BASED_IN": "LOCATED_IN",
    "LOCATED_IN_COUNTRY": "LOCATED_IN",
    "PARTNERED_WITH": "PARTNER_WITH",
    "PARTNERS_WITH": "PARTNER_WITH",
    "COMPETITOR_OF": "COMPETES_WITH",
    "COMPETES": "COMPETES_WITH",
    "ACQUIRES": "ACQUIRED",
    "BOUGHT": "ACQUIRED",
    "PURCHASED": "ACQUIRED",
    "EMPLOYS_PERSON": "EMPLOYS",
    "WORKS_FOR": "EMPLOYED_BY",
    "EMPLOYEE_OF": "EMPLOYED_BY",
    "INVESTED": "INVESTED_IN",
    "MAKES": "MANUFACTURES",
    "PRODUCED_BY": "MANUFACTURED_BY",
}

def canonical_relation(relation: str) -> str:
    """Normalize a raw relation string and fold known surface variants.

    Returns the canonical label, which may be outside the trained vocabulary
    (the caller decides whether to count it as unsupported or as a negative)."""
    rel = normalize_relation(relation)
    return RELATION_ALIASES.get(rel, rel)


def is_trainable_relation(relation: str) -> bool:
    """True when `relation` is one of the classes the pair head can predict."""
    return canonical_relation(relation) in RELATION_LABEL2ID


# ---------------------------------------------------------------
# Pair markers (§7.4 — "typed entity markers", the recommended option)
# ---------------------------------------------------------------
#
# The pair is presented *inside its own context*, with both entities marked in
# place and each marker carrying its own type:
#
#   [CLS] [H:PERSON] Elon Musk [/H] is the CEO of [T:ORGANIZATION] Tesla, Inc. [/T] . [SEP]

HEAD_OPEN, HEAD_CLOSE = "[H]", "[/H]"
TAIL_OPEN, TAIL_CLOSE = "[T]", "[/T]"


def clean_type(entity_type: str) -> str:
    """Coerce an arbitrary type string onto the six valid types (default MISC)."""
    etype = (entity_type or "MISC").strip().upper()
    return etype if etype in ENTITY_TYPES else "MISC"


def head_marker(entity_type: str, typed: bool = True) -> str:
    """Opening marker for the head entity, optionally type-tagged."""
    return f"[H:{clean_type(entity_type)}]" if typed else HEAD_OPEN


def tail_marker(entity_type: str, typed: bool = True) -> str:
    """Opening marker for the tail entity, optionally type-tagged."""
    return f"[T:{clean_type(entity_type)}]" if typed else TAIL_OPEN


def marker_tokens(typed: bool = True) -> list[str]:
    """Every marker string the pair tokenizer must register as a special token."""
    if not typed:

        return [HEAD_OPEN, TAIL_OPEN, HEAD_CLOSE, TAIL_CLOSE]
    return (
        [HEAD_CLOSE, TAIL_CLOSE]
        + [head_marker(t) for t in ENTITY_TYPES]
        + [tail_marker(t) for t in ENTITY_TYPES]
    )                                                   # 14 tokens
