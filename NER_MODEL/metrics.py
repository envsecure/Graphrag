"""
metrics.py — span P/R/F1: offset-exact (start, end, type) match, micro and
per type (docs/NER_model.md Part 9). stdlib only: the referee must keep
working on a machine that has never seen a GPU.

Empty-vs-empty scores 1.0; anything else with a zero denominator scores 0.0.
"""
from __future__ import annotations

from collections import Counter


def _prf(gold: Counter, pred: Counter) -> dict:
    tp = sum((gold & pred).values())
    p = tp / sum(pred.values()) if pred else (1.0 if not gold else 0.0)
    r = tp / sum(gold.values()) if gold else (1.0 if not pred else 0.0)
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"p": round(p, 4), "r": round(r, 4), "f1": round(f, 4),
            "gold": sum(gold.values()), "pred": sum(pred.values())}


def span_prf(gold, pred) -> dict:
    """gold/pred: iterables of (start, end, type). Multiplicity counts, so
    the same local span in two different rows scores as two occurrences."""
    g, p = Counter(gold), Counter(pred)
    out = {"micro": _prf(g, p), "per_type": {}}
    for t in sorted({x[2] for x in g} | {x[2] for x in p}):
        gt = Counter({k: v for k, v in g.items() if k[2] == t})
        pt = Counter({k: v for k, v in p.items() if k[2] == t})
        out["per_type"][t] = _prf(gt, pt)
    return out


if __name__ == "__main__":
    g = [(0, 1, "PERSON"), (5, 7, "ORGANIZATION"), (9, 9, "MISC")]
    p = [(0, 1, "PERSON"), (5, 7, "ORGANIZATION"), (12, 13, "ORGANIZATION")]
    m = span_prf(g, p)
    assert m["micro"]["p"] == round(2 / 3, 4) and m["micro"]["r"] == round(2 / 3, 4), m
    assert m["per_type"]["ORGANIZATION"]["p"] == 0.5 and m["per_type"]["ORGANIZATION"]["r"] == 1.0, m
    assert span_prf([], [])["micro"]["f1"] == 1.0
    assert span_prf([(0, 1, "PERSON")], [])["micro"]["f1"] == 0.0
    print("metrics self-test OK")
