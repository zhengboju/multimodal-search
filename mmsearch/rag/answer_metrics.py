"""Answer-level metrics for document VQA: ANLS (DocVQA/InfoVQA), relaxed accuracy (ChartQA), EM, token F1."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def normalize(s: str) -> str:
    s = (s or "").lower().strip()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(pred: str, golds: Sequence[str], tau: float = 0.5) -> float:
    p = " ".join((pred or "").lower().split())
    best = 0.0
    for g in golds:
        g = " ".join(str(g).lower().split())
        if not g and not p:
            return 1.0
        nl = levenshtein(p, g) / max(len(p), len(g), 1)
        best = max(best, 1 - nl if nl < tau else 0.0)
    return best


def _to_float(s: str) -> float | None:
    s = (s or "").strip().replace(",", "").replace("$", "")
    pct = s.endswith("%")
    s = s.rstrip("%").strip()
    try:
        v = float(s)
    except ValueError:
        return None
    return v / 100 if pct else v


def relaxed_accuracy(pred: str, golds: Sequence[str], tol: float = 0.05) -> float:
    """ChartQA: numeric answers within 5% relative error, otherwise exact (normalised) match."""
    for g in golds:
        pv, gv = _to_float(pred), _to_float(str(g))
        if pv is not None and gv is not None:
            # also accept the same number written with/without a % sign on one side
            for p in (pv, pv * 100, pv / 100):
                if (gv == 0 and p == 0) or (gv != 0 and abs(p - gv) / abs(gv) <= tol):
                    return 1.0
        elif normalize(pred) == normalize(str(g)):
            return 1.0
    return 0.0


def exact_match(pred: str, golds: Sequence[str]) -> float:
    return float(any(normalize(pred) == normalize(str(g)) for g in golds))


def token_f1(pred: str, golds: Sequence[str]) -> float:
    pt = normalize(pred).split()
    best = 0.0
    for g in golds:
        gt = normalize(str(g)).split()
        common = sum((Counter(pt) & Counter(gt)).values())
        if common == 0:
            continue
        p, r = common / len(pt), common / len(gt)
        best = max(best, 2 * p * r / (p + r))
    return best


METRICS = {"anls": anls, "relaxed": relaxed_accuracy, "em": exact_match, "f1": token_f1}


def metric_for(dataset_name: str, domain: str = "") -> str:
    n = (dataset_name + " " + domain).lower()
    if "chart" in n:
        return "relaxed"
    if "tatdqa" in n or "financial" in n:
        return "relaxed"
    return "anls"


def score(pred: str, golds: Sequence[str], metric: str) -> float:
    return METRICS[metric](pred, golds)
