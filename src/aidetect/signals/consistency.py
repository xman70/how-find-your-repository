"""Style consistency: sliding-window style drift and change-point detection.

Style differences inside a document do not imply AI use (people change tone,
quote sources, or write sections at different times). These signals contribute
only probabilistically, and detected change points are reported as
"potential authorship/style transitions", never as "AI begins here".
"""
from __future__ import annotations

import math

import numpy as np

from ..preprocessing import SentenceRecord
from .textstats import mattr

NAN = float("nan")
_POS = ["pos_ADJ", "pos_ADV", "pos_NOUN", "pos_VERB", "pos_AUX", "pos_PRON", "pos_PROPN", "pos_NUM", "pos_ADP",
        "pos_DET", "pos_CCONJ", "pos_SCONJ"]
_PUN = ["comma", "semicolon", "colon", "dash", "paren", "quote", "question", "exclaim", "ellipsis", "period"]
_FW = ["articles", "pron1s", "pron1p", "pron2", "pron3", "cconj", "sconj", "contraction", "function"]


def _style_parts(R: list[SentenceRecord]) -> list[np.ndarray]:
    parts = []
    for keys in (_POS, _PUN, _FW):
        parts.append(np.array([sum(r.c(k) for r in R) for k in keys], dtype=float))
    sl = [r.c("n_words") for r in R]
    parts.append(np.array([sum(x < 10 for x in sl), sum(10 <= x <= 25 for x in sl), sum(x > 25 for x in sl)], float))
    return parts


def js_divergence(p: np.ndarray, q: np.ndarray, alpha: float = 0.5) -> float:
    p = p + alpha
    q = q + alpha
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log2(p / m)) + np.sum(q * np.log2(q / m))))


def style_distance(A: list[SentenceRecord], B: list[SentenceRecord]) -> float:
    return float(np.mean([js_divergence(a, b) for a, b in zip(_style_parts(A), _style_parts(B))]))


def style_drift(R: list[SentenceRecord], window: int = 5, stride: int = 2) -> dict:
    """Drift between consecutive windows. Returns features and the series."""
    out = {"style_drift_mean": NAN, "style_drift_max": NAN, "style_drift_std": NAN, "series": [], "centers": []}
    if len(R) < window + stride:
        return out
    starts = list(range(0, len(R) - window + 1, stride))
    wins = [R[s:s + window] for s in starts]
    d = [style_distance(a, b) for a, b in zip(wins, wins[1:])]
    centers = [starts[i + 1] + window / 2 - 0.5 for i in range(len(d))]
    out.update(style_drift_mean=float(np.mean(d)), style_drift_max=float(np.max(d)),
               style_drift_std=float(np.std(d)) if len(d) > 1 else NAN, series=d, centers=centers)
    return out


def sentence_matrix(R: list[SentenceRecord]) -> np.ndarray:
    rows = []
    for r in R:
        n = max(r.c("n_words"), 1.0)
        a = max(r.c("n_alpha"), 1.0)
        rows.append([
            r.c("n_words"),
            float(np.mean(r.word_lens)) if r.word_lens else 0.0,
            float(np.mean(r.surprisal)) if len(r.surprisal) else 0.0,
            r.c("comma") / n,
            float(r.depth),
            r.c("function") / a,
            r.c("pos_NOUN") / n,
            float(len(r.transitions)),
        ])
    X = np.array(rows, dtype=float)
    if len(X) == 0:
        return X
    sd = X.std(axis=0)
    keep = sd > 1e-9
    return (X[:, keep] - X[:, keep].mean(axis=0)) / sd[keep]


def boundary_stats(Z: np.ndarray, window: int = 4, min_segment: int = 3) -> np.ndarray:
    """Two-sample statistic at each boundary b (between sentence b-1 and b)."""
    n = len(Z)
    T = np.full(n, np.nan)
    if n < 2 * min_segment or Z.shape[1] == 0:
        return T
    for b in range(min_segment, n - min_segment + 1):
        L = Z[max(0, b - window):b]
        Rr = Z[b:min(n, b + window)]
        diff = L.mean(axis=0) - Rr.mean(axis=0)
        T[b] = float(np.mean(diff ** 2) / (1.0 / len(L) + 1.0 / len(Rr)))
    return T


def cp_feature(R: list[SentenceRecord], window: int = 4, min_segment: int = 3) -> float:
    Z = sentence_matrix(R)
    T = boundary_stats(Z, window, min_segment)
    if np.all(np.isnan(T)):
        return NAN
    nb = int(np.sum(~np.isnan(T)))
    return float(np.nanmax(T) / (1.0 + math.log(nb)))


def detect_changepoints(R: list[SentenceRecord], K: np.ndarray | None = None, window: int = 4,
                        min_segment: int = 3, permutations: int = 200, alpha: float = 0.05,
                        seed: int = 0, extra: np.ndarray | None = None) -> dict:
    """Permutation-tested change points.

    ``extra`` optionally adds per-sentence series (e.g. sentence AI probability)
    to the statistic.
    """
    Z = sentence_matrix(R)
    if extra is not None and len(extra) == len(Z) and len(Z):
        e = np.asarray(extra, dtype=float).reshape(len(Z), -1)
        sd = e.std(axis=0)
        e = (e - e.mean(axis=0)) / np.where(sd > 1e-9, sd, 1.0)
        Z = np.hstack([Z, e]) if Z.size else e
    n = len(Z)
    res = {"series": [], "threshold": None, "p_value": None, "changepoints": []}
    if n < 2 * min_segment + 1 or Z.shape[1] == 0:
        return res
    T = boundary_stats(Z, window, min_segment)
    rng = np.random.default_rng(seed)
    null_max = np.empty(permutations)
    for p in range(permutations):
        null_max[p] = np.nanmax(boundary_stats(Z[rng.permutation(n)], window, min_segment))
    obs = float(np.nanmax(T))
    thr = float(np.quantile(null_max, 1 - alpha))
    res["series"] = [None if math.isnan(v) else float(v) for v in T]
    res["threshold"] = thr
    res["p_value"] = float((1 + np.sum(null_max >= obs)) / (permutations + 1))
    order = [b for b in np.argsort(-np.nan_to_num(T, nan=-1.0)) if not math.isnan(T[b]) and T[b] >= thr]
    chosen: list[int] = []
    for b in order:
        if all(abs(b - c) >= min_segment for c in chosen):
            chosen.append(int(b))
        if len(chosen) >= max(1, n // (2 * min_segment)):
            break
    for b in sorted(chosen):
        L = R[max(0, b - window):b]
        Rr = R[b:min(n, b + window)]
        p_val = float((1 + np.sum(null_max >= T[b])) / (permutations + 1))
        diffs = describe_difference(L, Rr, K, list(range(max(0, b - window), b)), list(range(b, min(n, b + window))))
        res["changepoints"].append({"boundary_before_sentence": b, "statistic": float(T[b]), "p_value": p_val,
                                    "label": "Potential authorship/style transition", "differences": diffs})
    return res


def describe_difference(L, Rr, K=None, li=None, ri=None) -> dict:
    def mean_s(S):
        v = np.concatenate([r.surprisal for r in S]) if S else np.zeros(0)
        return float(v.mean()) if len(v) else NAN

    out = {
        "style_js": style_distance(L, Rr),
        "perplexity_bits": abs(mean_s(L) - mean_s(Rr)),
        "vocabulary_mattr": abs(mattr([w for r in L for w in r.words]) - mattr([w for r in Rr for w in r.words])),
        "syntax_depth": abs(float(np.mean([r.depth for r in L])) - float(np.mean([r.depth for r in Rr]))),
        "sentence_length": abs(float(np.mean([r.c("n_words") for r in L])) - float(np.mean([r.c("n_words") for r in Rr]))),
        "semantic_distance": NAN,
    }
    if K is not None and li and ri:
        num = K[np.ix_(li, ri)].mean()
        den = math.sqrt(max(K[np.ix_(li, li)].mean(), 1e-12) * max(K[np.ix_(ri, ri)].mean(), 1e-12))
        out["semantic_distance"] = float(1 - num / den)
    return out
