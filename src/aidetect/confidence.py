"""Confidence model, kept separate from the AI likelihood.

AI likelihood answers "how strongly do the measured characteristics resemble
AI-generated text?". Confidence answers "how much should this estimate be
trusted?" and combines:

* length reliability - measured on truncated held-out documents (evaluate.py
  derives the table; defaults are used until then)
* detector agreement - weighted spread of the individual detectors
* feature coverage   - share of features inside the training population range
  (out-of-distribution text lowers confidence)
* committee stability - spread of the cross-validation committee members

Each component is reported so users can see why confidence is low.
"""
from __future__ import annotations

import numpy as np

DEFAULT_LENGTH_TABLE = [[0, 0.0], [50, 0.15], [100, 0.35], [250, 0.65], [500, 0.85], [1000, 1.0]]
EVIDENCE_TEXT = {  # length-based evidence levels (thresholds measured on held-out data)
    "insufficient": "Insufficient evidence (text too short)",
    "low": "Low confidence (short text)",
    "moderate": "Moderate confidence (text length)",
    "reliable": "More reliable statistical analysis (text length)",
}


def evidence_level(n_words: int, thresholds: dict) -> str:
    if n_words < thresholds.get("low", 100):
        return "insufficient"
    if n_words < thresholds.get("moderate", 250):
        return "low"
    if n_words < thresholds.get("reliable", 500):
        return "moderate"
    return "reliable"


def length_reliability(n_words: int, table=None) -> float:
    t = np.asarray(table or DEFAULT_LENGTH_TABLE, dtype=float)
    return float(np.clip(np.interp(n_words, t[:, 0], t[:, 1]), 0, 1))


def agreement(probs: dict[str, float], weights: dict[str, float]) -> float:
    names = [n for n in probs if weights.get(n, 0) > 0] or list(probs)
    p = np.array([probs[n] for n in names])
    w = np.array([max(weights.get(n, 0.0), 1e-3) for n in names])
    mean = np.average(p, weights=w)
    sd = np.sqrt(np.average((p - mean) ** 2, weights=w))
    return float(np.clip(1 - sd / 0.35, 0, 1))


def coverage(x: np.ndarray, names: list[str], population: dict) -> float:
    inside, total = 0, 0
    for v, n in zip(x, names):
        ref = population.get(n, {}).get("all")
        if ref is None or v != v:
            continue
        total += 1
        inside += ref["q005"] <= v <= ref["q995"]
    return inside / total if total else 1.0


def overall(n_words: int, probs: dict[str, float], weights: dict[str, float], stds: dict[str, float],
            cov: float, length_table=None, levels: dict | None = None) -> dict:
    lr = length_reliability(n_words, length_table)
    ag = agreement(probs, weights)
    st = float(np.clip(1 - 2 * np.mean(list(stds.values())), 0, 1)) if stds else 1.0
    ood = float(np.clip(cov, 0, 1)) ** 3
    value = lr * (0.6 + 0.4 * ag) * (0.5 + 0.5 * ood) * (0.7 + 0.3 * st)
    levels = levels or {"low": 0.4, "high": 0.7}
    level = "Low" if value < levels["low"] else ("High" if value >= levels["high"] else "Medium")
    return {"value": float(value), "level": level,
            "components": {"length_reliability": lr, "detector_agreement": ag, "feature_coverage": float(cov),
                           "committee_stability": st}}


def sentence_confidence(window_words: float, agreement_value: float, length_table=None) -> str:
    v = length_reliability(window_words, length_table) * (0.5 + 0.5 * agreement_value)
    return "Low" if v < 0.3 else ("High" if v >= 0.6 else "Medium")
