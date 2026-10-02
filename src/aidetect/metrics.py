"""Classification metrics with group-level bootstrap confidence intervals."""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from .calibration import balanced_weights, brier, ece


def binary_metrics(y, p, threshold: float = 0.5) -> dict:
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n_pos, n_neg = tp + fn, tn + fp
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / n_pos if n_pos else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec > 0 else float("nan")
    both = n_pos > 0 and n_neg > 0
    out = {
        "n": int(len(y)), "n_pos": n_pos, "n_neg": n_neg,
        "accuracy": (tp + tn) / len(y) if len(y) else float("nan"),
        "balanced_accuracy": 0.5 * (tp / n_pos + tn / n_neg) if both else float("nan"),
        "precision": prec, "recall": rec, "f1": f1,
        "fpr": fp / n_neg if n_neg else float("nan"),
        "fnr": fn / n_pos if n_pos else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if both else float("nan"),
        "brier_balanced": brier(y, p, balanced_weights(y)) if both else brier(y, p),
        "ece_balanced": ece(y, p, 10, balanced_weights(y)) if both else float("nan"),
    }
    if both:
        fpr, tpr, _ = roc_curve(y, p)
        for target in (0.01, 0.05):
            out[f"tpr_at_fpr_{int(target * 100)}pct"] = float(np.interp(target, fpr, tpr))
    return out


def bootstrap_ci(y, p, groups=None, n_boot: int = 500, seed: int = 0, threshold: float = 0.5,
                 keys=("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fpr", "fnr")) -> dict:
    """95 % percentile intervals; resamples whole groups when ``groups`` is given."""
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    rng = np.random.default_rng(seed)
    if groups is None:
        groups = np.arange(len(y))
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    vals: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        m = binary_metrics(y[idx], p[idx], threshold)
        for k in keys:
            v = m.get(k, float("nan"))
            if v == v:
                vals[k].append(v)
    return {k: ([float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) > 10 else None)
            for k, v in vals.items()}


def rate_ci(k: int, n: int, z: float = 1.96) -> list[float] | None:
    """Wilson score interval for a proportion (e.g. a false-positive rate)."""
    if n == 0:
        return None
    phat = k / n
    den = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / den
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / den
    return [max(0.0, centre - half), min(1.0, centre + half)]


def summarize(y, p, groups=None, threshold: float = 0.5, n_boot: int = 500, seed: int = 0) -> dict:
    m = binary_metrics(y, p, threshold)
    m["ci95"] = bootstrap_ci(y, p, groups, n_boot=n_boot, seed=seed, threshold=threshold)
    return m
