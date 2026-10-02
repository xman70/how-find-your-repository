"""Probability calibration: Platt scaling vs isotonic regression.

Raw classifier scores are not probabilities. Calibrators are fitted on a held-
out calibration split with class-balanced sample weights, so the displayed
probability corresponds to a 50 % prior (equal numbers of human and AI texts).
The real-world base rate in a given setting is unknown and changes the
posterior probability; this is stated in every report.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

P_MIN, P_MAX = 0.005, 0.995


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def balanced_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).astype(int)
    w = np.ones(len(y), dtype=float)
    for c in (0, 1):
        n = (y == c).sum()
        if n:
            w[y == c] = len(y) / (2.0 * n)
    return w


class PlattCalibrator:
    name = "platt"

    def fit(self, p: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        self.lr = LogisticRegression(C=1e4, max_iter=1000)
        self.lr.fit(logit(p).reshape(-1, 1), y, sample_weight=sample_weight)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return np.clip(self.lr.predict_proba(logit(p).reshape(-1, 1))[:, 1], P_MIN, P_MAX)

    @property
    def params(self) -> dict:
        return {"slope": float(self.lr.coef_[0, 0]), "intercept": float(self.lr.intercept_[0])}


class IsotonicCalibrator:
    name = "isotonic"

    def fit(self, p: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=P_MIN, y_max=P_MAX)
        self.iso.fit(np.asarray(p, float), np.asarray(y, float), sample_weight=sample_weight)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return np.clip(self.iso.predict(np.asarray(p, float)), P_MIN, P_MAX)


class IdentityCalibrator:
    name = "none"

    def fit(self, p, y, sample_weight=None):
        return self

    def transform(self, p):
        return np.clip(np.asarray(p, float), P_MIN, P_MAX)


CALIBRATORS = {"platt": PlattCalibrator, "isotonic": IsotonicCalibrator, "none": IdentityCalibrator}


def brier(y, p, w=None) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.average((p - y) ** 2, weights=w))


def log_loss(y, p, w=None) -> float:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return float(np.average(-(y * np.log(p) + (1 - y) * np.log(1 - p)), weights=w))


def reliability(y, p, n_bins: int = 10, w=None) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    w = np.ones_like(p) if w is None else np.asarray(w, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "count": 0, "mean_pred": None, "frac_pos": None})
            continue
        bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "count": int(m.sum()),
                     "mean_pred": float(np.average(p[m], weights=w[m])), "frac_pos": float(np.average(y[m], weights=w[m]))})
    return {"bins": bins}


def ece(y, p, n_bins: int = 10, w=None) -> float:
    """Expected Calibration Error with equal-width bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    w = np.ones_like(p) if w is None else np.asarray(w, float)
    total = w.sum()
    rel = reliability(y, p, n_bins, w)["bins"]
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    err = 0.0
    for b, info in enumerate(rel):
        if info["count"]:
            m = idx == b
            err += w[m].sum() / total * abs(info["mean_pred"] - info["frac_pos"])
    return float(err)


def compare_calibrators(p: np.ndarray, y: np.ndarray, methods=("platt", "isotonic"), folds: int = 5,
                        seed: int = 0, n_bins: int = 10) -> dict:
    """Cross-validated comparison on the calibration split (balanced weighting)."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(p))
    fold_of = np.empty(len(p), dtype=int)
    fold_of[order] = np.arange(len(p)) % folds
    results = {}
    for m in ("none",) + tuple(methods):
        out = np.empty(len(p))
        for k in range(folds):
            tr, te = fold_of != k, fold_of == k
            cal = CALIBRATORS[m]().fit(p[tr], y[tr], balanced_weights(y[tr]))
            out[te] = cal.transform(p[te])
        w = balanced_weights(y)
        results[m] = {"brier": brier(y, out, w), "ece": ece(y, out, n_bins, w), "log_loss": log_loss(y, out, w),
                      "reliability": reliability(y, out, n_bins, w)}
    best = min(methods, key=lambda m: (results[m]["brier"], results[m]["ece"]))
    return {"results": results, "selected": best}


def fit_best_calibrator(p, y, methods=("platt", "isotonic"), seed: int = 0, n_bins: int = 10):
    cmp = compare_calibrators(p, y, methods, seed=seed, n_bins=n_bins)
    cal = CALIBRATORS[cmp["selected"]]().fit(np.asarray(p, float), np.asarray(y, int), balanced_weights(y))
    return cal, cmp
