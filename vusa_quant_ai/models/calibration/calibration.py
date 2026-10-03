"""Probability calibration (Platt / isotonic) and conformal prediction intervals.

Calibration maps are fitted only on *out-of-sample* probabilities from earlier
walk-forward folds and evaluated on later folds, so the reported calibration is itself
out-of-sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...validation.metrics import brier, ece, log_loss


class ProbabilityCalibrator:
    def __init__(self, method: str = "auto"):
        self.method = method
        self.model_ = None
        self.chosen_ = "none"

    def fit(self, p: np.ndarray, y: np.ndarray) -> "ProbabilityCalibrator":
        p, y = np.asarray(p, float), np.asarray(y, float)
        m = ~(np.isnan(p) | np.isnan(y))
        p, y = p[m], y[m]
        if len(p) < 100 or len(np.unique(y)) < 2:
            self.chosen_ = "none (insufficient OOS data)"
            return self
        methods = ["platt", "isotonic"] if self.method == "auto" else [self.method]
        # choose method on a time-ordered split of the OOS predictions
        cut = int(len(p) * 0.6)
        best, best_m = brier(y[cut:], p[cut:]), "none"
        for meth in methods:
            mdl = self._fit_one(meth, p[:cut], y[:cut])
            b = brier(y[cut:], self._apply(mdl, meth, p[cut:]))
            if b < best - 1e-5:
                best, best_m = b, meth
        self.chosen_ = best_m
        if best_m != "none":
            self.model_ = self._fit_one(best_m, p, y)
        return self

    @staticmethod
    def _fit_one(method, p, y):
        if method == "platt":
            from sklearn.linear_model import LogisticRegression

            lp = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
            return LogisticRegression(C=1.0).fit(lp.reshape(-1, 1), y.astype(int))
        from sklearn.isotonic import IsotonicRegression

        return IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99).fit(p, y)

    @staticmethod
    def _apply(model, method, p):
        p = np.asarray(p, float)
        if method == "platt":
            lp = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
            return model.predict_proba(lp.reshape(-1, 1))[:, 1]
        return model.predict(p)

    def transform(self, p) -> np.ndarray:
        if self.model_ is None:
            return np.asarray(p, float)
        return np.clip(self._apply(self.model_, self.chosen_, p), 0.01, 0.99)


def calibration_report(y, p_raw, p_cal) -> dict:
    return {"raw": {"brier": brier(y, p_raw), "log_loss": log_loss(y, p_raw), "ece": ece(y, p_raw)},
            "calibrated": {"brier": brier(y, p_cal), "log_loss": log_loss(y, p_cal), "ece": ece(y, p_cal)}}


class ConformalRegressor:
    """Split / rolling conformal intervals on forecast residuals.

    * ``absolute``  : symmetric interval  point +/- q(|resid|)
    * ``quantile``  : asymmetric interval using empirical residual quantiles (better for skew)
    Only the most recent ``window`` OOS residuals are used, so the interval adapts to regime.
    The empirical residual distribution also serves as the predictive distribution for
    threshold probabilities (P(ret>5%) etc.).
    """

    def __init__(self, alpha: float = 0.2, window: int = 750, method: str = "quantile"):
        self.alpha, self.window, self.method = alpha, window, method
        self.resid_: np.ndarray = np.array([])

    def fit(self, y_true, y_pred) -> "ConformalRegressor":
        r = (np.asarray(y_true, float) - np.asarray(y_pred, float))
        r = r[~np.isnan(r)]
        self.resid_ = r[-self.window:]
        return self

    @property
    def n(self) -> int:
        return len(self.resid_)

    def interval(self, point: float | np.ndarray, alpha: float | None = None) -> tuple:
        a = self.alpha if alpha is None else alpha
        if self.n < 30:
            return (np.nan * np.asarray(point), np.nan * np.asarray(point))
        n = self.n
        if self.method == "absolute":
            q = np.quantile(np.abs(self.resid_), min(1.0, np.ceil((n + 1) * (1 - a)) / n))
            return point - q, point + q
        lo = np.quantile(self.resid_, max(0.0, np.floor((n + 1) * (a / 2)) / n))
        hi = np.quantile(self.resid_, min(1.0, np.ceil((n + 1) * (1 - a / 2)) / n))
        return point + lo, point + hi

    def prob_above(self, point: float, thr: float) -> float:
        if self.n < 30:
            return np.nan
        return float(np.mean(point + self.resid_ > thr))

    def prob_below(self, point: float, thr: float) -> float:
        if self.n < 30:
            return np.nan
        return float(np.mean(point + self.resid_ < thr))

    def quantiles(self, point: float, qs=(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)) -> dict:
        if self.n < 30:
            return {}
        return {q: float(point + np.quantile(self.resid_, q)) for q in qs}


def rolling_conformal_backtest(y_true: pd.Series, y_pred: pd.Series, alpha: float = 0.2, window: int = 500,
                               min_n: int = 100, gap: int = 0) -> pd.DataFrame:
    """Evaluate conformal coverage honestly: the interval at t uses only residuals whose
    labels were known at t (index position <= t - gap)."""
    yt, yp = y_true.values, y_pred.values
    res = yt - yp
    lo = np.full(len(yt), np.nan)
    hi = np.full(len(yt), np.nan)
    for t in range(len(yt)):
        end = t - gap
        if end < min_n:
            continue
        r = res[max(0, end - window): end]
        r = r[~np.isnan(r)]
        if len(r) < min_n:
            continue
        lo[t] = yp[t] + np.quantile(r, alpha / 2)
        hi[t] = yp[t] + np.quantile(r, 1 - alpha / 2)
    return pd.DataFrame({"y": yt, "pred": yp, "lo": lo, "hi": hi}, index=y_true.index)
