"""Ensembles and the stacking meta-model.

Inputs are *out-of-sample* base-model predictions from walk-forward validation. Every
ensemble weight / meta-model is estimated only from OOS predictions available before
the period it is evaluated on (expanding, purged by ``gap`` = horizon), so ensemble
comparisons are truly out-of-sample.

Methods compared: single best model, equal weight, performance weighted (inverse MSE),
stacked meta-model (ridge on base forecasts x regime features, learns *when* each model
works), Bayesian model averaging (weights proportional to OOS predictive likelihood).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...validation.metrics import brier, regression_metrics

METHODS = ["best_single", "equal_weight", "performance_weighted", "stacked", "bayesian"]


def _weights_perf(err: pd.DataFrame) -> pd.Series:
    mse = (err ** 2).mean()
    w = 1 / mse.replace(0, np.nan)
    return (w / w.sum()).fillna(0)


def _weights_bma(err: pd.DataFrame, sigma: pd.Series) -> pd.Series:
    # Gaussian predictive log-likelihood per model, tempered by sample size to avoid winner-takes-all
    ll = (-0.5 * (err ** 2) / (sigma ** 2) - np.log(sigma)).mean()
    ll = ll - ll.max()
    w = np.exp(ll * min(50, len(err)) / 10)
    return w / w.sum()


class StackingMetaModel:
    """Ridge regression on base forecasts, augmented with interactions between each base
    forecast and regime/state variables, so weights can vary by market regime."""

    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha

    def _design(self, P: pd.DataFrame, S: pd.DataFrame | None):
        cols = [P]
        if S is not None and not S.empty:
            Sz = (S - self.s_mu_) / self.s_sd_
            for sc in Sz.columns:
                cols.append(P.mul(Sz[sc].values, axis=0).add_suffix(f"*{sc}"))
        return pd.concat(cols, axis=1).fillna(0.0)

    def fit(self, P: pd.DataFrame, y: pd.Series, S: pd.DataFrame | None = None, task: str = "reg"):
        from sklearn.linear_model import LogisticRegression, Ridge

        self.task = task
        if S is not None and not S.empty:
            self.s_mu_, self.s_sd_ = S.mean(), S.std().replace(0, 1)
        D = self._design(P, S)
        m = y.notna().values
        if task == "reg":
            self.model_ = Ridge(alpha=self.alpha, fit_intercept=True).fit(D.values[m], y.values[m])
        else:
            lp = np.log(np.clip(D.values, 1e-3, 1 - 1e-3) / (1 - np.clip(D.values, 1e-3, 1 - 1e-3))) \
                if not any("*" in c for c in D.columns) else D.values
            self.model_ = LogisticRegression(C=1 / self.alpha).fit(lp[m], y.values[m].astype(int))
        self.cols_ = list(D.columns)
        return self

    def predict(self, P: pd.DataFrame, S: pd.DataFrame | None = None) -> np.ndarray:
        D = self._design(P, S)[self.cols_]
        if self.task == "reg":
            return self.model_.predict(D.values)
        lp = np.log(np.clip(D.values, 1e-3, 1 - 1e-3) / (1 - np.clip(D.values, 1e-3, 1 - 1e-3))) \
            if not any("*" in c for c in D.columns) else D.values
        return self.model_.predict_proba(lp)[:, 1]

    def base_weights(self) -> pd.Series:
        coef = pd.Series(np.ravel(self.model_.coef_), index=self.cols_)
        return coef[[c for c in self.cols_ if "*" not in c]]


def combine(method: str, P_ret: pd.DataFrame, P_up: pd.DataFrame, hist_err: pd.DataFrame | None,
            hist_y_ret: pd.Series | None = None, hist_P_ret: pd.DataFrame | None = None,
            hist_P_up: pd.DataFrame | None = None, hist_y_up: pd.Series | None = None,
            S_hist: pd.DataFrame | None = None, S_now: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Combine base forecasts for the rows of P_ret/P_up using only historical OOS info."""
    models = list(P_ret.columns)
    if method == "equal_weight" or hist_err is None or len(hist_err) < 60:
        w = pd.Series(1 / len(models), index=models)
        return P_ret.values @ w.values, P_up.values @ w.values, {"weights": w.to_dict(), "fallback": method != "equal_weight"}
    if method == "best_single":
        best = (hist_err ** 2).mean().idxmin()
        w = pd.Series(0.0, index=models)
        w[best] = 1.0
        return P_ret[best].values, P_up[best].values, {"weights": w.to_dict(), "best": best}
    if method == "performance_weighted":
        w = _weights_perf(hist_err)
        return P_ret.values @ w.values, P_up.values @ w.values, {"weights": w.to_dict()}
    if method == "bayesian":
        sig = hist_err.std().replace(0, np.nan).fillna(hist_err.std().mean())
        w = _weights_bma(hist_err, sig)
        return P_ret.values @ w.values, P_up.values @ w.values, {"weights": w.to_dict()}
    if method == "stacked":
        mr = StackingMetaModel().fit(hist_P_ret, hist_y_ret, S_hist, "reg")
        try:
            mc = StackingMetaModel(alpha=10.0).fit(hist_P_up, hist_y_up, None, "clf")
            pu = mc.predict(P_up)
        except Exception:
            pu = P_up.mean(axis=1).values
        w = mr.base_weights()
        return mr.predict(P_ret, S_now), pu, {"weights": (w / w.abs().sum()).to_dict() if w.abs().sum() else w.to_dict()}
    raise ValueError(method)


def walk_forward_ensembles(oos_ret: pd.DataFrame, oos_up: pd.DataFrame, y_ret: pd.Series, y_up: pd.Series,
                           label_end: pd.Series, state: pd.DataFrame | None = None, step: int = 63,
                           min_hist: int = 250) -> dict:
    """Evaluate every ensemble method out-of-sample on the OOS base predictions.

    At each block start t, weights are estimated from OOS rows whose labels had ended
    before t (no overlap), then applied to the next ``step`` rows."""
    idx = oos_ret.dropna(how="all").index
    oos_ret, oos_up = oos_ret.loc[idx].ffill(axis=1).bfill(axis=1), oos_up.loc[idx].ffill(axis=1).bfill(axis=1)
    y_ret, y_up, le = y_ret.reindex(idx), y_up.reindex(idx), label_end.reindex(idx)
    S = state.reindex(idx) if state is not None else None
    out = {m: (pd.Series(np.nan, index=idx), pd.Series(np.nan, index=idx)) for m in METHODS}
    for start in range(min_hist, len(idx), step):
        t0 = idx[start]
        hist = (le < t0).values & (np.arange(len(idx)) < start)
        if hist.sum() < min_hist // 2:
            continue
        blk = slice(start, min(len(idx), start + step))
        Pr, Pu = oos_ret.iloc[blk], oos_up.iloc[blk]
        err = oos_ret[hist].sub(y_ret[hist], axis=0)
        for m in METHODS:
            try:
                r, p, _ = combine(m, Pr, Pu, err, y_ret[hist], oos_ret[hist], oos_up[hist], y_up[hist],
                                  None if S is None else S[hist].fillna(0), None if S is None else S.iloc[blk].fillna(0))
            except Exception:
                r, p = Pr.mean(axis=1).values, Pu.mean(axis=1).values
            out[m][0].iloc[blk] = r
            out[m][1].iloc[blk] = np.clip(p, 0.01, 0.99)
    res = {}
    for m, (r, p) in out.items():
        ok = r.notna()
        res[m] = {**regression_metrics(y_ret[ok], r[ok]), "brier": brier(y_up[ok], p[ok]),
                  "pred_ret": r, "pred_up": p}
    return res
