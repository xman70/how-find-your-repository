"""Feature importance and its stability.

* Permutation importance on out-of-sample folds (model-agnostic, preferred)
* SHAP values for tree models when the ``shap`` package is installed (else reported
  as unavailable - never approximated under the SHAP name)
* Stability: importance is re-estimated in every walk-forward fold; mean, median,
  volatility (coefficient of variation) and per-regime importance are reported and
  each feature is classed Stable / Moderately stable / Unstable.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd

from ..validation.splits import walk_forward


def permutation_importance_oos(model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 5, seed: int = 0,
                               metric: str = "ic") -> pd.Series:
    rng = np.random.default_rng(seed)
    base_pred = model.predict(X)["ret"].values
    yv = y.values

    def score(p):
        m = ~np.isnan(yv)
        if metric == "ic":
            return pd.Series(p[m]).corr(pd.Series(yv[m]), method="spearman")
        return -np.mean((p[m] - yv[m]) ** 2)

    base = score(base_pred)
    out = {}
    for c in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[c] = rng.permutation(Xp[c].values)
            drops.append(base - score(model.predict(Xp)["ret"].values))
        out[c] = float(np.mean(drops))
    return pd.Series(out).sort_values(ascending=False)


def shap_values(model, X: pd.DataFrame, max_rows: int = 500) -> pd.DataFrame | None:
    if importlib.util.find_spec("shap") is None:
        return None
    import shap

    est = getattr(model, "reg_", None)
    pre = getattr(model, "pre_", None)
    if est is None or pre is None:
        return None
    Z = pre.transform(X.tail(max_rows))
    try:
        expl = shap.TreeExplainer(est)
        sv = expl.shap_values(Z)
    except Exception:
        try:
            expl = shap.Explainer(est.predict, Z[:100])
            sv = expl(Z).values
        except Exception:
            return None
    return pd.DataFrame(np.asarray(sv), index=X.tail(max_rows).index, columns=pre.cols_)


def shap_today(model, x_now: pd.DataFrame) -> pd.Series | None:
    sv = shap_values(model, x_now, 1)
    return None if sv is None else sv.iloc[-1].sort_values(key=np.abs, ascending=False)


def importance_stability(model_factory, X: pd.DataFrame, y_ret: pd.Series, y_up: pd.Series, label_end: pd.Series,
                         regimes: pd.Series | None = None, n_splits: int = 5, seed: int = 0) -> dict:
    """Permutation importance per fold -> stability statistics."""
    rows, by_regime = [], {}
    for sp in walk_forward(X.index, label_end, n_splits, max(500, int(label_end.notna().sum() * 0.4)), embargo=20):
        m = model_factory().fit(X.iloc[sp.train], y_ret.iloc[sp.train], y_up.iloc[sp.train])
        Xt, yt = X.iloc[sp.test], y_ret.iloc[sp.test]
        imp = permutation_importance_oos(m, Xt, yt, n_repeats=2, seed=seed)
        imp.name = sp.fold
        rows.append(imp)
        if regimes is not None:
            rg = regimes.reindex(Xt.index)
            for r in rg.dropna().unique():
                mask = (rg == r).values
                if mask.sum() >= 60:
                    by_regime.setdefault(r, []).append(permutation_importance_oos(m, Xt[mask], yt[mask], 1, seed))
    if not rows:
        return {"table": pd.DataFrame(), "by_regime": {}}
    M = pd.concat(rows, axis=1)
    tab = pd.DataFrame({"mean": M.mean(axis=1), "median": M.median(axis=1), "std": M.std(axis=1),
                        "share_positive": (M > 0).mean(axis=1)})
    tab["cv"] = tab["std"] / tab["mean"].abs().replace(0, np.nan)
    tab["stability"] = np.where((tab["share_positive"] >= 0.8) & (tab["mean"] > 0), "Stable",
                                np.where((tab["share_positive"] >= 0.6) & (tab["mean"] > 0), "Moderately stable", "Unstable"))
    reg_tab = {r: pd.concat(v, axis=1).mean(axis=1).sort_values(ascending=False).head(10).to_dict() for r, v in by_regime.items()}
    return {"table": tab.sort_values("mean", ascending=False), "by_regime": reg_tab, "per_fold": M}
