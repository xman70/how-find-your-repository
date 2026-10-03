"""Feature selection - always performed inside the training window of each fold.

Pipeline: coverage filter -> correlation filter -> relevance ranking (mutual information
+ L1 stability selection + tree importance, rank-averaged) -> top-K. Stability of the
selected set across folds and regimes is tracked; features selected in only a small
fraction of folds are flagged as unstable.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd


def coverage_filter(X: pd.DataFrame, min_cov: float = 0.8) -> list[str]:
    return [c for c in X.columns if X[c].notna().mean() >= min_cov and X[c].nunique(dropna=True) > 5]


def correlation_filter(X: pd.DataFrame, cols: list[str], thr: float = 0.92, priority: pd.Series | None = None) -> list[str]:
    order = list(priority.reindex(cols).sort_values(ascending=False).index) if priority is not None else cols
    C = X[order].corr(method="spearman").abs().fillna(0)
    keep: list[str] = []
    for c in order:
        if all(C.loc[c, k] < thr for k in keep):
            keep.append(c)
    return keep


def mutual_info_rank(X: pd.DataFrame, y: pd.Series, seed: int = 0) -> pd.Series:
    from sklearn.feature_selection import mutual_info_regression

    Z = X.fillna(X.median())
    m = y.notna().values
    mi = mutual_info_regression(Z.values[m], y.values[m], random_state=seed, n_neighbors=5)
    return pd.Series(mi, index=X.columns)


def stability_selection(X: pd.DataFrame, y: pd.Series, n_boot: int = 20, frac: float = 0.5, seed: int = 0,
                        block: int = 60) -> pd.Series:
    """Randomised-lasso style stability: fraction of block-bootstrap resamples in which a
    feature receives a non-zero L1 coefficient."""
    from sklearn.linear_model import LassoCV

    Z = X.fillna(X.median())
    Z = (Z - Z.mean()) / Z.std().replace(0, 1)
    m = y.notna().values
    Z, yy = Z.values[m], y.values[m]
    n = len(yy)
    rng = np.random.default_rng(seed)
    try:
        alpha = LassoCV(cv=3, n_alphas=20, max_iter=3000, random_state=seed).fit(Z, yy).alpha_
    except Exception:
        alpha = 1e-3
    from sklearn.linear_model import Lasso

    counts = np.zeros(Z.shape[1])
    for _ in range(n_boot):
        nb = max(1, int(n * frac / block))
        starts = rng.integers(0, max(1, n - block), nb)
        ids = np.concatenate([np.arange(s, min(n, s + block)) for s in starts])
        scale = rng.uniform(0.5, 1.0, Z.shape[1])  # randomised lasso
        coef = Lasso(alpha=alpha, max_iter=3000).fit(Z[ids] * scale, yy[ids]).coef_
        counts += np.abs(coef) > 1e-10
    return pd.Series(counts / n_boot, index=X.columns)


def tree_importance(X: pd.DataFrame, y: pd.Series, seed: int = 0) -> pd.Series:
    from sklearn.ensemble import ExtraTreesRegressor

    Z = X.fillna(X.median())
    m = y.notna().values
    et = ExtraTreesRegressor(n_estimators=150, max_depth=5, min_samples_leaf=40, max_features=0.3, n_jobs=-1,
                             random_state=seed).fit(Z.values[m], y.values[m])
    return pd.Series(et.feature_importances_, index=X.columns)


def select_features(X: pd.DataFrame, y: pd.Series, k: int = 40, seed: int = 0, use_stability: bool = True) -> tuple[list[str], pd.DataFrame]:
    cols = coverage_filter(X)
    if not cols:
        return [], pd.DataFrame()
    Xc = X[cols]
    m = y.notna()
    Xc, yy = Xc[m], y[m]
    if len(yy) > 3000:  # speed: subsample evenly in time
        step = len(yy) // 3000 + 1
        Xc, yy = Xc.iloc[::step], yy.iloc[::step]
    scores = pd.DataFrame(index=cols)
    scores["mi"] = mutual_info_rank(Xc, yy, seed)
    scores["tree"] = tree_importance(Xc, yy, seed)
    if use_stability:
        scores["stability"] = stability_selection(Xc, yy, seed=seed)
    ranks = scores.rank(pct=True).mean(axis=1)
    scores["rank_score"] = ranks
    keep = correlation_filter(Xc, cols, priority=ranks)
    sel = list(ranks.reindex(keep).sort_values(ascending=False).index[:k])
    return sel, scores.sort_values("rank_score", ascending=False)


class SelectionTracker:
    """Track which features get selected across folds / horizons / regimes."""

    def __init__(self):
        self.counts: Counter = Counter()
        self.n = 0
        self.by_regime: dict[str, Counter] = {}

    def update(self, selected: list[str], regime: str | None = None) -> None:
        self.n += 1
        self.counts.update(selected)
        if regime:
            self.by_regime.setdefault(regime, Counter()).update(selected)

    def table(self) -> pd.DataFrame:
        if not self.n:
            return pd.DataFrame()
        df = pd.DataFrame({"selected_frac": {k: v / self.n for k, v in self.counts.items()}})
        df["stability"] = pd.cut(df["selected_frac"], [-0.01, 0.34, 0.67, 1.01],
                                 labels=["Unstable", "Moderately stable", "Stable"])
        return df.sort_values("selected_frac", ascending=False)
