"""Learned traffic-light model (walk-forward) and honest blend selection.

The hand-set light weights are a reasonable prior, but nothing guarantees they are the best
way to combine the evidence. Here a strongly regularised logistic regression learns, from
history, how the reconstructable evidence components relate to the probability that the next
period's return is BETTER THAN NORMAL (above the median of the training period).

Leakage control
---------------
* Rows are sampled every ``step`` sessions; a training row is used only if its outcome window
  (h sessions) ended before the prediction date (purge gap = ceil(h/step)+1 rows).
* The model is refitted on an expanding window every ``refit_rows`` rows; predictions for a
  block always come from a model fitted before that block.

Selection
---------
Three candidates are compared by ranking skill (AUC: probability that a better-than-normal
period got a higher score than a worse one; 0.5 = no skill): the hand-set light (alpha=0), a
50/50 blend and a 75 % learned blend. The choice is made on the FIRST two-thirds of the
out-of-sample history only; the LAST third is reported as an untouched hold-out. The learned
model must beat the hand-set light by a margin, otherwise the simpler light is kept.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ALPHAS = (0.0, 0.5, 0.75)
MIN_GAIN = 0.01        # AUC points the learned blend must add in EACH half of the selection period
MIN_INDEPENDENT = 40   # independent (non-overlapping) outcome periods needed before learning is trusted


def learned_score(p) -> np.ndarray:
    """Probability of a better-than-normal period -> 0..100 light score (p=0.55 ~ 62, p=0.45 ~ 38)."""
    return 50 + 50 * np.tanh((np.asarray(p, float) - 0.5) / 0.2)


def _design(C: pd.DataFrame) -> np.ndarray:
    return ((C - 50.0) / 25.0).fillna(0.0).to_numpy()


def _fit(X: np.ndarray, yb: np.ndarray, C_reg: float):
    from sklearn.linear_model import LogisticRegression

    if len(np.unique(yb)) < 2:
        return None
    return LogisticRegression(C=C_reg, max_iter=500).fit(X, yb)


def walk_forward(C: pd.DataFrame, y: pd.Series, h: int, step: int, refit_rows: int = 50, min_train_rows: int = 150,
                 C_reg: float = 0.1) -> tuple[pd.Series, object, dict]:
    """Out-of-sample P(better than normal) for every row (NaN while there is too little history),
    plus a final model fitted on all rows whose outcome is known."""
    C = C.reindex(y.index)
    X = _design(C)
    yv = y.to_numpy(float)
    n = len(C)
    gap = int(np.ceil(h / max(step, 1))) + 1
    p = np.full(n, np.nan)
    for start in range(min_train_rows + gap, n, refit_rows):
        tr = np.arange(0, start - gap)
        tr = tr[np.isfinite(yv[tr])]
        if len(tr) < min_train_rows:
            continue
        med = np.median(yv[tr])
        m = _fit(X[tr], (yv[tr] > med).astype(int), C_reg)
        if m is None:
            continue
        end = min(n, start + refit_rows)
        p[start:end] = m.predict_proba(X[start:end])[:, 1]
    known = np.isfinite(yv)
    final = _fit(X[known], (yv[known] > np.median(yv[known])).astype(int), C_reg) if known.sum() >= min_train_rows else None
    coefs = {} if final is None else dict(zip(C.columns, np.round(final.coef_[0], 3)))
    return pd.Series(p, index=C.index), final, coefs


def auc(score: pd.Series, y: pd.Series) -> float:
    """Ranking skill of ``score`` for 'outcome above its median' (Mann-Whitney AUC)."""
    d = pd.DataFrame({"s": score, "y": y}).dropna()
    if len(d) < 30:
        return np.nan
    lab = d["y"] > d["y"].median()
    if lab.all() or not lab.any():
        return np.nan
    r = d["s"].rank()
    n1, n0 = lab.sum(), (~lab).sum()
    return float((r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def select_blend(heur: pd.Series, learned: pd.Series, y: pd.Series, h: int = 20, step: int = 5) -> dict:
    """Choose alpha on the first 2/3 of the period where the learned model exists; report the last 1/3.

    The learned blend is adopted only if (a) there are enough independent outcome periods to learn
    from, and (b) it beats the hand-set light by MIN_GAIN in BOTH halves of the selection period
    (stability), not just on average. The hold-out third is never used for the decision."""
    ok = learned.notna() & heur.notna() & y.notna()
    idx = heur.index[ok]
    n_indep = int(len(idx) * step // max(h, 1))
    if len(idx) < 90:
        return {"alpha": 0.0, "reason": f"only {len(idx)} out-of-sample rows for the learned model - hand-set light kept",
                "candidates": {}, "independent_periods": n_indep}
    cut = idx[int(len(idx) * 2 / 3)]
    sel, hold = idx[idx < cut], idx[idx >= cut]
    s1, s2 = sel[: len(sel) // 2], sel[len(sel) // 2:]
    cand = {}
    for a in ALPHAS:
        s = (1 - a) * heur + a * learned
        cand[a] = {"selection_auc": auc(s.loc[sel], y.loc[sel]), "selection_auc_first_half": auc(s.loc[s1], y.loc[s1]),
                   "selection_auc_second_half": auc(s.loc[s2], y.loc[s2]), "holdout_auc": auc(s.loc[hold], y.loc[hold]),
                   "all_auc": auc(s.loc[idx], y.loc[idx])}
    base = cand[0.0]

    def stable_gain(a):
        c = cand[a]
        vals = [c["selection_auc_first_half"] - base["selection_auc_first_half"],
                c["selection_auc_second_half"] - base["selection_auc_second_half"]]
        return min(vals) if all(np.isfinite(v) for v in vals) else -np.inf

    best = max((a for a in ALPHAS if a > 0), key=stable_gain)
    if n_indep < MIN_INDEPENDENT:
        best, why = 0.0, (f"only ~{n_indep} independent {h}-day outcome periods - too few to learn weights reliably; "
                          "hand-set light kept")
    elif stable_gain(best) < MIN_GAIN:
        best, why = 0.0, "learned model did not beat the hand-set light consistently on the selection period"
    else:
        why = (f"learned blend improved ranking skill in both halves of the selection period "
               f"({base['selection_auc']:.3f} -> {cand[best]['selection_auc']:.3f})")
    return {"alpha": float(best), "reason": why, "independent_periods": n_indep,
            "selection_period": [str(sel[0].date()), str(sel[-1].date())],
            "holdout_period": [str(hold[0].date()), str(hold[-1].date())],
            "candidates": {str(a): v for a, v in cand.items()},
            "holdout_auc_handset": base["holdout_auc"], "holdout_auc_chosen": cand[best]["holdout_auc"]}
