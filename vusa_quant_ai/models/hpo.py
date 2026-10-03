"""Nested hyper-parameter optimisation (Optuna TPE; randomized search fallback).

Called with the *outer* training window only; candidate parameters are scored on inner
purged walk-forward splits of that window, so the outer test fold never influences
tuning. The objective is robust, not raw accuracy:

    score = rank IC + (directional accuracy - base rate) - 2 * (Brier - 0.25)
            - 0.5 * std(IC across inner folds)            # stability penalty
            - 0.002 * complexity                         # leaves x trees / 1000
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd

from ..validation.metrics import brier, regression_metrics
from ..validation.splits import walk_forward
from .tree.models import LGBMForecaster, XGBForecaster

SPACES = {
    "lightgbm": lambda t: {"num_leaves": t.suggest_int("num_leaves", 4, 31), "learning_rate": t.suggest_float("learning_rate", 0.01, 0.1, log=True),
                           "n_estimators": t.suggest_int("n_estimators", 100, 500, step=50),
                           "min_child_samples": t.suggest_int("min_child_samples", 20, 200),
                           "colsample_bytree": t.suggest_float("colsample_bytree", 0.3, 0.9),
                           "reg_lambda": t.suggest_float("reg_lambda", 0.1, 20, log=True)},
    "xgboost": lambda t: {"max_depth": t.suggest_int("max_depth", 2, 5), "learning_rate": t.suggest_float("learning_rate", 0.01, 0.1, log=True),
                          "n_estimators": t.suggest_int("n_estimators", 100, 500, step=50),
                          "min_child_weight": t.suggest_int("min_child_weight", 5, 100),
                          "colsample_bytree": t.suggest_float("colsample_bytree", 0.3, 0.9),
                          "reg_lambda": t.suggest_float("reg_lambda", 0.1, 20, log=True)},
}
CLASSES = {"lightgbm": LGBMForecaster, "xgboost": XGBForecaster}


def _complexity(name: str, hp: dict) -> float:
    if name == "lightgbm":
        return hp.get("num_leaves", 8) * hp.get("n_estimators", 300) / 1000
    return (2 ** hp.get("max_depth", 3)) * hp.get("n_estimators", 300) / 1000


def tune(name: str, X: pd.DataFrame, y_ret: pd.Series, y_up: pd.Series, label_end: pd.Series, horizon: int,
         n_trials: int = 15, seed: int = 42, inner_splits: int = 3) -> dict:
    if name not in CLASSES or not CLASSES[name].info.available:
        return {"params": {}, "note": f"HPO unavailable for {name}"}
    le = label_end.where(label_end <= X.index[-1])
    splits = list(walk_forward(X.index, le, inner_splits, max(300, int(le.notna().sum() * 0.5)), embargo=max(5, horizon)))
    if not splits:
        return {"params": {}, "note": "not enough data for inner folds"}

    def evaluate(hp: dict) -> float:
        ics, das, bs = [], [], []
        for sp in splits:
            m = CLASSES[name](horizon, seed, **hp).fit(X.iloc[sp.train], y_ret.iloc[sp.train], y_up.iloc[sp.train])
            p = m.predict(X.iloc[sp.test])
            rm = regression_metrics(y_ret.iloc[sp.test], p["ret"])
            ics.append(rm.get("ic_spearman", 0) or 0)
            das.append((rm.get("directional_accuracy", 0.5) or 0.5) - float((y_ret.iloc[sp.test] > 0).mean()))
            bs.append(brier(y_up.iloc[sp.test], p["p_up"]))
        ics = np.nan_to_num(ics, nan=0.0)  # constant predictions -> undefined IC -> no skill
        val = float(np.mean(ics) + np.nanmean(das) - 2 * (np.nanmean(bs) - 0.25) - 0.5 * np.std(ics)
                    - 0.002 * _complexity(name, hp))
        return val if np.isfinite(val) else -1.0

    if importlib.util.find_spec("optuna") is not None:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(lambda t: evaluate(SPACES[name](t)), n_trials=n_trials, show_progress_bar=False)
        return {"params": study.best_params, "score": study.best_value, "method": "optuna-TPE", "n_trials": n_trials}
    # randomized-search fallback
    rng = np.random.default_rng(seed)
    best, best_s = {}, -np.inf

    class _T:
        def suggest_int(self, _, lo, hi, step=1):
            return int(rng.integers(lo // step, hi // step + 1) * step)

        def suggest_float(self, _, lo, hi, log=False):
            return float(np.exp(rng.uniform(np.log(lo), np.log(hi))) if log else rng.uniform(lo, hi))

    for _ in range(n_trials):
        hp = SPACES[name](_T())
        sc = evaluate(hp)
        if sc > best_s:
            best, best_s = hp, sc
    return {"params": best, "score": best_s, "method": "randomized search", "n_trials": n_trials}
