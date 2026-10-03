"""Walk-forward strategy backtest (BacktestService).

Historical signals are produced by models retrained every ``retrain_every`` sessions on
data available at that time only (expanding window, purged by the horizon). Each
historical decision uses the same decision rule (thresholds + hysteresis) on a
reduced component set that is reconstructable historically: ML probability, trend,
momentum, volatility and regime. News/valuation are excluded from the historical
simulation because they are not available point-in-time - this is reported.

The LEAKAGE AUDIT runs first; if any check FAILS the backtest is stopped.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config.settings import Settings
from ..features.builder import FeatureSet
from ..features.selection import select_features
from ..features.technical.indicators import technical_features
from ..labeling.labels import build_labels
from ..models.tree.models import HistGBForecaster, LGBMForecaster
from ..signals.components import momentum_component, trend_component, volatility_component
from ..validation.leakage import LeakageAudit, LeakageError, run_audit
from ..validation.metrics import block_bootstrap_ci, trading_metrics
from .engine import BacktestResult, run_backtest, signal_to_exposure


@dataclass
class WalkForwardBacktest:
    result: BacktestResult | None
    signals: pd.DataFrame
    audit: LeakageAudit
    robustness: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


def historical_signals(fs: FeatureSet, target: pd.DataFrame, regimes: pd.DataFrame, s: Settings, horizon: int = 20,
                       start: str | None = None, retrain_every: int | None = None, model: str = "lightgbm",
                       thresholds: tuple[float, float] | None = None, progress=None) -> pd.DataFrame:
    X = fs.frame[fs.model_columns()].copy()
    for c in ("regime_score", "hmm_stress_prob", "rv_pct"):
        if c in regimes:
            X[c] = regimes[c].reindex(X.index)
    lab = build_labels(target, [horizon])[horizon]
    y_ret, y_up, le = lab["fwd_ret"], lab["direction"], lab["label_end"]
    retrain_every = retrain_every or s.models.retrain_every_days
    idx = X.index
    first = idx.searchsorted(pd.Timestamp(start)) if start else max(1260, int(len(idx) * 0.4))
    cls = LGBMForecaster if model == "lightgbm" and LGBMForecaster.info.available else HistGBForecaster
    buy, sell = thresholds or (s.signals.buy_threshold, s.signals.sell_threshold)
    rows, m, cols = [], None, None
    prev_sig, dur = None, 0
    n_steps = len(idx) - first
    for k, i in enumerate(range(first, len(idx))):
        d = idx[i]
        if m is None or (i - first) % retrain_every == 0:
            trainable = (le.iloc[:i] < d).values & y_ret.iloc[:i].notna().values  # label known before d
            Xi = X.iloc[:i][trainable].iloc[252:]
            cols, _ = select_features(Xi, y_ret.loc[Xi.index], 30, s.models.random_seed, use_stability=False)
            cols = cols + [c for c in ("regime_score", "hmm_stress_prob") if c in X and c not in cols]
            m = cls(horizon, s.models.random_seed).fit(Xi[cols], y_ret.loc[Xi.index], y_up.loc[Xi.index])
            if progress:
                progress(f"Walk-forward backtest: retrained at {d.date()}", k / max(1, n_steps))
        p = m.predict(X.iloc[[i]][cols])
        fr = fs.frame.iloc[i]
        reg = {"vol_percentile": regimes["rv_pct"].iloc[i] if "rv_pct" in regimes else np.nan,
               "score": regimes["regime_score"].iloc[i] if "regime_score" in regimes else 50}
        comps = {"ml": 50 + 50 * np.tanh((p["p_up"].iloc[0] - 0.5) * 2 / 0.25), "trend": trend_component(fr).score,
                 "momentum": momentum_component(fr).score, "volatility": volatility_component(fr, reg).score,
                 "regime": reg["score"] if np.isfinite(reg["score"]) else 50}
        w = {"ml": 1.2, "trend": 1.2, "momentum": 1.0, "volatility": 0.9, "regime": 1.0}
        vals = {k2: v for k2, v in comps.items() if v is not None and np.isfinite(v)}
        score = sum(w[k2] * v for k2, v in vals.items()) / sum(w[k2] for k2 in vals)
        h = s.signals.hysteresis
        if prev_sig == "BUY":
            sig = "BUY" if score >= buy - h else ("SELL" if score <= sell else "HOLD")
        elif prev_sig == "SELL":
            sig = "SELL" if score <= sell + h else ("BUY" if score >= buy else "HOLD")
        else:
            sig = "BUY" if score >= buy else ("SELL" if score <= sell else "HOLD")
        dur = dur + 1 if sig == prev_sig else 1
        prev_sig = sig
        rows.append({"date": d, "score": score, "signal": sig, "p_up": float(p["p_up"].iloc[0]),
                     "exp_ret": float(p["ret"].iloc[0]), **{f"c_{k2}": v for k2, v in comps.items()}})
    return pd.DataFrame(rows).set_index("date")


def run_walkforward_backtest(fs: FeatureSet, target: pd.DataFrame, regimes: pd.DataFrame, s: Settings, pit=None,
                             horizon: int = 20, start: str | None = None, progress=None,
                             robustness: bool = True, retrain_every: int = 63) -> WalkForwardBacktest:
    X = fs.frame[fs.model_columns()]
    lab = build_labels(target, [horizon])[horizon]
    audit = run_audit(X.iloc[:-horizon - 1], lab["fwd_ret"].iloc[:-horizon - 1], target["close"], X.index[-1],
                      pit=pit, decision_times=fs.decision_times if pit is not None else None,
                      causality_fn=technical_features,
                      causality_data=target[["open", "high", "low", "close", "volume"]])
    if not audit.passed:
        raise LeakageError(audit)
    sig = historical_signals(fs, target, regimes, s, horizon, start, retrain_every=retrain_every, progress=progress)
    expo = signal_to_exposure(sig["signal"])
    px = target.loc[sig.index[0]:]
    res = run_backtest(px, expo, s.costs)
    out = WalkForwardBacktest(res, sig, audit)
    out.notes.append("Historical decision rule uses ML + trend + momentum + volatility + regime components only "
                     "(news, valuation and macro-agent views are not reconstructable point-in-time).")
    if "is_proxy" in target and target["is_proxy"].loc[sig.index[0]:].any():
        out.notes.append("Part of the backtest period uses the S&P 500 (EUR) proxy before VUSA's inception.")
    if robustness:
        out.robustness = robustness_checks(res, sig, px, s)
    return out


def robustness_checks(res: BacktestResult, sig: pd.DataFrame, px: pd.DataFrame, s: Settings) -> dict:
    """Does the strategy survive costs, sub-periods and parameter changes?"""
    from copy import deepcopy

    r = res.equity.pct_change().dropna()
    b = res.benchmark.pct_change().dropna()
    excess = (r - b.reindex(r.index)).dropna()
    out = {"excess_return_ci90_daily": block_bootstrap_ci(excess.values, np.mean, 500, 20, 0.1)}
    # sub-periods
    halves = np.array_split(r.index, 3)
    out["sub_periods"] = {f"{p[0].date()}..{p[-1].date()}": {"strategy": trading_metrics(r.loc[p]).get("sharpe"),
                                                            "buy_hold": trading_metrics(b.loc[p]).get("sharpe")}
                          for p in halves if len(p) > 60}
    # cost sensitivity
    cost_sens = {}
    for mult in (0.0, 2.0, 4.0):
        c = deepcopy(s.costs)
        c.spread_bps *= mult
        c.slippage_bps *= mult
        c.commission_pct *= mult
        c.commission_min *= mult
        rr = run_backtest(px, signal_to_exposure(sig["signal"]), c)
        cost_sens[f"costs x{mult:g}"] = {"cagr": rr.metrics.get("cagr"), "sharpe": rr.metrics.get("sharpe")}
    out["cost_sensitivity"] = cost_sens
    # threshold sensitivity (re-map scores; no retraining needed)
    thr = {}
    for db in (-5, 5):
        bthr, sthr = s.signals.buy_threshold + db, s.signals.sell_threshold + db
        alt = np.where(sig["score"] >= bthr, "BUY", np.where(sig["score"] <= sthr, "SELL", "HOLD"))
        rr = run_backtest(px, signal_to_exposure(pd.Series(alt, index=sig.index)), s.costs)
        thr[f"thresholds {bthr:.0f}/{sthr:.0f}"] = {"cagr": rr.metrics.get("cagr"), "sharpe": rr.metrics.get("sharpe"),
                                                   "max_dd": rr.metrics.get("max_drawdown")}
    out["threshold_sensitivity"] = thr
    out["beats_buy_and_hold_sharpe"] = bool((res.metrics.get("sharpe") or -9) > (res.benchmark_metrics.get("sharpe") or 9))
    out["beats_buy_and_hold_return"] = bool((res.metrics.get("cagr") or -9) > (res.benchmark_metrics.get("cagr") or 9))
    return out
