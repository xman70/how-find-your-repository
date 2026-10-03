"""Realistic event-driven backtester.

* Signals are generated after the close of day t from information known at t.
* Orders execute at the NEXT session's open (configurable delay) - never at the price
  that produced the signal. "close" execution is allowed only when explicitly selected
  (an executable closing-auction order placed before the close) and still uses t+delay.
* Costs: commission (pct with minimum), half bid/ask spread, slippage.
* Position constraints: long-only, max exposure, minimum trade size, whole units.
* Market holidays: only sessions present in the data are tradable.
* Optional taxes on realised gains are user-configured assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config.settings import CostSettings
from ..validation.metrics import trading_metrics


@dataclass
class BacktestResult:
    equity: pd.Series
    benchmark: pd.Series
    exposure: pd.Series
    trades: pd.DataFrame
    metrics: dict
    benchmark_metrics: dict
    costs_paid: float
    taxes_paid: float
    notes: list = field(default_factory=list)


def run_backtest(prices: pd.DataFrame, target_exposure: pd.Series, costs: CostSettings, initial_capital: float = 10000.0,
                 max_exposure: float = 1.0, min_trade_value: float = 50.0, whole_units: bool = True) -> BacktestResult:
    """``target_exposure`` is indexed by SIGNAL date (decision after that day's close)."""
    px = prices[["open", "close"]].copy()
    px["open"] = px["open"].fillna(px["close"])
    idx = px.index
    tgt = target_exposure.reindex(idx).ffill().fillna(0).clip(0, max_exposure)
    delay = max(1, int(costs.execution_delay_days))
    exec_col = "open" if costs.execution_price == "open" else "close"
    cash, units, cost_basis = initial_capital, 0.0, 0.0
    eq, expo, trades = [], [], []
    costs_paid = taxes = 0.0
    spread, slip = costs.spread_bps / 1e4 / 2, costs.slippage_bps / 1e4
    pending: list[tuple[int, float, pd.Timestamp]] = []  # (exec position, target, signal date)
    for i, d in enumerate(idx):
        # 1) execute orders scheduled for this session
        for order in [o for o in pending if o[0] == i]:
            _, w, sig_date = order
            price = px[exec_col].iloc[i]
            value = cash + units * price
            desired_units = w * value / price
            if whole_units:
                desired_units = np.floor(desired_units)
            delta = desired_units - units
            if abs(delta * price) >= min_trade_value:
                side = "BUY" if delta > 0 else "SELL"
                fill = price * (1 + spread + slip) if delta > 0 else price * (1 - spread - slip)
                notional = abs(delta) * fill
                comm = max(costs.commission_min, costs.commission_pct * notional)
                if delta > 0:
                    affordable = (cash - comm) / fill
                    if delta > affordable:
                        delta = np.floor(affordable) if whole_units else affordable
                        notional = abs(delta) * fill
                    if delta <= 0:
                        continue
                    cash -= notional + comm
                    cost_basis += notional + comm
                    units += delta
                else:
                    avg = cost_basis / units if units else 0.0
                    realised = abs(delta) * (fill - avg) - comm
                    tax = max(0.0, realised) * costs.tax_on_realized_gains_pct
                    cash += notional - comm - tax
                    taxes += tax
                    cost_basis -= avg * abs(delta)
                    units += delta
                costs_paid += comm + abs(delta) * price * (spread + slip)
                trades.append({"signal_date": sig_date.date(), "exec_date": d.date(), "side": side, "price": round(fill, 4),
                               "units": abs(delta), "commission": round(comm, 2), "target_exposure": w})
        pending = [o for o in pending if o[0] != i]
        # 2) mark to market at close
        c = px["close"].iloc[i]
        value = cash + units * c
        eq.append(value)
        expo.append(units * c / value if value else 0.0)
        # 3) after the close: new signal -> schedule for next session(s)
        cur_w = units * c / value if value else 0.0
        if abs(tgt.iloc[i] - cur_w) > 0.02 and i + delay < len(idx):
            pending = [o for o in pending if o[0] != i + delay]
            pending.append((i + delay, float(tgt.iloc[i]), d))
    equity = pd.Series(eq, index=idx)
    bench = initial_capital * px["close"] / px["close"].iloc[0]
    res = BacktestResult(equity, bench, pd.Series(expo, index=idx), pd.DataFrame(trades),
                         trading_metrics(equity.pct_change().dropna()), trading_metrics(bench.pct_change().dropna()),
                         costs_paid, taxes)
    res.metrics["n_trades"] = len(trades)
    res.metrics["turnover_per_year"] = float(len(trades) / max(1e-9, len(idx) / 252))
    res.metrics["avg_exposure"] = float(np.mean(expo))
    res.metrics["costs_paid"] = costs_paid
    res.metrics["costs_pct_of_initial"] = costs_paid / initial_capital
    res.notes.append(f"Execution: signal after close -> {exec_col} of session t+{delay}; spread {costs.spread_bps}bps, "
                     f"slippage {costs.slippage_bps}bps, commission {costs.commission_pct:.3%} (min {costs.commission_min}).")
    if costs.tax_on_realized_gains_pct:
        res.notes.append(f"Taxes: {costs.tax_on_realized_gains_pct:.0%} on realised gains (user assumption).")
    return res


def signal_to_exposure(signals: pd.Series, mapping: dict | None = None) -> pd.Series:
    mapping = mapping or {"BUY": 1.0, "HOLD": None, "SELL": 0.0}
    out, cur = [], 1.0
    for s in signals:
        m = mapping.get(s)
        if m is not None:
            cur = m
        out.append(cur)
    return pd.Series(out, index=signals.index)
