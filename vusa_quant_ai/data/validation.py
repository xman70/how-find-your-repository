"""Data-quality validation for OHLCV frames.

Checks are reported as structured issues with a severity. Severe issues block the
pipeline (the GUI shows them), minor ones lower the data-quality score. Prices are never
"repaired" with invented values: invalid rows are dropped and reported.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .calendar import trading_days


@dataclass
class ValidationReport:
    dataset: str
    rows_in: int
    rows_out: int = 0
    issues: list[dict] = field(default_factory=list)
    score: float = 100.0

    def add(self, check: str, severity: str, detail: str, count: int = 0) -> None:
        self.issues.append({"check": check, "severity": severity, "detail": detail, "count": int(count)})
        self.score -= {"critical": 40, "major": 12, "minor": 3, "info": 0}[severity]
        self.score = max(0.0, self.score)

    @property
    def ok(self) -> bool:
        return not any(i["severity"] == "critical" for i in self.issues)

    def summary(self) -> str:
        if not self.issues:
            return f"{self.dataset}: all checks passed ({self.rows_out} rows)"
        return f"{self.dataset}: score {self.score:.0f}/100; " + "; ".join(
            f"[{i['severity']}] {i['check']}: {i['detail']}" for i in self.issues)


def validate_ohlcv(df: pd.DataFrame, dataset: str, exchange: str = "XETRA", as_of: pd.Timestamp | None = None,
                   max_abs_return: float = 0.25, stale_days: int = 5, check_calendar: bool = True
                   ) -> tuple[pd.DataFrame, ValidationReport]:
    rep = ValidationReport(dataset, len(df) if df is not None else 0)
    if df is None or df.empty:
        rep.add("non_empty", "critical", "no rows")
        return pd.DataFrame(), rep
    d = df.copy().sort_index()

    dup = d.index.duplicated(keep="last")
    if dup.any():
        rep.add("duplicates", "minor", "duplicate dates removed (kept last)", dup.sum())
        d = d[~dup]

    if as_of is not None:
        future = d.index > pd.Timestamp(as_of).normalize()
        if future.any():
            rep.add("future_dates", "critical", f"rows dated after as-of {pd.Timestamp(as_of).date()} removed", future.sum())
            d = d[~future]

    bad_price = (d["close"] <= 0) | d["close"].isna()
    if bad_price.any():
        rep.add("non_positive_close", "major", "rows with missing/non-positive close dropped", bad_price.sum())
        d = d[~bad_price]

    if {"open", "high", "low"}.issubset(d.columns):
        o, h, l, c = d["open"], d["high"], d["low"], d["close"]
        incons = (h < l) | (h < np.fmax(o, c) * (1 - 1e-6)) | (l > np.fmin(o, c) * (1 + 1e-6))
        incons &= o.notna() & h.notna() & l.notna()
        if incons.any():
            rep.add("ohlc_consistency", "minor", "high/low inconsistent with open/close; high/low reset to envelope",
                    incons.sum())
            d.loc[incons, "high"] = np.fmax.reduce([o[incons], h[incons], c[incons]])
            d.loc[incons, "low"] = np.fmin.reduce([o[incons], l[incons], c[incons]])
        missing_ohl = d[["open", "high", "low"]].isna().any(axis=1)
        if missing_ohl.any():
            rep.add("missing_ohl", "info", "rows without open/high/low (close-only); range features use NaN",
                    missing_ohl.sum())

    r = np.log(d["close"]).diff()
    spikes = r.abs() > max_abs_return
    if spikes.any():
        # a spike immediately reversed is almost certainly a bad tick
        rev = spikes & (r.shift(-1) * r < 0) & (r.shift(-1).abs() > max_abs_return * 0.8)
        if rev.any():
            rep.add("bad_tick", "major", "single-day spike reversed next day; row dropped", rev.sum())
            d = d[~rev]
        if (spikes & ~rev).any():
            rep.add("large_move", "info", f"|daily log return| > {max_abs_return:.0%} retained (possibly genuine)",
                    (spikes & ~rev).sum())

    if "volume" in d.columns:
        zero_vol = (d["volume"].fillna(0) <= 0).mean()
        if zero_vol > 0.2:
            rep.add("volume", "info", f"{zero_vol:.0%} of rows without volume (index or illiquid data)")

    flat = (d["close"].diff() == 0).rolling(5).sum() >= 5
    if flat.any():
        rep.add("flat_prices", "minor", "5+ consecutive unchanged closes (stale feed?)", flat.sum())

    if check_calendar and len(d) > 20:
        expected = trading_days(d.index[0], d.index[-1], exchange)
        missing = expected.difference(d.index)
        if len(missing) > 0:
            frac = len(missing) / len(expected)
            sev = "major" if frac > 0.05 else ("minor" if frac > 0.005 else "info")
            rep.add("calendar_gaps", sev, f"{len(missing)} expected sessions missing ({frac:.1%}); not filled", len(missing))
        extra = d.index.difference(expected)
        if len(extra) > 0:
            rep.add("non_session_rows", "info", f"{len(extra)} rows on non-session days for {exchange} calendar",
                    len(extra))

    if as_of is not None and len(d):
        lag = len(trading_days(d.index[-1], as_of, exchange)) - 1
        if lag > stale_days:
            rep.add("staleness", "major", f"last row {d.index[-1].date()} is {lag} sessions old")
        elif lag > 1:
            rep.add("staleness", "minor", f"last row {d.index[-1].date()} is {lag} sessions old")

    rep.rows_out = len(d)
    if len(d) < 300:
        rep.add("history_length", "major", f"only {len(d)} rows - long-horizon models will be unreliable or skipped")
    return d, rep
