"""EVENT-IMPACT MODEL and NEWS MEMORY.

A historical event database is derived from data the application has *point-in-time*
(no hindsight): market events (volatility spikes, large drawdowns, gap shocks), macro
release surprises (CPI / payroll / unemployment moves vs trailing trend, stamped at the
publication time) and rate-policy changes (fed-funds steps). For any event category the
conditional forward returns (1D/5D/20D/60D) are computed: mean, median, std, P(positive),
5th percentile - and compared with the unconditional base rate.

News events extracted by the research agent are mapped onto these categories so the
system can answer "what happened the last time something similar occurred?".
Historical outcomes are descriptive evidence only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 5, 20, 60)
CATEGORY_MAP = {  # news category -> historical event categories
    "Monetary Policy": ["Fed rate hike", "Fed rate cut"], "Inflation": ["CPI upside surprise", "CPI downside surprise"],
    "Employment": ["Unemployment jump", "Strong payrolls"], "Volatility": ["Volatility spike (VIX +50% in 5d)"],
    "Market Drawdown": ["S&P drawdown hits -10%", "Large down day (< -3%)"], "Growth": ["Unemployment jump"],
    "Geopolitical": ["Volatility spike (VIX +50% in 5d)", "Large down day (< -3%)"],
    "Financial Conditions": ["Credit spread widening (+100bp in 20d)"], "Earnings": ["Large down day (< -3%)", "Large up day (> +3%)"],
}


def _declutter(dates: pd.DatetimeIndex, gap: int, index: pd.DatetimeIndex) -> list:
    out, last = [], -10 ** 9
    for d in dates:
        p = index.searchsorted(d)
        if p - last >= gap:
            out.append(index[min(p, len(index) - 1)])
            last = p
    return out


def build_event_database(close: pd.Series, assets: dict, pit=None, decision_times: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    idx = close.index
    r = close.pct_change()
    ev = []

    def add(cat, dates, gap=10):
        for d in _declutter(pd.DatetimeIndex(dates), gap, idx):
            ev.append({"date": d, "category": cat})

    add("Large down day (< -3%)", r[r < -0.03].index, 5)
    add("Large up day (> +3%)", r[r > 0.03].index, 5)
    dd = close / close.cummax() - 1
    cross = (dd < -0.10) & (dd.shift(1) >= -0.10)
    add("S&P drawdown hits -10%", dd[cross].index, 60)
    if "VIX" in assets:
        vix = assets["VIX"]["close"].reindex(idx).ffill()
        spike = vix / vix.shift(5) - 1 > 0.5
        add("Volatility spike (VIX +50% in 5d)", vix[spike].index, 20)
    if pit is not None and decision_times is not None:
        dts = pd.Series(decision_times, index=idx[: len(decision_times)])
        known = lambda sid, tr, per: pit.daily_view(sid, dts.values, tr, per).set_axis(dts.index)  # noqa: E731
        if "CPIAUCSL" in pit.series_ids:
            mom = known("CPIAUCSL", "yoy", 1)
            chg = mom.where(mom.diff() != 0)  # only on publication days
            trend = chg.dropna().rolling(12, min_periods=6).mean().shift(1)
            surprise = (chg.dropna() - trend).dropna()
            sd = surprise.rolling(36, min_periods=12).std().shift(1)
            add("CPI upside surprise", surprise[surprise > sd].index, 15)
            add("CPI downside surprise", surprise[surprise < -sd].index, 15)
        if "UNRATE" in pit.series_ids:
            u = known("UNRATE", "level", 1)
            du = u.diff()
            add("Unemployment jump", du[du >= 0.3].index, 20)
        if "PAYEMS" in pit.series_ids:
            p = known("PAYEMS", "diff", 1)
            pp = p.where(p.diff() != 0).dropna()
            add("Strong payrolls", pp[pp > pp.rolling(24, min_periods=12).quantile(0.8).shift(1)].index, 15)
        if "DFF" in pit.series_ids:
            ff = known("DFF", "level", 1)
            step = ff.diff(5)
            add("Fed rate hike", step[step >= 0.2].index, 30)
            add("Fed rate cut", step[step <= -0.2].index, 30)
        if "BAMLH0A0HYM2" in pit.series_ids:
            hy = known("BAMLH0A0HYM2", "level", 1)
            add("Credit spread widening (+100bp in 20d)", hy[hy.diff(20) > 1.0].index, 40)
    return pd.DataFrame(ev).sort_values("date").reset_index(drop=True) if ev else pd.DataFrame(columns=["date", "category"])


def conditional_outcomes(events: pd.DataFrame, close: pd.Series, as_of=None) -> pd.DataFrame:
    """Forward returns after each event category (only outcomes fully known by as-of)."""
    as_of = pd.Timestamp(as_of or close.index[-1])
    rows = []
    base = {h: close.pct_change(h).shift(-h) for h in HORIZONS}
    for cat, g in events.groupby("category"):
        rec = {"category": cat, "n_events": len(g), "last_event": str(pd.Timestamp(g["date"].max()).date())}
        for h in HORIZONS:
            vals = []
            for d in g["date"]:
                p = close.index.searchsorted(d)
                if p + h < len(close) and close.index[p + h] <= as_of:
                    vals.append(close.iloc[p + h] / close.iloc[p] - 1)
            v = np.array(vals)
            if len(v) >= 3:
                rec.update({f"{h}D_mean": v.mean(), f"{h}D_median": np.median(v), f"{h}D_std": v.std(),
                            f"{h}D_p_pos": (v > 0).mean(), f"{h}D_p05": np.percentile(v, 5), f"{h}D_n": len(v),
                            f"{h}D_base_mean": float(base[h].loc[:as_of].mean())})
        rows.append(rec)
    return pd.DataFrame(rows)


def match_news_events(news_events: list, outcomes: pd.DataFrame) -> list[dict]:
    """News memory lookup: map today's extracted events to historical reaction statistics."""
    out = []
    if outcomes is None or outcomes.empty:
        return out
    for e in news_events:
        cat = e["category"] if isinstance(e, dict) else e.category
        for hist_cat in CATEGORY_MAP.get(cat, []):
            row = outcomes[outcomes["category"] == hist_cat]
            if not row.empty:
                r = row.iloc[0].to_dict()
                out.append({"news_category": cat, "historical_category": hist_cat, "n": r.get("n_events"),
                            "5D_mean": r.get("5D_mean"), "20D_mean": r.get("20D_mean"), "20D_p_pos": r.get("20D_p_pos"),
                            "60D_mean": r.get("60D_mean"), "20D_base_mean": r.get("20D_base_mean")})
    seen, uniq = set(), []
    for o in out:
        k = (o["news_category"], o["historical_category"])
        if k not in seen:
            seen.add(k)
            uniq.append(o)
    return uniq
