"""Independent decision dimensions, each scored 0..100 (50 = neutral) with evidence.

Every score is a transparent function of named inputs; the evidence list records the
exact values used so the GUI and the report can show *why* a component scored as it did.
Missing inputs are reported as missing and the component weight is reduced - never
filled with invented values.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def squash(x: float, scale: float) -> float:
    """Map a signed quantity to 0..100 (50 neutral)."""
    if x is None or not np.isfinite(x):
        return np.nan
    return float(50 + 50 * np.tanh(x / scale))


@dataclass
class Component:
    name: str
    score: float
    evidence: list[str] = field(default_factory=list)
    inputs: dict = field(default_factory=dict)
    coverage: float = 1.0  # share of intended inputs available
    note: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "score": None if not np.isfinite(self.score) else round(self.score, 1),
                "evidence": self.evidence, "inputs": self.inputs, "coverage": round(self.coverage, 2), "note": self.note}


def _avg(parts: list[tuple[float, float]]) -> tuple[float, float]:
    vals = [(s, w) for s, w in parts if s is not None and np.isfinite(s)]
    if not vals:
        return np.nan, 0.0
    tot_w = sum(w for _, w in parts)
    return float(sum(s * w for s, w in vals) / sum(w for _, w in vals)), sum(w for _, w in vals) / tot_w


def _g(row: pd.Series, k: str):
    v = row.get(k, np.nan)
    return float(v) if v is not None and pd.notna(v) else np.nan


def trend_component(f: pd.Series) -> Component:
    d200, x50, sl, pers = _g(f, "dist_sma200"), _g(f, "sma50_sma200"), _g(f, "trend_slope_120"), _g(f, "trend_persistence_60")
    adx, di = _g(f, "adx14"), _g(f, "di_diff")
    parts = [(squash(d200, 0.06), 1.2), (squash(x50, 0.04), 1.0), (squash(sl, 0.25), 1.0),
             (squash((pers - 0.5) if np.isfinite(pers) else np.nan, 0.3), 0.6), (squash(di, 15), 0.5)]
    s, cov = _avg(parts)
    ev = [f"Price vs SMA200: {d200:+.1%}", f"SMA50 vs SMA200: {x50:+.1%}", f"120d trend slope (ann.): {sl:+.1%}",
          f"Days above SMA200 (60d): {pers:.0%}" if np.isfinite(pers) else "trend persistence: n/a",
          f"ADX {adx:.0f}, DI+ - DI- = {di:+.0f}" if np.isfinite(adx) else "ADX: n/a"]
    return Component("trend", s, ev, {"dist_sma200": d200, "sma50_sma200": x50, "slope120": sl}, cov)


def momentum_component(f: pd.Series) -> Component:
    r20, r60, m121, rsi, mh = (_g(f, k) for k in ("ret_20d", "ret_60d", "mom_12_1", "rsi14", "macd_hist_norm"))
    rsi_s = np.nan
    if np.isfinite(rsi):  # strong-but-not-extreme momentum preferred; overbought >78 penalised
        rsi_s = squash((rsi - 50) / 10, 1.5) if rsi <= 75 else squash((75 - rsi) / 5 + 1.5, 1.5)
    parts = [(squash(r20, 0.04), 1.0), (squash(r60, 0.07), 1.0), (squash(m121, 0.12), 1.2), (rsi_s, 0.6),
             (squash(mh, 0.003), 0.5)]
    s, cov = _avg(parts)
    ev = [f"20d return {r20:+.1%}", f"60d return {r60:+.1%}", f"12-1 month momentum {m121:+.1%}",
          f"RSI(14) {rsi:.0f}" + (" (overbought)" if rsi > 75 else " (oversold)" if rsi < 30 else ""),
          f"MACD histogram/price {mh:+.4f}"]
    return Component("momentum", s, ev, {"ret_20d": r20, "ret_60d": r60, "mom_12_1": m121, "rsi14": rsi}, cov)


def volatility_component(f: pd.Series, regime: dict) -> Component:
    pct, ratio, ts, vix = regime.get("vol_percentile", np.nan), _g(f, "rv_ratio_5_60"), _g(f, "vix_term_structure"), _g(f, "vix_level")
    parts = [(100 * (1 - pct) if np.isfinite(pct) else np.nan, 1.2), (squash(1 - ratio, 0.5) if np.isfinite(ratio) else np.nan, 0.6),
             (squash(ts, 0.08), 1.0), (squash((20 - vix) / 10, 1.0) if np.isfinite(vix) else np.nan, 0.8)]
    s, cov = _avg(parts)
    ev = [f"Realised vol percentile (5y): {pct:.0%}" if np.isfinite(pct) else "vol percentile n/a",
          f"5d/60d vol ratio: {ratio:.2f}", f"VIX: {vix:.1f}" if np.isfinite(vix) else "VIX: unavailable",
          f"VIX term structure (3M/1M-1): {ts:+.1%}" + (" backwardation = stress" if ts < 0 else "") if np.isfinite(ts)
          else "VIX term structure: unavailable"]
    return Component("volatility", s, ev, {"vol_pct": pct, "vix": vix, "vix_ts": ts}, cov,
                     "High score = calm volatility environment (favourable)")


def breadth_component(f: pd.Series) -> Component:
    a200, a50, adl, cyc, eq = (_g(f, k) for k in ("sector_pct_above_sma200_proxy", "sector_pct_above_sma50_proxy",
                                                  "sector_ad_line_slope_proxy", "cyclical_vs_defensive_60d", "equal_cap_ratio_mom"))
    parts = [(100 * a200 if np.isfinite(a200) else np.nan, 1.2), (100 * a50 if np.isfinite(a50) else np.nan, 0.8),
             (squash(adl, 1.0), 0.6), (squash(cyc, 0.04), 0.8), (squash(eq, 0.03), 0.6)]
    s, cov = _avg(parts)
    ev = [f"Sectors above SMA200 (proxy): {a200:.0%}" if np.isfinite(a200) else "breadth proxy unavailable",
          f"Sectors above SMA50 (proxy): {a50:.0%}" if np.isfinite(a50) else "",
          f"Cyclicals vs defensives 60d: {cyc:+.1%}" if np.isfinite(cyc) else "",
          f"Equal-weight vs cap-weight 60d: {eq:+.1%}" if np.isfinite(eq) else ""]
    return Component("breadth", s, [e for e in ev if e], {"pct_above_200": a200}, cov,
                     "Sector-ETF proxy, not constituent-level breadth")


def macro_component(f: pd.Series) -> Component:
    curve, hy, hyc, sahm, unr, infl_tr, nfci, ffc, cra = (_g(f, k) for k in (
        "curve_10y3m", "hy_oas", "hy_oas_chg_20d", "sahm_rt", "unrate_vs_12m_low", "inflation_trend_6m", "nfci",
        "fed_funds_chg_6m", "credit_risk_appetite"))
    if not np.isfinite(curve):
        curve = _g(f, "curve_10y_3m_mkt")
    rec = sahm if np.isfinite(sahm) else unr
    parts = [(squash(curve, 1.0), 0.6), (squash((4.5 - hy) / 1.5, 1.0) if np.isfinite(hy) else np.nan, 1.0),
             (squash(-hyc, 0.5), 0.8), (squash((0.3 - rec) / 0.3, 1.0) if np.isfinite(rec) else np.nan, 1.2),
             (squash(-infl_tr, 0.01), 0.6), (squash(-nfci, 0.4), 0.8), (squash(-ffc, 1.0), 0.4), (squash(cra, 0.02), 0.6)]
    s, cov = _avg(parts)
    ev = []
    if np.isfinite(curve):
        ev.append(f"Yield curve 10Y-3M: {curve:+.2f}pp" + (" (inverted)" if curve < 0 else ""))
    if np.isfinite(hy):
        ev.append(f"High-yield OAS: {hy:.2f}% (20d chg {hyc:+.2f})")
    if np.isfinite(rec):
        ev.append(f"Recession indicator (Sahm-style): {rec:.2f}" + (" - triggered" if rec >= 0.5 else ""))
    if np.isfinite(infl_tr):
        ev.append(f"Inflation trend (6m change in CPI YoY): {infl_tr:+.2%}")
    if np.isfinite(nfci):
        ev.append(f"Financial conditions (NFCI): {nfci:+.2f}")
    if np.isfinite(cra):
        ev.append(f"Credit risk appetite (HYG/LQD 20d): {cra:+.1%}")
    if not ev:
        ev.append("No macro inputs available (FRED unreachable) - component excluded")
    return Component("macro", s, ev, {"curve": curve, "hy_oas": hy, "recession": rec}, cov)


def valuation_component(val, f: pd.Series, close: pd.Series) -> Component:
    """Fundamental valuation if the user supplied it; else a labelled price-trend proxy."""
    parts, ev, note = [], [], ""
    if val is not None and (val.forward_pe or val.cape or val.trailing_pe):
        if val.forward_pe:
            parts.append((squash((18.5 - val.forward_pe) / 3.0, 1.0), 1.2))  # ~20y median fwd P/E ~ 16-19
            ev.append(f"Forward P/E {val.forward_pe:.1f} (long-run median ~16-19)")
        if val.cape:
            parts.append((squash((25 - val.cape) / 7.0, 1.0), 1.0))
            ev.append(f"CAPE {val.cape:.1f} (post-1990 median ~25)")
        ey = 100 / val.forward_pe if val.forward_pe else (100 / val.trailing_pe if val.trailing_pe else np.nan)
        ry = _g(f, "real_yield_10y")
        if np.isfinite(ey) and np.isfinite(ry):
            erp = ey - ry
            parts.append((squash((erp - 3.0) / 1.5, 1.0), 1.0))
            ev.append(f"Equity risk premium proxy (earnings yield {ey:.2f}% - real yield {ry:.2f}%): {erp:.2f}pp")
        if val.eps_revisions_3m_pct:
            parts.append((squash(val.eps_revisions_3m_pct / 3.0, 1.0), 0.6))
            ev.append(f"EPS revisions 3m: {val.eps_revisions_3m_pct:+.1f}%")
        note = f"User-supplied fundamentals as of {val.as_of or 'unknown date'} ({val.source or 'source not given'})"
    else:
        lp = np.log(close.dropna())
        if len(lp) > 2520:
            x = np.arange(2520)
            b = np.polyfit(x, lp.values[-2520:], 1)
            dev = lp.values[-1] - np.polyval(b, x[-1])
            parts.append((squash(-dev / 0.15, 1.0), 1.0))
            ev.append(f"PROXY: price vs 10-year log-trend {dev:+.1%} (not a fundamental valuation)")
        note = "No fundamental valuation data entered (Settings > Valuation). Price-based PROXY used with reduced weight."
    s, cov = _avg(parts) if parts else (np.nan, 0.0)
    return Component("valuation", s, ev or ["valuation unavailable"], {}, cov if val and val.forward_pe else 0.5 * cov, note)


def sentiment_component(news: dict | None, f: pd.Series) -> Component:
    parts, ev = [], []
    if news and news.get("n", 0) >= 5 and np.isfinite(news.get("score", np.nan)):
        parts.append((squash(news["score"], 0.25), 1.0))
        ev.append(f"News sentiment {news['score']:+.2f} from {news['n']} relevant articles (dispersion {news['dispersion']:.2f})")
    else:
        ev.append("News sentiment unavailable or too few relevant articles")
    vrp, vz = _g(f, "vrp"), _g(f, "vix_z_252")
    if np.isfinite(vz):
        parts.append((squash(-vz, 1.5), 0.8))
        ev.append(f"VIX z-score (1y): {vz:+.1f} (volatility sentiment)")
    if np.isfinite(vrp):
        parts.append((squash(vrp, 0.06), 0.4))
        ev.append(f"Variance risk premium (VIX - realised): {vrp:+.1%}")
    s, cov = _avg(parts) if parts else (np.nan, 0.0)
    return Component("sentiment", s, ev, {"news": news.get("score") if news else None}, cov)


def regime_component(regime: dict) -> Component:
    s = regime.get("score", np.nan)
    st = regime.get("hmm_stress_probability", np.nan)
    if np.isfinite(st):
        s = 0.8 * s + 0.2 * (100 * (1 - st))
    ev = [f"Rule-based regime: {regime.get('regime')} ({regime.get('days_in_regime')} sessions)",
          f"HMM stress-state probability: {st:.0%}" if np.isfinite(st) else "HMM: unavailable",
          f"Drawdown from peak: {regime.get('drawdown', np.nan):.1%}"]
    return Component("regime", float(s), ev, {"regime": regime.get("regime")})


def event_risk_component(events: list, shocks: dict, matched_hist: list) -> Component:
    neg = [e for e in events if (e["direction"] if isinstance(e, dict) else e.direction) == "Potentially bearish"]
    pos = [e for e in events if (e["direction"] if isinstance(e, dict) else e.direction) == "Potentially bullish"]
    mag = lambda es: sum((e["magnitude"] if isinstance(e, dict) else e.magnitude) * (e["confidence"] if isinstance(e, dict) else e.confidence) / 100 for e in es)  # noqa: E731
    net = (mag(pos) - mag(neg)) / 100
    s = squash(net, 2.0) if events else 50.0
    s = s - 25 * shocks.get("shock_level", 0)
    ev = [f"{len(neg)} potentially bearish / {len(pos)} potentially bullish extracted events"]
    ev += [f"Shock: {x}" for x in shocks.get("flags", [])]
    for m in matched_hist[:3]:
        if m.get("20D_mean") is not None and np.isfinite(m.get("20D_mean", np.nan)):
            ev.append(f"History after '{m['historical_category']}' (n={m['n']}): 20D mean {m['20D_mean']:+.1%} "
                      f"vs base {m.get('20D_base_mean', np.nan):+.1%}")
    return Component("event_risk", float(np.clip(s, 0, 100)), ev, {"n_events": len(events)},
                     note="High score = benign event environment")


def ml_component(hr) -> Component:
    if hr is None or not np.isfinite(hr.p_up):
        return Component("ml_forecast", np.nan, ["ML forecast unavailable"], {}, 0.0)
    lo, hi = hr.interval
    width = (hi - lo) if np.isfinite(lo) and np.isfinite(hi) else np.nan
    edge = (hr.p_up - 0.5) * 2
    s = 50 + 50 * np.tanh(edge / 0.25) * (0.5 + 0.5 * hr.confidence)
    ev = [f"{hr.horizon}D expected return {hr.expected_return:+.2%} (80% interval {lo:+.1%} .. {hi:+.1%})",
          f"Calibrated P(up) {hr.p_up:.0%} (raw {hr.p_up_raw:.0%}; calibration: {hr.calibration.get('method')})",
          f"Forecast confidence {hr.confidence:.0%}; model sign agreement "
          f"{(hr.disagreement.get('sign_agreement') or 0):.0%}"]
    return Component("ml_forecast", float(s), ev, {"p_up": hr.p_up, "exp_ret": hr.expected_return, "width": width})


def analogue_component(an: dict) -> Component:
    if not an or not an.get("summary"):
        return Component("analogues", np.nan, ["Analogue search unavailable"], {}, 0.0)
    s20 = an["summary"]["ret_20d"]
    base_m, base_p = an["base_rates"].get("base_20d_mean", 0), an["base_rates"].get("base_20d_p_pos", 0.5)
    rel = s20["weighted_mean"] - base_m
    s = 0.5 * squash(rel, 0.03) + 0.5 * (100 * s20["p_positive"] if s20["p_positive"] is not None else 50)
    s = 50 + (s - 50) * 0.8
    ev = [f"Top analogues: 20D weighted mean {s20['weighted_mean']:+.1%} vs unconditional {base_m:+.1%}",
          f"Analogues positive after 20D: {s20['p_positive']:.0%} (base rate {base_p:.0%})",
          f"Worst 60D drawdown among analogues: {an['summary']['max_dd_60d']['min']:.1%}"]
    return Component("analogues", float(s), ev, {"rel_20d": rel})


def risk_reward_component(hr, mc: dict | None) -> Component:
    if hr is None or not np.isfinite(hr.expected_return):
        return Component("risk_reward", np.nan, ["unavailable"], {}, 0.0)
    q = hr.quantiles or {}
    up, dn = q.get(0.9, np.nan), q.get(0.1, np.nan)
    ratio = (up / -dn) if np.isfinite(up) and np.isfinite(dn) and dn < 0 else np.nan
    pg, pl = hr.p_gt5, hr.p_lt5
    parts = [(squash(np.log(ratio) if np.isfinite(ratio) and ratio > 0 else np.nan, 0.5), 1.0),
             (squash((pg - pl) if np.isfinite(pg) and np.isfinite(pl) else np.nan, 0.15), 1.0)]
    if mc:
        parts.append((squash(mc.get("mean", np.nan) / (mc.get("CVaR95", np.nan) or np.nan), 0.3), 0.6))
    s, cov = _avg(parts)
    ev = [f"Upside P90 {up:+.1%} vs downside P10 {dn:+.1%} (ratio {ratio:.2f})" if np.isfinite(ratio) else "interval n/a",
          f"P(>+5%) {pg:.0%} vs P(<-5%) {pl:.0%}" if np.isfinite(pg) else ""]
    if mc:
        ev.append(f"Monte Carlo (block bootstrap) mean {mc.get('mean', np.nan):+.1%}, CVaR95 {mc.get('CVaR95', np.nan):.1%}")
    return Component("risk_reward", s, [e for e in ev if e], {"ratio": ratio}, cov)
