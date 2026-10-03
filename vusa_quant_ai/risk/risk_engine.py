"""Risk engine: VaR / CVaR, volatility, drawdown, tail statistics, beta, stress tests,
rolling correlations / correlation breakdowns, position sizing research and portfolio
exposure. Stress tests are labelled SCENARIOS - not predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ------------------------------------------------------------------ core risk statistics
def risk_metrics(close: pd.Series, bench: pd.Series | None = None) -> dict:
    r = np.log(close).diff().dropna()
    r1y = r.tail(252)
    s = {}
    s["vol_20d"] = float(r.tail(20).std() * np.sqrt(252))
    s["vol_60d"] = float(r.tail(60).std() * np.sqrt(252))
    s["vol_1y"] = float(r1y.std() * np.sqrt(252))
    s["vol_long_run"] = float(r.std() * np.sqrt(252))
    for lvl in (0.95, 0.99):
        q = np.quantile(r1y, 1 - lvl)
        s[f"VaR{int(lvl * 100)}_1d_hist"] = float(-q)
        s[f"CVaR{int(lvl * 100)}_1d_hist"] = float(-r1y[r1y <= q].mean())
        s[f"VaR{int(lvl * 100)}_1d_param"] = float(-(r1y.mean() + stats.norm.ppf(1 - lvl) * r1y.std()))
    # Cornish-Fisher modified VaR (skew / kurtosis aware)
    z = stats.norm.ppf(0.05)
    sk, ku = stats.skew(r1y), stats.kurtosis(r1y)
    zcf = z + (z ** 2 - 1) * sk / 6 + (z ** 3 - 3 * z) * ku / 24 - (2 * z ** 3 - 5 * z) * sk ** 2 / 36
    s["VaR95_1d_cornish_fisher"] = float(-(r1y.mean() + zcf * r1y.std()))
    s["VaR95_20d_hist"] = float(-np.quantile(close.pct_change(20).dropna().tail(2520), 0.05))
    dd = close / close.cummax() - 1
    s["drawdown_current"] = float(dd.iloc[-1])
    s["max_drawdown_1y"] = float((close.tail(252) / close.tail(252).cummax() - 1).min())
    s["max_drawdown_all"] = float(dd.min())
    s["skew_1y"], s["excess_kurtosis_1y"] = float(sk), float(ku)
    s["worst_day_1y"] = float(r1y.min())
    s["ulcer_index_1y"] = float(np.sqrt(((100 * (close.tail(252) / close.tail(252).cummax() - 1)) ** 2).mean()))
    if bench is not None:
        b = np.log(bench).diff().reindex(r.index)
        d = pd.concat([r, b], axis=1).dropna().tail(252)
        if len(d) > 60:
            s["beta_vs_spx_1y"] = float(np.cov(d.iloc[:, 0], d.iloc[:, 1])[0, 1] / d.iloc[:, 1].var())
            s["corr_vs_spx_1y"] = float(d.corr().iloc[0, 1])
    return s


# ------------------------------------------------------------------ stress testing
STRESS_SCENARIOS = {
    "Normal (median month)": {"spx": None, "desc": "Median 20-day outcome over history"},
    "Mild correction": {"spx": -0.05, "desc": "S&P 500 falls 5%"},
    "Bear market": {"spx": -0.20, "desc": "S&P 500 falls 15-25% (midpoint 20%)"},
    "Crash": {"spx": -0.33, "desc": "S&P 500 falls 30%+ (2008/2020-type)"},
    "Rate shock": {"rates_bp": 100, "desc": "US 10Y yield +100bp quickly"},
    "Inflation shock": {"rates_bp": 75, "vix": 8, "desc": "Unexpected inflation rise: yields +75bp, VIX +8"},
    "Volatility shock": {"vix": 15, "desc": "VIX +15 points"},
    "USD rally (EUR investor)": {"eurusd": -0.08, "desc": "EUR/USD -8% (USD up): positive FX translation for EUR investors"},
    "USD slump (EUR investor)": {"eurusd": +0.08, "desc": "EUR/USD +8% (USD down): FX translation loss for EUR investors"},
}

HISTORICAL_EPISODES = {
    "GFC 2008": ("2008-09-01", "2009-03-09"), "Euro crisis 2011": ("2011-07-22", "2011-10-03"),
    "China/oil 2015-16": ("2015-08-17", "2016-02-11"), "Volmageddon 2018": ("2018-01-26", "2018-02-08"),
    "Q4 2018": ("2018-09-20", "2018-12-24"), "COVID crash 2020": ("2020-02-19", "2020-03-23"),
    "Inflation bear 2022": ("2022-01-03", "2022-10-12"), "Tariff shock 2025": ("2025-02-19", "2025-04-08"),
}


def _beta(y: pd.Series, x: pd.Series, window: int = 756) -> tuple[float, float]:
    d = pd.concat([y, x], axis=1).dropna().tail(window)
    if len(d) < 60 or d.iloc[:, 1].var() == 0:
        return np.nan, np.nan
    b = np.cov(d.iloc[:, 0], d.iloc[:, 1])[0, 1] / d.iloc[:, 1].var()
    resid = d.iloc[:, 0] - b * d.iloc[:, 1]
    return float(b), float(resid.std())


def stress_test(target: pd.DataFrame, assets: dict[str, pd.DataFrame], currency: str = "EUR") -> pd.DataFrame:
    """Estimated VUSA impact per scenario from empirical (20-day) sensitivities.

    Uses non-overlapping 20-day returns over the last ~10 years; betas/sensitivities are
    descriptive and can break down in crises (reported alongside)."""
    c = target["close"]
    vr = c.pct_change(20).iloc[::20]
    rows = []
    get = lambda k: assets[k]["close"].reindex(c.index).ffill() if k in assets else None  # noqa: E731
    spx, tnx, vix, fx = get("SPX"), get("TNX"), get("VIX"), get("EURUSD")
    b_spx = _beta(vr, spx.pct_change(20).iloc[::20])[0] if spx is not None else np.nan
    b_rates = _beta(vr, (tnx / 10).diff(20).iloc[::20])[0] if tnx is not None else np.nan  # per 1.00 pct-pt
    b_vix = _beta(vr, vix.diff(20).iloc[::20])[0] if vix is not None else np.nan  # per VIX point
    b_fx = _beta(vr, fx.pct_change(20).iloc[::20])[0] if fx is not None else np.nan
    for name, sc in STRESS_SCENARIOS.items():
        impact, basis = np.nan, ""
        if sc.get("spx") is not None and np.isfinite(b_spx):
            impact, basis = b_spx * sc["spx"], f"beta to S&P (20d) = {b_spx:.2f}"
        elif name.startswith("Normal"):
            impact, basis = float(vr.dropna().median()), "historical median 20d return"
        else:
            parts, imp = [], 0.0
            if "rates_bp" in sc and np.isfinite(b_rates):
                imp += b_rates * sc["rates_bp"] / 100
                parts.append(f"rate sensitivity {b_rates:+.3f}/pp")
            if "vix" in sc and np.isfinite(b_vix):
                imp += b_vix * sc["vix"]
                parts.append(f"VIX sensitivity {b_vix:+.4f}/pt")
            if "eurusd" in sc and np.isfinite(b_fx):
                imp += b_fx * sc["eurusd"]
                parts.append(f"EUR/USD sensitivity {b_fx:+.2f}")
            if parts:
                impact, basis = imp, "; ".join(parts)
        rows.append({"scenario": name, "description": sc["desc"], "estimated_vusa_impact": impact,
                     "estimated_price": float(c.iloc[-1] * (1 + impact)) if np.isfinite(impact) else np.nan,
                     "basis": basis or "insufficient reference data", "type": "SCENARIO (not a prediction)"})
    for name, (a, b) in HISTORICAL_EPISODES.items():
        seg = c.loc[a:b]
        if len(seg) > 5:
            rows.append({"scenario": f"Replay: {name}", "description": f"{a} to {b}",
                         "estimated_vusa_impact": float(seg.iloc[-1] / seg.iloc[0] - 1),
                         "estimated_price": float(c.iloc[-1] * seg.iloc[-1] / seg.iloc[0]),
                         "basis": "realised path in the (possibly proxy-extended) history",
                         "type": "HISTORICAL REPLAY (not a prediction)"})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ correlations
def rolling_correlations(target: pd.Series, assets: dict[str, pd.DataFrame], sectors: dict[str, pd.DataFrame],
                         window: int = 60) -> pd.DataFrame:
    r = np.log(target).diff()
    cols = {}
    for k, df in {**assets, **sectors}.items():
        s = np.log(df["close"].reindex(target.index).ffill(limit=3)).diff()
        cols[k] = r.rolling(window, min_periods=int(window * 0.7)).corr(s)
    return pd.DataFrame(cols)


def correlation_breakdowns(rc: pd.DataFrame, long_window: int = 756, z_thr: float = 2.5) -> list[dict]:
    out = []
    for c in rc.columns:
        s = rc[c].dropna()
        if len(s) < 300:
            continue
        hist = s.iloc[:-20].tail(long_window)
        z = (s.iloc[-1] - hist.mean()) / (hist.std() or np.nan)
        if np.isfinite(z) and abs(z) > z_thr:
            out.append({"asset": c, "current_corr": float(s.iloc[-1]), "typical_corr": float(hist.mean()), "z": float(z)})
    return out


def correlation_matrix(target: pd.Series, assets: dict[str, pd.DataFrame], window: int = 252) -> pd.DataFrame:
    d = {"VUSA": np.log(target).diff()}
    for k, df in assets.items():
        d[k] = np.log(df["close"].reindex(target.index).ffill(limit=3)).diff()
    return pd.DataFrame(d).tail(window).corr()


# ------------------------------------------------------------------ risk score
def risk_score(rm: dict, regime: dict, mc: dict | None, shocks: dict | None, forecast_unc: float | None,
               p_dd10: float | None) -> tuple[float, dict]:
    """0 (low risk) .. 100 (high risk). Transparent weighted components."""
    comp = {}
    comp["volatility"] = float(np.clip(regime.get("vol_percentile", 0.5) * 100, 0, 100))
    comp["drawdown"] = float(np.clip(-rm.get("drawdown_current", 0) / 0.30 * 100, 0, 100))
    comp["tail"] = float(np.clip((rm.get("CVaR95_1d_hist", 0.02) - 0.01) / 0.04 * 100, 0, 100))
    comp["regime"] = float(100 - regime.get("score", 50))
    if mc:
        comp["simulated_downside"] = float(np.clip(mc.get("p_lt5", 0.15) / 0.4 * 100, 0, 100))
    if p_dd10 is not None and np.isfinite(p_dd10):
        comp["drawdown_probability"] = float(np.clip(p_dd10 / 0.3 * 100, 0, 100))
    if shocks:
        comp["shocks"] = float(shocks.get("shock_level", 0) * 100)
    if forecast_unc is not None and np.isfinite(forecast_unc):
        comp["forecast_uncertainty"] = float(np.clip(forecast_unc / 0.12 * 100, 0, 100))
    w = {"volatility": 1.2, "drawdown": 0.8, "tail": 0.8, "regime": 1.0, "simulated_downside": 1.0,
         "drawdown_probability": 1.0, "shocks": 0.8, "forecast_uncertainty": 0.6}
    tot = sum(w[k] * v for k, v in comp.items()) / sum(w[k] for k in comp)
    return float(tot), comp


# ------------------------------------------------------------------ position sizing (research only)
def position_sizing_research(close: pd.Series, signal_exposure: pd.Series | None = None, target_vol: float = 0.12,
                             max_leverage: float = 1.0, kelly_fraction: float = 0.25) -> dict:
    """Compare sizing rules historically (no leverage above ``max_leverage``; research only)."""
    r = close.pct_change().fillna(0)
    rv = np.log(close).diff().rolling(20).std().shift(1) * np.sqrt(252)
    base = signal_exposure.reindex(r.index).fillna(0).shift(1) if signal_exposure is not None else pd.Series(1.0, index=r.index)
    rules = {
        "Fixed 100%": base.clip(0, 1),
        "Fixed 60%": 0.6 * base.clip(0, 1),
        "Volatility targeting": (base * (target_vol / rv)).clip(0, max_leverage),
    }
    mu = r.rolling(756, min_periods=252).mean().shift(1) * 252
    var = (r.rolling(756, min_periods=252).std().shift(1) * np.sqrt(252)) ** 2
    rules["Fractional Kelly (25%)"] = (base * kelly_fraction * mu / var).clip(0, max_leverage)
    risk_budget = 0.01
    rules["Fixed-risk (1% daily VaR)"] = (base * risk_budget / (1.65 * rv / np.sqrt(252))).clip(0, max_leverage)
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    rules["Drawdown-aware"] = (base * np.where(dd.shift(1) < -0.15, 0.5, np.where(dd.shift(1) < -0.08, 0.75, 1.0))).clip(0, 1)
    from ..validation.metrics import trading_metrics

    out = {}
    for k, w in rules.items():
        sr = (w.fillna(0) * r).iloc[252:]
        out[k] = {**trading_metrics(sr), "avg_exposure": float(w.iloc[252:].mean())}
    return out


def portfolio_exposure(price: float, units: float, avg_price: float, cash: float, var95_20d: float,
                       horizon_p10: float | None, risk_tolerance: str) -> dict:
    value = units * price
    total = value + cash
    tol = {"conservative": 0.08, "moderate": 0.15, "aggressive": 0.25}.get(risk_tolerance, 0.15)
    out = {"position_value": value, "total_portfolio": total, "exposure_pct": value / total if total else np.nan,
           "unrealized_pnl": units * (price - avg_price) if avg_price else np.nan,
           "unrealized_pnl_pct": price / avg_price - 1 if avg_price else np.nan,
           "est_20d_VaR95_value": value * var95_20d,
           "est_20d_P10_loss_value": value * min(0.0, horizon_p10) if horizon_p10 is not None else np.nan,
           "loss_tolerance_pct": tol,
           "portfolio_impact_VaR_pct": (value * var95_20d / total) if total else np.nan}
    out["within_tolerance"] = bool(out["portfolio_impact_VaR_pct"] <= tol) if total else None
    return out
