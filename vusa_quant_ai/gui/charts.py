"""Plotly figure builders (Qt-independent, unit-testable)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TEMPLATE = "plotly_dark"
REGIME_COLORS = {"Strong bull": "#1a9850", "Bull": "#66bd63", "Recovery": "#a6d96a", "Neutral": "#bdbdbd",
                 "High volatility": "#fdae61", "Bear": "#f46d43", "Strong bear": "#d73027", "Crisis": "#7f0000"}
SIG_COLORS = {"BUY": "#2ecc71", "HOLD": "#f1c40f", "SELL": "#e74c3c"}


def _layout(fig, title, h=420):
    fig.update_layout(template=TEMPLATE, title=title, height=h, margin=dict(l=40, r=20, t=50, b=30),
                      legend=dict(orientation="h", y=-0.15))
    return fig


def empty(title: str, msg: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=16))
    return _layout(fig, title, 300)


def candlestick(df: pd.DataFrame, title="VUSA", years: float = 2, forecast: dict | None = None,
                sr: dict | None = None) -> go.Figure:
    d = df.tail(int(252 * years))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=d.index, open=d["open"], high=d["high"], low=d["low"], close=d["close"], name="VUSA"), 1, 1)
    full = df["close"]
    for n, c in ((20, "#f39c12"), (50, "#3498db"), (200, "#e84393")):
        fig.add_trace(go.Scatter(x=d.index, y=full.rolling(n).mean().reindex(d.index), name=f"SMA{n}",
                                 line=dict(width=1.2, color=c)), 1, 1)
    if forecast and np.isfinite(forecast.get("expected_return", np.nan)):
        p0, last = d["close"].iloc[-1], d.index[-1]
        h = forecast["horizon"]
        tgt = last + pd.tseries.offsets.BDay(h)
        lo, hi = forecast["interval"]
        fig.add_trace(go.Scatter(x=[last, tgt, tgt, last], y=[p0, p0 * (1 + hi), p0 * (1 + lo), p0], fill="toself",
                                 fillcolor="rgba(52,152,219,0.25)", line=dict(width=0),
                                 name=f"{int(100 * forecast.get('interval_level', .8))}% conformal interval"), 1, 1)
        fig.add_trace(go.Scatter(x=[last, tgt], y=[p0, p0 * (1 + forecast["expected_return"])], name=f"{h}D forecast",
                                 line=dict(dash="dash", color="#3498db", width=2)), 1, 1)
    if sr:
        for s in sr.get("support", [])[:3]:
            fig.add_hline(y=s["level"], line=dict(color="rgba(46,204,113,.5)", dash="dot"), row=1, col=1)
        for s in sr.get("resistance", [])[:3]:
            fig.add_hline(y=s["level"], line=dict(color="rgba(231,76,60,.5)", dash="dot"), row=1, col=1)
    if "volume" in d and d["volume"].notna().any():
        fig.add_trace(go.Bar(x=d.index, y=d["volume"], name="Volume", marker_color="#636e72"), 2, 1)
    if "is_proxy" in d and d["is_proxy"].any():
        px = d[d["is_proxy"] == 1]
        fig.add_vrect(x0=px.index[0], x1=px.index[-1], fillcolor="grey", opacity=0.15, annotation_text="proxy history")
    fig.update_xaxes(rangeslider_visible=False)
    return _layout(fig, title, 560)


def line(series: dict[str, pd.Series], title: str, h=380, yfmt: str | None = None, hlines: list | None = None) -> go.Figure:
    fig = go.Figure()
    for k, s in series.items():
        if s is not None and len(s.dropna()):
            fig.add_trace(go.Scatter(x=s.index, y=s.values, name=k, mode="lines"))
    for y in hlines or []:
        fig.add_hline(y=y, line=dict(dash="dot", color="grey"))
    if yfmt:
        fig.update_yaxes(tickformat=yfmt)
    return _layout(fig, title, h)


def drawdown(close: pd.Series, title="Drawdown") -> go.Figure:
    dd = close / close.cummax() - 1
    fig = go.Figure(go.Scatter(x=dd.index, y=dd, fill="tozeroy", line=dict(color="#e74c3c"), name="drawdown"))
    fig.update_yaxes(tickformat=".0%")
    return _layout(fig, title, 320)


def regime_chart(close: pd.Series, regimes: pd.DataFrame, years=8) -> go.Figure:
    d = regimes.dropna(subset=["regime_rule"]).tail(int(252 * years))
    c = close.reindex(d.index)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)
    for r, col in REGIME_COLORS.items():
        m = d["regime_rule"] == r
        if m.any():
            fig.add_trace(go.Scatter(x=d.index[m], y=c[m], mode="markers", marker=dict(size=3, color=col), name=r), 1, 1)
    if "hmm_stress_prob" in d:
        fig.add_trace(go.Scatter(x=d.index, y=d["hmm_stress_prob"], name="HMM stress-state P (filtered)",
                                 line=dict(color="#e67e22")), 2, 1)
    return _layout(fig, "Market regime (rule-based) and HMM stress probability (causal filter)", 560)


def forecast_bars(forecasts: dict) -> go.Figure:
    hs = sorted(forecasts, key=lambda k: int(k))
    er = [forecasts[h].get("expected_return") for h in hs]
    lo = [forecasts[h].get("interval", [np.nan, np.nan])[0] for h in hs]
    hi = [forecasts[h].get("interval", [np.nan, np.nan])[1] for h in hs]
    pu = [forecasts[h].get("p_up") for h in hs]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Expected return with 80% conformal interval", "Calibrated P(up)"))
    fig.add_trace(go.Scatter(x=[f"{h}D" for h in hs], y=er, mode="markers", marker=dict(size=12, color="#3498db"),
                             error_y=dict(type="data", symmetric=False, array=[(b or 0) - (a or 0) for a, b in zip(er, hi)],
                                          arrayminus=[(a or 0) - (c or 0) for a, c in zip(er, lo)]), name="expected"), 1, 1)
    fig.add_trace(go.Bar(x=[f"{h}D" for h in hs], y=pu, marker_color=["#2ecc71" if (p or 0) >= .5 else "#e74c3c" for p in pu],
                         name="P(up)"), 1, 2)
    fig.add_hline(y=0.5, row=1, col=2, line=dict(dash="dot"))
    fig.update_yaxes(tickformat=".1%", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", range=[0, 1], row=1, col=2)
    return _layout(fig, "Multi-horizon probabilistic forecasts", 420)


def distribution(quantiles: dict, resid_based_samples: np.ndarray | None, title: str) -> go.Figure:
    fig = go.Figure()
    if resid_based_samples is not None and len(resid_based_samples):
        fig.add_trace(go.Histogram(x=resid_based_samples, nbinsx=60, histnorm="probability density", name="predictive distribution"))
    for q, v in (quantiles or {}).items():
        fig.add_vline(x=v, line=dict(dash="dot"), annotation_text=f"q{float(q):.2f}")
    fig.update_xaxes(tickformat=".1%")
    return _layout(fig, title, 360)


def fan_chart(mc_result, s0: float, title: str) -> go.Figure:
    f = mc_result.fan
    x = list(range(0, len(f) + 1))
    fig = go.Figure()
    add = lambda a, b, col, nm: (fig.add_trace(go.Scatter(x=x, y=[s0] + list(f[b]), line=dict(width=0), showlegend=False)),  # noqa: E731
                                 fig.add_trace(go.Scatter(x=x, y=[s0] + list(f[a]), fill="tonexty", fillcolor=col,
                                                          line=dict(width=0), name=nm)))
    add("P10", "P90", "rgba(52,152,219,0.18)", "P10-P90")
    add("P25", "P75", "rgba(52,152,219,0.35)", "P25-P75")
    fig.add_trace(go.Scatter(x=x, y=[s0] + list(f["P50"]), line=dict(color="#f1c40f"), name="Median"))
    fig.update_xaxes(title="trading days ahead")
    return _layout(fig, title, 420)


def mc_terminal(mc: dict) -> go.Figure:
    fig = go.Figure()
    for k, r in mc.items():
        fig.add_trace(go.Histogram(x=r.terminal_returns, name=k, opacity=0.45, nbinsx=80, histnorm="probability density"))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(tickformat=".0%")
    return _layout(fig, "Terminal return distributions by simulation method", 420)


def calibration_curve(curve: list[dict]) -> go.Figure:
    if not curve:
        return empty("Reliability diagram", "insufficient OOS data")
    c = pd.DataFrame(curve)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dot", color="grey"), name="perfect"))
    fig.add_trace(go.Scatter(x=c["p_mean"], y=c["freq"], mode="lines+markers", marker=dict(size=4 + 20 * c["n"] / c["n"].max()),
                             name="calibrated OOS"))
    fig.update_xaxes(title="predicted P(up)", range=[0, 1])
    fig.update_yaxes(title="observed frequency", range=[0, 1])
    return _layout(fig, "Reliability diagram (out-of-sample, later folds)", 400)


def bar(s: pd.Series, title: str, h=400, fmt: str | None = None, color_sign=True) -> go.Figure:
    s = s.dropna()
    cols = ["#2ecc71" if v >= 0 else "#e74c3c" for v in s.values] if color_sign else None
    fig = go.Figure(go.Bar(x=s.values, y=[str(i) for i in s.index], orientation="h", marker_color=cols))
    fig.update_yaxes(autorange="reversed")
    if fmt:
        fig.update_xaxes(tickformat=fmt)
    return _layout(fig, title, max(h, 22 * len(s) + 80))


def components_chart(components: dict, contributions: dict) -> go.Figure:
    names = list(components)
    sc = [components[n].get("score") for n in names]
    ct = [contributions.get(n, 0) for n in names]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Component score (50 = neutral)", "Contribution to final score (pts)"))
    fig.add_trace(go.Bar(y=names, x=sc, orientation="h", marker_color=["#2ecc71" if (v or 50) >= 60 else "#e74c3c" if (v or 50) <= 40
                                                                        else "#f1c40f" for v in sc], name="score"), 1, 1)
    fig.add_trace(go.Bar(y=names, x=ct, orientation="h", marker_color=["#2ecc71" if v >= 0 else "#e74c3c" for v in ct],
                         name="contribution"), 1, 2)
    fig.add_vline(x=50, row=1, col=1, line=dict(dash="dot"))
    fig.update_xaxes(range=[0, 100], row=1, col=1)
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "Decision components", 460)


def heatmap(df: pd.DataFrame, title: str, h=520) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=df.values, x=list(df.columns), y=list(df.index), colorscale="RdBu", zmid=0,
                               text=np.round(df.values, 2), texttemplate="%{text}"))
    return _layout(fig, title, h)


def signal_history(close: pd.Series, sig: pd.DataFrame, title="Signal history") -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    c = close.reindex(sig.index) if len(sig) else close.tail(500)
    fig.add_trace(go.Scatter(x=close.loc[sig.index[0]:].index if len(sig) else c.index,
                             y=close.loc[sig.index[0]:].values if len(sig) else c.values, name="price", line=dict(color="#95a5a6")), 1, 1)
    for s_, col in SIG_COLORS.items():
        m = sig["signal"] == s_ if len(sig) else []
        if len(sig) and m.any():
            fig.add_trace(go.Scatter(x=sig.index[m], y=c[m], mode="markers", marker=dict(color=col, size=5), name=s_), 1, 1)
    if len(sig) and "score" in sig:
        fig.add_trace(go.Scatter(x=sig.index, y=sig["score"], name="final score", line=dict(color="#3498db")), 2, 1)
    return _layout(fig, title, 520)


def equity(bt) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    fig.add_trace(go.Scatter(x=bt.equity.index, y=bt.equity, name="strategy (after costs)"), 1, 1)
    fig.add_trace(go.Scatter(x=bt.benchmark.index, y=bt.benchmark, name="buy & hold"), 1, 1)
    fig.add_trace(go.Scatter(x=bt.exposure.index, y=bt.exposure, name="exposure", fill="tozeroy"), 2, 1)
    return _layout(fig, "Walk-forward backtest (next-open execution, costs included)", 520)


def prediction_errors(j: pd.DataFrame) -> go.Figure:
    r = j.dropna(subset=["actual_return"]) if not j.empty and "actual_return" in j else pd.DataFrame()
    if r.empty:
        return empty("Prediction vs actual", "No resolved predictions yet - they resolve automatically after their horizon")
    fig = go.Figure()
    for h, g in r.groupby("horizon"):
        fig.add_trace(go.Scatter(x=g["expected_return"], y=g["actual_return"], mode="markers", name=f"{h}D"))
    lim = float(np.nanmax(np.abs(r[["expected_return", "actual_return"]].values))) or 0.05
    fig.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], line=dict(dash="dot", color="grey"), name="perfect"))
    fig.update_xaxes(title="predicted", tickformat=".1%")
    fig.update_yaxes(title="actual", tickformat=".1%")
    return _layout(fig, "Prediction journal: predicted vs actual", 420)


def oos_performance(oos: pd.DataFrame, title: str) -> go.Figure:
    if oos is None or oos.empty:
        return empty(title)
    o = oos.dropna(subset=["ens_ret", "y_ret"])
    hit = (np.sign(o["ens_ret"]) == np.sign(o["y_ret"])).astype(float).rolling(126, min_periods=40).mean()
    cov = ((o["y_ret"] >= o["ci_lo"]) & (o["y_ret"] <= o["ci_hi"])).astype(float).where(o["ci_lo"].notna())\
        .rolling(126, min_periods=40).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hit.index, y=hit, name="rolling directional accuracy (126d)"))
    fig.add_trace(go.Scatter(x=cov.index, y=cov, name="rolling interval coverage (126d)"))
    fig.add_hline(y=0.5, line=dict(dash="dot"))
    fig.update_yaxes(tickformat=".0%")
    return _layout(fig, title, 380)


def analogues_chart(an: pd.DataFrame) -> go.Figure:
    if an is None or an.empty:
        return empty("Historical analogues", "no analogues")
    fig = go.Figure()
    for col, nm in (("ret_5d", "5D"), ("ret_20d", "20D"), ("ret_60d", "60D"), ("max_dd_60d", "max DD 60D")):
        fig.add_trace(go.Bar(x=[str(d) for d in an["date"]], y=an[col], name=nm))
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(barmode="group")
    return _layout(fig, "What happened after the most similar historical states", 400)
