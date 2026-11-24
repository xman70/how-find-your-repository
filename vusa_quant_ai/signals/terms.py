"""SHORT-TERM and LONG-TERM traffic lights.

The same 12 evidence components are combined twice, with weights that reflect what
matters at each time scale:

* SHORT TERM (next ~1-4 weeks): ML forecasts for 5/10/20 days, momentum, volatility,
  sentiment, event risk.
* LONG TERM (next ~6-12 months): ML forecasts for 60/120/252 days, trend, valuation,
  macro, regime.

The ML part of each light blends all horizons of that term, weighted by each horizon's
out-of-sample confidence, and measures the probability of a rise *relative to the base
rate* as well as in absolute terms: equities rise in most 12-month periods, so "75 %
chance of being up in a year" is ordinary, not a buy signal by itself.

Each light keeps its own history (hysteresis per term) and comes with a TRACK RECORD:
what actually happened after the light showed each colour in the past, using only
out-of-sample forecasts (see :func:`light_track_record`).
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from ..config.settings import SignalSettings
from .components import (Component, breadth_component, macro_component, momentum_component, sentiment_component,
                         trend_component, volatility_component)
from .decision import decide

TERMS = {
    "short": {"title": "SHORT TERM", "subtitle": "next 1-4 weeks", "horizons": (5, 10, 20), "eval_horizon": 20,
              "weights": {"trend": 0.8, "momentum": 1.2, "valuation": 0.1, "macro": 0.4, "sentiment": 0.8,
                          "volatility": 1.1, "breadth": 0.8, "regime": 0.8, "event_risk": 1.0, "ml_forecast": 1.5,
                          "analogues": 0.8, "risk_reward": 1.0}},
    "long": {"title": "LONG TERM", "subtitle": "next 6-12 months", "horizons": (60, 120, 252), "eval_horizon": 252,
             "weights": {"trend": 1.3, "momentum": 0.6, "valuation": 1.2, "macro": 1.3, "sentiment": 0.2,
                         "volatility": 0.4, "breadth": 0.7, "regime": 1.0, "event_risk": 0.3, "ml_forecast": 1.3,
                         "analogues": 0.8, "risk_reward": 0.8}},
}

PLAIN = {  # component -> (bullish phrase, bearish phrase)
    "trend": ("The price trend is up", "The price trend is down"),
    "momentum": ("Recent momentum is positive", "Recent momentum is negative"),
    "valuation": ("Valuation looks reasonable", "The market looks expensive"),
    "macro": ("The economic backdrop is supportive", "The economic backdrop is a headwind"),
    "sentiment": ("Market mood is calm / positive", "Market mood is nervous / negative"),
    "volatility": ("Volatility is low (calm market)", "Volatility is high (unstable market)"),
    "breadth": ("Most sectors are participating", "Few sectors are participating"),
    "regime": ("The market regime is bullish", "The market regime is bearish / stressed"),
    "event_risk": ("No worrying events detected", "Risky events or shocks detected"),
    "ml_forecast": ("The forecasting models lean up", "The forecasting models lean down"),
    "analogues": ("Similar past situations ended well", "Similar past situations ended badly"),
    "risk_reward": ("Upside outweighs downside", "Downside outweighs upside"),
    "history_model": ("Similar setups did better than normal in the past", "Similar setups did worse than normal in the past"),
}
COLOR = {"BUY": "GREEN", "HOLD": "YELLOW", "SELL": "RED"}


def base_rate(hr) -> float:
    b = (hr.model_preds or {}).get("naive_drift", {}).get("p_up")
    return float(b) if b is not None and np.isfinite(b) else 0.5


def ml_edge(p_up, base):
    """Blend of absolute edge (vs 50 %) and relative edge (vs the base rate)."""
    return 0.5 * (p_up - 0.5) + 0.5 * (p_up - base)


def ml_term_component(hrs: list) -> Component:
    hrs = [h for h in hrs if h is not None and np.isfinite(h.p_up)]
    if not hrs:
        return Component("ml_forecast", np.nan, ["No ML forecast for this term"], {}, 0.0)
    w = np.array([max(0.05, h.confidence if np.isfinite(h.confidence) else 0.05) for h in hrs])
    edges = np.array([ml_edge(h.p_up, base_rate(h)) for h in hrs])
    edge = float(np.average(edges, weights=w))
    conf = float(np.average([h.confidence if np.isfinite(h.confidence) else 0.05 for h in hrs], weights=w))
    s = 50 + 50 * np.tanh(2 * edge / 0.25) * (0.5 + 0.5 * conf)
    ev = [f"{h.horizon}D: expected {h.expected_return:+.1%}, P(up) {h.p_up:.0%} (normal for this horizon: "
          f"{base_rate(h):.0%}), confidence {h.confidence:.0%}" for h in hrs]
    return Component("ml_forecast", float(s), ev, {"edge": edge, "confidence": conf, "horizons": [h.horizon for h in hrs]})


def term_risk(term: str, risk: float, comps: dict[str, Component]) -> float:
    if term == "short":
        return float(risk)
    g = lambda k: comps[k].score if comps.get(k) is not None and np.isfinite(comps[k].score) else 50.0  # noqa: E731
    return float(0.4 * risk + 0.3 * (100 - g("macro")) + 0.3 * (100 - g("valuation")))


def plain_reasons(components: dict, contributions: dict, n: int = 3) -> tuple[list[str], list[str]]:
    pos = sorted([(v, k) for k, v in contributions.items() if v > 0.25], reverse=True)[:n]
    neg = sorted([(v, k) for k, v in contributions.items() if v < -0.25])[:n]
    return ([PLAIN.get(k, (k, k))[0] for _, k in pos], [PLAIN.get(k, (k, k))[1] for _, k in neg])


def decide_term(term: str, comps: dict[str, Component], forecast, risk: float, cfg: SignalSettings, history: pd.DataFrame | None,
                penalties: dict, ctx: dict, analogue_fn=None, risk_reward_fn=None, learned: dict | None = None) -> dict:
    spec = TERMS[term]
    hrs = [forecast.horizons.get(h) for h in spec["horizons"]] if forecast is not None else []
    hrs = [h for h in hrs if h is not None and np.isfinite(h.p_up)]
    c = dict(comps)
    c["ml_forecast"] = ml_term_component(hrs)
    main = max(hrs, key=lambda h: h.horizon) if hrs else None  # headline horizon of the term
    if risk_reward_fn is not None and main is not None:
        c["risk_reward"] = risk_reward_fn(main)
    if analogue_fn is not None:
        c["analogues"] = analogue_fn(term)
    tcfg = deepcopy(cfg)
    tcfg.weights = dict(spec["weights"])
    if learned and learned.get("alpha", 0) > 0 and np.isfinite(learned.get("learned_score", np.nan)):
        a = learned["alpha"]
        c["history_model"] = Component("history_model", learned["learned_score"],
                                       [f"Model trained on past outcomes: P(better than normal) "
                                        f"{learned['p_better_than_normal']:.0%} (weight {a:.0%} of the light)"])
        tcfg.weights["history_model"] = a / (1 - a) * sum(tcfg.weights.values())
    conf = float(np.mean([h.confidence for h in hrs if np.isfinite(h.confidence)])) if hrs else 0.2
    r = term_risk(term, risk, c)
    no_skill = False
    hold_auc = ((learned or {}).get("selection") or {}).get("holdout_auc_chosen")
    if hold_auc is not None and np.isfinite(hold_auc) and hold_auc < 0.5:
        penalties = {**penalties, "light_without_proven_skill": 0.3}  # do not show confidence the light has not earned
        no_skill = True
    d = decide(c, r, tcfg, conf, history, penalties, {**ctx, "p_up": main.p_up if main else None,
                                                      "p_lt5": main.p_lt5 if main else None}).to_dict()
    why, against = plain_reasons(d["components"], d["contributions"])
    d.update({"term": term, "title": spec["title"], "subtitle": spec["subtitle"], "color": COLOR[d["signal"]],
              "horizons_used": [h.horizon for h in hrs], "reasons_for": why, "reasons_against": against,
              "headline": None if main is None else {
                  "horizon": main.horizon, "expected_return": main.expected_return, "p_up": main.p_up,
                  "base_rate": base_rate(main), "interval": list(main.interval), "interval_level": main.interval_level,
                  "warnings": main.warnings}})
    if learned:
        d["calibration"] = {k: v for k, v in learned.items() if k != "track_record"}
    if no_skill:
        d["notes"].append(f"On unseen recent years this light ranked outcomes no better than chance (skill {hold_auc:.2f}); "
                          "its confidence is reduced.")
    if not hrs:
        d["notes"].append(f"No ML forecast available for horizons {spec['horizons']} in this mode.")
    return d


# ---------------------------------------------------------------------------------------- track record
RECON = ["ml_forecast", "trend", "momentum", "volatility", "breadth", "macro", "sentiment", "regime"]


def _row_components(f: pd.Series, rv_pct: float, regime_score: float, edge: float) -> dict[str, float]:
    reg = {"vol_percentile": rv_pct}
    return {"ml_forecast": 50 + 50 * np.tanh(2 * edge / 0.25) * 0.75 if np.isfinite(edge) else np.nan,
            "trend": trend_component(f).score, "momentum": momentum_component(f).score,
            "volatility": volatility_component(f, reg).score, "breadth": breadth_component(f).score,
            "macro": macro_component(f).score, "sentiment": sentiment_component(None, f).score,
            "regime": regime_score}


def _heuristic(row: dict, w: dict) -> float:
    ok = {k: v for k, v in row.items() if v is not None and np.isfinite(v)}
    return sum(w[k] * v for k, v in ok.items()) / sum(w[k] for k in ok) if ok else np.nan


def _term_hrs(term: str, forecast) -> list:
    hrs = [forecast.horizons.get(h) for h in TERMS[term]["horizons"]]
    return [h for h in hrs if h is not None and h.oos is not None and "ens_up" in h.oos]


def historical_components(term: str, forecast, features: pd.DataFrame, regimes: pd.DataFrame, step: int = 5) -> pd.DataFrame:
    """Reconstruct the light's evidence on past dates from information available then.

    Uses out-of-sample ensemble probabilities (walk-forward) and the causal components that can be
    rebuilt historically (trend, momentum, volatility, breadth, macro, regime, volatility
    sentiment). News, valuation inputs, analogues and risk/reward are not reconstructable and are
    left out. Every ``step``-th session is used (outcomes span 20-252 overlapping days, so daily
    rows add little information). Returns one column per component plus ``heuristic``."""
    hrs = _term_hrs(term, forecast)
    if not hrs:
        return pd.DataFrame()
    ml = pd.DataFrame({h.horizon: ml_edge(h.oos["ens_up"], base_rate(h)) for h in hrs}).dropna()
    if ml.empty:
        return pd.DataFrame()
    wts = np.array([max(0.05, h.confidence if np.isfinite(h.confidence) else 0.05) for h in hrs])
    edge = ((ml * wts).sum(axis=1) / wts.sum()).iloc[::step]
    idx = edge.index
    F, R = features.reindex(idx), regimes.reindex(idx)
    rv = R["rv_pct"] if "rv_pct" in R else pd.Series(np.nan, index=idx)
    rs = R["regime_score"] if "regime_score" in R else pd.Series(np.nan, index=idx)
    rows = [_row_components(F.loc[d], rv.at[d], rs.at[d], edge.at[d]) for d in idx]
    out = pd.DataFrame(rows, index=idx)[RECON]
    w = TERMS[term]["weights"]
    out["heuristic"] = [_heuristic(r, w) for r in rows]
    return out


def historical_term_scores(term: str, forecast, features: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    hc = historical_components(term, forecast, features, regimes)
    return pd.DataFrame() if hc.empty else hc[["heuristic"]].rename(columns={"heuristic": "score"})


def calibrate_light(term: str, forecast, features: pd.DataFrame, regimes: pd.DataFrame, cfg: SignalSettings,
                    today: pd.Timestamp, today_edge: float) -> dict:
    """Learn (walk-forward) how the evidence relates to better-than-normal outcomes, choose the
    blend with the hand-set light on early history, and build the track record of the chosen light."""
    from . import light_model as LM

    hrs = _term_hrs(term, forecast)
    out = {"alpha": 0.0, "selection": {"alpha": 0.0, "reason": "no out-of-sample history", "candidates": {}}}
    if not hrs:
        out["track_record"] = {"available": False, "reason": "no out-of-sample history for this term"}
        return out
    h = max(x.horizon for x in hrs)
    hc = historical_components(term, forecast, features, regimes)
    if hc.empty:
        out["track_record"] = {"available": False, "reason": "no out-of-sample history for this term"}
        return out
    y = forecast.horizons[h].oos["y_ret"].reindex(hc.index)
    p_oos, final, coefs = LM.walk_forward(hc[RECON], y, h, step=5)
    learned = pd.Series(LM.learned_score(p_oos), index=hc.index)
    sel = LM.select_blend(hc["heuristic"], learned, y, h=h, step=5)
    a = sel["alpha"]
    blended = hc["heuristic"] if a == 0 else ((1 - a) * hc["heuristic"] + a * learned).fillna(hc["heuristic"])
    out.update({"alpha": a, "selection": sel, "coefficients": coefs})
    # today's learned score from today's reconstructable evidence
    if final is not None and today in features.index:
        f = features.loc[today]
        rv = regimes["rv_pct"].get(today, np.nan) if "rv_pct" in regimes else np.nan
        rs = regimes["regime_score"].get(today, np.nan) if "regime_score" in regimes else np.nan
        row = pd.DataFrame([_row_components(f, rv, rs, today_edge)])[RECON]
        p_now = float(final.predict_proba(LM._design(row))[:, 1][0])
        out.update({"p_better_than_normal": p_now, "learned_score": float(LM.learned_score(p_now))})
    out["track_record"] = light_track_record(term, forecast, features, regimes, cfg, scores=blended.to_frame("score"),
                                             label="learned blend" if a else "hand-set weights")
    return out


def light_track_record(term: str, forecast, features: pd.DataFrame, regimes: pd.DataFrame, cfg: SignalSettings,
                       scores: pd.DataFrame | None = None, label: str = "hand-set weights") -> dict:
    """What happened after each colour in the past (out-of-sample)."""
    spec = TERMS[term]
    avail = [h for h in spec["horizons"] if forecast.horizons.get(h) is not None and forecast.horizons[h].oos is not None]
    if not avail:
        return {"available": False, "reason": "no out-of-sample history for this term"}
    h = max(avail)
    sc = scores if scores is not None else historical_term_scores(term, forecast, features, regimes)
    if sc.empty:
        return {"available": False, "reason": "no out-of-sample history for this term"}
    y = forecast.horizons[h].oos["y_ret"].reindex(sc.index)
    d = pd.DataFrame({"score": sc["score"], "y": y}).dropna()
    if len(d) < 100:
        return {"available": False, "reason": f"only {len(d)} historical observations"}
    d["light"] = np.where(d["score"] >= cfg.buy_threshold, "GREEN", np.where(d["score"] <= cfg.sell_threshold, "RED", "YELLOW"))
    base_up, base_mean = float((d["y"] > 0).mean()), float(d["y"].mean())
    step = max(1, int(np.median(np.diff(np.searchsorted(forecast.horizons[h].oos.index, d.index)))) if len(d) > 1 else 1)
    out = {"available": True, "horizon": h, "n": int(len(d)), "from": str(d.index[0].date()), "to": str(d.index[-1].date()),
           "independent_periods": int(len(d) * step // h), "base_p_up": base_up, "base_mean": base_mean, "by_light": {}}
    for col in ("GREEN", "YELLOW", "RED"):
        g = d[d["light"] == col]
        if len(g) == 0:
            out["by_light"][col] = {"n": 0, "share": 0.0}
            continue
        out["by_light"][col] = {"n": int(len(g)), "share": float(len(g) / len(d)), "p_up": float((g["y"] > 0).mean()),
                                "mean_return": float(g["y"].mean()), "worst": float(g["y"].min()),
                                "independent_periods": int(max(1, len(g) * step // h))}
    g, r = out["by_light"]["GREEN"], out["by_light"]["RED"]
    # Is GREEN really better than RED (or than average)? Judge against the noise level, counting
    # only independent (non-overlapping) periods.
    sd = float(d["y"].std())
    useful, diff, se = None, None, None
    need = max(3, 3 * h // step)  # at least ~3 independent periods
    if g.get("n", 0) >= need:
        if r.get("n", 0) >= need:
            diff = g["mean_return"] - r["mean_return"]
            se = sd * np.sqrt(1 / g["independent_periods"] + 1 / r["independent_periods"])
        else:
            diff = g["mean_return"] - base_mean
            se = sd * np.sqrt(1 / g["independent_periods"])
        useful = True if diff > 2 * se else (False if diff <= 0 else None)
    out.update({"informative": useful, "green_minus_red": diff, "noise_2se": None if se is None else 2 * se})
    if diff is None:
        out["verdict"] = "Too few past GREEN periods to judge this light."
    elif useful:
        out["verdict"] = "In the past, GREEN was followed by clearly better results than RED / average."
    elif useful is False:
        out["verdict"] = "In the past this light did NOT separate good from bad periods - treat it with caution."
    else:
        out["verdict"] = ("In the past GREEN did somewhat better, but the difference is within the range of chance - "
                          "the light is unproven.")
    out["light_version"] = label
    out["note"] = (f"Simplified historical reconstruction ({label}; ML + trend, momentum, volatility, breadth, macro, regime); "
                   f"overlapping {h}-day periods mean only ~{out['independent_periods']} independent observations. "
                   "Past results do not guarantee future results.")
    return out
