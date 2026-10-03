"""RESEARCH LAB / ABLATION STUDIES - the application tries to disprove itself.

All experiments use the same purged walk-forward protocol and a fixed reference model
(LightGBM, else HistGradientBoosting) so differences come from the *data/setup*, not
the model. Each answer is reported with its evidence and sample size; "cannot test" is
a valid answer (e.g. news sentiment without enough history).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..config.settings import Settings
from ..features.builder import FeatureSet
from ..labeling.labels import build_labels
from ..models.forecast_service import REGIME_FEATURES
from ..models.tree.models import HistGBForecaster, LGBMForecaster
from ..validation.metrics import brier, diebold_mariano, regression_metrics
from ..validation.splits import sample_weights, walk_forward

GROUPS = ["technical", "volatility", "cross_asset", "breadth", "macro", "sentiment"]


def _ref_model():
    return LGBMForecaster if LGBMForecaster.info.available else HistGBForecaster


def wf_eval(X: pd.DataFrame, lab: pd.DataFrame, h: int, seed: int = 42, weights: np.ndarray | None = None,
            n_splits: int = 5, start: pd.Timestamp | None = None, per_row_model: Callable | None = None) -> dict:
    X = X.iloc[252:]
    y_ret, y_up = lab["fwd_ret"].reindex(X.index), lab["direction"].reindex(X.index)
    le = lab["label_end"].reindex(X.index)
    le = le.where(le <= X.index[-1])
    min_train = max(500, int(le.notna().sum() * 0.4))
    preds, ups, idx = [], [], []
    cls = _ref_model()
    for sp in walk_forward(X.index, le, n_splits, min_train, embargo=max(5, min(h, 20))):
        tr = sp.train
        if start is not None:
            tr = tr[X.index[tr] >= start]
        w = None if weights is None else weights[252:][tr]
        if w is not None:
            tr, w = tr[w > 0], w[w > 0]
        if len(tr) < 250:
            continue
        if per_row_model is not None:
            p = per_row_model(X.iloc[tr], y_ret.iloc[tr], y_up.iloc[tr], X.iloc[sp.test])
        else:
            m = cls(h, seed).fit(X.iloc[tr], y_ret.iloc[tr], y_up.iloc[tr], w)
            p = m.predict(X.iloc[sp.test])
        preds.append(p["ret"])
        ups.append(p["p_up"])
        idx.append(X.index[sp.test])
    if not preds:
        return {"n": 0}
    pr, pu = pd.concat(preds), pd.concat(ups)
    yr, yu = y_ret.reindex(pr.index), y_up.reindex(pr.index)
    out = regression_metrics(yr, pr)
    out["brier"] = brier(yu, pu)
    out["_pred"] = pr
    out["_y"] = yr
    return out


def _strip(d: dict) -> dict:
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in d.items() if not k.startswith("_")}


def ablation_study(fs: FeatureSet, target: pd.DataFrame, regimes: pd.DataFrame | None, s: Settings, h: int = 20,
                   progress=None) -> pd.DataFrame:
    lab = build_labels(target, [h])[h]
    base_cols = fs.model_columns()
    reg_cols = [c for c in REGIME_FEATURES if regimes is not None and c in regimes]

    def design(cols):
        X = fs.frame[cols].copy()
        for c in reg_cols:
            X[c] = regimes[c].reindex(X.index)
        return X

    rows = []
    base = wf_eval(design(base_cols), lab, h, s.models.random_seed)
    rows.append({"experiment": "BASE (all groups)", "removed": "", **_strip(base)})
    for g in GROUPS:
        if g == "sentiment" and "sentiment" not in fs.groups:
            rows.append({"experiment": "remove sentiment", "removed": "sentiment",
                         "note": f"CANNOT TEST: only {fs.sentiment_coverage} days of news history "
                                 f"(need 250+). The value of news is unproven, not assumed."})
            continue
        if g not in fs.groups:
            rows.append({"experiment": f"remove {g}", "removed": g, "note": "group unavailable (no data)"})
            continue
        if progress:
            progress(f"Ablation: removing {g}", GROUPS.index(g) / len(GROUPS))
        cols = [c for c in base_cols if c not in fs.groups[g]]
        r = wf_eval(design(cols), lab, h, s.models.random_seed)
        dm = diebold_mariano(base["_pred"] - base["_y"], r["_pred"].reindex(base["_pred"].index) - base["_y"], h) \
            if r.get("n") else {}
        rows.append({"experiment": f"remove {g}", "removed": g, **_strip(r),
                     "delta_ic_vs_base": (r.get("ic_spearman") or 0) - (base.get("ic_spearman") or 0),
                     "delta_da_vs_base": (r.get("directional_accuracy") or 0) - (base.get("directional_accuracy") or 0),
                     "dm_p_value": dm.get("p_value"),
                     "verdict": _verdict(base, r, dm)})
    # remove the single strongest feature
    from ..features.selection import select_features

    X_all = design(base_cols)
    lab_ok = lab["fwd_ret"].reindex(X_all.index)
    sel, scores = select_features(X_all.iloc[252:-h - 1], lab_ok.iloc[252:-h - 1], 10, s.models.random_seed, use_stability=False)
    if sel:
        strongest = sel[0]
        r = wf_eval(X_all.drop(columns=[strongest]), lab, h, s.models.random_seed)
        rows.append({"experiment": f"remove strongest feature ({strongest})", "removed": strongest, **_strip(r),
                     "delta_ic_vs_base": (r.get("ic_spearman") or 0) - (base.get("ic_spearman") or 0),
                     "delta_da_vs_base": (r.get("directional_accuracy") or 0) - (base.get("directional_accuracy") or 0),
                     "verdict": "robust" if (r.get("ic_spearman") or 0) > 0.5 * (base.get("ic_spearman") or 0) else
                                "fragile: edge depends heavily on one feature"})
    # technical only
    tech = fs.groups.get("technical", []) + fs.groups.get("volatility", [])
    r = wf_eval(design(tech), lab, h, s.models.random_seed)
    rows.append({"experiment": "ONLY technical + volatility", "removed": "all non-technical", **_strip(r),
                 "delta_ic_vs_base": (r.get("ic_spearman") or 0) - (base.get("ic_spearman") or 0),
                 "delta_da_vs_base": (r.get("directional_accuracy") or 0) - (base.get("directional_accuracy") or 0)})
    return pd.DataFrame(rows)


def _verdict(base: dict, r: dict, dm: dict) -> str:
    if not r.get("n"):
        return "n/a"
    d_ic = (r.get("ic_spearman") or 0) - (base.get("ic_spearman") or 0)
    sig = dm.get("p_value") is not None and dm["p_value"] < 0.10
    if d_ic < -0.02:
        return "group HELPS" + (" (significant)" if sig else " (not significant)")
    if d_ic > 0.02:
        return "group HURTS / adds noise" + (" (significant)" if sig else " (not significant)")
    return "no measurable effect"


def regime_option_comparison(fs: FeatureSet, target: pd.DataFrame, regimes: pd.DataFrame, s: Settings, h: int = 20) -> dict:
    """Option A: separate models per regime bucket; Option B: regime variables in one model; plus no-regime baseline."""
    lab = build_labels(target, [h])[h]
    base = fs.frame[fs.model_columns()].copy()
    XB = base.copy()
    for c in REGIME_FEATURES:
        if c in regimes:
            XB[c] = regimes[c].reindex(XB.index)
    bucket = regimes["regime_rule"].reindex(base.index).map(
        lambda r: "up" if r in ("Strong bull", "Bull", "Recovery") else ("stress" if r in ("High volatility", "Crisis",
                                                                                          "Strong bear", "Bear") else "neutral"))
    cls = _ref_model()

    def per_regime(Xtr, ytr, utr, Xte):
        out_r = pd.Series(np.nan, index=Xte.index)
        out_p = pd.Series(np.nan, index=Xte.index)
        glob = cls(h, s.models.random_seed).fit(Xtr, ytr, utr)
        gp = glob.predict(Xte)
        for b in bucket.reindex(Xte.index).dropna().unique():
            mtr = (bucket.reindex(Xtr.index) == b).values
            mte = (bucket.reindex(Xte.index) == b).values
            if mtr.sum() >= 300:
                p = cls(h, s.models.random_seed).fit(Xtr[mtr], ytr[mtr], utr[mtr]).predict(Xte[mte])
            else:
                p = gp[mte]
            out_r[mte], out_p[mte] = p["ret"].values, p["p_up"].values
        out_r = out_r.fillna(gp["ret"])
        out_p = out_p.fillna(gp["p_up"])
        return pd.DataFrame({"ret": out_r, "p_up": out_p})

    res = {"No regime information": _strip(wf_eval(base, lab, h, s.models.random_seed)),
           "Option B: regime variables in unified model": _strip(wf_eval(XB, lab, h, s.models.random_seed)),
           "Option A: separate models per regime bucket": _strip(wf_eval(base, lab, h, s.models.random_seed,
                                                                         per_row_model=per_regime))}
    best = max(res, key=lambda k: (res[k].get("ic_spearman") or -9))
    return {"results": res, "best": best}


def history_and_window_study(fs: FeatureSet, target: pd.DataFrame, s: Settings, h: int = 20) -> dict:
    lab = build_labels(target, [h])[h]
    X = fs.frame[fs.model_columns()]
    out = {}
    for name, scheme in (("full history", "full"), (f"recent {s.models.recent_window_years:g}y window", "recent"),
                         (f"exp. weighted (half-life {s.models.exp_halflife_years:g}y)", "exp_weighted")):
        w = sample_weights(X.index, scheme, s.models.exp_halflife_years, s.models.recent_window_years)
        out[name] = _strip(wf_eval(X, lab, h, s.models.random_seed, weights=w))
    for yrs in (10, 20):
        start = X.index[-1] - pd.DateOffset(years=yrs)
        out[f"only last {yrs} years of training data"] = _strip(wf_eval(X, lab, h, s.models.random_seed,
                                                                        start=pd.Timestamp(start)))
    return out


def custom_experiment(fs: FeatureSet, target: pd.DataFrame, regimes: pd.DataFrame | None, s: Settings, groups: list[str],
                      h: int = 20, window: str = "full", years: int | None = None, regime_vars: bool = True) -> dict:
    lab = build_labels(target, [h])[h]
    cols = [c for g in groups for c in fs.groups.get(g, [])]
    if not cols:
        return {"error": "no features in the selected groups"}
    X = fs.frame[cols].copy()
    if regime_vars and regimes is not None:
        for c in REGIME_FEATURES:
            if c in regimes:
                X[c] = regimes[c].reindex(X.index)
    w = sample_weights(X.index, window, s.models.exp_halflife_years, s.models.recent_window_years)
    start = pd.Timestamp(X.index[-1] - pd.DateOffset(years=years)) if years else None
    return _strip(wf_eval(X, lab, h, s.models.random_seed, weights=w, start=start))


def research_questions(forecast, ablation: pd.DataFrame | None, regime_cmp: dict | None, window: dict | None,
                       backtest=None) -> list[dict]:
    """The application's attempt to disprove its own assumptions."""
    q = []
    prim = None
    if forecast is not None:
        prim = forecast.horizons.get(20) or next(iter(forecast.horizons.values()), None)

    def row(question, answer, evidence):
        q.append({"question": question, "answer": answer, "evidence": evidence})

    if ablation is not None and not ablation.empty:
        for g, qq in (("sentiment", "Does news sentiment improve forecasting?"), ("macro", "Does macro data improve forecasting?")):
            r = ablation[ablation["removed"] == g]
            if r.empty:
                row(qq, "not tested", "")
            elif "note" in r and isinstance(r.iloc[0].get("note"), str) and r.iloc[0].get("note"):
                row(qq, "CANNOT TEST YET", r.iloc[0]["note"])
            else:
                row(qq, r.iloc[0].get("verdict", "n/a"), f"ΔIC when removed {r.iloc[0].get('delta_ic_vs_base', np.nan):+.3f}, "
                    f"DM p={r.iloc[0].get('dm_p_value')}")
        st = ablation[ablation["experiment"].str.startswith("remove strongest")]
        if not st.empty:
            row("Do results survive removing the strongest feature?", st.iloc[0].get("verdict"),
                f"ΔIC {st.iloc[0].get('delta_ic_vs_base', np.nan):+.3f}")
    if prim is not None and prim.model_scores:
        sc = prim.model_scores
        dl = {k: v for k, v in sc.items() if k in ("lstm", "gru", "tcn", "transformer", "nbeats")}
        tr = {k: v for k, v in sc.items() if k in ("xgboost", "lightgbm", "catboost", "hist_gb", "random_forest", "extra_trees")}
        if dl and tr:
            bdl = max(dl.items(), key=lambda kv: kv[1].get("ic_spearman") or -9)
            btr = max(tr.items(), key=lambda kv: kv[1].get("ic_spearman") or -9)
            row("Does deep learning outperform tree models?",
                "YES" if (bdl[1].get("ic_spearman") or -9) > (btr[1].get("ic_spearman") or -9) else "NO",
                f"best DL {bdl[0]} IC {bdl[1].get('ic_spearman', np.nan):+.3f} vs best tree {btr[0]} IC {btr[1].get('ic_spearman', np.nan):+.3f}")
        else:
            row("Does deep learning outperform tree models?", "not tested in this mode", "run Deep Research")
        ec = prim.ensemble_comparison
        if ec:
            ens_best = max((m for m in ec if m != "best_single"), key=lambda m: ec[m].get("ic_spearman") or -9)
            row("Does ensemble forecasting outperform individual models?",
                "YES" if (ec[ens_best].get("ic_spearman") or -9) > (ec.get("best_single", {}).get("ic_spearman") or -9) else "NO",
                f"{ens_best} IC {ec[ens_best].get('ic_spearman', np.nan):+.3f} vs best single (selected OOS) "
                f"{ec.get('best_single', {}).get('ic_spearman', np.nan):+.3f}")
        naive = sc.get("naive_drift", {})
        ens = ec.get(prim.ensemble_method, {}) if ec else {}
        row("Does the model beat a naive drift benchmark?",
            "YES" if (ens.get("directional_accuracy") or 0) > (naive.get("directional_accuracy") or 0) + 0.005 else "NO",
            f"ensemble DA {ens.get('directional_accuracy', np.nan):.1%} vs naive {naive.get('directional_accuracy', np.nan):.1%}")
    if regime_cmp:
        r = regime_cmp["results"]
        row("Does regime detection improve robustness?", f"best setup: {regime_cmp['best']}",
            "; ".join(f"{k}: IC {v.get('ic_spearman', np.nan):+.3f}" for k, v in r.items()))
    if window:
        best = max(window, key=lambda k: window[k].get("ic_spearman") or -9)
        row("Do results survive different time periods / training windows?", f"best: {best}",
            "; ".join(f"{k}: IC {v.get('ic_spearman', np.nan):+.3f}, DA {v.get('directional_accuracy', np.nan):.1%}" for k, v in window.items()))
    if backtest is not None and backtest.result is not None:
        rb = backtest.robustness
        row("Do signals remain profitable after transaction costs?",
            "YES" if (backtest.result.metrics.get("cagr") or -1) > 0 else "NO",
            f"CAGR {backtest.result.metrics.get('cagr', np.nan):.1%} vs buy&hold {backtest.result.benchmark_metrics.get('cagr', np.nan):.1%}; "
            f"cost sensitivity {rb.get('cost_sensitivity')}")
        row("Do results survive parameter changes?", "see threshold sensitivity", str(rb.get("threshold_sensitivity")))
        row("Does the strategy beat buy-and-hold (risk-adjusted)?", "YES" if rb.get("beats_buy_and_hold_sharpe") else "NO",
            f"Sharpe {backtest.result.metrics.get('sharpe', np.nan):.2f} vs {backtest.result.benchmark_metrics.get('sharpe', np.nan):.2f}")
    return q
