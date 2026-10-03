"""MonitoringService: prediction journal, live scorecard, model drift, feature drift,
champion/challenger promotion, model retirement and the end-of-run SELF-AUDIT."""
from __future__ import annotations

import json
import uuid

import numpy as np
import pandas as pd
from scipy import stats

from ..core.runtime import utcnow_iso
from ..database.db import Database
from ..validation.metrics import brier, diebold_mariano, ece, trading_metrics


# ------------------------------------------------------------------ prediction journal
def journal_predictions(db: Database, run_id: str, as_of: str, price: float, forecast, signal: str,
                        regime: str | None, target_dates: dict[int, str]) -> list[str]:
    ids = []
    for h, hr in forecast.horizons.items():
        if not np.isfinite(hr.expected_return):
            continue
        pid = f"{run_id}_H{h}"
        db.upsert("predictions", {
            "prediction_id": pid, "run_id": run_id, "as_of": as_of, "horizon": h,
            "model_id": hr.model_ids.get("ensemble", "ensemble"), "price": price, "expected_return": hr.expected_return,
            "p_up": hr.p_up, "p_down": hr.p_down, "p_gt5": hr.p_gt5, "p_lt5": hr.p_lt5, "p_dd10": hr.p_dd10,
            "confidence": hr.confidence, "signal": signal, "target_date": target_dates.get(h), "actual_return": None,
            "error": None, "resolved_at": None, "regime": regime})
        db.upsert("prediction_intervals", {"prediction_id": pid, "level": hr.interval_level, "lower": hr.interval[0],
                                           "upper": hr.interval[1], "method": "conformal (OOS residual quantiles)"})
        for name, p in hr.model_preds.items():
            db.upsert("predictions", {
                "prediction_id": f"{pid}_{name}", "run_id": run_id, "as_of": as_of, "horizon": h,
                "model_id": hr.model_ids.get(name, name), "price": price, "expected_return": p["ret"], "p_up": p["p_up"],
                "p_down": 1 - p["p_up"], "p_gt5": None, "p_lt5": None, "p_dd10": None, "confidence": None, "signal": None,
                "target_date": target_dates.get(h), "actual_return": None, "error": None, "resolved_at": None, "regime": regime})
        ids.append(pid)
    return ids


def resolve_predictions(db: Database, close: pd.Series, open_: pd.Series | None = None) -> int:
    """Fill actual outcomes for predictions whose target date has passed (same executable
    convention as training labels: entry at next open after as-of, exit at target close)."""
    rows = db.query("SELECT prediction_id, as_of, horizon, price, expected_return, target_date FROM predictions "
                    "WHERE actual_return IS NULL AND target_date IS NOT NULL")
    n = 0
    last = close.index[-1]
    for r in rows:
        td = pd.Timestamp(r["target_date"])
        if td > last:
            continue
        a = pd.Timestamp(r["as_of"][:10])
        pos = close.index.searchsorted(a, side="right")
        if pos >= len(close):
            continue
        entry = open_.iloc[pos] if open_ is not None and pd.notna(open_.iloc[pos]) else close.iloc[pos]
        exit_ = close.loc[:td].iloc[-1]
        actual = float(exit_ / entry - 1)
        db.execute("UPDATE predictions SET actual_return=?, error=?, resolved_at=? WHERE prediction_id=?",
                   [actual, actual - (r["expected_return"] or 0), utcnow_iso(), r["prediction_id"]])
        n += 1
    return n


def journal_frame(db: Database, ensemble_only: bool = True) -> pd.DataFrame:
    df = db.query_df("SELECT * FROM predictions ORDER BY as_of DESC, horizon")
    if df.empty:
        return df
    if ensemble_only:
        df = df[df["signal"].notna()]
    iv = db.query_df("SELECT * FROM prediction_intervals")
    if not iv.empty:
        df = df.merge(iv[["prediction_id", "lower", "upper"]], on="prediction_id", how="left")
    return df


# ------------------------------------------------------------------ live scorecard
def live_scorecard(db: Database) -> pd.DataFrame:
    df = db.query_df("SELECT * FROM predictions WHERE actual_return IS NOT NULL AND signal IS NOT NULL")
    if df.empty:
        return pd.DataFrame()
    rows = []
    df["regime"] = df["regime"].fillna("unknown")
    groups = [((h, "ALL"), g) for h, g in df.groupby("horizon")] + \
             [((h, r), g) for (h, r), g in df.groupby(["horizon", "regime"])]
    for (h, rg), g in groups:
        y = (g["actual_return"] > 0).astype(float)
        p = g["p_up"].astype(float)
        e = g["actual_return"] - g["expected_return"]
        strat = np.where(g["expected_return"] > 0, g["actual_return"], 0.0)
        rec = {"horizon": h, "regime": rg, "n": len(g),
               "directional_accuracy": float((np.sign(g["expected_return"]) == np.sign(g["actual_return"])).mean()),
               "brier": brier(y, p), "ece": ece(y, p) if len(g) >= 20 else np.nan, "mae": float(e.abs().mean()),
               "rmse": float(np.sqrt((e ** 2).mean()))}
        tp = ((p >= 0.5) & (y == 1)).sum()
        fp = ((p >= 0.5) & (y == 0)).sum()
        fn = ((p < 0.5) & (y == 1)).sum()
        rec["precision"] = tp / (tp + fp) if tp + fp else np.nan
        rec["recall"] = tp / (tp + fn) if tp + fn else np.nan
        rec["f1"] = 2 * rec["precision"] * rec["recall"] / (rec["precision"] + rec["recall"]) \
            if rec["precision"] and rec["recall"] and np.isfinite(rec["precision"]) and np.isfinite(rec["recall"]) else np.nan
        tm = trading_metrics(pd.Series(strat)) if len(g) > 5 else {}
        rec.update({k: tm.get(k) for k in ("sharpe", "sortino", "max_drawdown", "profit_factor")})
        rows.append(rec)
    out = pd.DataFrame(rows)
    now = utcnow_iso()
    db.upsert_many("performance_metrics", [{"scope": "live_ensemble", "horizon": int(r["horizon"]), "regime": str(r["regime"]),
                                            "metric": m, "value": float(r[m]) if r[m] is not None and pd.notna(r[m]) else None,
                                            "n": int(r["n"]), "updated_at": now}
                                           for _, r in out.iterrows() for m in ("directional_accuracy", "brier", "mae")])
    return out


# ------------------------------------------------------------------ model drift
def model_drift(db: Database, forecast=None, min_live: int = 30, tolerance: float = 0.06) -> dict:
    """Compare recent live accuracy (resolved journal) with OOS validation accuracy."""
    out = {"alert": False, "checks": [], "message": ""}
    df = db.query_df("SELECT * FROM predictions WHERE actual_return IS NOT NULL AND signal IS NOT NULL")
    if forecast is None:
        return out
    now = utcnow_iso()
    for h, hr in forecast.horizons.items():
        val = hr.ensemble_comparison.get(hr.ensemble_method, {})
        hist_da, _hist_b = val.get("directional_accuracy"), val.get("brier")
        # recent OOS period inside validation itself (last 20% of OOS) vs earlier
        if hr.oos is not None and len(hr.oos.dropna(subset=["ens_ret", "y_ret"])) > 300:
            o = hr.oos.dropna(subset=["ens_ret", "y_ret"])
            cut = int(len(o) * 0.8)
            early = (np.sign(o["ens_ret"].iloc[:cut]) == np.sign(o["y_ret"].iloc[:cut])).mean()
            late = (np.sign(o["ens_ret"].iloc[cut:]) == np.sign(o["y_ret"].iloc[cut:])).mean()
            drift = late < early - tolerance
            out["checks"].append({"horizon": h, "source": "OOS recent vs earlier", "historical": float(early),
                                  "recent": float(late), "drift": bool(drift)})
            if drift:
                out["alert"] = True
        g = df[df["horizon"] == h] if not df.empty else pd.DataFrame()
        if len(g) >= min_live and hist_da is not None:
            live_da = float((np.sign(g["expected_return"]) == np.sign(g["actual_return"])).mean())
            drift = live_da < hist_da - tolerance
            out["checks"].append({"horizon": h, "source": "live journal vs validation", "historical": hist_da,
                                  "recent": live_da, "drift": bool(drift), "n_live": len(g)})
            out["alert"] |= drift
        db.upsert("model_drift", {"checked_at": now, "model_id": hr.model_ids.get("ensemble", f"ENS_H{h}"),
                                  "metric": "directional_accuracy", "historical": hist_da,
                                  "recent": out["checks"][-1]["recent"] if out["checks"] else None,
                                  "drift": int(any(c["drift"] for c in out["checks"] if c["horizon"] == h)),
                                  "payload": json.dumps([c for c in out["checks"] if c["horizon"] == h], default=str)})
    if out["alert"]:
        hs = sorted({c["horizon"] for c in out["checks"] if c["drift"]})
        out["message"] = f"⚠ MODEL DRIFT: recent accuracy deteriorated for horizon(s) {hs}. Confidence reduced; " \
                         f"retraining with challenger comparison recommended (old models preserved)."
    return out


# ------------------------------------------------------------------ feature drift
def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    e = expected[~np.isnan(expected)]
    a = actual[~np.isnan(actual)]
    if len(e) < 50 or len(a) < 5:
        return np.nan
    qs = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return np.nan
    qs[0], qs[-1] = -np.inf, np.inf
    pe = np.histogram(e, qs)[0] / len(e)
    pa = np.histogram(a, qs)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def mahalanobis_ood(train: pd.DataFrame, today: pd.Series) -> dict:
    """Distance of today's input vector from the training distribution (Ledoit-Wolf
    shrunk covariance), expressed as a percentile of the training rows' own distances."""
    from sklearn.covariance import LedoitWolf

    T = train.dropna()
    cols = [c for c in T.columns if pd.notna(today.get(c))]
    if len(T) < 300 or len(cols) < 3:
        return {"percentile": np.nan, "distance": np.nan}
    T = T[cols]
    mu, sd = T.mean(), T.std().replace(0, 1)
    Z = ((T - mu) / sd).values
    z0 = ((today[cols] - mu) / sd).values.astype(float)
    lw = LedoitWolf().fit(Z)
    P = lw.precision_
    d_train = np.sqrt(np.einsum("ij,jk,ik->i", Z, P, Z))
    d0 = float(np.sqrt(z0 @ P @ z0))
    return {"distance": d0, "percentile": float((d_train < d0).mean())}


def feature_drift(db: Database | None, X: pd.DataFrame, features: list[str], recent: int = 252,
                  train_end=None) -> dict:
    """PSI / KS per feature (recent year vs earlier history, informational: persistent
    features naturally drift) plus the decisive out-of-distribution test on *today's*
    vector: per-feature 99% range check and multivariate Mahalanobis percentile."""
    train = X.loc[:train_end].iloc[:-recent] if train_end is not None else X.iloc[:-recent]
    cur = X.iloc[-recent:]
    today = X.iloc[-1]
    rows, ood = [], []
    for f in features:
        if f not in X:
            continue
        tr = train[f].dropna().values
        if len(tr) < 200:
            continue
        p = psi(tr, cur[f].values)
        ks = stats.ks_2samp(tr, cur[f].dropna().values) if cur[f].notna().sum() > 10 else (np.nan, np.nan)
        lo, hi = np.quantile(tr, [0.005, 0.995])
        out_range = bool(pd.notna(today[f]) and (today[f] < lo or today[f] > hi))
        if out_range:
            ood.append(f)
        rows.append({"feature": f, "psi": p, "ks_stat": float(ks[0]), "ks_p": float(ks[1]), "out_of_range": int(out_range)})
    tab = pd.DataFrame(rows)
    if db is not None and not tab.empty:
        now = utcnow_iso()
        db.upsert_many("feature_drift", [{"checked_at": now, **r} for r in tab.to_dict("records")])
    n_high = int((tab["psi"] > 0.25).sum()) if not tab.empty else 0
    share_ood = len(ood) / max(1, len(rows))
    feats = [f for f in features if f in X]
    mh = mahalanobis_ood(X[feats].iloc[:-1], today[feats]) if feats else {"percentile": np.nan, "distance": np.nan}
    mpct = mh["percentile"]
    alert = bool(share_ood >= 0.15 or (np.isfinite(mpct) and mpct >= 0.99))
    msg = ""
    if alert:
        msg = (f"⚠ OUT-OF-DISTRIBUTION MARKET: today's input vector is at the {100 * mpct:.1f}th percentile of "
               f"training distances (Mahalanobis); {len(ood)} of {len(rows)} inputs outside their 99% training range"
               f"{' (' + ', '.join(ood[:6]) + ')' if ood else ''}. Confidence reduced.")
    pen = min(0.4, share_ood * 1.2 + (max(0.0, mpct - 0.95) * 4 if np.isfinite(mpct) else 0.0))
    return {"alert": alert, "message": msg, "table": tab, "ood_features": ood, "share_ood": share_ood,
            "n_high_psi": n_high, "mahalanobis_percentile": mpct, "mahalanobis_distance": mh["distance"],
            "penalty": float(pen)}


# ------------------------------------------------------------------ champion / challenger
def champion_challenger(db: Database, forecast, horizon: int) -> dict:
    """Promote the challenger (today's new ensemble) only if it beats the production model
    out-of-sample on the same rows with statistical support (Diebold-Mariano p<0.10)."""
    hr = forecast.horizons.get(horizon)
    if hr is None or hr.oos is None:
        return {"decision": "no challenger"}
    prod = db.query("SELECT * FROM model_versions WHERE horizon=? AND model_name='ensemble' AND role='production' "
                    "ORDER BY created_at DESC", [horizon]) if db else []
    challenger_id = hr.model_ids.get("ensemble")
    if len(prod) <= 1:
        return {"decision": "first production model", "production": challenger_id}
    old = prod[1]
    old_val = json.loads(old["validation"] or "{}")
    new_val = hr.ensemble_comparison.get(hr.ensemble_method, {})
    # compare against the best single base model on identical OOS rows as a proxy for the stored champion
    o = hr.oos.dropna(subset=["ens_ret", "y_ret"])
    best = None
    for c in [c for c in o.columns if c.startswith("m_") and c != "m_naive_drift"]:
        e = (o[c] - o["y_ret"]).dropna()
        if best is None or (e ** 2).mean() < best[1]:
            best = (c, (e ** 2).mean())
    dm = diebold_mariano(o["ens_ret"] - o["y_ret"], o[best[0]] - o["y_ret"], horizon) if best else {}
    improved = (new_val.get("rmse", 9) <= (old_val.get("rmse") or 9)) and (dm.get("stat") or 0) <= 0
    if not improved and db:
        db.execute("UPDATE model_versions SET role='challenger' WHERE model_id=?", [challenger_id])
        db.execute("UPDATE model_versions SET role='production' WHERE model_id=?", [old["model_id"]])
    return {"decision": "promoted" if improved else "kept previous production", "challenger": challenger_id,
            "previous": old["model_id"], "challenger_rmse": new_val.get("rmse"), "previous_rmse": old_val.get("rmse"),
            "dm_vs_best_single": dm, "significant": bool(dm.get("p_value", 1) < 0.10)}


def retirement_flags(forecast) -> list[dict]:
    """Flag base models that underperform the naive benchmark, are badly calibrated or unstable."""
    out = []
    for h, hr in forecast.horizons.items():
        naive = hr.model_scores.get("naive_drift", {})
        for name, sc in hr.model_scores.items():
            if name == "naive_drift":
                continue
            reasons = []
            if naive and (sc.get("rmse") or 0) > (naive.get("rmse") or 9) * 1.05:
                reasons.append("RMSE worse than naive benchmark")
            if (sc.get("ece") or 0) > 0.12:
                reasons.append(f"poor calibration (ECE {sc['ece']:.2f})")
            if (sc.get("directional_accuracy") or 1) < 0.48:
                reasons.append("directional accuracy below 48%")
            by_reg = hr.scores_by_regime.get(name, {})
            if by_reg:
                das = [v["directional_accuracy"] for v in by_reg.values()]
                if len(das) > 1 and np.std(das) > 0.12:
                    reasons.append("unstable across regimes")
            if reasons:
                out.append({"horizon": h, "model": name, "model_id": hr.model_ids.get(name), "reasons": reasons,
                            "action": "flag for retirement from production (history preserved)"})
    return out


# ------------------------------------------------------------------ self audit
def self_audit(state: dict) -> dict:
    fc = state.get("forecast_primary", {})
    ens = fc.get("ensemble_oos", {})
    naive = fc.get("model_scores", {}).get("naive_drift", {})
    scores = fc.get("model_scores", {})
    strengths, weaknesses, lim, data_lim, leak, regime_dep = [], [], [], [], [], []
    if ens:
        da = ens.get("directional_accuracy") or np.nan
        nda = naive.get("directional_accuracy") or np.nan
        (strengths if da > nda + 0.01 else weaknesses).append(
            f"Ensemble OOS directional accuracy {da:.1%} vs naive drift {nda:.1%} ({fc.get('horizon')}D, n={ens.get('n')}).")
        ic = ens.get("ic_spearman")
        if ic is not None:
            (strengths if ic > 0.05 else weaknesses).append(f"Ensemble OOS rank IC {ic:+.3f}.")
    cal = fc.get("calibration", {})
    if cal:
        e = cal.get("calibrated", {}).get("ece")
        if e is not None:
            (strengths if e < 0.05 else weaknesses).append(f"OOS calibration ECE {e:.3f} (method {cal.get('method')}).")
    cov = fc.get("conformal", {})
    if cov.get("coverage") is not None:
        (strengths if abs(cov["coverage"] - cov["nominal"]) < 0.05 else weaknesses).append(
            f"Conformal coverage {cov['coverage']:.0%} vs nominal {cov['nominal']:.0%}.")
    if scores:
        best = max(scores.items(), key=lambda kv: kv[1].get("ic_spearman") or -9)
        worst = min(scores.items(), key=lambda kv: kv[1].get("ic_spearman") or 9)
        strengths.append(f"Best base model by IC: {best[0]} ({best[1].get('ic_spearman', np.nan):+.3f}).")
        weaknesses.append(f"Weakest base model by IC: {worst[0]} ({worst[1].get('ic_spearman', np.nan):+.3f}).")
    by_reg = fc.get("scores_by_regime", {})
    for m, d in list(by_reg.items())[:1]:
        regime_dep.append(f"{m} accuracy by regime: " + ", ".join(f"{k} {v['directional_accuracy']:.0%} (n={v['n']})" for k, v in d.items()))
    lim += ["Daily data only; intraday dynamics and overnight US moves after the XETRA close are not modelled at decision time.",
            "Breadth uses sector-ETF proxies, not constituent-level data.",
            "Forecasts are probabilistic and regime-dependent; past relationships may break.",
            "Historical backtest decision rule excludes news/valuation (not reconstructable point-in-time)."]
    if state.get("proxy_note"):
        data_lim.append(state["proxy_note"])
    for k, v in state.get("macro_status", {}).items():
        if not str(v).startswith("ok") and v != "SYNTHETIC":
            data_lim.append(f"Macro {k}: {v}")
    if not state.get("news", {}).get("available"):
        data_lim.append(f"No current news: {state.get('news', {}).get('reason', 'not run')}")
    if not state.get("valuation_supplied"):
        data_lim.append("No fundamental valuation inputs (P/E, CAPE) - valuation uses a price-based proxy.")
    if state.get("is_synthetic"):
        data_lim.insert(0, "SYNTHETIC DEMO DATA - results say nothing about the real market.")
    for c in state.get("leakage_audit", {}).get("checks", []):
        if c["status"] != "PASS":
            leak.append(f"[{c['status']}] {c['name']}: {c['detail']}")
    if not leak:
        leak.append("Leakage audit passed all checks.")
    research_risks = ["Multiple models / ensembles / thresholds were compared: some selection bias remains even with "
                      "nested selection; treat the OOS edge as an upper bound.",
                      "Overlapping multi-day labels make effective sample sizes much smaller than row counts."]
    dis = fc.get("disagreement", {})
    return {"MODEL STRENGTHS": strengths, "MODEL WEAKNESSES": weaknesses, "KNOWN LIMITATIONS": lim,
            "DATA LIMITATIONS": data_lim, "POSSIBLE LEAKAGE": leak, "REGIME DEPENDENCE": regime_dep or ["n/a"],
            "MODEL DISAGREEMENT": [f"Sign agreement {dis.get('sign_agreement', 0) or 0:.0%}, forecast spread "
                                   f"{dis.get('std_of_model_returns', np.nan):.2%}"] if dis else ["n/a"],
            "RECENT PERFORMANCE": [c["message"] for c in [state.get("model_drift", {})] if c.get("message")] or
                                  [f"{len(state.get('scorecard', []))} live scorecard rows"],
            "OUT-OF-SAMPLE PERFORMANCE": [f"{k}: DA {v.get('directional_accuracy', np.nan):.1%}, IC {v.get('ic_spearman', np.nan):+.3f}"
                                          for k, v in fc.get("ensemble_comparison", {}).items()],
            "RESEARCH RISKS": research_risks}


def new_run_id(prefix: str = "RUN") -> str:
    return f"{prefix}_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
