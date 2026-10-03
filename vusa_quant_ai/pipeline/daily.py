"""DAILY AUTOMATIC PIPELINE (the "ANALYZE VUSA NOW" / "RUN DEEP RESEARCH" engine).

01 market status       02 download data        03 validate data       04 update database
05 retrieve news       06 analyse sentiment    07 extract events      08 update macro
09 technical features  10 detect regime        11 check anomalies     12 predictions
13 calibration         14 scenario analysis    15 risk                16 skeptic analysis
17 final signal        18 save decision state  19 compare vs yesterday 20 daily report

Every step records status / duration / message. A failing optional step is reported and
the pipeline continues with that information marked as unavailable; a failing critical
step (no price data) stops the run with an explicit error - no fabricated output.
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ..agents.committee import run_committee
from ..agents.news_agent.agent import MarketResearchAgent
from ..alerts.engine import generate_alerts
from ..analysis.analogues import find_analogues
from ..analysis.counterfactual import run_counterfactuals
from ..analysis.event_impact import build_event_database, conditional_outcomes, match_news_events
from ..analysis.leading_indicators import leading_indicator_table
from ..config.settings import DATA_DIR, MODE_PROFILES, Settings
from ..core.runtime import environment_fingerprint, get_logger, set_seeds, utcnow_iso
from ..data.calendar import market_status
from ..data.service import DataService, MarketDataBundle
from ..database.db import Database
from ..explainability.importance import shap_today
from ..explainability.narrative import llm_explanation, template_explanation
from ..features.builder import build_features
from ..features.sentiment.sentiment_features import daily_sentiment_from_articles
from ..features.technical.indicators import support_resistance
from ..models.forecast_service import ForecastResult, ForecastService
from ..models.registry import ModelRegistry, model_catalog
from ..monitoring.monitor import (champion_challenger, feature_drift, journal_predictions, live_scorecard, model_drift,
                                  new_run_id, resolve_predictions, retirement_flags, self_audit)
from ..regimes.anomaly import detect_shocks
from ..regimes.changepoint import structural_change_report
from ..regimes.detector import current_regime_summary, detect_regimes
from ..reports.daily_report import build_report
from ..risk.risk_engine import (correlation_breakdowns, correlation_matrix, portfolio_exposure, position_sizing_research,
                                risk_metrics, risk_score, rolling_correlations, stress_test)
from ..signals import components as C
from ..signals.decision import decide
from ..simulation.montecarlo import mc_comparison_table, run_monte_carlo

log = get_logger("vusa.pipeline")

STEPS = ["Check market status", "Download data", "Validate data", "Update database", "Retrieve news",
         "Analyze sentiment", "Extract events", "Update macro data", "Update technical features", "Detect regime",
         "Check anomalies", "Generate predictions", "Probabilistic calibration", "Scenario analysis", "Calculate risk",
         "Skeptic analysis", "Generate final signal", "Save decision state", "Compare against yesterday",
         "Generate daily report"]


@dataclass
class Artifacts:
    """Rich in-memory objects for the GUI (charts); not persisted."""
    bundle: MarketDataBundle | None = None
    features: object = None
    regimes: pd.DataFrame | None = None
    forecast: ForecastResult | None = None
    monte_carlo: dict = field(default_factory=dict)
    stress: pd.DataFrame | None = None
    rolling_corr: pd.DataFrame | None = None
    corr_matrix: pd.DataFrame | None = None
    analogues: pd.DataFrame | None = None
    counterfactuals: pd.DataFrame | None = None
    leading: pd.DataFrame | None = None
    feature_drift: pd.DataFrame | None = None
    change_series: dict = field(default_factory=dict)
    event_outcomes: pd.DataFrame | None = None
    shap_now: pd.Series | None = None
    importance_stability: pd.DataFrame | None = None


@dataclass
class PipelineResult:
    ok: bool
    run_id: str
    state: dict
    artifacts: Artifacts
    steps: list[dict]
    error: str = ""


class DailyPipeline:
    def __init__(self, settings: Settings, db: Database | None = None, progress: Callable[[str, float], None] | None = None):
        self.s = settings
        self.db = db or Database(settings.db_path)
        self.progress = progress or (lambda m, f: None)
        self.registry = ModelRegistry(self.db, DATA_DIR / "models")

    # ------------------------------------------------------------------ public
    def run(self, mode: str | None = None, offline: bool = False, synthetic: bool | None = None, as_of=None,
            use_llm: bool = False) -> PipelineResult:
        mode = mode or self.s.mode
        prof = dict(MODE_PROFILES.get(mode, MODE_PROFILES["balanced"]))
        set_seeds(self.s.models.random_seed)
        run_id = new_run_id("DEEP" if mode == "deep" else "RUN")
        steps: list[dict] = []
        state: dict = {"run_id": run_id, "mode": mode, "started_at": utcnow_iso(), "warnings": []}
        art = Artifacts()
        ctx = {"t": time.perf_counter()}

        def step(i: int, status: str, msg: str = ""):
            dt = time.perf_counter() - ctx["t"]
            ctx["t"] = time.perf_counter()
            steps.append({"step": i, "name": STEPS[i - 1], "status": status, "seconds": round(dt, 2), "message": msg})
            self.progress(f"[{i:02d}/20] {STEPS[i - 1]}: {status}" + (f" - {msg}" if msg else ""), i / 20)

        self.db.upsert("research_runs", {"run_id": run_id, "kind": "deep_research" if mode == "deep" else "analysis",
                                         "started_at": state["started_at"], "finished_at": None, "status": "running",
                                         "mode": mode, "steps": "[]", "environment": json.dumps(environment_fingerprint()),
                                         "summary": ""})
        try:
            return self._run(mode, prof, offline, synthetic, as_of, use_llm, run_id, state, art, steps, step)
        except Exception as exc:  # noqa: BLE001
            log.exception("pipeline failed")
            err = f"{type(exc).__name__}: {exc}"
            state["error"] = err
            state["traceback"] = traceback.format_exc()[-3000:]
            self.db.upsert("research_runs", {"run_id": run_id, "kind": "analysis", "started_at": state["started_at"],
                                             "finished_at": utcnow_iso(), "status": "failed", "mode": mode,
                                             "steps": json.dumps(steps), "environment": json.dumps(environment_fingerprint()),
                                             "summary": err})
            return PipelineResult(False, run_id, state, art, steps, err)

    # ------------------------------------------------------------------ implementation
    def _run(self, mode, prof, offline, synthetic, as_of, use_llm, run_id, state, art, steps, step) -> PipelineResult:
        s = self.s
        # 01 ------------------------------------------------------------------
        ms = market_status(exchange=s.exchange)
        state["market_status"] = ms
        step(1, "ok", f"{ms['exchange']} {ms['state']}; last complete session {ms['last_complete_session']}")
        # 02-04 ---------------------------------------------------------------
        ds = DataService(s, self.db)
        bundle = ds.load(offline=offline, include_macro=prof["macro"], synthetic=synthetic, as_of=as_of)
        art.bundle = bundle
        state.update({"data_current": bundle.data_current, "is_synthetic": bundle.is_synthetic,
                      "data_status": bundle.status_label, "proxy_note": bundle.proxy_note,
                      "macro_status": bundle.macro_status, "symbol": bundle.target_symbol, "currency": s.currency})
        state["warnings"] += bundle.warnings
        if bundle.target.empty:
            step(2, "FAILED", "no VUSA price data from any provider or cache")
            raise RuntimeError("No VUSA price data available from any provider or the local cache. "
                               "Check your internet connection / providers (Data Quality tab). "
                               "No analysis was produced (nothing is fabricated).")
        step(2, "ok", f"{len(bundle.target)} VUSA rows ({bundle.status_label}); {len(bundle.assets)} reference assets")
        crit = [v for v in bundle.validation if v.dataset == "VUSA"]
        step(3, "ok" if all(v.ok for v in crit) else "WARN", "; ".join(v.summary() for v in crit)[:300])
        state["validation"] = [{"dataset": v.dataset, "score": v.score, "issues": v.issues} for v in bundle.validation]
        state["data_quality_score"] = float(np.mean([v.score for v in crit])) if crit else 0.0
        n_macro = self._persist_inputs(bundle)
        step(4, "ok", f"prices cached; {n_macro} macro releases stored point-in-time")
        target = bundle.target
        as_of_ts = target.index[-1]
        state["as_of"] = str(as_of_ts.date())
        state["price"] = float(target["close"].iloc[-1])

        # 05-07 news ----------------------------------------------------------
        news = {"available": False, "reason": "disabled in Fast mode"}
        if prof["news"] and not bundle.is_synthetic:
            news = MarketResearchAgent(s, ds.health).run(offline=offline)
            if news.get("available"):
                self.db.upsert_many("news_articles", [{k: a[k] for k in ("article_id", "title", "summary", "url", "source",
                                    "published_at", "retrieved_at", "relevance", "source_quality", "sentiment", "label",
                                    "topics", "duplicate_of")} for a in news["articles"]])
                self.db.upsert_many("news_events", [{k: e[k] for k in ("event_id", "article_id", "event_date", "category",
                                    "description", "direction", "magnitude", "confidence", "horizon", "available_at",
                                    "payload")} for e in news["events"]])
        elif bundle.is_synthetic:
            news = {"available": False, "reason": "synthetic demo mode - no real news used"}
        if news.get("available") and news.get("sentiment", {}).get("n"):
            sn = news["sentiment"]
            self.db.upsert("sentiment", {"date": str(pd.Timestamp.now(tz="UTC").date()), "scope": "news_all",
                                         "score": sn.get("score"), "dispersion": sn.get("dispersion"), "n_articles": sn.get("n"),
                                         "available_at": utcnow_iso()})
            for topic, v in sn.get("by_topic", {}).items():
                self.db.upsert("sentiment", {"date": str(pd.Timestamp.now(tz="UTC").date()), "scope": topic, "score": v["score"],
                                             "dispersion": None, "n_articles": v["n"], "available_at": utcnow_iso()})
        step(5, "ok" if news.get("available") else "UNAVAILABLE", news.get("reason", f"{len(news.get('articles', []))} articles"))
        step(6, "ok" if news.get("available") else "skipped",
             f"sentiment {news.get('sentiment', {}).get('score')}" if news.get("available") else "")
        step(7, "ok" if news.get("available") else "skipped", f"{len(news.get('events', []))} events")
        state["news"] = {k: v for k, v in news.items() if k != "articles"}
        state["news"]["top_articles"] = sorted([a for a in news.get("articles", []) if not a.get("duplicate_of")],
                                               key=lambda a: -(a["relevance"] * a["source_quality"]))[:40]

        # 08-09 features ------------------------------------------------------
        step(8, "ok" if bundle.pit.series_ids else "UNAVAILABLE", f"{len(bundle.pit.series_ids)} macro series point-in-time")
        arts_hist = self.db.query_df("SELECT * FROM news_articles")
        news_daily = daily_sentiment_from_articles(arts_hist) if not arts_hist.empty else None
        fs = build_features(bundle, s.exchange, news_daily)
        art.features = fs
        state["feature_notes"] = fs.notes
        state["warnings"] += fs.notes
        fnow = fs.frame.iloc[-1]
        state["features_now"] = {k: (float(v) if pd.notna(v) else None) for k, v in fnow.items()}
        state["feature_groups"] = {g: len(c) for g, c in fs.groups.items()}
        state["support_resistance"] = support_resistance(target)
        lt = fs.lineage_table()
        state["lineage"] = lt.round(6).to_dict("records")
        self._lineage_cache = [{"run_id": run_id, "feature": r["feature"], "source": r["source"],
                                "calculated_at": r["calculated_at"], "transformation": r["transformation"],
                                "original_value": None if pd.isna(r["original_value"]) else float(r["original_value"]),
                                "normalized_value": None if pd.isna(r["normalized_value"]) else float(r["normalized_value"]),
                                "missing_status": r["missing_status"]} for r in lt.to_dict("records")]
        step(9, "ok", f"{fs.frame.shape[1]} features in {len(fs.groups)} groups (version {fs.version})")
        state["macro_snapshot"] = {c: state["features_now"].get(c) for c in fs.groups.get("macro", [])}
        state["pit_quality"] = ("Macro history uses latest-vintage data with release-lag timing (revision risk); add a FRED "
                                "API key for ALFRED vintages.") if any(bundle.pit.method(x) == "release_lag"
                                                                      for x in bundle.pit.series_ids) else None
        # stats for quant agent
        r = np.log(target["close"]).diff().dropna()
        state["stats"] = {"autocorr_1d": float(r.tail(756).autocorr()), "drift_1y_ann": float(r.tail(252).mean() * 252),
                          "drift_long_ann": float(r.mean() * 252), "skew_60d": float(r.tail(60).skew()),
                          "hit_rate_60d": float((r.tail(60) > 0).mean())}

        # 10 regime ------------------------------------------------------------
        vix = bundle.assets["VIX"]["close"].reindex(target.index).ffill() if "VIX" in bundle.assets else None
        reg = detect_regimes(target["close"], fs.frame, vix, s.models.random_seed)
        art.regimes = reg
        regime = current_regime_summary(reg)
        state["regime"] = regime
        state["regime_history"] = {str(k.date()): v for k, v in reg["regime_rule"].dropna().tail(30).astype(str).items()}
        step(10, "ok", f"{regime['regime']} / {regime['volatility_label']} volatility")

        # 11 anomalies / change points ---------------------------------------------
        cp = structural_change_report(target["close"])
        art.change_series = cp.pop("series")
        state["structural_change"] = cp
        shocks = detect_shocks(target, fs.frame, regime, cp, news.get("news_shock", False),
                               macro_shock=bool((state["features_now"].get("hy_oas_chg_20d") or 0) > 1.0 or
                                                abs(state["features_now"].get("us10y_chg_20d") or 0) > 0.6))
        shocks.pop("series", None)
        state["shocks"] = shocks
        step(11, "ok" if not shocks["flags"] else "FLAGS", "; ".join(shocks["flags"])[:300])

        # 12-13 forecasts ------------------------------------------------------
        models = prof["models"] or [m for m in s.models.enabled_models] + ["sarima", "state_space", "nbeats"]
        models = list(dict.fromkeys(models))
        horizons = prof["horizons"] or s.models.horizons
        if s.models.primary_horizon not in horizons:
            horizons = sorted(set(horizons) | {s.models.primary_horizon})
        hpo_trials = s.models.optuna_trials or (10 if mode == "deep" else 0)
        fsvc = ForecastService(s, self.registry, progress=lambda m, f: self.progress(f"[12/20] {m}", 0.55 + 0.05 * f),
                               hpo_trials=hpo_trials)
        fc = fsvc.run(fs, target, reg, horizons, models, as_of_ts, pit=bundle.pit, full_validation=prof["full_validation"])
        art.forecast = fc
        state["leakage_audit"] = fc.audit.to_dict() if fc.audit else {}
        state["training_window"] = {"chosen": fc.training_window, "comparison": fc.window_comparison}
        state["feature_selection_stability"] = fc.selection.head(40).reset_index().rename(columns={"index": "feature"}) \
            .to_dict("records") if not fc.selection.empty else []
        state["forecasts"] = {h: _hr_summary(hr) for h, hr in fc.horizons.items()}
        prim = fc.horizons.get(s.models.primary_horizon)
        state["forecast_primary"] = _hr_summary(prim, full=True) if prim else {}
        step(12, "ok" if prim and np.isfinite(prim.expected_return) else "WARN",
             f"{len(fc.horizons)} horizons x {len(models)} models; audit {'PASSED' if fc.audit.passed else 'FAILED'}")
        step(13, "ok", f"primary calibration: {prim.calibration.get('method') if prim else 'n/a'}; conformal coverage "
             f"{(prim.conformal.get('coverage') if prim else None)}")
        state["model_catalog"] = model_catalog()

        # 14 scenario analysis ----------------------------------------------------
        mc_paths = prof["mc_paths"] or s.monte_carlo_paths
        h_mc = s.models.primary_horizon
        resid = None
        if prim is not None and prim.oos is not None:
            o = prim.oos.dropna(subset=["ens_ret", "y_ret"])
            resid = (o["y_ret"] - o["ens_ret"]).values[-750:]
        mc = run_monte_carlo(target["close"], h_mc, mc_paths, s.models.random_seed,
                             prim.expected_return if prim else None, resid, regime.get("hmm_stress_probability"))
        art.monte_carlo = mc
        state["monte_carlo"] = mc_comparison_table(mc).round(5).to_dict("index")
        state["monte_carlo_primary"] = mc["Block bootstrap"].stats
        stress = stress_test(target, bundle.assets, s.currency)
        art.stress = stress
        state["stress"] = stress.to_dict("records")
        if prim and prim.final_models:
            x_now = fsvc.design_matrix(fs, reg).loc[[as_of_ts]]
            cf = run_counterfactuals(prim.final_models, x_now, prim.ensemble_weights)
            art.counterfactuals = cf
            state["counterfactuals"] = cf.to_dict("records") if not cf.empty else []
            for nm in ("lightgbm", "xgboost", "hist_gb", "random_forest"):
                if nm in prim.final_models:
                    sv = shap_today(prim.final_models[nm], x_now)
                    if sv is not None:
                        art.shap_now = sv
                        state["shap_today"] = {"model": nm, "values": sv.head(15).round(6).to_dict()}
                    break
        an = find_analogues(fs.frame, target["close"], reg["regime_rule"], as_of=as_of_ts) if prof["analogues"] else {}
        art.analogues = an.get("analogues")
        state["analogues"] = {k: (v.to_dict("records") if isinstance(v, pd.DataFrame) else v) for k, v in an.items()}
        ev_db = build_event_database(target["close"], bundle.assets, bundle.pit, fs.decision_times)
        outcomes = conditional_outcomes(ev_db, target["close"], as_of_ts)
        art.event_outcomes = outcomes
        state["event_study"] = outcomes.round(5).to_dict("records")
        state["news_memory"] = match_news_events(news.get("events", []), outcomes)
        lead = leading_indicator_table(fs.frame, target["close"])
        art.leading = lead
        state["leading_indicators"] = lead.round(4).to_dict("records") if not lead.empty else []
        step(14, "ok", f"Monte Carlo {mc_paths:,} paths x {len(mc)} methods; {len(stress)} stress scenarios; "
             f"{len(an.get('analogues', [])) if an else 0} analogues")

        # 15 risk -------------------------------------------------------------
        spx = bundle.assets["SPX"]["close"] if "SPX" in bundle.assets else None
        rm = risk_metrics(target["close"], spx.reindex(target.index).ffill() if spx is not None else None)
        state["risk_metrics"] = rm
        rc = rolling_correlations(target["close"], bundle.assets, bundle.sectors)
        art.rolling_corr = rc
        art.corr_matrix = correlation_matrix(target["close"], bundle.assets)
        state["correlation_breakdowns"] = correlation_breakdowns(rc)
        epi = prim.uncertainty.get("aleatoric_std") if prim else None
        rscore, rcomp = risk_score(rm, regime, mc["Block bootstrap"].stats, shocks, epi, prim.p_dd10 if prim else None)
        state["risk_components"] = rcomp
        if s.portfolio.enabled and (s.portfolio.holdings_units or s.portfolio.cash_available):
            state["portfolio"] = portfolio_exposure(state["price"], s.portfolio.holdings_units, s.portfolio.avg_purchase_price,
                                                    s.portfolio.cash_available, rm["VaR95_20d_hist"],
                                                    prim.quantiles.get(0.1) if prim and prim.quantiles else None,
                                                    s.portfolio.risk_tolerance)
            state["risk_tolerance"] = s.portfolio.risk_tolerance
        if mode == "deep":
            state["position_sizing"] = position_sizing_research(target["close"])
            if prim is not None and prim.selected_features:
                from ..explainability.importance import importance_stability
                from ..labeling.labels import build_labels
                from ..models.tree.models import HistGBForecaster

                lab = build_labels(target, [prim.horizon])[prim.horizon]
                Xi = fs.frame[prim.selected_features].iloc[252:]
                le = lab["label_end"].reindex(Xi.index)
                stab = importance_stability(lambda: HistGBForecaster(prim.horizon, s.models.random_seed), Xi,
                                            lab["fwd_ret"].reindex(Xi.index), lab["direction"].reindex(Xi.index),
                                            le.where(le <= as_of_ts), reg["regime_rule"], 4, s.models.random_seed)
                art.importance_stability = stab["table"]
                state["importance_stability"] = stab["table"].round(5).reset_index().rename(
                    columns={"index": "feature"}).to_dict("records")
                state["importance_by_regime"] = stab["by_regime"]
        step(15, "ok", f"risk score {rscore:.0f}/100")

        # drift monitors (feed confidence penalties) -----------------------------
        resolved = resolve_predictions(self.db, target["close"], target["open"])
        mdrift = model_drift(self.db, fc)
        sel_feats = prim.selected_features if prim else []
        fdrift = feature_drift(self.db, fs.frame, sel_feats)
        art.feature_drift = fdrift.pop("table")
        state["model_drift"], state["feature_drift"] = mdrift, fdrift
        state["resolved_predictions"] = resolved

        # 17 (components + decision) before 16 skeptic so the skeptic can attack it --------
        comps = {
            "trend": C.trend_component(fnow), "momentum": C.momentum_component(fnow),
            "valuation": C.valuation_component(s.valuation, fnow, target["close"]),
            "macro": C.macro_component(fnow), "sentiment": C.sentiment_component(news.get("sentiment"), fnow),
            "volatility": C.volatility_component(fnow, regime), "breadth": C.breadth_component(fnow),
            "regime": C.regime_component(regime),
            "event_risk": C.event_risk_component(news.get("events", []), shocks, state["news_memory"]),
            "ml_forecast": C.ml_component(prim), "analogues": C.analogue_component(an),
            "risk_reward": C.risk_reward_component(prim, mc["Block bootstrap"].stats),
        }
        state["valuation_supplied"] = bool(s.valuation.forward_pe or s.valuation.cape or s.valuation.trailing_pe)
        hist = self.db.query_df("SELECT * FROM signals ORDER BY as_of")
        hist = hist[hist["as_of"] < state["as_of"]] if not hist.empty else hist
        penalties = {
            "shocks": min(0.4, shocks["shock_level"]),
            "model_drift": 0.25 if mdrift["alert"] else 0.0,
            "out_of_distribution": fdrift["penalty"] if fdrift["alert"] else fdrift["penalty"] * 0.5,
            "data_not_current": 0.0 if bundle.data_current else 0.3,
            "synthetic_data": 0.5 if bundle.is_synthetic else 0.0,
            "leakage_audit": 0.5 if not fc.audit.passed else 0.0,
            "structural_change": 0.1 if cp.get("structural_change_detected") else 0.0,
        }
        sma200 = float(target["close"].rolling(200).mean().iloc[-1])
        decision = decide(comps, rscore, s.signals, prim.confidence if prim and np.isfinite(prim.confidence) else 0.2, hist,
                          penalties, {"price": state["price"], "sma200": sma200,
                                      "vix_now": state["features_now"].get("vix_level"),
                                      "p_up": prim.p_up if prim else None, "p_lt5": prim.p_lt5 if prim else None})
        state["decision"] = decision.to_dict()

        # 16 committee + skeptic ---------------------------------------------------
        state["source_health"] = ds.health.table().to_dict("records") if not ds.health.table().empty else []
        outputs, committee = run_committee(state)
        state["agents"] = {k: v.to_dict() for k, v in outputs.items()}
        state["committee"] = committee
        step(16, "ok", f"8 agents; skeptic counter-case strength {committee['skeptic'].get('strength_of_counter_case')}")
        step(17, "ok", f"{decision.label}; opportunity {decision.opportunity}, risk {decision.risk}, "
             f"confidence {decision.confidence:.0%}")
        state["self_audit"] = self_audit(state)
        state["retirement_flags"] = retirement_flags(fc)
        state["champion_challenger"] = champion_challenger(self.db, fc, s.models.primary_horizon)
        state["scorecard"] = live_scorecard(self.db).round(4).to_dict("records")
        # explanation (structured JSON -> text; LLM optional with fabrication guard)
        payload = {k: state.get(k) for k in ("as_of", "price", "currency", "decision", "forecast_primary", "regime",
                                             "risk_metrics", "committee", "data_status")}
        payload["analogues"] = {"summary": state["analogues"].get("summary")} if state.get("analogues") else {}
        payload = _jsonable(payload)
        if use_llm:
            text, meta = llm_explanation(payload, s.api_keys.anthropic)
        else:
            text, meta = template_explanation(payload), {"mode": "template"}
        state["explanation"], state["explanation_meta"] = text, meta

        # 18 save --------------------------------------------------------------
        state["environment"] = environment_fingerprint()
        state["settings_snapshot"] = s.to_dict()
        state["data_snapshot"] = {"last_rows": target.tail(5).reset_index().astype(str).to_dict("records"),
                                  "dataset_version": fs.dataset_version, "feature_version": fs.version,
                                  "n_rows": len(target), "first_date": str(target.index[0].date())}
        state["model_versions"] = {h: hr.model_ids for h, hr in fc.horizons.items()}
        state["steps"] = steps
        self._persist(run_id, state, decision, fc)
        step(18, "ok", "decision trace, features, predictions and agent outputs stored")

        # 19 compare -------------------------------------------------------------
        prev = self._previous_trace(state["as_of"], run_id)
        state["vs_previous"] = _compare(prev, state) if prev else {"note": "no previous run"}
        alerts = generate_alerts(state, prev, s.alerts)
        self.db.upsert_many("alerts", alerts)
        state["alerts"] = alerts
        step(19, "ok", f"{len(alerts)} alerts")

        # 20 report ------------------------------------------------------------
        state["steps"] = steps
        md, html = build_report(state)
        self.db.upsert("daily_reports", {"run_id": run_id, "as_of": state["as_of"], "markdown": md, "html": html,
                                         "created_at": utcnow_iso()})
        rep_dir = DATA_DIR / "reports"
        rep_dir.mkdir(parents=True, exist_ok=True)
        (rep_dir / f"VUSA_report_{state['as_of']}_{run_id}.md").write_text(md, encoding="utf-8")
        (rep_dir / f"VUSA_report_{state['as_of']}_{run_id}.html").write_text(html, encoding="utf-8")
        state["report_markdown"] = md
        step(20, "ok", f"report saved to {rep_dir}")
        state["steps"] = steps
        state["finished_at"] = utcnow_iso()
        self.db.save_trace(run_id, state["as_of"], decision.signal, _jsonable(state))
        self.db.upsert("research_runs", {"run_id": run_id, "kind": "deep_research" if mode == "deep" else "analysis",
                                         "started_at": state["started_at"], "finished_at": state["finished_at"],
                                         "status": "ok", "mode": mode, "steps": json.dumps(steps),
                                         "environment": json.dumps(state["environment"]),
                                         "summary": f"{decision.label} opp {decision.opportunity} risk {decision.risk}"})
        return PipelineResult(True, run_id, state, art, steps)

    # ------------------------------------------------------------------ persistence
    def _persist(self, run_id: str, state: dict, decision, fc: ForecastResult) -> None:
        db = self.db
        db.upsert("signals", {"run_id": run_id, "as_of": state["as_of"], "signal": decision.signal,
                              "strength": decision.strength, "opportunity": decision.opportunity, "risk": decision.risk,
                              "confidence": decision.confidence, "agreement": decision.agreement,
                              "final_score": decision.final_score, "previous_signal": decision.previous_signal,
                              "duration_days": decision.duration_days, "change_reason": decision.change_reason,
                              "data_current": int(bool(state.get("data_current"))), "created_at": utcnow_iso()})
        # keep only the latest run per as_of date in the signal history used for hysteresis
        db.execute("DELETE FROM signals WHERE as_of=? AND run_id<>?", [state["as_of"], run_id])
        db.upsert_many("signal_components", [{"run_id": run_id, "component": k, "score": v.get("score"),
                                              "weight": self.s.signals.weights.get(k), "contribution": decision.contributions.get(k),
                                              "evidence": json.dumps(v.get("evidence"))} for k, v in decision.components.items()])
        db.upsert_many("agent_outputs", [{"run_id": run_id, "agent": k, "stance": v["stance"], "score": v["score"],
                                          "payload": json.dumps(v, default=str)} for k, v in state["agents"].items()])
        fn = state["features_now"]
        db.upsert_many("features", [{"run_id": run_id, "date": state["as_of"], "feature": k, "value": v}
                                    for k, v in fn.items() if v is not None])
        if fc and state.get("forecast_primary"):
            from ..data.calendar import add_trading_days as atd
            tds = {h: str(atd(state["as_of"], h + 1, self.s.exchange).date()) for h in fc.horizons}
            journal_predictions(db, run_id, state["as_of"], state["price"], fc, decision.signal,
                                state["regime"]["regime"], tds)
        lt = self._lineage_rows(run_id)
        db.upsert_many("feature_lineage", lt)
        db.upsert("regimes", {"date": state["as_of"], "method": "rule+hmm", "regime": state["regime"]["regime"],
                              "probability": state["regime"].get("hmm_stress_probability"),
                              "payload": json.dumps(state["regime"], default=str)})

    def _persist_inputs(self, bundle) -> int:
        """Store point-in-time macro releases (+ series metadata) and implied price adjustments."""
        from ..data.macro.fred import MACRO_SERIES

        now = utcnow_iso()
        n = 0
        for sid in bundle.pit.series_ids:
            r = bundle.pit.records(sid)
            if r.empty:
                continue
            desc, lag, _ = MACRO_SERIES.get(sid, (sid, None, None))
            self.db.upsert("macro_series", {"series_id": sid, "title": desc, "frequency": None, "units": None,
                                            "source": str(r["source"].iloc[-1]), "publication_lag_days": lag,
                                            "notes": f"pit_method={bundle.pit.method(sid)}", "updated_at": now})
            rows = [{"series_id": sid, "effective_date": str(pd.Timestamp(e).date()), "value": float(v),
                     "published_at": pd.Timestamp(pb).isoformat(), "vintage": str(vt), "pit_method": pm, "source": src,
                     "retrieved_at": now}
                    for e, v, pb, vt, pm, src in r[["effective_date", "value", "published_at", "vintage", "pit_method",
                                                    "source"]].itertuples(index=False)]
            self.db.upsert_many("macro_releases", rows)
            n += len(rows)
        t = bundle.target
        if {"adj_close", "close"} <= set(t.columns) and not bundle.is_synthetic:
            ratio = (t["adj_close"] / t["close"]).dropna()
            jumps = ratio[ratio.diff().abs() > 1e-4]
            self.db.upsert_many("adjustments", [{"symbol": bundle.target_symbol, "date": str(d.date()),
                                                 "kind": "implied_distribution_adjustment", "value": float(v),
                                                 "source": "adj_close/close", "retrieved_at": now} for d, v in jumps.items()])
        return n

    def _lineage_rows(self, run_id):
        return getattr(self, "_lineage_cache", [])

    def _previous_trace(self, as_of: str, run_id: str) -> dict | None:
        rows = self.db.query("SELECT trace FROM decision_trace WHERE run_id<>? ORDER BY created_at DESC LIMIT 1", [run_id])
        return json.loads(rows[0]["trace"]) if rows else None


# ---------------------------------------------------------------------- helpers
def _hr_summary(hr, full: bool = False) -> dict:
    if hr is None:
        return {}
    d = {"horizon": hr.horizon, "expected_return": hr.expected_return, "p_up": hr.p_up, "p_down": hr.p_down,
         "p_up_raw": hr.p_up_raw, "p_gt5": hr.p_gt5, "p_lt5": hr.p_lt5, "p_dd10": hr.p_dd10,
         "interval": list(hr.interval), "interval_level": hr.interval_level, "confidence": hr.confidence,
         "ensemble_method": hr.ensemble_method, "model_preds": hr.model_preds, "warnings": hr.warnings,
         "model_errors": hr.model_errors, "n_oos": hr.n_oos, "n_train": hr.n_train}
    if full:
        d.update({"quantiles": {str(k): v for k, v in hr.quantiles.items()}, "ensemble_weights": hr.ensemble_weights,
                  "ensemble_comparison": hr.ensemble_comparison, "ensemble_oos": hr.ensemble_comparison.get(hr.ensemble_method, {}),
                  "model_scores": hr.model_scores, "scores_by_regime": hr.scores_by_regime, "dm_tests": hr.dm_tests,
                  "calibration": hr.calibration, "calibration_curve": hr.calibration_curve, "conformal": hr.conformal,
                  "disagreement": hr.disagreement, "uncertainty": hr.uncertainty, "selected_features": hr.selected_features,
                  "folds": hr.folds, "model_ids": hr.model_ids,
                  "hpo": {k: [x.get("params") for x in v] for k, v in hr.hpo.items()}})
    return d


def _compare(prev: dict, cur: dict) -> dict:
    pd_, cd = prev.get("decision", {}), cur.get("decision", {})
    out = {"previous_as_of": prev.get("as_of"), "previous_signal": pd_.get("signal"), "signal": cd.get("signal"),
           "opportunity_change": (cd.get("opportunity") or 0) - (pd_.get("opportunity") or 0),
           "risk_change": (cd.get("risk") or 0) - (pd_.get("risk") or 0),
           "price_change": (cur.get("price") or 0) / (prev.get("price") or 1) - 1 if prev.get("price") else None}
    pp, cp_ = prev.get("forecast_primary", {}).get("p_up"), cur.get("forecast_primary", {}).get("p_up")
    if pp is not None and cp_ is not None:
        out["p_up_change"] = cp_ - pp
    comp_prev, comp_cur = pd_.get("components", {}), cd.get("components", {})
    out["component_changes"] = {k: (comp_cur[k].get("score") or 0) - (comp_prev.get(k, {}).get("score") or 0)
                                for k in comp_cur if k in comp_prev}
    return out


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    if isinstance(o, (pd.DataFrame,)):
        return o.to_dict("records")
    if isinstance(o, (pd.Series,)):
        return o.to_dict()
    return str(o)


def _clean_floats(o):
    if isinstance(o, dict):
        return {str(k): _clean_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_floats(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def _sanitize(o):
    """Recursively convert to JSON-safe primitives (str keys, finite floats or None)."""
    if isinstance(o, dict):
        return {str(k.date()) if isinstance(k, pd.Timestamp) else str(k): _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_sanitize(v) for v in o]
    if isinstance(o, (pd.DataFrame, pd.Series)):
        return _sanitize(o.to_dict("records") if isinstance(o, pd.DataFrame) else o.to_dict())
    if isinstance(o, (str, bool, int)) or o is None:
        return o
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    return _sanitize(_json_default(o)) if not isinstance(_json_default(o), str) or isinstance(o, (np.floating, np.integer, np.bool_)) \
        else _json_default(o)


def _jsonable(state: dict) -> dict:
    return _clean_floats(json.loads(json.dumps(_sanitize(state), default=_json_default)))
