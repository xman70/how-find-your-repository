"""Backend service layer (GUI-independent API).

The GUI only talks to these services, so the same backend can later be exposed as a web
API (e.g. FastAPI) without touching the quantitative code.
"""
from __future__ import annotations

import json
from typing import Callable

import pandas as pd

from ..backtesting.walkforward import run_walkforward_backtest
from ..config.settings import DATA_DIR, Settings
from ..data.calendar import is_trading_day, market_status
from ..data.service import DataService  # noqa: F401  (DataService is part of the public API)
from ..database.db import Database
from ..features.builder import build_features
from ..models.forecast_service import ForecastService
from ..models.registry import ModelRegistry
from ..monitoring.monitor import journal_frame, live_scorecard
from ..pipeline import research_lab as lab
from ..pipeline.daily import DailyPipeline, PipelineResult
from ..regimes.detector import detect_regimes
from ..risk.risk_engine import risk_metrics, stress_test


class Backend:
    """Facade bundling all services around one settings object and one database."""

    def __init__(self, settings: Settings, db: Database | None = None):
        self.s = settings
        self.db = db or Database(settings.db_path)
        self.data = DataService(settings, self.db)
        self.registry = ModelRegistry(self.db, DATA_DIR / "models")
        self.last: PipelineResult | None = None

    # ------------------------------------------------------------- SignalService / ResearchAgent
    def analyze(self, mode: str | None = None, offline: bool = False, synthetic: bool | None = None, as_of=None,
                use_llm: bool = False, progress: Callable | None = None) -> PipelineResult:
        res = DailyPipeline(self.s, self.db, progress).run(mode, offline, synthetic, as_of, use_llm)
        if res.ok:
            self.last = res
        return res

    def deep_research(self, offline=False, synthetic=None, progress=None, use_llm: bool = False) -> PipelineResult:
        return self.analyze("deep", offline, synthetic, None, use_llm, progress)

    # ------------------------------------------------------------- BacktestService
    def backtest(self, horizon: int = 20, start: str | None = None, progress=None, retrain_every: int = 63):
        if self.last is None:
            raise RuntimeError("Run an analysis first (data and features are reused).")
        a = self.last.artifacts
        bt = run_walkforward_backtest(a.features, a.bundle.target, a.regimes, self.s, a.bundle.pit, horizon, start,
                                      progress, retrain_every=retrain_every)
        bt_id = f"BT_{pd.Timestamp.now(tz="UTC").strftime('%Y%m%d_%H%M%S')}"
        tr = bt.result.trades
        if not tr.empty:
            self.db.upsert_many("trades", [{"backtest_id": bt_id, "trade_no": i, "signal_date": str(r["signal_date"]),
                                            "exec_date": str(r["exec_date"]), "side": r["side"], "price": float(r["price"]),
                                            "units": float(r["units"]), "cost": float(r["commission"])}
                                           for i, r in enumerate(tr.to_dict("records"))])
        self.db.upsert("backtests", {
            "backtest_id": bt_id, "created_at": str(pd.Timestamp.now(tz="UTC")),
            "config": json.dumps({"horizon": horizon, "start": start, "costs": self.s.costs.__dict__, "retrain_every": retrain_every}),
            "metrics": json.dumps({"strategy": bt.result.metrics, "buy_hold": bt.result.benchmark_metrics,
                                   "robustness": bt.robustness}, default=str),
            "leakage_audit": json.dumps(bt.audit.to_dict()),
            "equity_curve": bt.result.equity.iloc[::5].to_json(), "status": "ok"})
        return bt

    # ------------------------------------------------------------- Research lab
    def ablation(self, horizon: int = 20, progress=None):
        a = self._need()
        return lab.ablation_study(a.features, a.bundle.target, a.regimes, self.s, horizon, progress)

    def regime_options(self, horizon: int = 20):
        a = self._need()
        return lab.regime_option_comparison(a.features, a.bundle.target, a.regimes, self.s, horizon)

    def window_study(self, horizon: int = 20):
        a = self._need()
        return lab.history_and_window_study(a.features, a.bundle.target, self.s, horizon)

    def custom_experiment(self, groups, horizon=20, window="full", years=None, regime_vars=True):
        a = self._need()
        return lab.custom_experiment(a.features, a.bundle.target, a.regimes, self.s, groups, horizon, window, years, regime_vars)

    def _need(self):
        if self.last is None:
            raise RuntimeError("Run an analysis first.")
        return self.last.artifacts

    # ------------------------------------------------------------- MonitoringService / ReportService
    def journal(self) -> pd.DataFrame:
        return journal_frame(self.db)

    def scorecard(self) -> pd.DataFrame:
        return live_scorecard(self.db)

    def signal_history(self) -> pd.DataFrame:
        return self.db.query_df("SELECT * FROM signals ORDER BY as_of")

    def alerts(self, limit: int = 200) -> pd.DataFrame:
        return self.db.query_df("SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?", [limit])

    def reports(self) -> pd.DataFrame:
        return self.db.query_df("SELECT run_id, as_of, created_at FROM daily_reports ORDER BY created_at DESC")

    def report(self, run_id: str) -> dict | None:
        r = self.db.query("SELECT * FROM daily_reports WHERE run_id=?", [run_id])
        return r[0] if r else None

    def model_versions(self) -> pd.DataFrame:
        return self.db.query_df("SELECT * FROM model_versions ORDER BY created_at DESC")

    def sentiment_history(self) -> pd.DataFrame:
        return self.db.query_df("SELECT * FROM sentiment WHERE scope='news_all' ORDER BY date")

    def data_quality(self) -> pd.DataFrame:
        return self.db.query_df("SELECT * FROM data_quality ORDER BY checked_at DESC LIMIT 400")

    # ------------------------------------------------------------- decision trace / reproducibility
    def why(self, date: str) -> dict | None:
        """'Why was BUY generated on this date?' -> exact stored reasoning state."""
        return self.db.trace_for_date(date)

    def traces(self) -> pd.DataFrame:
        return self.db.query_df("SELECT run_id, as_of, signal, created_at FROM decision_trace ORDER BY created_at DESC")

    def reproduce(self, run_id: str, progress=None) -> dict:
        """Re-run a stored analysis with the same settings snapshot, seed and as-of date
        using cached data, then compare with the stored decision state."""
        tr = self.db.load_trace(run_id)
        if not tr:
            return {"ok": False, "error": f"no trace for {run_id}"}
        from copy import deepcopy

        from ..config.settings import _merge

        s2 = deepcopy(self.s)
        snap = dict(tr.get("settings_snapshot", {}))
        snap.pop("api_keys", None)
        _merge(s2, snap)
        res = DailyPipeline(s2, self.db, progress).run(tr.get("mode"), offline=True, synthetic=tr.get("is_synthetic"),
                                                       as_of=tr.get("as_of"))
        if not res.ok:
            return {"ok": False, "error": res.error}
        a, b = tr.get("decision", {}), res.state.get("decision", {})
        diffs = {}
        for k in ("signal", "opportunity", "risk", "final_score"):
            if a.get(k) != b.get(k):
                diffs[k] = {"stored": a.get(k), "reproduced": b.get(k)}
        fa, fb = tr.get("forecast_primary", {}), res.state.get("forecast_primary", {})
        for k in ("expected_return", "p_up"):
            if fa.get(k) is not None and fb.get(k) is not None and abs(fa[k] - fb[k]) > 1e-9:
                diffs[f"forecast.{k}"] = {"stored": fa[k], "reproduced": fb[k]}
        env_a, env_b = tr.get("environment", {}), res.state.get("environment", {})
        pkg_diff = {k: (env_a.get("packages", {}).get(k), v) for k, v in env_b.get("packages", {}).items()
                    if env_a.get("packages", {}).get(k) != v}
        notes = []
        if tr.get("news", {}).get("available"):
            notes.append("Original run used live news; the reproduction runs offline, so news-dependent components "
                         "(sentiment, event risk) can differ.")
        if pkg_diff:
            notes.append(f"Package versions differ: {pkg_diff}")
        if tr.get("data_snapshot", {}).get("dataset_version") != res.state.get("data_snapshot", {}).get("dataset_version"):
            notes.append("Dataset version differs (provider revised historical prices or cache changed).")
        return {"ok": True, "identical": not diffs, "differences": diffs, "notes": notes,
                "reproduction_run_id": res.run_id}

    # ------------------------------------------------------------- misc
    def market_status(self) -> dict:
        return market_status(exchange=self.s.exchange)

    def is_trading_day(self, d) -> bool:
        return is_trading_day(d, self.s.exchange)


# ---------------------------------------------------------------------------------------------
# Named services (spec section 74). Each is a small, stateless facade over the engine modules so
# a future web API (e.g. FastAPI routes) can expose them one-to-one.
# ---------------------------------------------------------------------------------------------
class FeatureService:
    build = staticmethod(build_features)


class ForecastServiceAPI:
    def __init__(self, settings: Settings, registry: ModelRegistry | None = None):
        self.svc = ForecastService(settings, registry)

    def forecast(self, fs, target, regimes, horizons, models, as_of=None, pit=None):
        return self.svc.run(fs, target, regimes, horizons, models, as_of, pit=pit)


class RegimeService:
    detect = staticmethod(detect_regimes)

    @staticmethod
    def summary(regimes):
        from ..regimes.detector import current_regime_summary

        return current_regime_summary(regimes)


class RiskService:
    metrics = staticmethod(risk_metrics)
    stress = staticmethod(stress_test)


class NewsService:
    def __init__(self, settings: Settings):
        from ..agents.news_agent.agent import MarketResearchAgent

        self.agent = MarketResearchAgent(settings)

    def research(self, offline: bool = False) -> dict:
        return self.agent.run(offline)


ResearchAgent = NewsService


class SignalService:
    def __init__(self, backend: Backend):
        self.b = backend

    def current(self) -> dict | None:
        return self.b.last.state["decision"] if self.b.last else None

    def history(self):
        return self.b.signal_history()

    def why(self, date: str):
        return self.b.why(date)


class BacktestService:
    def __init__(self, backend: Backend):
        self.b = backend

    def run(self, horizon: int = 20, start: str | None = None, retrain_every: int = 63):
        return self.b.backtest(horizon, start, None, retrain_every)


class MonitoringService:
    def __init__(self, backend: Backend):
        self.b = backend

    def journal(self):
        return self.b.journal()

    def scorecard(self):
        return self.b.scorecard()

    def drift(self) -> dict:
        st = self.b.last.state if self.b.last else {}
        return {"model": st.get("model_drift"), "feature": st.get("feature_drift")}


class ReportService:
    def __init__(self, backend: Backend):
        self.b = backend

    def latest(self) -> str | None:
        return self.b.last.state.get("report_markdown") if self.b.last else None

    def list(self):
        return self.b.reports()
