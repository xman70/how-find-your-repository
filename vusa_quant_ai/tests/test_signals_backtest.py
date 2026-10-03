"""Decision engine, backtest engine, Monte Carlo, narrative guard and API-failure tests."""
import numpy as np
import pandas as pd
import pytest

from vusa_quant_ai.backtesting.engine import run_backtest, signal_to_exposure
from vusa_quant_ai.config.settings import CostSettings, SignalSettings
from vusa_quant_ai.signals.components import Component
from vusa_quant_ai.signals.decision import decide


def comps(score: float, **override):
    names = ["trend", "momentum", "valuation", "macro", "sentiment", "volatility", "breadth", "regime", "event_risk",
             "ml_forecast", "analogues", "risk_reward"]
    return {n: Component(n, override.get(n, score), [f"{n} evidence"]) for n in names}


def hist(sig, dur=10):
    return pd.DataFrame([{"signal": sig, "duration_days": dur}])


def test_buy_hold_sell_thresholds():
    cfg = SignalSettings()
    assert decide(comps(80), 30, cfg, 0.6, None, {}, {}).signal == "BUY"
    assert decide(comps(20), 70, cfg, 0.6, None, {}, {}).signal == "SELL"
    assert decide(comps(52), 40, cfg, 0.6, None, {}, {}).signal == "HOLD"


def test_hysteresis_keeps_signal_stable():
    cfg = SignalSettings()
    # score slightly below the BUY threshold: a previous BUY persists, a previous HOLD does not upgrade
    c = comps(60)  # between exit level (62-4) and entry level (62)
    d_prev_buy = decide(c, 50, cfg, 0.6, hist("BUY"), {}, {})
    d_prev_hold = decide(c, 50, cfg, 0.6, hist("HOLD"), {}, {})
    assert d_prev_buy.signal == "BUY"
    assert d_prev_hold.signal == "HOLD"
    assert not d_prev_buy.changed


def test_single_bullish_subsystem_is_downgraded():
    cfg = SignalSettings(buy_threshold=52)
    c = comps(50, ml_forecast=100)  # only ML is bullish
    d = decide(c, 30, cfg, 0.6, None, {}, {})
    assert d.signal == "HOLD" and d.strength == "WEAK BULLISH"


def test_strong_buy_requires_confirmations():
    cfg = SignalSettings()
    strong = decide(comps(85), 20, cfg, 0.7, None, {}, {})
    assert strong.signal == "BUY" and strong.strength == "STRONG"
    weak_conf = decide(comps(85, ml_forecast=45, regime=45, risk_reward=45), 20, cfg, 0.7, None, {}, {})
    assert weak_conf.strength != "STRONG"


def test_disagreement_and_what_would_change_reported():
    cfg = SignalSettings()
    d = decide(comps(80, valuation=20, macro=25), 30, cfg, 0.6, None, {"shocks": 0.2}, {"price": 100, "sma200": 95,
                                                                                         "p_up": 0.6, "p_lt5": 0.1})
    assert any("disagreement" in c.lower() for c in d.contradictions)
    assert len(d.what_would_change) >= 3
    assert d.confidence < 0.95
    assert sum(d.contributions.values()) == pytest.approx(d.opportunity - 50, abs=0.1)


def test_penalties_reduce_confidence():
    cfg = SignalSettings()
    a = decide(comps(75), 30, cfg, 0.7, None, {}, {})
    b = decide(comps(75), 30, cfg, 0.7, None, {"model_drift": 0.25, "data_not_current": 0.3}, {})
    assert b.confidence < a.confidence


def _prices():
    idx = pd.bdate_range("2024-01-01", periods=60)
    o = np.linspace(100, 130, 60)
    return pd.DataFrame({"open": o, "close": o + 0.5}, index=idx)


def test_backtest_executes_next_open_with_costs():
    px = _prices()
    tgt = pd.Series(0.0, index=px.index)
    tgt.iloc[10:] = 1.0  # signal after close of day 10
    c = CostSettings(commission_pct=0.001, commission_min=1.0, spread_bps=10, slippage_bps=5)
    r = run_backtest(px, tgt, c, 10000)
    first = r.trades.iloc[0]
    assert str(first["exec_date"]) == str(px.index[11].date())  # next session
    assert first["price"] == pytest.approx(px["open"].iloc[11] * (1 + 0.0005 + 0.0005))
    free = run_backtest(px, tgt, CostSettings(commission_pct=0, commission_min=0, spread_bps=0, slippage_bps=0), 10000)
    assert r.equity.iloc[-1] < free.equity.iloc[-1]
    assert r.metrics["n_trades"] >= 1


def test_signal_to_exposure_hold_keeps_position():
    s = pd.Series(["BUY", "HOLD", "SELL", "HOLD", "BUY"])
    assert list(signal_to_exposure(s)) == [1.0, 1.0, 0.0, 0.0, 1.0]


def test_monte_carlo_methods(bundle):
    from vusa_quant_ai.simulation.montecarlo import run_monte_carlo

    mc = run_monte_carlo(bundle.target["close"], 20, 2000, 0, forecast_ret=0.01, conformal_resid=np.random.default_rng(0).normal(0, .04, 500))
    assert len(mc) == 6
    for r in mc.values():
        s = r.stats
        assert s["P10"] <= s["P25"] <= s["P50"] <= s["P75"] <= s["P90"]
        assert r.fan.shape[0] == 20
    assert mc["Model-driven"].stats["mean"] == pytest.approx(0.01, abs=0.01)


def test_fabrication_guard():
    from vusa_quant_ai.explainability.narrative import fabrication_check, template_explanation

    payload = {"decision": {"label": "BUY", "opportunity": 72.4, "risk": 31.0}, "forecast_primary": {"p_up": 0.64}}
    assert fabrication_check("Opportunity is 72.4 and P(up) is 64%.", payload) == []
    assert fabrication_check("The S&P will reach 7215 and return 13.7%.", payload) != []
    assert "not a guarantee" in template_explanation({"decision": {"label": "BUY", "contributions": {}}})


def test_llm_explanation_without_key_falls_back():
    from vusa_quant_ai.explainability.narrative import llm_explanation

    text, meta = llm_explanation({"decision": {"label": "HOLD", "contributions": {}}}, api_key="")
    assert meta["mode"] == "template" and "HOLD" in text


def test_provider_fallback_chain_and_health():
    from vusa_quant_ai.data.sources.base import SourceHealth, run_with_fallback

    h = SourceHealth()

    def boom():
        raise ConnectionError("network down")

    ok_df = pd.DataFrame({"close": [1.0]}, index=pd.to_datetime(["2024-01-02"]))
    res, attempts = run_with_fallback("VUSA", [("A", boom), ("B", lambda: pd.DataFrame()), ("C", lambda: ok_df)], h)
    assert res.ok and res.source == "C"
    assert [a.ok for a in attempts] == [False, False, True]
    t = h.table()
    assert set(t["status"]) == {"failed", "ok"}
    res2, _ = run_with_fallback("VUSA", [("A", boom)], h)
    assert not res2.ok and "network down" in res2.error


def test_all_providers_down_no_fabrication(settings, tmp_path, monkeypatch):
    """Without network and without cache the pipeline must stop with an explicit error."""
    from copy import deepcopy

    from vusa_quant_ai.database.db import Database
    from vusa_quant_ai.pipeline.daily import DailyPipeline

    s = deepcopy(settings)
    s.providers.market_priority = []  # no providers available
    db = Database(tmp_path / "empty.sqlite")
    r = DailyPipeline(s, db).run(mode="fast", offline=True, synthetic=False)
    assert not r.ok
    assert "No VUSA price data" in r.error
    assert "decision" not in r.state


def test_cache_used_when_offline_and_flagged(settings, bundle, tmp_path):
    from copy import deepcopy

    from vusa_quant_ai.data.service import DataService
    from vusa_quant_ai.database.db import Database

    db = Database(tmp_path / "c.sqlite")
    s = deepcopy(settings)
    db.save_prices(s.yahoo_symbol, bundle.target, "yfinance")
    b = DataService(s, db).load(offline=True, include_macro=False)
    assert not b.target.empty
    assert not b.data_current and "DATA NOT CURRENT" in b.status_label
    assert any(r.from_cache for r in b.fetch_log)


def test_scheduler_skips_weekends_and_holidays():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from vusa_quant_ai.pipeline.scheduler import next_run_time, should_run_now

    sat = datetime(2026, 10, 3, 19, 0, tzinfo=ZoneInfo("Europe/Nicosia"))
    ok, why = should_run_now(sat, None, "18:30", "Europe/Nicosia", "XETRA")
    assert not ok and "not a trading day" in why
    nr = next_run_time(sat, "18:30", "Europe/Nicosia", "XETRA")
    assert nr.weekday() == 0
    gf = datetime(2026, 4, 3, 19, 0, tzinfo=ZoneInfo("Europe/Nicosia"))  # Good Friday
    assert not should_run_now(gf, None, "18:30", "Europe/Nicosia", "XETRA")[0]


def test_low_confidence_suppresses_directional_signal():
    cfg = SignalSettings()
    d = decide(comps(85), 20, cfg, 0.7, None, {"synthetic_data": 0.5, "data_not_current": 0.3, "model_drift": 0.25,
                                              "shocks": 0.4}, {})
    assert d.signal == "HOLD" and "LOW CONFIDENCE" in d.strength
    assert any("suppressed" in n for n in d.notes)


def test_high_risk_blocks_buy_and_says_so():
    cfg = SignalSettings()
    d = decide(comps(95), 95, cfg, 0.7, None, {}, {})
    assert d.signal != "BUY"
    assert any("BUY withheld" in n for n in d.notes)
