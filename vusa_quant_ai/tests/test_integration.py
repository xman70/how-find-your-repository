"""Integration tests: full 20-step pipeline (fast mode, synthetic data), decision trace,
prediction journal resolution, drift monitoring, reproduction, report, and a GUI smoke test.

These cover the MINIMUM ACCEPTANCE TEST items that can be verified offline; live-provider
downloads are exercised by scripts/acceptance_test.py when internet access is available.
"""
import json
from copy import deepcopy

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def pipeline_result(settings, tmp_path_factory):
    from vusa_quant_ai.database.db import Database
    from vusa_quant_ai.pipeline.daily import DailyPipeline

    db = Database(tmp_path_factory.mktemp("db") / "it.sqlite")
    s = deepcopy(settings)
    p = DailyPipeline(s, db)
    res = p.run(mode="fast", synthetic=True)
    assert res.ok, res.error + str(res.state.get("traceback"))
    return p, res


def test_pipeline_runs_all_20_steps(pipeline_result):
    _, res = pipeline_result
    assert [s["step"] for s in res.steps] == list(range(1, 21))
    st = res.state
    d = st["decision"]
    assert d["signal"] in ("BUY", "HOLD", "SELL")
    assert 0 <= d["opportunity"] <= 100 and 0 <= d["risk"] <= 100
    assert len(d["components"]) == 12
    assert st["leakage_audit"]["passed"]
    assert st["is_synthetic"] and "SYNTHETIC" in st["report_markdown"]
    for k in ("monte_carlo", "stress", "analogues", "committee", "self_audit", "regime", "forecasts"):
        assert st.get(k), k
    assert set(st["agents"]) == {"Quant", "Macro", "News", "Technical", "Risk Manager", "ML Scientist",
                                 "Portfolio Analyst", "Skeptic"}
    com = st["committee"]
    assert "bull_case" in com and "bear_case" in com and "neutral_case" in com
    assert d["what_would_change"]
    assert set(st["self_audit"]) >= {"MODEL STRENGTHS", "MODEL WEAKNESSES", "POSSIBLE LEAKAGE", "RESEARCH RISKS"}


def test_decision_trace_is_stored_and_reconstructable(pipeline_result):
    p, res = pipeline_result
    tr = p.db.load_trace(res.run_id)
    assert tr["decision"]["signal"] == res.state["decision"]["signal"]
    assert tr["model_versions"] and tr["environment"]["python"]
    assert p.db.trace_for_date(res.state["as_of"])["run_id"] == res.run_id
    json.dumps(tr)  # fully serialisable
    lin = p.db.query("SELECT COUNT(*) AS n FROM feature_lineage WHERE run_id=?", [res.run_id])[0]["n"]
    assert lin > 50


def test_prediction_journal_resolves(pipeline_result, bundle):
    from vusa_quant_ai.monitoring.monitor import live_scorecard, resolve_predictions

    p, res = pipeline_result
    db = p.db
    # simulate a prediction made 30 sessions ago that has now matured
    idx = bundle.target.index
    asof, tgt = idx[-30], idx[-10]
    db.upsert("predictions", {"prediction_id": "OLD_H20", "run_id": "OLD", "as_of": str(asof.date()), "horizon": 20,
                              "model_id": "x", "price": 1.0, "expected_return": 0.01, "p_up": 0.6, "p_down": 0.4,
                              "p_gt5": None, "p_lt5": None, "p_dd10": None, "confidence": 0.5, "signal": "BUY",
                              "target_date": str(tgt.date()), "actual_return": None, "error": None, "resolved_at": None,
                              "regime": "Bull"})
    n = resolve_predictions(db, bundle.target["close"], bundle.target["open"])
    assert n >= 1
    row = db.query("SELECT * FROM predictions WHERE prediction_id='OLD_H20'")[0]
    exp = bundle.target["close"].loc[tgt] / bundle.target["open"].iloc[idx.get_loc(asof) + 1] - 1
    assert row["actual_return"] == pytest.approx(exp)
    sc = live_scorecard(db)
    assert not sc.empty and "directional_accuracy" in sc


def test_second_run_compares_with_previous_and_hysteresis(pipeline_result):
    p, res = pipeline_result
    res2 = p.run(mode="fast", synthetic=True)
    assert res2.ok
    assert res2.state["vs_previous"].get("previous_signal") is not None
    assert res2.state["decision"]["previous_signal"] is None or isinstance(res2.state["decision"]["previous_signal"], str)


def test_feature_drift_flags_out_of_distribution(features):
    from vusa_quant_ai.monitoring.monitor import feature_drift

    X = features.frame[["rv_20", "dist_sma200", "rsi14", "vix_level"]].copy()
    normal = feature_drift(None, X, list(X.columns))
    X.iloc[-1] = X.iloc[-1] + 25 * X.std()
    shocked = feature_drift(None, X, list(X.columns))
    assert shocked["alert"] and "OUT-OF-DISTRIBUTION" in shocked["message"]
    assert shocked["penalty"] >= normal["penalty"]


def test_backtest_and_research_lab(pipeline_result, settings):
    from vusa_quant_ai.backtesting.walkforward import run_walkforward_backtest
    from vusa_quant_ai.pipeline.research_lab import ablation_study, research_questions

    _, res = pipeline_result
    a = res.artifacts
    bt = run_walkforward_backtest(a.features, a.bundle.target, a.regimes, settings, a.bundle.pit, 20,
                                  start=str(a.bundle.target.index[-600].date()), retrain_every=126)
    assert bt.audit.passed
    assert len(bt.signals) > 500
    assert set(bt.signals["signal"].unique()) <= {"BUY", "HOLD", "SELL"}
    assert "cost_sensitivity" in bt.robustness
    # trades never execute on the signal date
    if not bt.result.trades.empty:
        assert (pd.to_datetime(bt.result.trades["exec_date"]) > pd.to_datetime(bt.result.trades["signal_date"])).all()
    ab = ablation_study(a.features, a.bundle.target, a.regimes, settings, 20)
    assert "BASE (all groups)" in set(ab["experiment"])
    sent = ab[ab["removed"] == "sentiment"]
    assert "CANNOT TEST" in str(sent.iloc[0]["note"])
    q = research_questions(a.forecast, ab, None, None, bt)
    assert any("news sentiment" in x["question"] for x in q)


def test_reproduce_analysis(settings, tmp_path):
    from vusa_quant_ai.database.db import Database
    from vusa_quant_ai.services.api import Backend

    s = deepcopy(settings)
    b = Backend(s, Database(tmp_path / "r.sqlite"))
    r = b.analyze("fast", synthetic=True)
    assert r.ok
    rep = b.reproduce(r.run_id)
    assert rep["ok"]
    assert rep["identical"], rep["differences"]


def test_gui_smoke(pipeline_result, settings):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    try:
        from vusa_quant_ai.gui.main_window import MainWindow
    except ImportError as exc:  # missing system GL libraries in headless CI
        pytest.skip(str(exc))
    app = QApplication.instance() or QApplication([])
    w = MainWindow(deepcopy(settings))
    w.backend.db = pipeline_result[0].db
    w.populate(pipeline_result[1])
    assert "SYNTHETIC" in w.banner.text()
    assert w.tabs.count() == 16
    w.close()
    app.processEvents()
