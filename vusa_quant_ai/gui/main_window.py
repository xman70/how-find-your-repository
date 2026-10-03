"""VUSA AI QUANT TERMINAL - main window."""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSpinBox,
                               QSplitter, QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

from .. import APP_NAME, __version__
from ..config.settings import Settings
from ..core.runtime import detect_hardware
from ..pipeline.scheduler import next_run_time, should_run_now
from ..services.api import Backend
from . import charts as CH
from .settings_tab import SettingsTab
from .widgets import ChartView, DataTable, kpi_row, start_worker


def _pct(x, d=1, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{100 * x:{'+' if sign else ''}.{d}f}%"


def _scroll(inner: QWidget) -> QScrollArea:
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setWidget(inner)
    return sa


class MainWindow(QMainWindow):
    TABS = ["Dashboard", "Forecasts", "Signals", "Technical", "Macro", "News AI", "Market Regime", "Machine Learning",
            "Model Lab", "Backtesting", "Monte Carlo", "Risk", "Explainability", "Prediction Journal", "Data Quality",
            "Settings"]

    def __init__(self, settings: Settings):
        super().__init__()
        self.s = settings
        self.backend = Backend(settings)
        self.result = None
        self._threads = []
        self.setWindowTitle(f"{APP_NAME} v{__version__} - decision support, not investment advice")
        self.resize(1550, 980)
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.addWidget(self._header())
        self.banner = QLabel("No analysis yet. Click ANALYZE VUSA NOW.")
        self.banner.setObjectName("banner")
        self.banner.setWordWrap(True)
        root.addWidget(self.banner)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)
        self.pages = {}
        for t in self.TABS:
            builder = getattr(self, f"_build_{t.lower().replace(' ', '_')}")
            w = builder()
            self.pages[t] = w
            self.tabs.addTab(w if t in ("Settings",) else _scroll(w), t)
        self.statusBar().showMessage(f"Hardware: {detect_hardware()}")
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(60_000)
        self._last_sched_session = None
        self._update_schedule_label()
        if self.s.schedule.run_at_launch:
            QTimer.singleShot(1500, lambda: self.run_analysis(self.s.mode))

    # ================================================================== header
    def _header(self) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        title = QLabel("<span style='font-size:18pt;font-weight:bold;color:#58a6ff'>VUSA AI QUANT TERMINAL</span>"
                       "<br><span style='color:#8b949e'>Vanguard S&P 500 UCITS ETF - research & decision support</span>")
        lay.addWidget(title)
        lay.addStretch(1)
        self.mode = QComboBox()
        self.mode.addItems(["fast", "balanced", "deep", "continuous"])
        self.mode.setCurrentText(self.s.mode)
        lay.addWidget(QLabel("Mode:"))
        lay.addWidget(self.mode)
        self.offline = QCheckBox("Offline (cache only)")
        self.demo = QCheckBox("Synthetic DEMO data")
        self.demo.setToolTip("Clearly-labelled synthetic data for exploring the app when no provider is reachable.")
        self.llm = QCheckBox("LLM explanation")
        self.llm.setToolTip("Uses Claude (ANTHROPIC_API_KEY) to phrase the explanation from structured JSON; "
                            "a fabrication guard rejects any untraceable number.")
        for c in (self.offline, self.demo, self.llm):
            lay.addWidget(c)
        self.btn_analyze = QPushButton("ANALYZE VUSA NOW")
        self.btn_analyze.clicked.connect(lambda: self.run_analysis(self.mode.currentText()))
        self.btn_deep = QPushButton("RUN DEEP RESEARCH")
        self.btn_deep.setObjectName("deep")
        self.btn_deep.clicked.connect(lambda: self.run_analysis("deep"))
        self.btn_repro = QPushButton("Reproduce analysis")
        self.btn_repro.setObjectName("secondary")
        self.btn_repro.clicked.connect(self.reproduce)
        for b in (self.btn_analyze, self.btn_deep, self.btn_repro):
            lay.addWidget(b)
        v = QVBoxLayout()
        self.progress = QProgressBar()
        self.progress.setMaximum(1000)
        self.progress.setMinimumWidth(260)
        self.progress_lbl = QLabel("")
        self.progress_lbl.setMaximumWidth(420)
        self.sched_lbl = QLabel("")
        v.addWidget(self.progress)
        v.addWidget(self.progress_lbl)
        v.addWidget(self.sched_lbl)
        lay.addLayout(v)
        return w

    def _busy(self, on: bool):
        for b in (self.btn_analyze, self.btn_deep, self.btn_repro):
            b.setEnabled(not on)

    def _on_progress(self, msg: str, frac: float):
        self.progress.setValue(int(1000 * max(0.0, min(1.0, frac))))
        self.progress_lbl.setText(msg[:120])

    # ================================================================== actions
    def run_analysis(self, mode: str):
        self._busy(True)
        self.progress.setValue(0)
        th, wk = start_worker(self, self.backend.analyze, self._done, self._fail, self._on_progress, mode,
                              self.offline.isChecked(), True if self.demo.isChecked() else None, None, self.llm.isChecked())
        self._threads.append((th, wk))

    def _fail(self, msg: str):
        self._busy(False)
        QMessageBox.critical(self, "Analysis failed", msg[:4000])
        self.banner.setText("⚠ Analysis failed - see message. No result was fabricated.")
        self.banner.setStyleSheet("background:#5c1a1a;color:white")

    def _done(self, res):
        self._busy(False)
        if not res.ok:
            self._fail(res.error + "\n\n" + str(res.state.get("traceback", ""))[-1500:])
            return
        self.result = res
        self._last_sched_session = res.state.get("as_of")
        try:
            self.populate(res)
        except Exception as exc:  # noqa: BLE001
            import traceback

            QMessageBox.warning(self, "Display error", f"Analysis finished but a view failed: {exc}\n{traceback.format_exc()[-1500:]}")

    def reproduce(self):
        if not self.result:
            QMessageBox.information(self, "Reproduce", "Run or load an analysis first.")
            return
        self._busy(True)
        th, wk = start_worker(self, self.backend.reproduce, self._repro_done, self._fail, self._on_progress, self.result.run_id)
        self._threads.append((th, wk))

    def _repro_done(self, rep: dict):
        self._busy(False)
        QMessageBox.information(self, "Reproduction result", json.dumps(rep, indent=2, default=str)[:4000])

    def _tick(self):
        self._update_schedule_label()
        if not self.s.schedule.daily_enabled and self.mode.currentText() != "continuous":
            return
        if not self.btn_analyze.isEnabled():
            return
        ok, why = should_run_now(datetime.now().astimezone(), self._last_sched_session, self.s.schedule.daily_time,
                                 self.s.schedule.timezone, self.s.exchange)
        if ok:
            self._last_sched_session = str(datetime.now().date())
            self.run_analysis(self.mode.currentText() if self.mode.currentText() != "continuous" else "balanced")

    def _update_schedule_label(self):
        try:
            nr = next_run_time(datetime.now().astimezone(), self.s.schedule.daily_time, self.s.schedule.timezone, self.s.exchange)
            self.sched_lbl.setText(f"Next scheduled run: {nr:%a %Y-%m-%d %H:%M} ({self.s.schedule.timezone})"
                                   if self.s.schedule.daily_enabled else "Daily schedule disabled")
        except Exception:
            self.sched_lbl.setText("")

    # ================================================================== tab builders
    def _build_dashboard(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        r1, self.k1 = kpi_row(["Price", "Signal", "Opportunity", "Risk", "Confidence", "Model agreement"])
        r2, self.k2 = kpi_row(["20D expected return", "80% range", "P(up)", "P(down)", "Regime", "Data quality"])
        lay.addWidget(r1)
        lay.addWidget(r2)
        self.dash_chart = ChartView(min_height=520)
        lay.addWidget(self.dash_chart)
        split = QSplitter(Qt.Horizontal)
        self.consensus_tbl = DataTable()
        self.components_chart = ChartView(min_height=420)
        split.addWidget(self.consensus_tbl)
        split.addWidget(self.components_chart)
        split.setSizes([500, 900])
        lay.addWidget(split)
        self.decision_text = QTextBrowser()
        self.decision_text.setMinimumHeight(700)
        lay.addWidget(self.decision_text)
        self.alerts_tbl = DataTable()
        self.alerts_tbl.setMinimumHeight(200)
        lay.addWidget(QLabel("Alerts"))
        lay.addWidget(self.alerts_tbl)
        return w

    def _build_forecasts(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.fc_chart = ChartView()
        self.fc_tbl = DataTable()
        self.fc_tbl.setMinimumHeight(260)
        self.dist_chart = ChartView()
        self.cf_tbl = DataTable()
        self.cf_tbl.setMinimumHeight(250)
        lay.addWidget(self.fc_chart)
        lay.addWidget(self.fc_tbl)
        lay.addWidget(self.dist_chart)
        lay.addWidget(QLabel("Counterfactual scenarios (model sensitivity - SCENARIOS, not predictions)"))
        lay.addWidget(self.cf_tbl)
        return w

    def _build_signals(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.sig_chart = ChartView()
        self.sig_tbl = DataTable()
        self.sig_tbl.setMinimumHeight(250)
        h = QHBoxLayout()
        self.why_date = QLineEdit()
        self.why_date.setPlaceholderText("YYYY-MM-DD")
        b = QPushButton("Why was this signal generated on this date?")
        b.clicked.connect(self.why)
        h.addWidget(self.why_date)
        h.addWidget(b)
        self.why_text = QTextBrowser()
        self.why_text.setMinimumHeight(400)
        lay.addWidget(self.sig_chart)
        lay.addWidget(self.sig_tbl)
        lay.addLayout(h)
        lay.addWidget(self.why_text)
        return w

    def _build_technical(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.tech_chart = ChartView(min_height=560)
        self.tech_osc = ChartView()
        self.tech_vol = ChartView()
        self.tech_breadth = ChartView()
        self.tech_tbl = DataTable()
        self.tech_tbl.setMinimumHeight(400)
        for x in (self.tech_chart, self.tech_osc, self.tech_vol, self.tech_breadth, self.tech_tbl):
            lay.addWidget(x)
        return w

    def _build_macro(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.macro_rates = ChartView()
        self.macro_infl = ChartView()
        self.macro_credit = ChartView()
        self.lead_tbl = DataTable()
        self.lead_tbl.setMinimumHeight(320)
        self.macro_tbl = DataTable()
        self.macro_tbl.setMinimumHeight(300)
        for x in (self.macro_rates, self.macro_infl, self.macro_credit):
            lay.addWidget(x)
        lay.addWidget(QLabel("Leading-indicator dashboard (statistical associations, not causation)"))
        lay.addWidget(self.lead_tbl)
        lay.addWidget(QLabel("Macro series status / point-in-time method"))
        lay.addWidget(self.macro_tbl)
        return w

    def _build_news_ai(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.news_text = QTextBrowser()
        self.news_text.setMinimumHeight(260)
        self.news_events = DataTable()
        self.news_events.setMinimumHeight(260)
        self.news_articles = DataTable()
        self.news_articles.setMinimumHeight(360)
        self.event_tbl = DataTable()
        self.event_tbl.setMinimumHeight(300)
        self.sent_chart = ChartView(min_height=300)
        lay.addWidget(self.news_text)
        lay.addWidget(self.sent_chart)
        lay.addWidget(QLabel("Extracted events"))
        lay.addWidget(self.news_events)
        lay.addWidget(QLabel("Relevant articles (de-duplicated, scored)"))
        lay.addWidget(self.news_articles)
        lay.addWidget(QLabel("Event-impact study / news memory: historical conditional outcomes"))
        lay.addWidget(self.event_tbl)
        return w

    def _build_market_regime(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.reg_text = QTextBrowser()
        self.reg_text.setMinimumHeight(220)
        self.reg_chart = ChartView(min_height=560)
        self.cp_chart = ChartView()
        lay.addWidget(self.reg_text)
        lay.addWidget(self.reg_chart)
        lay.addWidget(self.cp_chart)
        return w

    def _build_machine_learning(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.ml_scores = DataTable()
        self.ml_scores.setMinimumHeight(300)
        self.ml_ens = DataTable()
        self.ml_ens.setMinimumHeight(200)
        self.ml_cal = ChartView()
        self.ml_oos = ChartView()
        self.ml_text = QTextBrowser()
        self.ml_text.setMinimumHeight(420)
        lay.addWidget(QLabel("Base models - out-of-sample walk-forward metrics (primary horizon)"))
        lay.addWidget(self.ml_scores)
        lay.addWidget(QLabel("Ensemble methods - out-of-sample comparison"))
        lay.addWidget(self.ml_ens)
        lay.addWidget(self.ml_cal)
        lay.addWidget(self.ml_oos)
        lay.addWidget(self.ml_text)
        return w

    def _build_model_lab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        g = QGroupBox("Research Lab - test the system's own assumptions (walk-forward, purged)")
        gl = QGridLayout(g)
        self.lab_groups = {}
        for i, grp in enumerate(["technical", "volatility", "cross_asset", "breadth", "macro", "sentiment"]):
            cb = QCheckBox(grp)
            cb.setChecked(True)
            self.lab_groups[grp] = cb
            gl.addWidget(cb, 0, i)
        self.lab_window = QComboBox()
        self.lab_window.addItems(["full", "recent", "exp_weighted"])
        self.lab_years = QSpinBox()
        self.lab_years.setRange(0, 40)
        self.lab_years.setSpecialValueText("all years")
        self.lab_regime = QCheckBox("regime variables")
        self.lab_regime.setChecked(True)
        gl.addWidget(QLabel("Training window"), 1, 0)
        gl.addWidget(self.lab_window, 1, 1)
        gl.addWidget(QLabel("Use only last N years"), 1, 2)
        gl.addWidget(self.lab_years, 1, 3)
        gl.addWidget(self.lab_regime, 1, 4)
        btns = [("Run custom experiment", self.lab_custom), ("Run ablation study", self.lab_ablation),
                ("Regime Option A vs B", self.lab_regimes), ("Training window / history study", self.lab_window_study)]
        for i, (t, f) in enumerate(btns):
            b = QPushButton(t)
            b.clicked.connect(f)
            gl.addWidget(b, 2, i)
        lay.addWidget(g)
        self.lab_tbl = DataTable()
        self.lab_tbl.setMinimumHeight(360)
        self.lab_q = DataTable()
        self.lab_q.setMinimumHeight(300)
        self.models_tbl = DataTable()
        self.models_tbl.setMinimumHeight(300)
        lay.addWidget(self.lab_tbl)
        lay.addWidget(QLabel("Does the system disprove itself? (research questions)"))
        lay.addWidget(self.lab_q)
        lay.addWidget(QLabel("Model registry (versions, roles: production / challenger; retired models keep history)"))
        lay.addWidget(self.models_tbl)
        self._lab_cache = {}
        return w

    def _build_backtesting(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        h = QHBoxLayout()
        self.bt_h = QComboBox()
        self.bt_h.addItems(["20", "5", "60"])
        self.bt_retrain = QSpinBox()
        self.bt_retrain.setRange(5, 252)
        self.bt_retrain.setValue(63)
        b = QPushButton("Run walk-forward backtest (leakage audit first)")
        b.clicked.connect(self.run_backtest)
        h.addWidget(QLabel("Horizon"))
        h.addWidget(self.bt_h)
        h.addWidget(QLabel("Retrain every (sessions)"))
        h.addWidget(self.bt_retrain)
        h.addWidget(b)
        h.addStretch(1)
        lay.addLayout(h)
        self.bt_chart = ChartView(min_height=520)
        self.bt_text = QTextBrowser()
        self.bt_text.setMinimumHeight(420)
        self.bt_trades = DataTable()
        self.bt_trades.setMinimumHeight(260)
        lay.addWidget(self.bt_chart)
        lay.addWidget(self.bt_text)
        lay.addWidget(self.bt_trades)
        return w

    def _build_monte_carlo(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.mc_fan = ChartView()
        self.mc_dist = ChartView()
        self.mc_tbl = DataTable()
        self.mc_tbl.setMinimumHeight(260)
        lay.addWidget(self.mc_fan)
        lay.addWidget(self.mc_dist)
        lay.addWidget(self.mc_tbl)
        return w

    def _build_risk(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.risk_text = QTextBrowser()
        self.risk_text.setMinimumHeight(260)
        self.stress_tbl = DataTable()
        self.stress_tbl.setMinimumHeight(360)
        self.dd_chart = ChartView()
        self.corr_chart = ChartView(min_height=560)
        self.rc_chart = ChartView()
        lay.addWidget(self.risk_text)
        lay.addWidget(QLabel("Stress tests and historical replays (SCENARIOS - not predictions)"))
        lay.addWidget(self.stress_tbl)
        for x in (self.dd_chart, self.corr_chart, self.rc_chart):
            lay.addWidget(x)
        return w

    def _build_explainability(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.exp_text = QTextBrowser()
        self.exp_text.setMinimumHeight(520)
        self.shap_chart = ChartView()
        self.fi_chart = ChartView()
        self.fsel_tbl = DataTable()
        self.fsel_tbl.setMinimumHeight(300)
        self.lineage_tbl = DataTable()
        self.lineage_tbl.setMinimumHeight(360)
        self.analog_tbl = DataTable()
        self.analog_tbl.setMinimumHeight(260)
        self.analog_chart = ChartView()
        self.istab_tbl = DataTable()
        self.istab_tbl.setMinimumHeight(260)
        lay.addWidget(self.exp_text)
        lay.addWidget(self.shap_chart)
        lay.addWidget(self.fi_chart)
        lay.addWidget(QLabel("Feature selection stability across folds"))
        lay.addWidget(self.fsel_tbl)
        lay.addWidget(QLabel("Permutation-importance stability across walk-forward folds (Deep mode)"))
        lay.addWidget(self.istab_tbl)
        lay.addWidget(QLabel("Historical analogues"))
        lay.addWidget(self.analog_chart)
        lay.addWidget(self.analog_tbl)
        lay.addWidget(QLabel("Data lineage (every model input: source, transformation, original & normalised value)"))
        lay.addWidget(self.lineage_tbl)
        return w

    def _build_prediction_journal(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.pj_chart = ChartView()
        self.pj_tbl = DataTable()
        self.pj_tbl.setMinimumHeight(360)
        self.sc_tbl = DataTable()
        self.sc_tbl.setMinimumHeight(260)
        lay.addWidget(self.pj_chart)
        lay.addWidget(QLabel("Every stored prediction (resolved automatically once the horizon has passed)"))
        lay.addWidget(self.pj_tbl)
        lay.addWidget(QLabel("Live scorecard (by horizon and regime)"))
        lay.addWidget(self.sc_tbl)
        return w

    def _build_data_quality(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.dq_text = QTextBrowser()
        self.dq_text.setMinimumHeight(300)
        self.dq_tbl = DataTable()
        self.dq_tbl.setMinimumHeight(360)
        self.dq_val = DataTable()
        self.dq_val.setMinimumHeight(260)
        self.dq_steps = DataTable()
        self.dq_steps.setMinimumHeight(420)
        lay.addWidget(self.dq_text)
        lay.addWidget(QLabel("Source health (fallback chain attempts)"))
        lay.addWidget(self.dq_tbl)
        lay.addWidget(QLabel("Validation issues"))
        lay.addWidget(self.dq_val)
        lay.addWidget(QLabel("Pipeline steps (last run)"))
        lay.addWidget(self.dq_steps)
        return w

    def _build_settings(self):
        return SettingsTab(self.s, on_saved=self._settings_saved)

    def _settings_saved(self):
        self.backend = Backend(self.s)
        self._update_schedule_label()

    # ================================================================== population
    def populate(self, res):
        st, a = res.state, res.artifacts
        d, fc, rg = st["decision"], st.get("forecast_primary", {}), st.get("regime", {})
        cur = "€" if st.get("currency") == "EUR" else ""
        sig_col = {"BUY": "#2ecc71", "HOLD": "#f1c40f", "SELL": "#e74c3c"}[d["signal"]]
        if st.get("is_synthetic"):
            self.banner.setText("⚠ SYNTHETIC DEMO DATA - NOT REAL MARKET DATA. Nothing shown describes the real market.")
            self.banner.setStyleSheet("background:#7f1d1d;color:white")
        elif not st.get("data_current"):
            self.banner.setText("🟡 DATA NOT CURRENT - " + " | ".join(st.get("warnings", []))[:400])
            self.banner.setStyleSheet("background:#5c4a00;color:white")
        else:
            self.banner.setText(f"Data current as of session {st['as_of']} (daily close data; not real-time).")
            self.banner.setStyleSheet("background:#0f3d1f;color:white")
        dis = fc.get("disagreement", {}) or {}
        self.k1["Price"].set(f"{cur}{st['price']:.2f}", f"{st.get('symbol')} close {st['as_of']}")
        self.k1["Signal"].set(f"{d['label']}", f"previous: {d.get('previous_signal')} | {d.get('duration_days')} session(s)", sig_col)
        self.k1["Opportunity"].set(f"{d['opportunity']:.0f} / 100", d["quadrant"])
        self.k1["Risk"].set(f"{d['risk']:.0f} / 100", "", "#e74c3c" if d["risk"] >= 60 else None)
        self.k1["Confidence"].set(f"{100 * d['confidence']:.0f}%", "penalties: " + ", ".join(k for k, v in d["penalties"].items() if v))
        self.k1["Model agreement"].set(_pct(dis.get("sign_agreement"), 0, False), f"{dis.get('n_models')} models")
        lo, hi = (fc.get("interval") or [np.nan, np.nan])
        self.k2["20D expected return"].set(_pct(fc.get("expected_return")), f"ensemble: {fc.get('ensemble_method')}")
        self.k2["80% range"].set(f"{_pct(lo)} → {_pct(hi)}", f"{cur}{st['price'] * (1 + (lo or 0)):.2f} → {cur}{st['price'] * (1 + (hi or 0)):.2f}")
        self.k2["P(up)"].set(_pct(fc.get("p_up"), 0, False), f"raw {_pct(fc.get('p_up_raw'), 0, False)}")
        self.k2["P(down)"].set(_pct(fc.get("p_down"), 0, False), f"P(<-5%) {_pct(fc.get('p_lt5'), 0, False)}")
        self.k2["Regime"].set(f"{rg.get('regime')}", f"{rg.get('volatility_label')} VOLATILITY")
        self.k2["Data quality"].set(f"{st.get('data_quality_score', 0):.0f}/100", st.get("data_status", ""))
        tgt = a.bundle.target
        self.dash_chart.set_figure(CH.candlestick(tgt, f"VUSA ({st.get('symbol')})", 2, fc, st.get("support_resistance")))
        cons = pd.DataFrame([{"model": m, "expected_return": p["ret"], "p_up": p["p_up"],
                              "weight": (fc.get("ensemble_weights") or {}).get(m)} for m, p in (fc.get("model_preds") or {}).items()])
        if not cons.empty:
            cons = pd.concat([cons.sort_values("expected_return", ascending=False),
                              pd.DataFrame([{"model": f"ENSEMBLE ({fc.get('ensemble_method')})", "expected_return": fc.get("expected_return"),
                                             "p_up": fc.get("p_up"), "weight": None}])])
        self.consensus_tbl.set_df(cons)
        self.components_chart.set_figure(CH.components_chart(d["components"], d["contributions"]))
        self.decision_text.setHtml(self._decision_html(st))
        self.alerts_tbl.set_df(pd.DataFrame(st.get("alerts", []))[["created_at", "kind", "severity", "message"]]
                               if st.get("alerts") else pd.DataFrame())
        # forecasts
        self.fc_chart.set_figure(CH.forecast_bars(st["forecasts"]))
        self.fc_tbl.set_df(pd.DataFrame([{"horizon": f"{h}D", "expected": f.get("expected_return"), "P(up)": f.get("p_up"),
                                          "lower80": f.get("interval", [None, None])[0], "upper80": f.get("interval", [None, None])[1],
                                          "P(>+5%)": f.get("p_gt5"), "P(<-5%)": f.get("p_lt5"), "P(maxDD>10%)": f.get("p_dd10"),
                                          "confidence": f.get("confidence"), "ensemble": f.get("ensemble_method"),
                                          "n_oos": f.get("n_oos"), "warnings": "; ".join(f.get("warnings", []))}
                                         for h, f in sorted(st["forecasts"].items(), key=lambda kv: int(kv[0]))]))
        prim = a.forecast.horizons.get(self.s.models.primary_horizon) if a.forecast else None
        if prim is not None and prim.oos is not None:
            o = prim.oos.dropna(subset=["ens_ret", "y_ret"])
            samples = prim.expected_return + (o["y_ret"] - o["ens_ret"]).values[-750:]
            self.dist_chart.set_figure(CH.distribution(prim.quantiles, samples, f"{prim.horizon}D predictive distribution "
                                                                                 "(ensemble + conformal residuals)"))
        self.cf_tbl.set_df(a.counterfactuals if a.counterfactuals is not None else pd.DataFrame())
        # signals
        hist = self.backend.signal_history()
        sig_df = hist.set_index(pd.to_datetime(hist["as_of"]))[["signal", "final_score"]].rename(columns={"final_score": "score"}) \
            if not hist.empty else pd.DataFrame(columns=["signal", "score"])
        self.sig_chart.set_figure(CH.signal_history(tgt["close"], sig_df, "Stored daily signals"))
        self.sig_tbl.set_df(hist.sort_values("as_of", ascending=False) if not hist.empty else hist)
        # technical
        f = a.features.frame
        self.tech_chart.set_figure(CH.candlestick(tgt, "Price, moving averages, support/resistance", 3, None, st.get("support_resistance")))
        self.tech_osc.set_figure(CH.line({"RSI(14)": f.get("rsi14"), "Stochastic %K": f.get("stoch_k"),
                                          "Williams %R + 100": f.get("williams_r") + 100 if "williams_r" in f else None},
                                         "Momentum oscillators", hlines=[30, 70]))
        self.tech_vol.set_figure(CH.line({k: f.get(k) for k in ("rv_20", "parkinson_20", "garman_klass_20", "rogers_satchell_20",
                                                                "vix_level") if k in f} | ({"VIX/100": f["vix_level"] / 100} if "vix_level" in f else {}),
                                         "Volatility estimators (annualised)"))
        self.tech_breadth.set_figure(CH.line({k: f[k] for k in ("sector_pct_above_sma50_proxy", "sector_pct_above_sma200_proxy",
                                                                "cyclical_vs_defensive_60d") if k in f}, "Breadth (sector-ETF proxy)"))
        tech_now = pd.DataFrame([{"feature": k, "value": v} for k, v in st["features_now"].items()
                                 if k in a.features.groups.get("technical", []) + a.features.groups.get("volatility", [])])
        self.tech_tbl.set_df(tech_now)
        # macro
        self.macro_rates.set_figure(CH.line({k: f[k] for k in ("ust10y", "ust2y", "fed_funds", "real_yield_10y", "curve_10y3m",
                                                               "curve_10y2y", "us10y_level") if k in f}, "Rates & yield curve (point-in-time)"))
        self.macro_infl.set_figure(CH.line({k: f[k] for k in ("cpi_yoy", "core_cpi_yoy", "core_pce_yoy", "payrolls_yoy", "indpro_yoy")
                                            if k in f}, "Inflation & growth (vintage known at each date)", yfmt=".1%"))
        self.macro_credit.set_figure(CH.line({k: f[k] for k in ("hy_oas", "ig_oas", "nfci", "unrate", "sahm_rt") if k in f},
                                             "Credit, financial conditions, labour"))
        self.lead_tbl.set_df(a.leading)
        self.macro_tbl.set_df(pd.DataFrame([{"series": k, "status": v} for k, v in st.get("macro_status", {}).items()]))
        # news
        n = st.get("news", {})
        if n.get("available"):
            s_ = n.get("sentiment", {})
            self.news_text.setHtml("<h3>Market intelligence report</h3><ul>" + "".join(f"<li>{x}</li>" for x in n.get("steps", [])) +
                                   f"</ul><p>Weighted sentiment <b>{s_.get('score')}</b> (dispersion {s_.get('dispersion')}, "
                                   f"n={s_.get('n')}, model {s_.get('model')}). Labels: {s_.get('label_counts')}</p>"
                                   f"<p>Source tiers: {n.get('source_tiers')}</p><p>Contradictions: {n.get('contradictions')}</p>")
        else:
            self.news_text.setHtml(f"<h3>News unavailable</h3><p>{n.get('reason')}</p><p>No headlines are invented. "
                                   "Configure RSS feeds / NewsAPI in Settings.</p>")
        sh = self.backend.sentiment_history()
        if sh.empty:
            self.sent_chart.set_figure(CH.empty("News sentiment history", "No stored news sentiment yet - it accumulates "
                                                "from daily runs (no historical archive is fabricated)"))
        else:
            self.sent_chart.set_figure(CH.line({"weighted news sentiment": sh.set_index(pd.to_datetime(sh["date"]))["score"]},
                                               "News sentiment history (application's own news memory)", hlines=[0]))
        self.news_events.set_df(pd.DataFrame(n.get("events", [])))
        self.news_articles.set_df(pd.DataFrame(n.get("top_articles", [])))
        self.event_tbl.set_df(a.event_outcomes)
        # regime
        cp = st.get("structural_change", {})
        self.reg_text.setHtml(f"<h3>{rg.get('regime')} / {rg.get('volatility_label')} volatility</h3>"
                              f"<p>Days in regime: {rg.get('days_in_regime')} (previous: {rg.get('previous_regime')}); HMM stress "
                              f"probability {_pct(rg.get('hmm_stress_probability'), 0, False)}; realised vol 20d "
                              f"{_pct(rg.get('realized_vol_20d'), 1, False)} (percentile {_pct(rg.get('vol_percentile'), 0, False)}); "
                              f"drawdown {_pct(rg.get('drawdown'))}.</p><p>1-year regime distribution: {rg.get('distribution_1y')}</p>"
                              f"<h4>Change-point / structural-break tests</h4><pre>{json.dumps(cp, indent=1, default=str)}</pre>"
                              f"<h4>Shocks</h4><p>{'; '.join(st.get('shocks', {}).get('flags', [])) or 'none'}</p>")
        self.reg_chart.set_figure(CH.regime_chart(tgt["close"], a.regimes))
        bo = a.change_series.get("bocpd")
        if bo is not None:
            self.cp_chart.set_figure(CH.line({"BOCPD P(change point in last 5 sessions)": bo["bocpd_recent_cp_prob"],
                                              "KS p-value (recent 6m vs history)": a.change_series["ks"]["ks_p"]},
                                             "Change-point detection"))
        # ML
        if fc:
            sc = pd.DataFrame(fc.get("model_scores", {})).T.reset_index().rename(columns={"index": "model"})
            dm = fc.get("dm_tests", {})
            if not sc.empty:
                sc["DM_p_vs_naive"] = sc["model"].map(lambda m: (dm.get(m) or {}).get("p_value"))
                sc["model_id"] = sc["model"].map(lambda m: (fc.get("model_ids") or {}).get(m))
            self.ml_scores.set_df(sc)
            self.ml_ens.set_df(pd.DataFrame(fc.get("ensemble_comparison", {})).T.reset_index().rename(columns={"index": "method"}))
            self.ml_cal.set_figure(CH.calibration_curve(fc.get("calibration_curve", [])))
            if prim is not None:
                self.ml_oos.set_figure(CH.oos_performance(prim.oos, "Rolling out-of-sample accuracy and interval coverage"))
            sa = st.get("self_audit", {})
            errs = {**fc.get("model_errors", {})}
            self.ml_text.setHtml("<h3>Self-audit</h3>" + "".join(f"<h4>{k}</h4><ul>" + "".join(f"<li>{x}</li>" for x in v) + "</ul>"
                                                               for k, v in sa.items()) +
                                 f"<h4>Calibration</h4><pre>{json.dumps(fc.get('calibration'), indent=1, default=str)}</pre>"
                                 f"<h4>Conformal coverage (OOS)</h4><pre>{json.dumps(fc.get('conformal'), indent=1, default=str)}</pre>"
                                 f"<h4>Uncertainty decomposition</h4><pre>{json.dumps(fc.get('uncertainty'), indent=1, default=str)}</pre>"
                                 f"<h4>Training window</h4><pre>{json.dumps(st.get('training_window'), indent=1, default=str)}</pre>"
                                 f"<h4>Walk-forward folds</h4><ul>" + "".join(f"<li>{x}</li>" for x in fc.get("folds", [])) + "</ul>"
                                 f"<h4>Leakage audit</h4><pre>{json.dumps(st.get('leakage_audit'), indent=1)}</pre>"
                                 f"<h4>Model drift</h4><pre>{json.dumps(st.get('model_drift'), indent=1, default=str)}</pre>"
                                 f"<h4>Feature drift</h4><pre>{json.dumps({k: v for k, v in st.get('feature_drift', {}).items()}, indent=1, default=str)}</pre>"
                                 f"<h4>Champion / challenger</h4><pre>{json.dumps(st.get('champion_challenger'), indent=1, default=str)}</pre>"
                                 f"<h4>Retirement flags</h4><pre>{json.dumps(st.get('retirement_flags'), indent=1, default=str)}</pre>"
                                 f"<h4>Skipped / failed models</h4><pre>{json.dumps(errs, indent=1)}</pre>")
        self.models_tbl.set_df(self.backend.model_versions())
        # Monte Carlo
        if a.monte_carlo:
            self.mc_fan.set_figure(CH.fan_chart(a.monte_carlo["Block bootstrap"], st["price"],
                                                f"Monte Carlo fan chart - block bootstrap ({a.monte_carlo['Block bootstrap'].n_paths:,} paths)"))
            self.mc_dist.set_figure(CH.mc_terminal(a.monte_carlo))
            self.mc_tbl.set_df(pd.DataFrame(st["monte_carlo"]).T.reset_index().rename(columns={"index": "method"}))
        # risk
        rm = st.get("risk_metrics", {})
        self.risk_text.setHtml("<h3>Risk metrics</h3><table>" + "".join(f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in rm.items()
                                                                       if isinstance(v, float)) +
                               f"</table><h4>Risk score components</h4><pre>{json.dumps(st.get('risk_components'), indent=1)}</pre>"
                               f"<h4>Correlation breakdowns</h4><pre>{json.dumps(st.get('correlation_breakdowns'), indent=1)}</pre>"
                               f"<h4>Portfolio</h4><pre>{json.dumps(st.get('portfolio', 'not configured'), indent=1, default=str)}</pre>"
                               f"<h4>Position sizing research (Deep mode)</h4><pre>{json.dumps(st.get('position_sizing', 'run Deep Research'), indent=1, default=str)}</pre>")
        self.stress_tbl.set_df(a.stress)
        self.dd_chart.set_figure(CH.drawdown(tgt["close"]))
        if a.corr_matrix is not None:
            self.corr_chart.set_figure(CH.heatmap(a.corr_matrix.round(2), "Correlation matrix (last 252 sessions, daily log returns)"))
        if a.rolling_corr is not None:
            self.rc_chart.set_figure(CH.line({k: a.rolling_corr[k] for k in ("NDX", "RUT", "TLT", "GOLD", "VIX", "DXY", "OIL")
                                              if k in a.rolling_corr}, "Rolling 60d correlation with VUSA"))
        # explainability
        self.exp_text.setPlainText(st.get("explanation", "") + "\n\n" + json.dumps(st.get("explanation_meta"), default=str) +
                                   "\n\nCOMMITTEE\n" + st.get("committee", {}).get("summary", ""))
        if a.shap_now is not None:
            self.shap_chart.set_figure(CH.bar(a.shap_now.head(20), f"SHAP contributions today ({st.get('shap_today', {}).get('model')})"))
        else:
            self.shap_chart.set_figure(CH.empty("SHAP", "SHAP unavailable (install 'shap' or no tree model in this mode)"))
        if prim is not None and prim.feature_scores is not None:
            self.fi_chart.set_figure(CH.bar(prim.feature_scores["rank_score"].head(25), "Feature relevance (MI + tree + stability, rank-avg)",
                                            color_sign=False))
        self.fsel_tbl.set_df(pd.DataFrame(st.get("feature_selection_stability", [])))
        self.analog_tbl.set_df(a.analogues)
        self.analog_chart.set_figure(CH.analogues_chart(a.analogues))
        self.istab_tbl.set_df(pd.DataFrame(st.get("importance_stability", [])) if st.get("importance_stability")
                              else pd.DataFrame([{"note": "Run Deep Research to compute importance stability"}]))
        self.lineage_tbl.set_df(pd.DataFrame(st.get("lineage", [])))
        # journal
        j = self.backend.journal()
        self.pj_chart.set_figure(CH.prediction_errors(j))
        self.pj_tbl.set_df(j)
        self.sc_tbl.set_df(self.backend.scorecard())
        # data quality
        self.dq_text.setHtml(f"<h3>{st.get('data_status')}</h3><p>Market: {st.get('market_status')}</p>"
                             f"<p>{st.get('proxy_note') or ''}</p><p>{st.get('pit_quality') or ''}</p><ul>" +
                             "".join(f"<li>{w}</li>" for w in st.get("warnings", [])) + "</ul>")
        self.dq_tbl.set_df(pd.DataFrame(st.get("source_health", [])))
        self.dq_val.set_df(pd.DataFrame([{**i, "dataset": v["dataset"]} for v in st.get("validation", []) for i in v["issues"]]))
        self.dq_steps.set_df(pd.DataFrame(st.get("steps", [])))
        self._lab_cache = {}
        self._refresh_questions()

    def _decision_html(self, st: dict) -> str:
        d, com = st["decision"], st.get("committee", {})
        fc = st.get("forecast_primary", {})
        cur = "€" if st.get("currency") == "EUR" else ""
        rows = "".join(f"<tr><td>{h}D</td><td>{_pct(f.get('expected_return'))}</td><td>{_pct(f.get('p_up'), 0, False)}</td></tr>"
                       for h, f in sorted(st["forecasts"].items(), key=lambda kv: int(kv[0])))
        lo, hi = fc.get("interval", [0, 0])
        lst = lambda xs: "<ul>" + "".join(f"<li>{x}</li>" for x in xs) + "</ul>"  # noqa: E731
        why = "".join(f"<tr><td>{k}</td><td>{v:+.1f}</td></tr>" for k, v in sorted(d["contributions"].items(), key=lambda kv: -kv[1]))
        return (f"<h2>VUSA AI RESEARCH DECISION</h2><table cellpadding=4>"
                f"<tr><td>SIGNAL</td><td><b>{d['label']}</b></td></tr><tr><td>OPPORTUNITY</td><td>{d['opportunity']:.0f}/100</td></tr>"
                f"<tr><td>RISK</td><td>{d['risk']:.0f}/100</td></tr><tr><td>CONFIDENCE</td><td>{100 * d['confidence']:.0f}%</td></tr>"
                f"<tr><td>MODEL AGREEMENT</td><td>{_pct((fc.get('disagreement') or {}).get('sign_agreement'), 0, False)}</td></tr>"
                f"<tr><td>CURRENT PRICE</td><td>{cur}{st['price']:.2f}</td></tr></table>"
                f"<table cellpadding=4><tr><th>Horizon</th><th>Forecast</th><th>P(up)</th></tr>{rows}</table>"
                f"<p>20D P(UP) {_pct(fc.get('p_up'), 0, False)} | P(DOWN) {_pct(fc.get('p_down'), 0, False)} | 20D RANGE "
                f"{cur}{st['price'] * (1 + (lo or 0)):.2f} → {cur}{st['price'] * (1 + (hi or 0)):.2f}</p>"
                f"<p>MARKET REGIME: {st['regime']['regime']} | DATA QUALITY: {st.get('data_status')} | MODEL STATUS: "
                f"{'⚠ DRIFT' if st.get('model_drift', {}).get('alert') else 'no drift alarm'}"
                f"{' | ⚠ OUT-OF-DISTRIBUTION' if st.get('feature_drift', {}).get('alert') else ''}</p>"
                + (f"<p><b>Signal changed:</b> {d['change_reason']}</p>" if d.get("changed") else "") +
                "<h3>WHY THE SIGNAL EXISTS (contribution in points vs neutral)</h3><table>" + why + "</table>"
                "<h3>BULL CASE</h3>" + lst(com.get("bull_case", [])) + "<h3>BEAR CASE</h3>" + lst(com.get("bear_case", [])) +
                "<h3>NEUTRAL CASE</h3>" + lst(com.get("neutral_case", [])) + "<h3>KEY RISKS</h3>" + lst(com.get("key_risks", [])) +
                ("<h3>⚠ CONTRADICTIONS</h3>" + lst(d.get("contradictions", [])) if d.get("contradictions") else "") +
                "<h3>SKEPTIC</h3>" + lst((com.get("skeptic") or {}).get("process_risks", [])) +
                "<h3>WHAT WOULD CHANGE THE SIGNAL?</h3>" + lst(d.get("what_would_change", [])) +
                "<h3>HISTORICAL EVIDENCE</h3>" + lst([f"Analogues 20D: {json.dumps((st.get('analogues') or {}).get('summary', {}).get('ret_20d'), default=str)}"] +
                                                    [f"{m['news_category']} → {m['historical_category']}: 20D mean {m.get('20D_mean')}" for m in st.get("news_memory", [])[:4]]) +
                "<p><i>Decision support only. Probabilistic research output, not investment advice. No trades are placed.</i></p>")

    # ================================================================== signals tab
    def why(self):
        d = self.why_date.text().strip()
        tr = self.backend.why(d)
        if not tr:
            self.why_text.setPlainText(f"No stored decision trace for {d}.")
            return
        dd = tr.get("decision", {})
        self.why_text.setHtml(f"<h3>{dd.get('label')} on {tr.get('as_of')} (run {tr.get('run_id')})</h3>"
                              f"<p>Opportunity {dd.get('opportunity')}, risk {dd.get('risk')}, final score {dd.get('final_score')}, "
                              f"confidence {dd.get('confidence')}</p><h4>Components</h4><pre>{json.dumps(dd.get('components'), indent=1)[:6000]}</pre>"
                              f"<h4>Forecast</h4><pre>{json.dumps({k: v for k, v in tr.get('forecast_primary', {}).items() if k in ('expected_return', 'p_up', 'interval', 'ensemble_method', 'model_preds', 'model_ids')}, indent=1)}</pre>"
                              f"<h4>Committee</h4><pre>{json.dumps(tr.get('committee', {}), indent=1)[:6000]}</pre>"
                              f"<h4>Model versions</h4><pre>{json.dumps(tr.get('model_versions'), indent=1)}</pre>"
                              f"<h4>Data snapshot</h4><pre>{json.dumps(tr.get('data_snapshot'), indent=1)}</pre>"
                              f"<h4>Environment</h4><pre>{json.dumps(tr.get('environment'), indent=1)}</pre>")

    # ================================================================== model lab
    def _lab_run(self, fn, *args, done=None):
        if not self.result:
            QMessageBox.information(self, "Research lab", "Run an analysis first.")
            return
        self._busy(True)

        def ok(r):
            self._busy(False)
            (done or self._lab_show)(r)

        th, wk = start_worker(self, lambda *a, progress=None: fn(*a), ok, self._fail, self._on_progress, *args)
        self._threads.append((th, wk))

    def _lab_show(self, r):
        if isinstance(r, pd.DataFrame):
            self.lab_tbl.set_df(r)
        elif isinstance(r, dict) and "results" in r:
            self.lab_tbl.set_df(pd.DataFrame(r["results"]).T.reset_index().rename(columns={"index": "setup"}))
        elif isinstance(r, dict):
            first = next(iter(r.values()), None)
            self.lab_tbl.set_df(pd.DataFrame(r).T.reset_index() if isinstance(first, dict) else pd.DataFrame([r]))

    def lab_custom(self):
        groups = [g for g, cb in self.lab_groups.items() if cb.isChecked()]
        self._lab_run(self.backend.custom_experiment, groups, self.s.models.primary_horizon, self.lab_window.currentText(),
                      self.lab_years.value() or None, self.lab_regime.isChecked())

    def lab_ablation(self):
        def done(r):
            self._lab_cache["ablation"] = r
            self._lab_show(r)
            self._refresh_questions()
        self._lab_run(self.backend.ablation, self.s.models.primary_horizon, done=done)

    def lab_regimes(self):
        def done(r):
            self._lab_cache["regimes"] = r
            self._lab_show(r)
            self._refresh_questions()
        self._lab_run(self.backend.regime_options, self.s.models.primary_horizon, done=done)

    def lab_window_study(self):
        def done(r):
            self._lab_cache["window"] = r
            self._lab_show(r)
            self._refresh_questions()
        self._lab_run(self.backend.window_study, self.s.models.primary_horizon, done=done)

    def _refresh_questions(self):
        from ..pipeline.research_lab import research_questions

        fc = self.result.artifacts.forecast if self.result else None
        q = research_questions(fc, self._lab_cache.get("ablation"), self._lab_cache.get("regimes"),
                               self._lab_cache.get("window"), self._lab_cache.get("backtest"))
        self.lab_q.set_df(pd.DataFrame(q))

    # ================================================================== backtest
    def run_backtest(self):
        if not self.result:
            QMessageBox.information(self, "Backtest", "Run an analysis first.")
            return
        self._busy(True)
        th, wk = start_worker(self, self.backend.backtest, self._bt_done, self._bt_fail, self._on_progress,
                              int(self.bt_h.currentText()), None, retrain_every=self.bt_retrain.value())
        self._threads.append((th, wk))

    def _bt_fail(self, msg):
        self._busy(False)
        self.bt_text.setPlainText("BACKTEST STOPPED\n\n" + msg)

    def _bt_done(self, bt):
        self._busy(False)
        self._lab_cache["backtest"] = bt
        self.bt_chart.set_figure(CH.equity(bt.result))
        m, b = bt.result.metrics, bt.result.benchmark_metrics
        tab = pd.DataFrame({"strategy": m, "buy_and_hold": b})
        self.bt_text.setHtml(f"<h3>{bt.audit.summary().splitlines()[0]}</h3><pre>{bt.audit.summary()}</pre>"
                             f"{tab.to_html(float_format=lambda x: f'{x:.4f}')}"
                             f"<h4>Robustness</h4><pre>{json.dumps(bt.robustness, indent=1, default=str)}</pre>"
                             f"<h4>Notes</h4><ul>" + "".join(f"<li>{x}</li>" for x in bt.notes + bt.result.notes) + "</ul>")
        self.bt_trades.set_df(bt.result.trades)
        self._refresh_questions()

    def closeEvent(self, ev):
        for th, _ in self._threads:
            if th.isRunning():
                th.quit()
                th.wait(2000)
        super().closeEvent(ev)
