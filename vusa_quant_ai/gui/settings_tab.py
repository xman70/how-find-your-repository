"""Settings tab: every configurable parameter in one place. API keys go to the local
.env file only (never to the settings JSON or the database)."""
from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget)

from ..config.settings import LISTINGS, Settings, normalize_ticker, save_api_keys, save_settings


def _dspin(v, lo, hi, step=0.01, dec=4):
    w = QDoubleSpinBox()
    w.setDecimals(dec)
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setValue(float(v))
    return w


def _spin(v, lo, hi):
    w = QSpinBox()
    w.setRange(lo, hi)
    w.setValue(int(v))
    return w


class SettingsTab(QScrollArea):
    def __init__(self, s: Settings, on_saved=None):
        super().__init__()
        self.s, self.on_saved = s, on_saved
        self.setWidgetResizable(True)
        inner = QWidget()
        self.setWidget(inner)
        lay = QVBoxLayout(inner)

        g = QGroupBox("Instrument")
        f = QFormLayout(g)
        self.ticker = QLineEdit(s.ticker)
        self.listing = QComboBox()
        self.listing.addItems(list(LISTINGS))
        self.listing.setCurrentText(s.listing)
        self.currency = QLineEdit(s.currency)
        f.addRow("Ticker (VUSAA is accepted as alias)", self.ticker)
        f.addRow("Exchange / listing", self.listing)
        f.addRow("Currency", self.currency)
        lay.addWidget(g)

        g = QGroupBox("Data providers (fallback order) & API keys")
        f = QFormLayout(g)
        self.market_prio = QLineEdit(",".join(s.providers.market_priority))
        self.macro_prio = QLineEdit(",".join(s.providers.macro_priority))
        self.news_prio = QLineEdit(",".join(s.providers.news_priority))
        self.timeout = _dspin(s.providers.request_timeout, 1, 120, 1, 0)
        self.allow_demo = QCheckBox("Allow synthetic demo data (always labelled)")
        self.allow_demo.setChecked(s.providers.allow_synthetic_demo)
        self.keys = {}
        for k, lbl in (("fred", "FRED API key (enables true ALFRED vintages)"), ("alphavantage", "Alpha Vantage key"),
                       ("twelvedata", "Twelve Data key"), ("newsapi", "NewsAPI key"),
                       ("anthropic", "Anthropic API key (optional LLM explanation)")):
            e = QLineEdit(getattr(s.api_keys, k))
            e.setEchoMode(QLineEdit.Password)
            self.keys[k] = e
            f.addRow(lbl, e)
        f.addRow("Market providers", self.market_prio)
        f.addRow("Macro providers", self.macro_prio)
        f.addRow("News providers", self.news_prio)
        f.addRow("Request timeout (s)", self.timeout)
        f.addRow(self.allow_demo)
        lay.addWidget(g)

        g = QGroupBox("News sources (RSS / official feeds, one per line)")
        v = QVBoxLayout(g)
        self.feeds = QPlainTextEdit("\n".join(s.news.rss_feeds))
        self.news_lookback = _spin(s.news.lookback_days, 1, 30)
        v.addWidget(self.feeds)
        h = QHBoxLayout()
        h.addWidget(QLabel("Lookback days"))
        h.addWidget(self.news_lookback)
        v.addLayout(h)
        lay.addWidget(g)

        g = QGroupBox("Models & forecasting")
        f = QFormLayout(g)
        self.horizons = QLineEdit(",".join(map(str, s.models.horizons)))
        self.primary = _spin(s.models.primary_horizon, 1, 252)
        self.models = QLineEdit(",".join(s.models.enabled_models))
        self.window = QComboBox()
        self.window.addItems(["full", "recent", "exp_weighted", "auto"])
        self.window.setCurrentText(s.models.training_window)
        self.recent_years = _dspin(s.models.recent_window_years, 1, 40, 1, 1)
        self.hist_years = _spin(s.models.history_years, 5, 40)
        self.n_splits = _spin(s.models.n_splits, 2, 12)
        self.embargo = _spin(s.models.embargo_days, 0, 60)
        self.alpha = _dspin(s.models.conformal_alpha, 0.01, 0.5, 0.05, 2)
        self.maxf = _spin(s.models.max_features, 5, 200)
        self.seed = _spin(s.models.random_seed, 0, 10 ** 6)
        self.dl_epochs = _spin(s.models.deep_learning_epochs, 1, 500)
        self.optuna = _spin(s.models.optuna_trials, 0, 500)
        self.mc_paths = QComboBox()
        self.mc_paths.addItems(["10000", "50000", "100000"])
        self.mc_paths.setCurrentText(str(s.monte_carlo_paths))
        for lbl, w in (("Horizons (trading days)", self.horizons), ("Primary horizon", self.primary),
                       ("Enabled models (Deep mode)", self.models), ("Training window", self.window),
                       ("Recent window (years)", self.recent_years), ("History to download (years)", self.hist_years),
                       ("Walk-forward folds", self.n_splits), ("Embargo (sessions)", self.embargo),
                       ("Conformal alpha (0.2 = 80% interval)", self.alpha), ("Max selected features", self.maxf),
                       ("Random seed", self.seed), ("Deep-learning epochs", self.dl_epochs),
                       ("Optuna trials (Deep mode HPO; 0 = off)", self.optuna), ("Monte Carlo paths", self.mc_paths)):
            f.addRow(lbl, w)
        lay.addWidget(g)

        g = QGroupBox("Signal thresholds & alerts")
        f = QFormLayout(g)
        self.buy = _dspin(s.signals.buy_threshold, 50, 100, 1, 1)
        self.sell = _dspin(s.signals.sell_threshold, 0, 50, 1, 1)
        self.hyst = _dspin(s.signals.hysteresis, 0, 20, 0.5, 1)
        self.min_days = _spin(s.signals.min_signal_days, 1, 30)
        self.max_risk = _dspin(s.signals.max_risk_for_buy, 0, 100, 1, 0)
        self.n_conf = _spin(s.signals.strong_confirmations_required, 1, 4)
        self.alert_prob = _dspin(s.alerts.probability_change, 0.01, 0.5, 0.01, 2)
        self.alert_dd = _dspin(s.alerts.drawdown_pct, 0.01, 0.6, 0.01, 2)
        for lbl, w in (("BUY threshold", self.buy), ("SELL threshold", self.sell), ("Hysteresis (pts)", self.hyst),
                       ("Minimum signal duration (sessions)", self.min_days), ("Max risk score for BUY", self.max_risk),
                       ("Confirmations for STRONG signal (of ML/trend/risk-reward/regime)", self.n_conf),
                       ("Alert: probability change", self.alert_prob), ("Alert: drawdown", self.alert_dd)):
            f.addRow(lbl, w)
        lay.addWidget(g)

        g = QGroupBox("Transaction costs & taxes (explicit assumptions - verify with a tax adviser)")
        f = QFormLayout(g)
        c = s.costs
        self.comm = _dspin(c.commission_pct, 0, 0.05, 0.0001, 5)
        self.comm_min = _dspin(c.commission_min, 0, 100, 0.5, 2)
        self.spread = _dspin(c.spread_bps, 0, 200, 0.5, 1)
        self.slip = _dspin(c.slippage_bps, 0, 200, 0.5, 1)
        self.delay = _spin(c.execution_delay_days, 1, 5)
        self.exec_price = QComboBox()
        self.exec_price.addItems(["open", "close"])
        self.exec_price.setCurrentText(c.execution_price)
        self.tax_cg = _dspin(c.tax_on_realized_gains_pct, 0, 1, 0.01, 3)
        self.tax_div = _dspin(c.tax_on_dividends_pct, 0, 1, 0.01, 3)
        for lbl, w in (("Commission (fraction)", self.comm), ("Minimum commission", self.comm_min), ("Bid/ask spread (bps)", self.spread),
                       ("Slippage (bps)", self.slip), ("Execution delay (sessions)", self.delay),
                       ("Execution price (close only if an executable close is modelled)", self.exec_price),
                       ("Tax on realised gains (assumption)", self.tax_cg), ("Tax on dividends (assumption)", self.tax_div)):
            f.addRow(lbl, w)
        f.addRow(QLabel(c.tax_note))
        lay.addWidget(g)

        g = QGroupBox("Portfolio (optional, research only - the app never trades)")
        f = QFormLayout(g)
        p = s.portfolio
        self.pf_on = QCheckBox("Enable portfolio-aware analysis")
        self.pf_on.setChecked(p.enabled)
        self.pf_units = _dspin(p.holdings_units, 0, 1e9, 1, 4)
        self.pf_avg = _dspin(p.avg_purchase_price, 0, 1e6, 0.1, 4)
        self.pf_cash = _dspin(p.cash_available, 0, 1e10, 100, 2)
        self.pf_inv = _dspin(p.investment_amount, 0, 1e10, 100, 2)
        self.pf_risk = QComboBox()
        self.pf_risk.addItems(["conservative", "moderate", "aggressive"])
        self.pf_risk.setCurrentText(p.risk_tolerance)
        f.addRow(self.pf_on)
        for lbl, w in (("VUSA units held", self.pf_units), ("Average purchase price", self.pf_avg), ("Cash available", self.pf_cash),
                       ("Investment amount", self.pf_inv), ("Risk tolerance", self.pf_risk)):
            f.addRow(lbl, w)
        lay.addWidget(g)

        g = QGroupBox("Valuation inputs (S&P 500; leave 0 if unknown - never invented)")
        f = QFormLayout(g)
        va = s.valuation
        self.v_fpe = _dspin(va.forward_pe, 0, 100, 0.1, 2)
        self.v_tpe = _dspin(va.trailing_pe, 0, 100, 0.1, 2)
        self.v_cape = _dspin(va.cape, 0, 100, 0.1, 2)
        self.v_dy = _dspin(va.dividend_yield_pct, 0, 20, 0.01, 2)
        self.v_rev = _dspin(va.eps_revisions_3m_pct, -50, 50, 0.1, 2)
        self.v_asof = QLineEdit(va.as_of)
        self.v_src = QLineEdit(va.source)
        for lbl, w in (("Forward P/E", self.v_fpe), ("Trailing P/E", self.v_tpe), ("CAPE", self.v_cape), ("Dividend yield %", self.v_dy),
                       ("EPS revisions 3m %", self.v_rev), ("As of (date)", self.v_asof), ("Source", self.v_src)):
            f.addRow(lbl, w)
        lay.addWidget(g)

        g = QGroupBox("Schedule")
        f = QFormLayout(g)
        sc = s.schedule
        self.run_launch = QCheckBox("Run analysis at application launch")
        self.run_launch.setChecked(sc.run_at_launch)
        self.daily = QCheckBox("Run automatically every trading day")
        self.daily.setChecked(sc.daily_enabled)
        self.daily_time = QLineEdit(sc.daily_time)
        self.tz = QLineEdit(sc.timezone)
        self.def_mode = QComboBox()
        self.def_mode.addItems(["fast", "balanced", "deep", "continuous"])
        self.def_mode.setCurrentText(s.mode)
        f.addRow(self.run_launch)
        f.addRow(self.daily)
        f.addRow("Daily time (HH:MM, local)", self.daily_time)
        f.addRow("Time zone", self.tz)
        f.addRow("Default mode", self.def_mode)
        lay.addWidget(g)

        b = QPushButton("Save settings")
        b.clicked.connect(self.save)
        lay.addWidget(b)
        lay.addStretch(1)

    def save(self):
        s = self.s
        try:
            s.ticker = normalize_ticker(self.ticker.text())
            s.listing = self.listing.currentText()
            s.currency = self.currency.text().strip().upper() or "EUR"
            s.providers.market_priority = [x.strip() for x in self.market_prio.text().split(",") if x.strip()]
            s.providers.macro_priority = [x.strip() for x in self.macro_prio.text().split(",") if x.strip()]
            s.providers.news_priority = [x.strip() for x in self.news_prio.text().split(",") if x.strip()]
            s.providers.request_timeout = self.timeout.value()
            s.providers.allow_synthetic_demo = self.allow_demo.isChecked()
            for k, e in self.keys.items():
                setattr(s.api_keys, k, e.text().strip())
            s.news.rss_feeds = [x.strip() for x in self.feeds.toPlainText().splitlines() if x.strip()]
            s.news.lookback_days = self.news_lookback.value()
            s.models.horizons = sorted({int(x) for x in self.horizons.text().split(",") if x.strip()})
            s.models.primary_horizon = self.primary.value()
            s.models.enabled_models = [x.strip() for x in self.models.text().split(",") if x.strip()]
            s.models.training_window = self.window.currentText()
            s.models.recent_window_years = self.recent_years.value()
            s.models.history_years = self.hist_years.value()
            s.models.n_splits = self.n_splits.value()
            s.models.embargo_days = self.embargo.value()
            s.models.conformal_alpha = self.alpha.value()
            s.models.max_features = self.maxf.value()
            s.models.random_seed = self.seed.value()
            s.models.deep_learning_epochs = self.dl_epochs.value()
            s.models.optuna_trials = self.optuna.value()
            s.monte_carlo_paths = int(self.mc_paths.currentText())
            s.signals.buy_threshold, s.signals.sell_threshold = self.buy.value(), self.sell.value()
            s.signals.hysteresis, s.signals.min_signal_days = self.hyst.value(), self.min_days.value()
            s.signals.max_risk_for_buy, s.signals.strong_confirmations_required = self.max_risk.value(), self.n_conf.value()
            s.alerts.probability_change, s.alerts.drawdown_pct = self.alert_prob.value(), self.alert_dd.value()
            c = s.costs
            c.commission_pct, c.commission_min, c.spread_bps, c.slippage_bps = (self.comm.value(), self.comm_min.value(),
                                                                                self.spread.value(), self.slip.value())
            c.execution_delay_days, c.execution_price = self.delay.value(), self.exec_price.currentText()
            c.tax_on_realized_gains_pct, c.tax_on_dividends_pct = self.tax_cg.value(), self.tax_div.value()
            p = s.portfolio
            p.enabled, p.holdings_units, p.avg_purchase_price = self.pf_on.isChecked(), self.pf_units.value(), self.pf_avg.value()
            p.cash_available, p.investment_amount, p.risk_tolerance = self.pf_cash.value(), self.pf_inv.value(), self.pf_risk.currentText()
            v = s.valuation
            v.forward_pe, v.trailing_pe, v.cape = self.v_fpe.value(), self.v_tpe.value(), self.v_cape.value()
            v.dividend_yield_pct, v.eps_revisions_3m_pct = self.v_dy.value(), self.v_rev.value()
            v.as_of, v.source = self.v_asof.text().strip(), self.v_src.text().strip()
            sc = s.schedule
            sc.run_at_launch, sc.daily_enabled = self.run_launch.isChecked(), self.daily.isChecked()
            sc.daily_time, sc.timezone = self.daily_time.text().strip(), self.tz.text().strip()
            s.mode = self.def_mode.currentText()
            save_settings(s)
            save_api_keys(s.api_keys)
            QMessageBox.information(self, "Settings", "Settings saved (API keys stored in the local .env file).")
            if self.on_saved:
                self.on_saved()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Settings", f"Could not save settings: {exc}")
