"""Application configuration.

Settings are layered:
1. Built-in defaults (this file)
2. ``config/user_settings.json`` (written by the Settings tab)
3. Environment variables / ``.env`` (API keys only)

Nothing in here hard-codes country specific tax rules: the tax section is a set of
explicit, user-editable assumptions with neutral defaults (0 %).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("VUSA_DATA_DIR", PROJECT_ROOT / "storage"))
SETTINGS_FILE = PROJECT_ROOT / "config" / "user_settings.json"
ENV_FILE = PROJECT_ROOT / ".env"

# User-entered aliases. "VUSAA" is a common typo / broker code for VUSA.
TICKER_ALIASES = {"VUSAA": "VUSA", "VUSA": "VUSA"}

# Listings of the Vanguard S&P 500 UCITS ETF (distributing share class).
LISTINGS = {
    "XETRA (EUR)": {"yahoo": "VUSA.DE", "stooq": "vusa.de", "currency": "EUR", "exchange": "XETRA"},
    "Euronext Amsterdam (EUR)": {"yahoo": "VUSA.AS", "stooq": "vusa.nl", "currency": "EUR", "exchange": "XAMS"},
    "London (GBP)": {"yahoo": "VUSA.L", "stooq": "vusa.uk", "currency": "GBP", "exchange": "XLON"},
    "Borsa Italiana (EUR)": {"yahoo": "VUSA.MI", "stooq": "vusa.it", "currency": "EUR", "exchange": "XMIL"},
}


@dataclass
class ProviderSettings:
    market_priority: list[str] = field(default_factory=lambda: ["yfinance", "stooq", "twelvedata", "alphavantage"])
    macro_priority: list[str] = field(default_factory=lambda: ["fred_api", "fred_csv"])
    news_priority: list[str] = field(default_factory=lambda: ["rss", "newsapi"])
    request_timeout: float = 20.0
    cache_max_age_hours: float = 12.0
    allow_synthetic_demo: bool = False  # never on by default; demo data is always labelled


@dataclass
class ApiKeys:
    fred: str = ""
    alphavantage: str = ""
    twelvedata: str = ""
    newsapi: str = ""
    anthropic: str = ""


@dataclass
class ModelSettings:
    horizons: list[int] = field(default_factory=lambda: [1, 3, 5, 10, 20, 60, 120, 252])
    primary_horizon: int = 20
    enabled_models: list[str] = field(default_factory=lambda: [
        "naive_drift", "elastic_net", "random_forest", "extra_trees", "hist_gb",
        "xgboost", "lightgbm", "catboost", "arima", "ets", "lstm", "gru", "tcn", "transformer",
    ])
    training_window: str = "full"  # full | recent | exp_weighted | auto
    recent_window_years: float = 8.0
    exp_halflife_years: float = 5.0
    history_years: int = 25
    n_splits: int = 5
    embargo_days: int = 5
    conformal_alpha: float = 0.2  # 80 % interval
    max_features: int = 40
    random_seed: int = 42
    optuna_trials: int = 0  # 0 disables HPO (fast); Deep Research sets this
    deep_learning_epochs: int = 30
    retrain_every_days: int = 21


@dataclass
class SignalSettings:
    buy_threshold: float = 62.0
    sell_threshold: float = 38.0
    hysteresis: float = 4.0
    min_signal_days: int = 3
    strong_signal_threshold: float = 72.0
    strong_confirmations_required: int = 3  # out of ML, trend, risk/reward, regime
    max_risk_for_buy: float = 70.0
    min_confidence_for_directional: float = 0.15  # below this BUY/SELL is shown as HOLD (LOW CONFIDENCE)
    weights: dict[str, float] = field(default_factory=lambda: {
        "trend": 1.2, "momentum": 1.0, "valuation": 0.6, "macro": 0.9, "sentiment": 0.5,
        "volatility": 0.9, "breadth": 0.7, "regime": 1.0, "event_risk": 0.6,
        "ml_forecast": 1.2, "analogues": 0.8, "risk_reward": 1.0,
    })


@dataclass
class CostSettings:
    commission_pct: float = 0.0005
    commission_min: float = 1.0
    spread_bps: float = 4.0
    slippage_bps: float = 2.0
    execution_delay_days: int = 1  # signal after close -> execute next session open
    execution_price: str = "open"  # open | close (only if executable close is explicitly modelled)
    # Tax assumptions: explicit and editable. No legal claims are made by the application.
    tax_on_realized_gains_pct: float = 0.0
    tax_on_dividends_pct: float = 0.0
    tax_note: str = ("Tax assumptions are user supplied. Verify with a qualified tax adviser for your "
                     "jurisdiction (e.g. Cyprus). The application makes no legal or tax claims.")


@dataclass
class PortfolioSettings:
    enabled: bool = False
    investment_amount: float = 0.0
    holdings_units: float = 0.0
    avg_purchase_price: float = 0.0
    cash_available: float = 0.0
    risk_tolerance: str = "moderate"  # conservative | moderate | aggressive


@dataclass
class ScheduleSettings:
    run_at_launch: bool = False
    daily_enabled: bool = True
    daily_time: str = "18:30"  # local time, after XETRA close
    timezone: str = "Europe/Nicosia"


@dataclass
class AlertSettings:
    probability_change: float = 0.10
    volatility_z: float = 2.5
    drawdown_pct: float = 0.08
    breakout_lookback: int = 60


@dataclass
class NewsSettings:
    rss_feeds: list[str] = field(default_factory=lambda: [
        "https://www.federalreserve.gov/feeds/press_all.xml",
        "https://www.ecb.europa.eu/rss/press.html",
        "https://www.bls.gov/feed/bls_latest.rss",
        "https://www.bea.gov/news/rss.xml",
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://feeds.content.dowjones.io/public/rss/mw_topstories",
        "https://www.investing.com/rss/news_25.rss",
        "https://finance.yahoo.com/news/rssindex",
    ])
    max_articles: int = 300
    lookback_days: int = 3


@dataclass
class ValuationSettings:
    """Fundamental valuation inputs for the S&P 500.

    Free APIs do not reliably provide point-in-time index valuation, so values are entered
    by the user (with their date and source, e.g. S&P Dow Jones Indices / FactSet / multpl).
    Empty values are never invented: the valuation component then falls back to an
    explicitly-labelled *price-based proxy* with reduced weight.
    """
    trailing_pe: float = 0.0
    forward_pe: float = 0.0
    cape: float = 0.0
    dividend_yield_pct: float = 0.0
    eps_growth_fwd_pct: float = 0.0
    eps_revisions_3m_pct: float = 0.0
    as_of: str = ""
    source: str = ""


@dataclass
class Settings:
    ticker: str = "VUSA"
    listing: str = "XETRA (EUR)"
    currency: str = "EUR"
    mode: str = "balanced"  # fast | balanced | deep | continuous
    monte_carlo_paths: int = 10000
    providers: ProviderSettings = field(default_factory=ProviderSettings)
    api_keys: ApiKeys = field(default_factory=ApiKeys)
    models: ModelSettings = field(default_factory=ModelSettings)
    signals: SignalSettings = field(default_factory=SignalSettings)
    costs: CostSettings = field(default_factory=CostSettings)
    portfolio: PortfolioSettings = field(default_factory=PortfolioSettings)
    schedule: ScheduleSettings = field(default_factory=ScheduleSettings)
    alerts: AlertSettings = field(default_factory=AlertSettings)
    news: NewsSettings = field(default_factory=NewsSettings)
    valuation: ValuationSettings = field(default_factory=ValuationSettings)
    db_url: str = ""  # empty -> sqlite file in DATA_DIR

    # ------------------------------------------------------------------ helpers
    @property
    def yahoo_symbol(self) -> str:
        return LISTINGS.get(self.listing, LISTINGS["XETRA (EUR)"])["yahoo"]

    @property
    def stooq_symbol(self) -> str:
        return LISTINGS.get(self.listing, LISTINGS["XETRA (EUR)"])["stooq"]

    @property
    def exchange(self) -> str:
        return LISTINGS.get(self.listing, LISTINGS["XETRA (EUR)"])["exchange"]

    @property
    def db_path(self) -> Path:
        return DATA_DIR / "vusa_quant.sqlite"

    def to_dict(self, include_keys: bool = False) -> dict[str, Any]:
        d = asdict(self)
        if not include_keys:
            d["api_keys"] = {k: ("***" if v else "") for k, v in d["api_keys"].items()}
        return d


def normalize_ticker(user_input: str) -> str:
    t = (user_input or "").strip().upper()
    return TICKER_ALIASES.get(t, t)


def _load_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _merge(obj: Any, data: dict[str, Any]) -> Any:
    for f in fields(obj):
        if f.name not in data:
            continue
        cur = getattr(obj, f.name)
        val = data[f.name]
        if is_dataclass(cur) and isinstance(val, dict):
            _merge(cur, val)
        else:
            setattr(obj, f.name, val)
    return obj


def load_settings(path: Path | None = None) -> Settings:
    s = Settings()
    p = path or SETTINGS_FILE
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            data.pop("api_keys", None)  # keys never live in the settings json
            _merge(s, data)
        except (json.JSONDecodeError, OSError):
            pass
    env = {**_load_env_file(ENV_FILE), **os.environ}
    s.api_keys.fred = env.get("FRED_API_KEY", s.api_keys.fred)
    s.api_keys.alphavantage = env.get("ALPHAVANTAGE_API_KEY", s.api_keys.alphavantage)
    s.api_keys.twelvedata = env.get("TWELVEDATA_API_KEY", s.api_keys.twelvedata)
    s.api_keys.newsapi = env.get("NEWSAPI_KEY", s.api_keys.newsapi)
    s.api_keys.anthropic = env.get("ANTHROPIC_API_KEY", s.api_keys.anthropic)
    s.ticker = normalize_ticker(s.ticker)
    return s


def save_settings(s: Settings, path: Path | None = None) -> None:
    p = path or SETTINGS_FILE
    d = asdict(s)
    d.pop("api_keys", None)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2), encoding="utf-8")


def save_api_keys(keys: ApiKeys, path: Path | None = None) -> None:
    """Persist API keys to the local .env file (never to the settings json / database)."""
    p = path or ENV_FILE
    existing = _load_env_file(p)
    existing.update({
        "FRED_API_KEY": keys.fred, "ALPHAVANTAGE_API_KEY": keys.alphavantage,
        "TWELVEDATA_API_KEY": keys.twelvedata, "NEWSAPI_KEY": keys.newsapi,
        "ANTHROPIC_API_KEY": keys.anthropic,
    })
    p.write_text("\n".join(f"{k}={v}" for k, v in existing.items()) + "\n", encoding="utf-8")


MODE_PROFILES = {
    "fast": {"models": ["naive_drift", "elastic_net", "hist_gb", "lightgbm"], "horizons": [1, 5, 20],
             "news": False, "macro": False, "deep": False, "mc_paths": 2000, "analogues": True,
             "stress": True, "full_validation": False},
    "balanced": {"models": ["naive_drift", "elastic_net", "random_forest", "hist_gb", "xgboost", "lightgbm", "arima"],
                 "horizons": [1, 5, 10, 20, 60, 252], "news": True, "macro": True, "deep": False,
                 "mc_paths": 10000, "analogues": True, "stress": True, "full_validation": False},
    "deep": {"models": None, "horizons": None, "news": True, "macro": True, "deep": True,
             "mc_paths": None, "analogues": True, "stress": True, "full_validation": True},
}
MODE_PROFILES["continuous"] = dict(MODE_PROFILES["balanced"])
