"""DataService - orchestrates every data source, the cache and the point-in-time store.

Order of preference for each dataset: configured live providers (A -> B -> C), then the
local cache (marked ``DATA NOT CURRENT`` if older than the last completed session) and
only - if the user explicitly enabled it - the synthetic demo generator (marked
``SYNTHETIC DEMO`` everywhere). Nothing is silently fabricated.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config.settings import Settings
from ..core.runtime import get_logger, utcnow_iso
from ..database.db import Database
from .calendar import market_status, trading_days
from .macro.fred import MACRO_SERIES, load_macro_into_store
from .market.providers import (AlphaVantageProvider, StooqProvider, SyntheticProvider, TwelveDataProvider,
                               YFinanceProvider, _common_factor, _finalize)
from .market.universe import REFERENCE_ASSETS, SECTORS, SYNTHETIC_BASES, SYNTHETIC_BETAS
from .point_in_time.engine import PointInTimeStore
from .sources.base import FetchResult, SourceHealth, run_with_fallback
from .validation import ValidationReport, validate_ohlcv

log = get_logger("vusa.data")

VUSA_INCEPTION = "2012-05-22"


@dataclass
class MarketDataBundle:
    target: pd.DataFrame  # VUSA (possibly extended with a labelled proxy before inception)
    target_symbol: str
    assets: dict[str, pd.DataFrame] = field(default_factory=dict)
    sectors: dict[str, pd.DataFrame] = field(default_factory=dict)
    pit: PointInTimeStore = field(default_factory=PointInTimeStore)
    macro_status: dict[str, str] = field(default_factory=dict)
    validation: list[ValidationReport] = field(default_factory=list)
    fetch_log: list[FetchResult] = field(default_factory=list)
    data_current: bool = False
    is_synthetic: bool = False
    proxy_used: bool = False
    proxy_note: str = ""
    as_of: str = ""
    market_status: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def status_label(self) -> str:
        if self.is_synthetic:
            return "SYNTHETIC DEMO DATA - NOT REAL MARKET DATA"
        return "DATA CURRENT" if self.data_current else "DATA NOT CURRENT"


class DataService:
    def __init__(self, settings: Settings, db: Database | None = None):
        self.s = settings
        self.db = db
        self.health = SourceHealth()

    # ------------------------------------------------------------------ providers
    def _market_attempts(self, symbol: str, start: str, exchange: str):
        t = self.s.providers.request_timeout
        provs = {"yfinance": YFinanceProvider(t), "stooq": StooqProvider(t),
                 "twelvedata": TwelveDataProvider(self.s.api_keys.twelvedata, t),
                 "alphavantage": AlphaVantageProvider(self.s.api_keys.alphavantage, t)}
        return [(n, (lambda p=provs[n]: p.history(symbol, start, None, exchange)))
                for n in self.s.providers.market_priority if n in provs]

    def fetch_symbol(self, key: str, symbol: str, start: str, exchange: str, offline: bool = False
                     ) -> tuple[pd.DataFrame, FetchResult]:
        res = FetchResult(key, "none", False, error="offline mode")
        if not offline:
            res, attempts = run_with_fallback(key, self._market_attempts(symbol, start, exchange), self.health)
            if res.ok:
                if self.db:
                    self.db.save_prices(symbol, res.data, res.source)
                return res.data, res
        if self.db:
            cached = self.db.load_prices(symbol)
            if not cached.empty:
                cached = cached[["open", "high", "low", "close", "adj_close", "volume", "available_at"]]
                cres = FetchResult(key, "cache", True, cached, rows=len(cached), from_cache=True,
                                   last_date=str(cached.index[-1].date()),
                                   error="live providers failed; using local cache" if not offline else "offline")
                self.health.record(cres)
                return cached, cres
        return pd.DataFrame(), res

    # ------------------------------------------------------------------ main entry
    def load(self, offline: bool = False, include_macro: bool = True, synthetic: bool | None = None,
             as_of: pd.Timestamp | None = None) -> MarketDataBundle:
        synthetic = self.s.providers.allow_synthetic_demo if synthetic is None else synthetic
        start = (pd.Timestamp.today() - pd.DateOffset(years=self.s.models.history_years)).strftime("%Y-%m-%d")
        exch = self.s.exchange
        ms = market_status(exchange=exch)
        explicit_as_of = as_of is not None
        as_of = pd.Timestamp(as_of or ms["last_complete_session"])
        bundle = MarketDataBundle(pd.DataFrame(), self.s.yahoo_symbol, as_of=str(as_of.date()), market_status=ms)

        if synthetic:
            return self._synthetic_bundle(start, as_of, bundle)

        target, res = self.fetch_symbol("VUSA", self.s.yahoo_symbol, start, exch, offline)
        bundle.fetch_log.append(res)
        if explicit_as_of and not target.empty:  # historical / reproduction run: only data up to as-of
            target = target.loc[:as_of]
        for key, (sym, _, _) in REFERENCE_ASSETS.items():
            df, r = self.fetch_symbol(key, sym, start, "NYSE", offline)
            bundle.fetch_log.append(r)
            if explicit_as_of and not df.empty:
                df = df.loc[:as_of]
            if not df.empty:
                df, rep = validate_ohlcv(df, key, "NYSE", as_of, max_abs_return=0.6 if key in ("VIX", "VIX3M", "OIL") else 0.25,
                                         check_calendar=key not in ("EURUSD", "GOLD", "OIL", "DXY"))
                bundle.validation.append(rep)
                bundle.assets[key] = df
        for sym in SECTORS:
            df, r = self.fetch_symbol(f"sector:{sym}", sym, start, "NYSE", offline)
            bundle.fetch_log.append(r)
            if explicit_as_of and not df.empty:
                df = df.loc[:as_of]
            if not df.empty:
                bundle.sectors[sym] = df

        if target.empty:
            bundle.warnings.append(f"No VUSA data from any provider ({res.error}). Analysis cannot run on real data.")
            return bundle
        target, rep = validate_ohlcv(target, "VUSA", exch, as_of)
        bundle.validation.append(rep)
        bundle.target = self._extend_with_proxy(target, bundle)

        last = bundle.target.index[-1] if not bundle.target.empty else None
        bundle.data_current = bool(last is not None and last >= pd.Timestamp(ms["last_complete_session"])
                                   and not res.from_cache and res.ok and not explicit_as_of)
        if not bundle.data_current:
            bundle.warnings.append(f"DATA NOT CURRENT: last VUSA bar {last.date() if last is not None else 'n/a'}; "
                                   f"last completed session {ms['last_complete_session']}.")

        if include_macro:
            bundle.macro_status = load_macro_into_store(
                bundle.pit, list(MACRO_SERIES), self.s.api_keys.fred,
                [] if offline else self.s.providers.macro_priority, self.s.providers.request_timeout,
                start, self.health)
            missing = [k for k, v in bundle.macro_status.items() if not v.startswith("ok")]
            if missing:
                bundle.warnings.append(f"Macro series unavailable: {', '.join(missing[:8])}"
                                       f"{'...' if len(missing) > 8 else ''}")
        self._log_quality(bundle)
        return bundle

    # ------------------------------------------------------------------ helpers
    def _extend_with_proxy(self, target: pd.DataFrame, bundle: MarketDataBundle) -> pd.DataFrame:
        """VUSA exists only since 2012. For long-horizon training the history before the
        first VUSA bar is extended with the S&P 500 converted to EUR (explicitly labelled
        ``is_proxy``); this proxy excludes the ETF's fees/dividend timing."""
        target = target.copy()
        target["is_proxy"] = 0
        spx, fx = bundle.assets.get("SPX"), bundle.assets.get("EURUSD")
        if spx is None or spx.empty:
            return target
        first = target.index[0]
        conv = spx["close"].copy()
        if self.s.currency == "EUR" and fx is not None and not fx.empty:
            conv = conv / fx["close"].reindex(conv.index).ffill()
        conv = conv.dropna()
        pre = conv[conv.index < first]
        if len(pre) < 250 or first not in conv.index and conv.index.searchsorted(first) >= len(conv):
            return target
        anchor = conv.asof(first)
        scale = target["close"].iloc[0] / anchor
        p = pd.DataFrame(index=pre.index)
        spx_pre = spx.loc[pre.index]
        ratio = (pre / spx_pre["close"]).values * scale
        for c in ("open", "high", "low", "close"):
            p[c] = spx_pre[c].values * ratio
        p["adj_close"] = p["close"]
        p["volume"] = np.nan
        p["available_at"] = spx_pre["available_at"]
        p["is_proxy"] = 1
        bundle.proxy_used = True
        bundle.proxy_note = (f"History before {first.date()} uses S&P 500 converted to {self.s.currency} as a "
                             f"labelled proxy ({len(p)} rows); excludes ETF fees and dividend timing.")
        return pd.concat([p, target]).sort_index()

    def _synthetic_bundle(self, start: str, as_of: pd.Timestamp, bundle: MarketDataBundle) -> MarketDataBundle:
        gen = SyntheticProvider(seed=self.s.models.random_seed)
        idx = trading_days(start, as_of, "XETRA")
        common = _common_factor(gen.seed, idx)
        bundle.is_synthetic = True
        bundle.warnings.append("SYNTHETIC DEMO DATA: generated locally, NOT real market data. "
                               "Use only to explore the application.")
        tgt = gen.history("VUSA", start, str(as_of.date()), "XETRA", base=30.0)
        tgt, rep = validate_ohlcv(tgt, "VUSA", "XETRA", as_of)
        bundle.validation.append(rep)
        tgt["is_proxy"] = 0
        bundle.target = tgt
        rng = np.random.default_rng(gen.seed)
        rv = pd.Series(common, index=idx).rolling(21, min_periods=5).std().bfill().values * np.sqrt(252)
        for key, beta in SYNTHETIC_BETAS.items():
            if key in ("VIX", "VIX3M"):
                lvl = 8 + 100 * rv * (1.0 if key == "VIX" else 0.85) + rng.normal(0, 0.8, len(idx))
                lvl = np.maximum(lvl, 9)
                df = pd.DataFrame({"open": lvl, "high": lvl * 1.03, "low": lvl * 0.97, "close": lvl, "adj_close": lvl,
                                   "volume": 0.0}, index=idx)
                df = _finalize(df, "NYSE")
            else:
                base = SYNTHETIC_BASES.get(key, 100.0)
                n = len(idx)
                r = beta * 0.3 * common + rng.normal(0, 0.006, n) if key in ("TNX", "IRX", "EURUSD", "DXY") else \
                    beta * common + rng.standard_t(5, n) * 0.004 + 0.0002
                c = base * np.exp(np.cumsum(r))
                df = pd.DataFrame({"open": c, "high": c * 1.004, "low": c * 0.996, "close": c, "adj_close": c,
                                   "volume": rng.lognormal(14, 0.3, n)}, index=idx)
                df = _finalize(df, "NYSE")
            bundle.assets[key] = df
        for i, sym in enumerate(SECTORS):
            n = len(idx)
            r = (0.7 + 0.06 * i) * common + rng.standard_t(5, n) * 0.006 + 0.0002
            c = 50 * np.exp(np.cumsum(r))
            bundle.sectors[sym] = _finalize(pd.DataFrame({"open": c, "high": c * 1.005, "low": c * 0.995, "close": c,
                                                          "adj_close": c, "volume": 1e6}, index=idx), "NYSE")
        # synthetic macro with release lags so the PIT path is exercised identically
        months = pd.date_range(start, as_of, freq="MS")
        cpi = 100 * np.exp(np.cumsum(rng.normal(0.002, 0.002, len(months))))
        unrate = np.clip(5 + np.cumsum(rng.normal(0, 0.12, len(months))), 3, 11)
        bundle.pit.add_latest_with_lag("CPIAUCSL", pd.Series(cpi, months), 16, "synthetic")
        bundle.pit.add_latest_with_lag("UNRATE", pd.Series(unrate, months), 8, "synthetic")
        days = pd.DatetimeIndex(idx)
        y10 = np.clip(3 + np.cumsum(rng.normal(0, 0.04, len(days))), 0.5, 8)
        y2 = np.clip(y10 - 0.5 + np.cumsum(rng.normal(0, 0.03, len(days))) * 0.3, 0.1, 8)
        hy = np.clip(4 + 40 * (rv - rv.mean()) + rng.normal(0, 0.1, len(days)), 2.5, 15)
        for sid, vals in (("DGS10", y10), ("DGS2", y2), ("T10Y2Y", y10 - y2), ("BAMLH0A0HYM2", hy)):
            bundle.pit.add_latest_with_lag(sid, pd.Series(vals, days), 1, "synthetic")
        bundle.macro_status = {k: "SYNTHETIC" for k in bundle.pit.series_ids}
        bundle.data_current = False
        self.health.record(FetchResult("VUSA", "synthetic", True, rows=len(tgt), is_synthetic=True,
                                       last_date=str(tgt.index[-1].date())))
        return bundle

    def _log_quality(self, bundle: MarketDataBundle) -> None:
        if not self.db:
            return
        now = utcnow_iso()
        rows = []
        for r in bundle.fetch_log:
            h = r.health_row()
            rows.append({"checked_at": now, "dataset": h["dataset"], "source": h["source"], "status": h["status"],
                         "latency_ms": h["latency_ms"], "rows": h["rows"], "last_date": h["last_date"],
                         "issues": h["issues"]})
        for v in bundle.validation:
            rows.append({"checked_at": now, "dataset": f"validation:{v.dataset}", "source": "validator",
                         "status": "ok" if v.ok else "failed", "latency_ms": 0, "rows": v.rows_out,
                         "last_date": None, "issues": v.summary()})
        self.db.upsert_many("data_quality", rows)
