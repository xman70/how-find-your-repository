"""Market-data providers: yfinance, Stooq, Twelve Data, Alpha Vantage.

All providers return a DataFrame indexed by session date (naive, normalised) with
columns open, high, low, close, adj_close, volume. Daily bars are stamped with
``available_at`` = official close of that session in UTC, which is the earliest moment
the bar could be known (point-in-time alignment).

A synthetic generator exists for tests and for an explicitly-enabled demo mode. Its
output is always flagged ``is_synthetic`` and the GUI shows a red banner; it is never
used silently in place of real data.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from ..calendar import market_close_utc, trading_days
from ..sources.base import ProviderUnavailable, http_get

COLS = ["open", "high", "low", "close", "adj_close", "volume"]


def _finalize(df: pd.DataFrame, exchange: str = "NYSE") -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    for c in COLS:
        if c not in df.columns:
            df[c] = df["close"] if c == "adj_close" else np.nan
    df = df[COLS].astype(float)
    df = df.dropna(subset=["close"])
    df["available_at"] = [market_close_utc(d, exchange if exchange in ("XETRA", "XAMS", "XMIL", "XLON") else "NYSE")
                          .isoformat() for d in df.index]
    return df


class YFinanceProvider:
    name = "yfinance"

    def __init__(self, timeout: float = 20.0):
        self.timeout = timeout

    def history(self, symbol: str, start: str, end: str | None = None, exchange: str = "NYSE") -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ProviderUnavailable("yfinance not installed") from exc
        raw = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False, threads=False,
                          timeout=self.timeout)
        if raw is None or raw.empty:
            raise ProviderUnavailable(f"yfinance returned no rows for {symbol}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close",
                                  "Adj Close": "adj_close", "Volume": "volume"})
        return _finalize(raw, exchange)


class StooqProvider:
    name = "stooq"
    SYMBOL_MAP = {"^GSPC": "^spx", "^NDX": "^ndx", "^DJI": "^dji", "^RUT": "^rut", "^VIX": "^vix",
                  "EURUSD=X": "eurusd", "GC=F": "gc.f", "CL=F": "cl.f", "DX-Y.NYB": "dx.f"}

    def __init__(self, timeout: float = 20.0):
        self.timeout = timeout

    def to_stooq(self, symbol: str) -> str:
        if symbol in self.SYMBOL_MAP:
            return self.SYMBOL_MAP[symbol]
        s = symbol.lower()
        for suf, rep in ((".de", ".de"), (".as", ".nl"), (".l", ".uk"), (".mi", ".it")):
            if s.endswith(suf):
                return s[: -len(suf)] + rep
        if "." not in s and "^" not in s and "=" not in s:
            return s + ".us"
        return s

    def history(self, symbol: str, start: str, end: str | None = None, exchange: str = "NYSE") -> pd.DataFrame:
        sym = self.to_stooq(symbol)
        params = {"s": sym, "i": "d", "d1": pd.Timestamp(start).strftime("%Y%m%d")}
        if end:
            params["d2"] = pd.Timestamp(end).strftime("%Y%m%d")
        r = http_get("https://stooq.com/q/d/l/", timeout=self.timeout, params=params)
        txt = r.text
        if not txt or txt.strip().lower().startswith("no data") or "Date" not in txt[:50]:
            raise ProviderUnavailable(f"stooq has no data for {sym}")
        df = pd.read_csv(io.StringIO(txt), parse_dates=["Date"], index_col="Date")
        df = df.rename(columns=str.lower)
        df["adj_close"] = df["close"]  # stooq prices are split/dividend adjusted
        return _finalize(df, exchange)


class TwelveDataProvider:
    name = "twelvedata"

    def __init__(self, api_key: str, timeout: float = 20.0):
        self.api_key, self.timeout = api_key, timeout

    def history(self, symbol: str, start: str, end: str | None = None, exchange: str = "NYSE") -> pd.DataFrame:
        if not self.api_key:
            raise ProviderUnavailable("Twelve Data API key not configured")
        sym = symbol.split(".")[0].replace("^", "")
        params = {"symbol": sym, "interval": "1day", "start_date": start, "outputsize": 5000,
                  "apikey": self.api_key, "format": "JSON"}
        if exchange in ("XETRA", "XAMS", "XLON", "XMIL"):
            params["exchange"] = {"XETRA": "XETR", "XAMS": "Euronext", "XLON": "LSE", "XMIL": "MTA"}[exchange]
        js = http_get("https://api.twelvedata.com/time_series", timeout=self.timeout, params=params).json()
        if js.get("status") == "error" or "values" not in js:
            raise ProviderUnavailable(f"Twelve Data: {js.get('message', 'no values')}")
        df = pd.DataFrame(js["values"]).set_index("datetime")
        df = df.rename(columns=str.lower).apply(pd.to_numeric, errors="coerce")
        return _finalize(df, exchange)


class AlphaVantageProvider:
    name = "alphavantage"

    def __init__(self, api_key: str, timeout: float = 20.0):
        self.api_key, self.timeout = api_key, timeout

    def history(self, symbol: str, start: str, end: str | None = None, exchange: str = "NYSE") -> pd.DataFrame:
        if not self.api_key:
            raise ProviderUnavailable("Alpha Vantage API key not configured")
        sym = symbol.replace(".DE", ".DEX").replace(".L", ".LON").replace(".AS", ".AMS")
        js = http_get("https://www.alphavantage.co/query", timeout=self.timeout,
                      params={"function": "TIME_SERIES_DAILY", "symbol": sym, "outputsize": "full",
                              "apikey": self.api_key}).json()
        key = next((k for k in js if "Time Series" in k), None)
        if not key:
            raise ProviderUnavailable(f"Alpha Vantage: {js.get('Note') or js.get('Information') or js.get('Error Message')}")
        df = pd.DataFrame(js[key]).T
        df.columns = [c.split(". ")[-1] for c in df.columns]
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df[df.index >= start]
        return _finalize(df, exchange)


class SyntheticProvider:
    """Regime-switching synthetic market generator for tests and the labelled demo mode.

    Produces fat-tailed, volatility-clustered prices. Output is NEVER real data.
    """

    name = "synthetic"

    def __init__(self, seed: int = 7):
        self.seed = seed

    def history(self, symbol: str, start: str, end: str | None = None, exchange: str = "XETRA",
                base: float = 100.0, beta: float = 1.0, drift: float = 0.0003) -> pd.DataFrame:
        idx = trading_days(start, end or pd.Timestamp.today(), exchange)
        rng = np.random.default_rng(self.seed + sum(map(ord, symbol)))
        n = len(idx)
        common = _common_factor(self.seed, idx)
        idio = rng.standard_t(5, n) * 0.004
        ret = drift + beta * common + idio
        close = base * np.exp(np.cumsum(ret))
        gap = rng.normal(0, 0.002, n)
        open_ = close * np.exp(-ret + gap)
        rng_hl = np.abs(rng.normal(0, 0.006, n)) + np.abs(ret) / 2
        high = np.maximum(open_, close) * (1 + rng_hl)
        low = np.minimum(open_, close) * (1 - rng_hl)
        vol = rng.lognormal(12, 0.4, n) * (1 + 20 * np.abs(common))
        df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "adj_close": close,
                           "volume": vol}, index=idx)
        return _finalize(df, exchange)


def _common_factor(seed: int, idx: pd.DatetimeIndex) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(idx)
    state = np.zeros(n, dtype=int)
    P = np.array([[0.995, 0.005], [0.03, 0.97]])
    for t in range(1, n):
        state[t] = rng.choice(2, p=P[state[t - 1]])
    mu = np.where(state == 0, 0.0004, -0.0012)
    sig = np.where(state == 0, 0.008, 0.022)
    # GARCH-ish clustering
    eps = rng.standard_t(4, n) / np.sqrt(2)
    h = np.empty(n)
    h[0] = sig[0] ** 2
    out = np.empty(n)
    for t in range(n):
        if t > 0:
            h[t] = 0.05 * sig[t] ** 2 + 0.08 * out[t - 1] ** 2 + 0.87 * h[t - 1]
        out[t] = mu[t] + np.sqrt(h[t]) * eps[t]
    return out
