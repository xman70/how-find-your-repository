"""Technical indicator library (pure pandas / numpy, causal - every value at t uses only
data up to and including t)."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def wma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype=float)
    return s.rolling(n, min_periods=n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def hma(s: pd.Series, n: int) -> pd.Series:
    return wma(2 * wma(s, n // 2) - wma(s, n), int(np.sqrt(n)))


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(s: pd.Series, fast=12, slow=26, signal=9):
    m = ema(s, fast) - ema(s, slow)
    sig = ema(m, signal)
    return m, sig, m - sig


def true_range(h, l, c):
    pc = c.shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def atr(h, l, c, n=14):
    return true_range(h, l, c).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def adx(h, l, c, n=14):
    up, dn = h.diff(), -l.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = atr(h, l, c, n)
    pdi = 100 * pd.Series(plus_dm, h.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / tr
    mdi = 100 * pd.Series(minus_dm, h.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / tr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean(), pdi, mdi


def stochastic(h, l, c, n=14, d=3):
    ll, hh = l.rolling(n).min(), h.rolling(n).max()
    k = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    return k, k.rolling(d).mean()


def williams_r(h, l, c, n=14):
    hh, ll = h.rolling(n).max(), l.rolling(n).min()
    return -100 * (hh - c) / (hh - ll).replace(0, np.nan)


def cci(h, l, c, n=20):
    tp = (h + l + c) / 3
    md = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - tp.rolling(n).mean()) / (0.015 * md.replace(0, np.nan))


def obv(c, v):
    return (np.sign(c.diff()).fillna(0) * v.fillna(0)).cumsum()


def realized_vol(r: pd.Series, n: int) -> pd.Series:
    return r.rolling(n, min_periods=max(5, n // 2)).std() * np.sqrt(TRADING_DAYS)


def parkinson_vol(h, l, n=20):
    x = np.log(h / l) ** 2 / (4 * np.log(2))
    return np.sqrt(x.rolling(n).mean() * TRADING_DAYS)


def garman_klass_vol(o, h, l, c, n=20):
    x = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    return np.sqrt(x.clip(lower=0).rolling(n).mean() * TRADING_DAYS)


def rogers_satchell_vol(o, h, l, c, n=20):
    x = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    return np.sqrt(x.clip(lower=0).rolling(n).mean() * TRADING_DAYS)


def slope(s: pd.Series, n: int) -> pd.Series:
    """OLS slope of log(s) over a rolling window, annualised."""
    x = np.arange(n) - (n - 1) / 2
    denom = (x ** 2).sum()
    ls = np.log(s.where(s > 0))
    return ls.rolling(n, min_periods=n).apply(lambda y: np.dot(x, y) / denom, raw=True) * TRADING_DAYS


def rolling_zscore(s: pd.Series, n: int, min_periods: int | None = None) -> pd.Series:
    m = s.rolling(n, min_periods=min_periods or n // 2).mean()
    sd = s.rolling(n, min_periods=min_periods or n // 2).std()
    return (s - m) / sd.replace(0, np.nan)


def rolling_percentile(s: pd.Series, n: int, min_periods: int | None = None) -> pd.Series:
    return s.rolling(n, min_periods=min_periods or n // 4).apply(lambda x: (x[:-1] < x[-1]).mean() if len(x) > 1 else np.nan,
                                                                  raw=True)


def drawdown(c: pd.Series) -> pd.Series:
    return c / c.cummax() - 1


def technical_features(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Build the technical feature block for an OHLCV frame. Returns causal features only."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    v = df["volume"] if "volume" in df else pd.Series(np.nan, df.index)
    r = np.log(c).diff()
    f = {}
    # ---------------------------------------------------------------- price
    for n in (1, 2, 3, 5, 10, 20, 60, 120, 252):
        f[f"ret_{n}d"] = c.pct_change(n)
    f["logret_1d"] = r
    f["gap"] = np.log(o / c.shift(1))
    f["hl_range"] = (h - l) / c
    f["close_location"] = ((c - l) / (h - l).replace(0, np.nan)) - 0.5
    f["drawdown"] = drawdown(c)
    f["drawdown_252"] = c / c.rolling(252, min_periods=60).max() - 1
    for n in (20, 60, 252):
        f[f"dist_high_{n}"] = c / h.rolling(n, min_periods=n // 2).max() - 1
        f[f"dist_low_{n}"] = c / l.rolling(n, min_periods=n // 2).min() - 1
    # ---------------------------------------------------------------- trend
    for n in (10, 20, 50, 100, 200):
        f[f"dist_sma{n}"] = c / sma(c, n) - 1
    for n in (12, 26, 50):
        f[f"dist_ema{n}"] = c / ema(c, n) - 1
    f["dist_hma20"] = c / hma(c, 20) - 1
    f["sma50_sma200"] = sma(c, 50) / sma(c, 200) - 1
    f["sma20_sma50"] = sma(c, 20) / sma(c, 50) - 1
    for n in (20, 60, 120):
        f[f"trend_slope_{n}"] = slope(c, n)
    a, pdi, mdi = adx(h, l, c)
    f["adx14"] = a
    f["di_diff"] = pdi - mdi
    above200 = (c > sma(c, 200)).astype(float).where(sma(c, 200).notna())
    f["trend_persistence_60"] = above200.rolling(60, min_periods=20).mean()
    f["up_days_20"] = (r > 0).astype(float).rolling(20).mean()
    if v.notna().sum() > 100 and (v.fillna(0) > 0).mean() > 0.5:
        vwap20 = (c * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
        f["dist_vwap20"] = c / vwap20 - 1
    # ---------------------------------------------------------------- momentum
    f["rsi14"] = rsi(c, 14)
    f["rsi2"] = rsi(c, 2)
    m, sig, hist = macd(c)
    f["macd_norm"] = m / c
    f["macd_hist_norm"] = hist / c
    for n in (10, 20, 60):
        f[f"roc_{n}"] = c.pct_change(n)
    k, d = stochastic(h, l, c)
    f["stoch_k"], f["stoch_d"] = k, d
    f["williams_r"] = williams_r(h, l, c)
    f["cci20"] = cci(h, l, c)
    f["mom_accel"] = c.pct_change(20) - c.pct_change(20).shift(20)
    f["mom_12_1"] = c.shift(21) / c.shift(252) - 1
    # ---------------------------------------------------------------- volatility
    f["atr14_pct"] = atr(h, l, c) / c
    for n in (5, 10, 20, 60, 120, 252):
        f[f"rv_{n}"] = realized_vol(r, n)
    f["parkinson_20"] = parkinson_vol(h, l)
    f["garman_klass_20"] = garman_klass_vol(o, h, l, c)
    f["rogers_satchell_20"] = rogers_satchell_vol(o, h, l, c)
    f["vol_of_vol_60"] = f["rv_20"].rolling(60, min_periods=30).std()
    f["rv_ratio_5_60"] = f["rv_5"] / f["rv_60"]
    f["rv_ratio_20_252"] = f["rv_20"] / f["rv_252"]
    f["downside_vol_20"] = r.clip(upper=0).rolling(20).std() * np.sqrt(TRADING_DAYS)
    f["skew_60"] = r.rolling(60).skew()
    f["kurt_60"] = r.rolling(60).kurt()
    f["rv20_z_5y"] = rolling_zscore(f["rv_20"], 1260, 250)
    # ---------------------------------------------------------------- volume
    if v.notna().sum() > 100 and (v.fillna(0) > 0).mean() > 0.5:
        ob = obv(c, v)
        f["obv_slope_20"] = (ob - ob.shift(20)) / v.rolling(20).mean().replace(0, np.nan) / 20
        f["volume_z_20"] = rolling_zscore(np.log1p(v), 60, 20)
        f["volume_trend"] = v.rolling(10).mean() / v.rolling(60).mean().replace(0, np.nan) - 1
        f["volume_pressure"] = (np.sign(r) * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    out = pd.DataFrame(f, index=df.index)
    out = out.replace([np.inf, -np.inf], np.nan)
    if prefix:
        out.columns = [f"{prefix}{col}" for col in out.columns]
    return out


def support_resistance(df: pd.DataFrame, lookback: int = 250, window: int = 10, n_levels: int = 4) -> dict:
    """Pivot-based support / resistance levels from local extrema (descriptive only)."""
    d = df.tail(lookback)
    if len(d) < window * 3:
        return {"support": [], "resistance": []}
    h, l, c = d["high"], d["low"], d["close"].iloc[-1]
    piv_h = h[(h == h.rolling(2 * window + 1, center=True).max())].dropna()
    piv_l = l[(l == l.rolling(2 * window + 1, center=True).min())].dropna()

    def cluster(levels):
        levels = sorted(levels)
        out = []
        for x in levels:
            if out and abs(x / out[-1][0] - 1) < 0.01:
                out[-1] = ((out[-1][0] * out[-1][1] + x) / (out[-1][1] + 1), out[-1][1] + 1)
            else:
                out.append((x, 1))
        return out

    res = [(round(x, 2), n) for x, n in cluster(list(piv_h.values) + list(piv_l.values))]
    sup = sorted([x for x in res if x[0] < c], key=lambda x: -x[0])[:n_levels]
    rst = sorted([x for x in res if x[0] > c], key=lambda x: x[0])[:n_levels]
    return {"support": [{"level": x, "touches": n} for x, n in sup],
            "resistance": [{"level": x, "touches": n} for x, n in rst]}
