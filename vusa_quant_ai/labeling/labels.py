"""Labels / targets.

All labels are *forward-looking* by construction and are kept strictly separate from
features. Each label records ``label_end`` (the last date whose price enters the label)
so purging/embargo can remove overlapping observations during validation.

Execution convention: a decision taken after the close of day t can be executed at the
earliest at the next session's open (t+1). Forward returns are therefore measured from
the open of t+1 ("executable") by default, with close-to-close returns available for
research comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RETURN_BUCKETS = [-np.inf, -0.05, -0.02, 0.0, 0.02, 0.05, np.inf]
BUCKET_NAMES = ["< -5%", "-5..-2%", "-2..0%", "0..+2%", "+2..+5%", "> +5%"]


def forward_return(df: pd.DataFrame, h: int, executable: bool = True) -> pd.Series:
    c = df["close"]
    if executable and "open" in df and df["open"].notna().mean() > 0.9:
        entry = df["open"].shift(-1).fillna(c.shift(-1))
        exit_ = c.shift(-h)
        return exit_ / entry - 1
    return c.shift(-h) / c - 1


def label_end_dates(index: pd.DatetimeIndex, h: int) -> pd.Series:
    pos = np.arange(len(index)) + h
    ends = [index[p] if p < len(index) else pd.NaT for p in pos]
    return pd.Series(ends, index=index)


def forward_max_drawdown(close: pd.Series, h: int) -> pd.Series:
    """Worst peak-to-trough decline within the next h sessions (starting from today's close)."""
    c = close.values
    out = np.full(len(c), np.nan)
    for i in range(len(c) - h):
        path = np.concatenate([[c[i]], c[i + 1: i + h + 1]])
        out[i] = (path / np.maximum.accumulate(path) - 1).min()
    return pd.Series(out, index=close.index)


def triple_barrier(close: pd.Series, h: int, vol: pd.Series | None = None, up_mult: float = 1.5,
                   dn_mult: float = 1.5, min_barrier: float = 0.01) -> pd.DataFrame:
    """Triple-barrier labels (Lopez de Prado).

    Barriers scale with trailing daily volatility * sqrt(h). Returns label in {1 (UP),
    -1 (DOWN), 0 (TIMEOUT)}, the touch time and the realised return.
    """
    r = np.log(close).diff()
    vol = vol if vol is not None else r.ewm(span=60, min_periods=20).std()
    width = (vol * np.sqrt(h)).clip(lower=min_barrier)
    c = close.values
    n = len(c)
    lab = np.full(n, np.nan)
    t_touch = np.full(n, np.nan)
    ret = np.full(n, np.nan)
    for i in range(n - h):
        w = width.iloc[i]
        if np.isnan(w):
            continue
        path = c[i + 1: i + h + 1] / c[i] - 1
        up_hit = np.where(path >= up_mult * w)[0]
        dn_hit = np.where(path <= -dn_mult * w)[0]
        fu = up_hit[0] if len(up_hit) else h + 1
        fd = dn_hit[0] if len(dn_hit) else h + 1
        if fu == fd == h + 1:
            lab[i], t_touch[i], ret[i] = 0, h, path[-1]
        elif fu < fd:
            lab[i], t_touch[i], ret[i] = 1, fu + 1, path[fu]
        else:
            lab[i], t_touch[i], ret[i] = -1, fd + 1, path[fd]
    return pd.DataFrame({"tb_label": lab, "tb_t": t_touch, "tb_ret": ret, "tb_width": width}, index=close.index)


def build_labels(df: pd.DataFrame, horizons: list[int]) -> dict[int, pd.DataFrame]:
    out = {}
    for h in horizons:
        fr = forward_return(df, h)
        lab = pd.DataFrame(index=df.index)
        lab["fwd_ret"] = fr
        lab["direction"] = np.where(fr.isna(), np.nan, (fr > 0).astype(float))
        lab["bucket"] = pd.cut(fr, RETURN_BUCKETS, labels=False)
        lab["gt5"] = np.where(fr.isna(), np.nan, (fr > 0.05).astype(float))
        lab["lt_5"] = np.where(fr.isna(), np.nan, (fr < -0.05).astype(float))
        mdd = forward_max_drawdown(df["close"], h)
        lab["fwd_mdd"] = mdd
        lab["dd10"] = np.where(mdd.isna(), np.nan, (mdd < -0.10).astype(float))
        tb = triple_barrier(df["close"], h)
        lab = lab.join(tb)
        lab["label_end"] = label_end_dates(df.index, h + 1)  # +1: execution at next open
        out[h] = lab
    return out
