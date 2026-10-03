"""Market-breadth features.

True constituent-level breadth (advance/decline line, % of 500 members above SMA200,
new highs/lows) requires a constituent data licence. Without it the application uses
transparent *proxies* built from the 11 SPDR sector ETFs and the equal-weight/cap-weight
ratio. Feature names carry ``_proxy`` so the user always knows what was measured.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..cross_asset.cross import aligned_close


def breadth_features(sectors: dict[str, pd.DataFrame], decision_times: pd.DatetimeIndex, index: pd.DatetimeIndex
                     ) -> tuple[pd.DataFrame, dict[str, str]]:
    if not sectors:
        return pd.DataFrame(index=index), {}
    closes = pd.DataFrame({k: aligned_close(sectors, k, decision_times, index) for k in sectors})
    valid = closes.notna().sum(axis=1)
    enough = valid >= max(5, int(0.6 * closes.shape[1]))
    f = {}
    for n in (50, 200):
        above = (closes > closes.rolling(n, min_periods=n).mean()).astype(float).where(closes.rolling(n, min_periods=n).mean().notna())
        f[f"sector_pct_above_sma{n}_proxy"] = above.mean(axis=1).where(enough)
    rets = closes.pct_change()
    adv = (rets > 0).sum(axis=1)
    dec = (rets < 0).sum(axis=1)
    ad = (adv - dec).where(enough)
    f["sector_adv_decl_proxy"] = ad / valid.replace(0, np.nan)
    f["sector_ad_line_slope_proxy"] = ad.cumsum().diff(20) / 20
    hi = (closes >= closes.rolling(252, min_periods=126).max()).sum(axis=1)
    lo = (closes <= closes.rolling(252, min_periods=126).min()).sum(axis=1)
    f["sector_new_highs_proxy"] = hi.where(enough) / valid
    f["sector_new_lows_proxy"] = lo.where(enough) / valid
    r20 = closes.pct_change(20)
    f["sector_dispersion_20d"] = r20.std(axis=1).where(enough)
    defensives = [c for c in ("XLU", "XLP", "XLV") if c in closes]
    cyclicals = [c for c in ("XLY", "XLK", "XLI", "XLF") if c in closes]
    if defensives and cyclicals:
        f["cyclical_vs_defensive_60d"] = closes[cyclicals].pct_change(60).mean(axis=1) - \
            closes[defensives].pct_change(60).mean(axis=1)
    out = pd.DataFrame(f, index=index)
    return out, {k: "SPDR sector ETFs (proxy)" for k in out.columns}
