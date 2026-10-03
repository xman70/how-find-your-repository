"""FeatureService: assembles every feature block, records lineage, and guarantees each
row is computed only from information available at that row's decision time."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.runtime import stable_hash, utcnow_iso
from ..data.calendar import market_close_utc
from ..data.service import MarketDataBundle
from .breadth.breadth import breadth_features
from .cross_asset.cross import cross_asset_features
from .macro.macro_features import macro_features
from .sentiment.sentiment_features import MIN_COVERAGE_DAYS, sentiment_features
from .technical.indicators import technical_features

FEATURE_VERSION = "fv3"
GROUP_PREFIXES = {
    "technical": None, "volatility": ("rv_", "atr", "parkinson", "garman", "rogers", "vol_of_vol", "downside_vol",
                                      "skew_", "kurt_", "rv20", "rv_ratio"),
    "macro": None, "sentiment": ("news_",), "breadth": ("sector_", "cyclical_"), "cross_asset": None,
}
NON_MODEL_PREFIXES = ("staleness_",)


@dataclass
class FeatureSet:
    frame: pd.DataFrame
    groups: dict[str, list[str]]
    lineage: dict[str, dict]
    decision_times: pd.DatetimeIndex
    version: str
    dataset_version: str
    sentiment_coverage: int = 0
    notes: list[str] = field(default_factory=list)

    def model_columns(self, exclude_groups: tuple[str, ...] = ()) -> list[str]:
        cols = []
        for g, cs in self.groups.items():
            if g in exclude_groups:
                continue
            cols += [c for c in cs if not c.startswith(NON_MODEL_PREFIXES)]
        return cols

    def lineage_table(self, as_of=None) -> pd.DataFrame:
        row = self.frame.loc[:as_of].iloc[-1] if as_of is not None else self.frame.iloc[-1]
        recs = []
        for name, meta in self.lineage.items():
            orig = row.get(name, np.nan)
            col = self.frame[name].loc[:row.name].dropna().tail(1260)
            z = (orig - col.mean()) / col.std() if len(col) > 30 and col.std() > 0 else np.nan
            recs.append({"feature": name, "group": meta["group"], "source": meta["source"],
                         "calculated_at": meta["calculated_at"], "transformation": meta["transformation"],
                         "original_value": orig, "normalized_value": z,
                         "missing_status": "missing" if pd.isna(orig) else "ok"})
        return pd.DataFrame(recs)


def decision_timestamps(index: pd.DatetimeIndex, exchange: str, buffer_minutes: int = 60) -> pd.DatetimeIndex:
    """Decision time of each session = official close + buffer (UTC)."""
    return pd.DatetimeIndex([market_close_utc(d, exchange) + pd.Timedelta(minutes=buffer_minutes) for d in index])


def build_features(bundle: MarketDataBundle, exchange: str = "XETRA", news_daily: pd.DataFrame | None = None,
                   include: tuple[str, ...] = ("technical", "cross_asset", "breadth", "macro", "sentiment")
                   ) -> FeatureSet:
    tgt = bundle.target
    idx = tgt.index
    dts = decision_timestamps(idx, exchange)
    if "is_proxy" in tgt and tgt["is_proxy"].any():
        # proxy rows are S&P 500 (EUR) closes -> decided after the NYSE close of that day
        proxy = tgt["is_proxy"].fillna(0).astype(bool).values
        dts_us = decision_timestamps(idx[proxy], "NYSE")
        vals = dts.values.copy()
        vals[proxy] = dts_us.values
        dts = pd.DatetimeIndex(vals).tz_localize("UTC") if pd.DatetimeIndex(vals).tz is None else pd.DatetimeIndex(vals)
    now = utcnow_iso()
    blocks, lineage, groups, notes = [], {}, {}, []

    tech = technical_features(tgt)
    vol_cols = [c for c in tech.columns if c.startswith(GROUP_PREFIXES["volatility"])]
    groups["technical"] = [c for c in tech.columns if c not in vol_cols]
    groups["volatility"] = vol_cols
    blocks.append(tech)
    for c in tech.columns:
        lineage[c] = {"group": "volatility" if c in vol_cols else "technical", "source": f"VUSA OHLCV ({bundle.target_symbol})",
                      "transformation": _describe(c), "calculated_at": now}

    if "cross_asset" in include and bundle.assets:
        ca, src = cross_asset_features(bundle.assets, tgt["close"], dts)
        blocks.append(ca)
        groups["cross_asset"] = list(ca.columns)
        for c in ca.columns:
            lineage[c] = {"group": "cross_asset", "source": src[c], "transformation": _describe(c), "calculated_at": now}
    if "breadth" in include and bundle.sectors:
        br, src = breadth_features(bundle.sectors, dts, idx)
        blocks.append(br)
        groups["breadth"] = list(br.columns)
        for c in br.columns:
            lineage[c] = {"group": "breadth", "source": src[c], "transformation": _describe(c), "calculated_at": now}
    if "macro" in include and bundle.pit.series_ids:
        mf, src = macro_features(bundle.pit, dts, idx)
        blocks.append(mf)
        groups["macro"] = list(mf.columns)
        for c in mf.columns:
            lineage[c] = {"group": "macro", "source": src[c], "transformation": "point-in-time as-of join"
                          + (" + derived" if "derived" in src[c] else ""), "calculated_at": now}
    cov = 0
    if "sentiment" in include and news_daily is not None and not news_daily.empty:
        sf, src, cov = sentiment_features(news_daily, dts, idx)
        if cov >= MIN_COVERAGE_DAYS:
            blocks.append(sf)
            groups["sentiment"] = list(sf.columns)
            for c in sf.columns:
                lineage[c] = {"group": "sentiment", "source": src[c], "transformation": _describe(c), "calculated_at": now}
        else:
            notes.append(f"News sentiment has only {cov} days of history (< {MIN_COVERAGE_DAYS}); excluded from model "
                         "training, used only for the current-day assessment.")
    frame = pd.concat(blocks, axis=1)
    frame = frame.loc[:, ~frame.columns.duplicated()].astype(float)
    if "is_proxy" in tgt:
        frame["is_proxy_history"] = tgt["is_proxy"].astype(float)
        lineage["is_proxy_history"] = {"group": "meta", "source": "data service", "transformation": "flag",
                                       "calculated_at": now}
    groups = {g: [c for c in cs if c in frame.columns] for g, cs in groups.items()}
    return FeatureSet(frame=frame, groups=groups, lineage=lineage, decision_times=dts, version=FEATURE_VERSION,
                      dataset_version=stable_hash(tgt[["close"]].tail(500)), sentiment_coverage=cov, notes=notes)


def _describe(name: str) -> str:
    rules = [
        ("ret_", "simple return over n sessions"), ("logret", "log return"), ("dist_sma", "close / SMA(n) - 1"),
        ("dist_ema", "close / EMA(n) - 1"), ("dist_hma", "close / Hull MA - 1"), ("trend_slope", "annualised OLS slope of log price"),
        ("rsi", "Wilder RSI"), ("macd", "MACD / price"), ("roc_", "rate of change"), ("stoch", "stochastic oscillator"),
        ("williams", "Williams %R"), ("cci", "commodity channel index"), ("rv_", "annualised realised volatility (close-close)"),
        ("parkinson", "Parkinson high-low volatility"), ("garman", "Garman-Klass volatility"),
        ("rogers", "Rogers-Satchell volatility"), ("zscore_5y", "rolling 5-year z-score"), ("_z_", "rolling z-score"),
        ("_pct_5y", "rolling 5-year percentile"), ("drawdown", "close / running max - 1"), ("corr_", "rolling 60d correlation of log returns"),
        ("vix_term", "VIX3M / VIX - 1"), ("vrp", "VIX/100 - realised vol 20d"), ("adx", "Wilder ADX"),
        ("obv", "on-balance volume slope"), ("volume", "volume statistic"), ("news_", "weighted news sentiment"),
        ("sector_", "sector-ETF breadth proxy"), ("atr", "ATR / price"), ("gap", "log(open / prev close)"),
    ]
    for k, v in rules:
        if k in name:
            return v
    return "see features module"
