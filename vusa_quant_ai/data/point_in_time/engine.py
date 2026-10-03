"""POINT-IN-TIME DATA ENGINE.

Every observation is stored with:

* ``effective_date``  - the period the number describes (e.g. CPI for 2024-03)
* ``published_at``    - when the number became public (UTC)
* ``retrieved_at``    - when this application downloaded it
* ``vintage``         - revision identifier where available
* ``source`` / ``pit_method`` - how publication time was determined

``pit_method`` values:

* ``vintage``         - true real-time vintages (ALFRED); fully point-in-time incl. revisions
* ``release_lag``     - publication time approximated by a conservative release lag applied
                        to the *latest* vintage. Timing is safe, but later revisions may leak
                        - this is reported explicitly in the leakage audit.
* ``market_close``    - market data, available at the official session close
* ``timestamp``       - news with an explicit publication timestamp

``value_as_of(t)`` returns only information whose ``published_at <= t``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PITRecord:
    series_id: str
    effective_date: pd.Timestamp
    value: float
    published_at: pd.Timestamp
    vintage: str = ""
    pit_method: str = "vintage"
    source: str = ""


def to_utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


class PointInTimeStore:
    def __init__(self):
        self._frames: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------ ingest
    def add_records(self, series_id: str, df: pd.DataFrame, pit_method: str, source: str) -> None:
        """``df`` must have columns effective_date, value, published_at (+ optional vintage)."""
        d = df.copy()
        d["effective_date"] = pd.to_datetime(d["effective_date"])
        d["published_at"] = pd.to_datetime(d["published_at"], utc=True)
        d["value"] = pd.to_numeric(d["value"], errors="coerce")
        d = d.dropna(subset=["value"])
        if "vintage" not in d:
            d["vintage"] = ""
        d["pit_method"] = pit_method
        d["source"] = source
        d = d[["effective_date", "value", "published_at", "vintage", "pit_method", "source"]]
        if series_id in self._frames:
            d = pd.concat([self._frames[series_id], d])
        d = d.drop_duplicates(["effective_date", "published_at"], keep="last")
        self._frames[series_id] = d.sort_values(["published_at", "effective_date"]).reset_index(drop=True)

    def add_latest_with_lag(self, series_id: str, s: pd.Series, lag_days: int, source: str,
                            release_hour_utc: int = 13) -> None:
        """Latest-vintage series -> approximate publication = effective period end + lag."""
        eff = pd.to_datetime(s.index)
        period_end = _period_end(eff, s)
        pub = (period_end + pd.to_timedelta(lag_days, unit="D")).normalize() + pd.Timedelta(hours=release_hour_utc)
        self.add_records(series_id, pd.DataFrame({"effective_date": eff, "value": s.values, "published_at": pub}),
                         "release_lag", source)

    # ------------------------------------------------------------------ queries
    @property
    def series_ids(self) -> list[str]:
        return list(self._frames)

    def method(self, series_id: str) -> str:
        f = self._frames.get(series_id)
        return "" if f is None or f.empty else str(f["pit_method"].iloc[-1])

    def records(self, series_id: str) -> pd.DataFrame:
        return self._frames.get(series_id, pd.DataFrame()).copy()

    def history_as_of(self, series_id: str, as_of) -> pd.Series:
        """Full history of a series exactly as it was known at ``as_of`` (latest vintage per period)."""
        f = self._frames.get(series_id)
        if f is None:
            return pd.Series(dtype=float)
        t = to_utc(as_of)
        known = f[f["published_at"] <= t]
        known = known.sort_values("published_at").drop_duplicates("effective_date", keep="last")
        return known.set_index("effective_date")["value"].sort_index()

    def value_as_of(self, series_id: str, as_of) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None]:
        h = self.history_as_of(series_id, as_of)
        if h.empty:
            return np.nan, None, None
        f = self._frames[series_id]
        eff = h.index[-1]
        pub = f[(f["effective_date"] == eff) & (f["published_at"] <= to_utc(as_of))]["published_at"].max()
        return float(h.iloc[-1]), eff, pub

    def daily_view(self, series_id: str, decision_times: pd.Series | pd.DatetimeIndex,
                   transform: str = "level", periods: int = 12) -> pd.Series:
        """Value known at each decision time (sweep algorithm, O(n_records + n_days)).

        ``transform``: ``level`` | ``yoy`` (pct change vs ``periods`` earlier, computed on the
        vintage known at that time) | ``diff`` (change vs ``periods`` earlier).
        """
        f = self._frames.get(series_id)
        times = pd.DatetimeIndex(pd.to_datetime(decision_times, utc=True))
        if f is None or f.empty:
            return pd.Series(np.nan, index=times)
        pubs = f["published_at"].values
        effs = f["effective_date"].values
        vals = f["value"].values
        import bisect

        known: dict = {}
        keys: list = []  # sorted effective dates known so far
        out = np.full(len(times), np.nan)
        order = np.argsort(times.values)
        j = 0
        for i in order:
            t = times.values[i]
            while j < len(pubs) and pubs[j] <= t:
                if effs[j] not in known:
                    bisect.insort(keys, effs[j])
                known[effs[j]] = vals[j]
                j += 1
            if not known:
                continue
            if transform == "level":
                out[i] = known[keys[-1]]
            else:
                if len(keys) > periods:
                    cur, prev = known[keys[-1]], known[keys[-1 - periods]]
                    if transform == "yoy" and prev != 0 and not np.isnan(prev):
                        out[i] = cur / prev - 1.0
                    elif transform == "diff":
                        out[i] = cur - prev
        return pd.Series(out, index=times)

    def staleness_days(self, series_id: str, decision_times: pd.DatetimeIndex) -> pd.Series:
        """Days since the last publication at each decision time (data freshness feature)."""
        f = self._frames.get(series_id)
        times = pd.DatetimeIndex(pd.to_datetime(decision_times, utc=True))
        if f is None or f.empty:
            return pd.Series(np.nan, index=times)
        pubs = np.sort(f["published_at"].values)
        pos = np.searchsorted(pubs, times.values, side="right") - 1
        res = np.where(pos >= 0, (times.values - pubs[np.clip(pos, 0, None)]) / np.timedelta64(1, "D"), np.nan)
        return pd.Series(res, index=times)


def _period_end(eff: pd.DatetimeIndex, s: pd.Series) -> pd.DatetimeIndex:
    freq = pd.infer_freq(eff[-min(len(eff), 24):]) if len(eff) >= 3 else None
    if freq and freq.startswith(("M", "MS")):
        return eff + pd.offsets.MonthEnd(0)
    if freq and freq.startswith(("Q", "QS")):
        return eff + pd.offsets.QuarterEnd(0)
    if freq and freq.startswith(("A", "Y")):
        return eff + pd.offsets.YearEnd(0)
    if freq and freq.startswith("W"):
        return eff + pd.Timedelta(days=6)
    gaps = np.median(np.diff(eff.values).astype("timedelta64[D]").astype(int)) if len(eff) > 2 else 1
    if gaps >= 80:
        return eff + pd.offsets.QuarterEnd(0)
    if gaps >= 25:
        return eff + pd.offsets.MonthEnd(0)
    if gaps >= 6:
        return eff + pd.Timedelta(days=6)
    return eff


def align_market_to_decision(df: pd.DataFrame, decision_times: pd.DatetimeIndex) -> pd.DataFrame:
    """As-of join of a market frame (with ``available_at``) onto decision timestamps.

    Used to align US-index closes (16:00 New York) to European decision times so that a
    US close that happens *after* a European decision is not used for that decision.
    """
    if df.empty:
        return pd.DataFrame(index=decision_times)
    left = pd.DataFrame({"decision_time": pd.to_datetime(decision_times, utc=True)})
    right = df.copy()
    right["available_at"] = pd.to_datetime(right["available_at"], utc=True)
    right = right.sort_values("available_at")
    right["source_date"] = right.index
    merged = pd.merge_asof(left.sort_values("decision_time"), right.reset_index(drop=True),
                           left_on="decision_time", right_on="available_at", direction="backward")
    merged.index = decision_times
    return merged.drop(columns=["decision_time"])
