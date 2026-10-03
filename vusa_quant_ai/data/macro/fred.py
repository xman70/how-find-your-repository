"""FRED / ALFRED macro data with point-in-time handling.

* With a FRED API key, true real-time vintages are downloaded from ALFRED
  (``realtime_start``/``realtime_end``) so each value carries its actual publication date
  and revisions are tracked.
* Without a key, the public ``fredgraph.csv`` endpoint gives only the *latest* vintage.
  Publication time is then approximated with a conservative release lag per series, and
  the leakage audit flags the residual revision risk. This is reported, never hidden.
"""
from __future__ import annotations

import io

import pandas as pd

from ..point_in_time.engine import PointInTimeStore
from ..sources.base import ProviderUnavailable, http_get

# series_id -> (description, conservative publication lag in days after period end, revised?)
MACRO_SERIES = {
    "DGS10": ("10-Year Treasury constant maturity", 1, False),
    "DGS2": ("2-Year Treasury constant maturity", 1, False),
    "DGS3MO": ("3-Month Treasury constant maturity", 1, False),
    "T10Y2Y": ("10Y-2Y Treasury spread", 1, False),
    "T10Y3M": ("10Y-3M Treasury spread", 1, False),
    "DFII10": ("10-Year TIPS real yield", 1, False),
    "T10YIE": ("10-Year breakeven inflation", 1, False),
    "DFF": ("Effective federal funds rate", 1, False),
    "BAMLH0A0HYM2": ("ICE BofA US High Yield OAS", 1, False),
    "BAMLC0A0CM": ("ICE BofA US Corporate OAS", 1, False),
    "NFCI": ("Chicago Fed National Financial Conditions Index", 5, True),
    "CPIAUCSL": ("CPI all items (SA)", 16, True),
    "CPILFESL": ("Core CPI (SA)", 16, True),
    "PCEPILFE": ("Core PCE price index", 32, True),
    "UNRATE": ("Unemployment rate", 8, True),
    "PAYEMS": ("Nonfarm payrolls", 8, True),
    "ICSA": ("Initial jobless claims", 6, True),
    "INDPRO": ("Industrial production", 17, True),
    "GDPC1": ("Real GDP", 32, True),
    "UMCSENT": ("UMich consumer sentiment", 3, True),
    "WALCL": ("Fed total assets (liquidity)", 2, False),
    "M2SL": ("M2 money stock", 28, True),
    "DEXUSEU": ("USD per EUR", 2, False),
    "SAHMREALTIME": ("Sahm rule real-time recession indicator", 8, False),
    "USREC": ("NBER recession indicator (published with long delay)", 365, True),
}


class FredApiProvider:
    name = "fred_api"

    def __init__(self, api_key: str, timeout: float = 20.0):
        self.api_key, self.timeout = api_key, timeout

    def vintages(self, series_id: str, start: str = "1995-01-01") -> pd.DataFrame:
        if not self.api_key:
            raise ProviderUnavailable("FRED API key not configured (ALFRED vintages unavailable)")
        js = http_get("https://api.stlouisfed.org/fred/series/observations", timeout=self.timeout, params={
            "series_id": series_id, "api_key": self.api_key, "file_type": "json", "observation_start": start,
            "realtime_start": "1776-07-04", "realtime_end": "9999-12-31"}).json()
        if "observations" not in js:
            raise ProviderUnavailable(f"FRED: {js.get('error_message', 'no observations')}")
        df = pd.DataFrame(js["observations"])
        df = df[df["value"] != "."]
        if df.empty:
            raise ProviderUnavailable(f"FRED returned no numeric values for {series_id}")
        lag = MACRO_SERIES.get(series_id, ("", 1, True))[1]
        out = pd.DataFrame({
            "effective_date": pd.to_datetime(df["date"]),
            "value": pd.to_numeric(df["value"], errors="coerce"),
            # realtime_start is the vintage date (US). Releases are assumed public by 14:00 UTC.
            "published_at": pd.to_datetime(df["realtime_start"]) + pd.Timedelta(hours=14),
            "vintage": df["realtime_start"],
        })
        # ALFRED's first vintage for very old observations is the series' creation date,
        # which can be far later than true publication; floor at effective date + lag is NOT
        # applied (later is safe, earlier would leak).
        out["published_at"] = out["published_at"].where(
            out["published_at"] >= out["effective_date"] + pd.Timedelta(days=min(lag, 1)),
            out["effective_date"] + pd.Timedelta(days=lag))
        return out


class FredCsvProvider:
    name = "fred_csv"

    def __init__(self, timeout: float = 20.0):
        self.timeout = timeout

    def latest(self, series_id: str, start: str = "1995-01-01") -> pd.Series:
        r = http_get("https://fred.stlouisfed.org/graph/fredgraph.csv", timeout=self.timeout,
                     params={"id": series_id, "cosd": start})
        df = pd.read_csv(io.StringIO(r.text))
        if df.shape[1] < 2:
            raise ProviderUnavailable(f"FRED csv malformed for {series_id}")
        df.columns = ["date", "value"]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        s = df.dropna().set_index(pd.to_datetime(df.dropna()["date"]))["value"]
        if s.empty:
            raise ProviderUnavailable(f"FRED csv empty for {series_id}")
        return s


def load_macro_into_store(store: PointInTimeStore, series: list[str], api_key: str, priority: list[str],
                          timeout: float, start: str, health=None) -> dict[str, str]:
    """Load every requested series using the configured provider priority. Returns status per series."""
    from ..sources.base import run_with_fallback

    status: dict[str, str] = {}
    api, csv = FredApiProvider(api_key, timeout), FredCsvProvider(timeout)
    for sid in series:
        lag = MACRO_SERIES.get(sid, ("", 30, True))[1]
        attempts = []
        for p in priority:
            if p == "fred_api":
                attempts.append(("fred_api", lambda sid=sid: api.vintages(sid, start)))
            elif p == "fred_csv":
                attempts.append(("fred_csv", lambda sid=sid: csv.latest(sid, start)))
        res, _ = run_with_fallback(f"macro:{sid}", attempts, health)
        if not res.ok:
            status[sid] = f"unavailable ({res.error[:120]})"
            continue
        if res.source == "fred_api":
            store.add_records(sid, res.data, "vintage", "FRED/ALFRED")
            status[sid] = "ok (true vintages)"
        else:
            store.add_latest_with_lag(sid, res.data, lag, "FRED csv (latest vintage)")
            status[sid] = f"ok (latest vintage, {lag}d release-lag approximation)"
    return status
