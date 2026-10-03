"""Trading calendars and market-close alignment.

Holiday rules are computed algorithmically (no network needed) for the exchanges the
application uses: XETRA / Euronext / London / Milan for VUSA listings and NYSE for the
US reference indices. When ``pandas_market_calendars`` is installed it is preferred,
otherwise these built-in rules are used. Ad-hoc closures (e.g. national mourning days)
cannot be predicted - the data-validation layer reports sessions that are missing from
the data feed rather than inventing prices.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import pandas as pd

EXCHANGE_INFO = {
    "XETRA": {"tz": "Europe/Berlin", "close": time(17, 30), "open": time(9, 0)},
    "XAMS": {"tz": "Europe/Amsterdam", "close": time(17, 30), "open": time(9, 0)},
    "XMIL": {"tz": "Europe/Rome", "close": time(17, 30), "open": time(9, 0)},
    "XLON": {"tz": "Europe/London", "close": time(16, 30), "open": time(8, 0)},
    "NYSE": {"tz": "America/New_York", "close": time(16, 0), "open": time(9, 30)},
}


def easter_sunday(year: int) -> date:
    """Anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    while d.weekday() != weekday:
        d += timedelta(days=1)
    return d + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    d = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    while d.weekday() != weekday:
        d -= timedelta(days=1)
    return d


def _observed(d: date) -> date:
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=256)
def holidays(exchange: str, year: int) -> frozenset[date]:
    e = easter_sunday(year)
    h: set[date] = set()
    if exchange in ("XETRA", "XAMS", "XMIL"):
        h |= {date(year, 1, 1), e - timedelta(days=2), e + timedelta(days=1), date(year, 5, 1),
              date(year, 12, 24), date(year, 12, 25), date(year, 12, 26), date(year, 12, 31)}
        if exchange == "XMIL":
            h |= {date(year, 8, 15)}
        if exchange == "XAMS":
            h -= {date(year, 12, 24), date(year, 12, 31)}  # Euronext: early close, not closed
    elif exchange == "XLON":
        h |= {_observed(date(year, 1, 1)), e - timedelta(days=2), e + timedelta(days=1),
              _nth_weekday(year, 5, 0, 1), _last_weekday(year, 5, 0), _last_weekday(year, 8, 0)}
        xmas, box = date(year, 12, 25), date(year, 12, 26)
        if xmas.weekday() >= 5:
            xmas = xmas + timedelta(days=(7 - xmas.weekday()))
        if box.weekday() >= 5 or box == xmas:
            box = xmas + timedelta(days=1)
            while box.weekday() >= 5:
                box += timedelta(days=1)
        h |= {xmas, box}
    elif exchange == "NYSE":
        h |= {_observed(date(year, 1, 1)), _nth_weekday(year, 1, 0, 3), _nth_weekday(year, 2, 0, 3),
              e - timedelta(days=2), _last_weekday(year, 5, 0), _observed(date(year, 7, 4)),
              _nth_weekday(year, 9, 0, 1), _nth_weekday(year, 11, 3, 4), _observed(date(year, 12, 25))}
        if year >= 2022:
            h.add(_observed(date(year, 6, 19)))
        # New Year's Day falling on Saturday is NOT observed on the previous Friday by NYSE
        if date(year, 1, 1).weekday() == 5:
            h.discard(date(year - 1, 12, 31))
    return frozenset(d for d in h if d.weekday() < 5)


def is_trading_day(d: date | datetime | pd.Timestamp, exchange: str = "XETRA") -> bool:
    d = pd.Timestamp(d).date()
    if d.weekday() >= 5:
        return False
    return d not in holidays(exchange, d.year)


def trading_days(start, end, exchange: str = "XETRA") -> pd.DatetimeIndex:
    days = pd.bdate_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize())
    return pd.DatetimeIndex([d for d in days if is_trading_day(d, exchange)])


def next_trading_day(d, exchange: str = "XETRA") -> pd.Timestamp:
    d = pd.Timestamp(d).normalize() + pd.Timedelta(days=1)
    while not is_trading_day(d, exchange):
        d += pd.Timedelta(days=1)
    return d


def previous_trading_day(d, exchange: str = "XETRA") -> pd.Timestamp:
    d = pd.Timestamp(d).normalize() - pd.Timedelta(days=1)
    while not is_trading_day(d, exchange):
        d -= pd.Timedelta(days=1)
    return d


def add_trading_days(d, n: int, exchange: str = "XETRA") -> pd.Timestamp:
    cur = pd.Timestamp(d).normalize()
    for _ in range(n):
        cur = next_trading_day(cur, exchange)
    return cur


def market_close_utc(d, exchange: str = "XETRA") -> pd.Timestamp:
    """UTC timestamp of the official close of session ``d`` (timezone normalised)."""
    info = EXCHANGE_INFO[exchange]
    local = datetime.combine(pd.Timestamp(d).date(), info["close"], tzinfo=ZoneInfo(info["tz"]))
    return pd.Timestamp(local).tz_convert("UTC")


def market_status(now: datetime | None = None, exchange: str = "XETRA") -> dict:
    info = EXCHANGE_INFO[exchange]
    tz = ZoneInfo(info["tz"])
    now = (now or datetime.now(tz)).astimezone(tz)
    today = now.date()
    trading = is_trading_day(today, exchange)
    if not trading:
        state = "closed (weekend)" if today.weekday() >= 5 else "closed (holiday)"
    elif now.time() < info["open"]:
        state = "pre-open"
    elif now.time() <= info["close"]:
        state = "open"
    else:
        state = "closed (after close)"
    last_complete = today if (trading and now.time() > info["close"]) else previous_trading_day(today, exchange).date()
    return {"exchange": exchange, "local_time": now.isoformat(timespec="minutes"), "is_trading_day": trading,
            "state": state, "last_complete_session": str(last_complete),
            "next_session": str(next_trading_day(today, exchange).date())}
