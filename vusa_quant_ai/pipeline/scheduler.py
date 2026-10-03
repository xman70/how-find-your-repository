"""Daily schedule logic shared by the GUI timer and the headless runner.

* Runs only on trading days of the configured exchange (weekends / holidays skipped -
  the system never pretends a non-existent session happened).
* Runs once per completed session, at the configured local time (default after close).
"""
from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from ..data.calendar import EXCHANGE_INFO, is_trading_day, next_trading_day


def next_run_time(now: datetime, daily_time: str, tz: str, exchange: str) -> datetime:
    z = ZoneInfo(tz)
    now = now.astimezone(z)
    hh, mm = (int(x) for x in daily_time.split(":"))
    cand = datetime.combine(now.date(), time(hh, mm), tzinfo=z)
    if cand <= now or not is_trading_day(cand.date(), exchange):
        d = next_trading_day(now.date(), exchange).date()
        cand = datetime.combine(d, time(hh, mm), tzinfo=z)
    # never earlier than the exchange close on that day
    info = EXCHANGE_INFO[exchange]
    close_local = datetime.combine(cand.date(), info["close"], tzinfo=ZoneInfo(info["tz"])).astimezone(z)
    if cand < close_local + timedelta(minutes=15):
        cand = close_local + timedelta(minutes=15)
    return cand


def should_run_now(now: datetime, last_run_session: str | None, daily_time: str, tz: str, exchange: str) -> tuple[bool, str]:
    z = ZoneInfo(tz)
    now = now.astimezone(z)
    if not is_trading_day(now.date(), exchange):
        return False, f"{now.date()} is not a trading day on {exchange} (weekend/holiday) - no new session to analyse"
    hh, mm = (int(x) for x in daily_time.split(":"))
    if now.time() < time(hh, mm):
        return False, f"before scheduled time {daily_time}"
    if last_run_session == str(now.date()):
        return False, "already analysed today's session"
    return True, "scheduled run due"
