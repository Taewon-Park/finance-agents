"""NYSE 거래일 캘린더 (exchange_calendars XNYS)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd

_CALENDAR_START = "2000-01-01"


@lru_cache(maxsize=1)
def _nyse() -> xcals.ExchangeCalendar:
    return xcals.get_calendar("XNYS", start=_CALENDAR_START)


def nyse_sessions(start: date, end: date) -> pd.DatetimeIndex:
    """[start, end] 구간의 NYSE 거래일 (자정 기준, tz 없음)."""
    sessions = _nyse().sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    return pd.DatetimeIndex(sessions).tz_localize(None).astype("datetime64[ns]")


def last_completed_session(now: datetime | None = None, buffer_minutes: int = 30) -> date:
    """장 마감 후 buffer_minutes가 지난 가장 최근 거래일.

    장중에 수집하면 당일 봉이 미완성이므로 저장하지 않기 위해 쓴다.
    """
    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    cal = _nyse()
    candidates = cal.sessions_in_range(
        pd.Timestamp(now_utc.date() - timedelta(days=14)), pd.Timestamp(now_utc.date())
    )
    cutoff = pd.Timestamp(now_utc) - pd.Timedelta(minutes=buffer_minutes)
    for session in reversed(candidates):
        if cal.session_close(session) <= cutoff:
            return session.date()
    raise RuntimeError("최근 14일 안에 마감된 거래일이 없습니다")
