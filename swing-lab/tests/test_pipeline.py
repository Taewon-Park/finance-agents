"""수집(증분·재수집), 정제 점검, 캘린더, 소스 어댑터 — 네트워크 없이 검증한다."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from swinglab.config import load_backtest_config, load_universe
from swinglab.pipeline.calendar import last_completed_session, nyse_sessions
from swinglab.pipeline.clean import run_checks
from swinglab.pipeline.ingest import ingest
from swinglab.pipeline.storage import clean_path, raw_path, read_prices
from swinglab.sources import yfinance_source
from swinglab.sources.base import FetchError, normalize_prices
from swinglab.sources.yfinance_source import YFinanceSource

PROJECT = Path(__file__).resolve().parents[1]


class FakeSource:
    """날짜 범위 안의 저장된 '수정주가'를 돌려준다. factor로 배당 반영(과거 값 변경)을 흉내 낸다."""

    name = "fake"

    def __init__(self, tickers: list[str], sessions: pd.DatetimeIndex) -> None:
        self.sessions = sessions
        self.base = {t: 100 + np.arange(len(sessions), dtype=float) + i for i, t in enumerate(tickers)}
        self.factor = 1.0
        self.calls: list[tuple[str, date, date]] = []

    def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        frames = []
        for t in tickers:
            self.calls.append((t, start, end))
            mask = (self.sessions >= pd.Timestamp(start)) & (self.sessions <= pd.Timestamp(end))
            px = self.base[t][mask] * self.factor
            frames.append(
                pd.DataFrame(
                    {"date": self.sessions[mask], "ticker": t, "open": px, "high": px + 1, "low": px - 1,
                     "close": px, "volume": 1000, "source": self.name}
                )
            )
        return normalize_prices(pd.concat(frames))


@pytest.fixture
def sessions() -> pd.DatetimeIndex:
    return nyse_sessions(date(2024, 1, 2), date(2024, 3, 28))


def test_full_then_incremental_ingest(tmp_path: Path, sessions: pd.DatetimeIndex) -> None:
    src = FakeSource(["AAA", "BBB"], sessions)
    first_end = sessions[29].date()
    rep = ingest(["AAA", "BBB"], src, date(2024, 1, 1), first_end, tmp_path)
    assert [r.mode for r in rep.results] == ["full", "full"]
    assert rep.clean_rows == 60

    src.calls.clear()
    second_end = sessions[39].date()
    rep = ingest(["AAA", "BBB"], src, date(2024, 1, 1), second_end, tmp_path)
    assert [r.mode for r in rep.results] == ["incremental", "incremental"]
    assert [r.rows_added for r in rep.results] == [10, 10]
    # 최근 5거래일만 겹쳐서 요청하고, 처음부터 다시 받지 않는다
    assert all(start == sessions[25].date() for _, start, _ in src.calls)

    stored = read_prices(raw_path(tmp_path, "fake", "AAA"))
    assert len(stored) == 40 and stored["date"].is_unique
    np.testing.assert_allclose(stored["close"], src.base["AAA"][:40])
    assert len(read_prices(clean_path(tmp_path))) == 80


def test_up_to_date_does_not_fetch(tmp_path: Path, sessions: pd.DatetimeIndex) -> None:
    src = FakeSource(["AAA"], sessions)
    end = sessions[19].date()
    ingest(["AAA"], src, date(2024, 1, 1), end, tmp_path)
    src.calls.clear()
    rep = ingest(["AAA"], src, date(2024, 1, 1), end, tmp_path)
    assert rep.results[0].mode == "up_to_date"
    assert src.calls == []


def test_adjustment_change_triggers_full_refetch(tmp_path: Path, sessions: pd.DatetimeIndex) -> None:
    src = FakeSource(["AAA"], sessions)
    ingest(["AAA"], src, date(2024, 1, 1), sessions[29].date(), tmp_path)
    src.factor = 0.99  # 새 배당 → 과거 수정주가 전체가 바뀜
    rep = ingest(["AAA"], src, date(2024, 1, 1), sessions[39].date(), tmp_path)
    assert rep.results[0].mode == "refetch"
    stored = read_prices(raw_path(tmp_path, "fake", "AAA"))
    np.testing.assert_allclose(stored["close"], src.base["AAA"][:40] * 0.99)  # 기준이 섞이지 않음


def test_rounding_noise_does_not_trigger_refetch(tmp_path: Path, sessions: pd.DatetimeIndex) -> None:
    src = FakeSource(["AAA"], sessions)
    ingest(["AAA"], src, date(2024, 1, 1), sessions[29].date(), tmp_path)
    src.factor = 1 + 2e-6  # 실측한 Yahoo 반올림 잡음 수준
    rep = ingest(["AAA"], src, date(2024, 1, 1), sessions[39].date(), tmp_path)
    assert rep.results[0].mode == "incremental"


def test_failed_ticker_does_not_stop_others(tmp_path: Path, sessions: pd.DatetimeIndex) -> None:
    src = FakeSource(["AAA"], sessions)
    original = src.fetch

    def flaky(tickers: list[str], start: date, end: date) -> pd.DataFrame:
        if tickers == ["BAD"]:
            raise FetchError("BAD: 실패")
        return original(tickers, start, end)

    src.fetch = flaky  # type: ignore[method-assign]
    rep = ingest(["BAD", "AAA"], src, date(2024, 1, 1), sessions[9].date(), tmp_path)
    assert [r.mode for r in rep.results] == ["failed", "full"]
    assert len(rep.failed) == 1
    assert rep.clean_rows == 10


def test_quality_checks_detect_each_issue(sessions: pd.DatetimeIndex) -> None:
    days = sessions[:10]
    df = pd.DataFrame(
        {"date": days, "ticker": "AAA", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
         "volume": 1000, "source": "t"}
    )
    df.loc[3, "close"] = 130.0  # +30% (그리고 다음 날 -23%)
    df.loc[3, "high"] = 131.0
    df.loc[5, ["high", "low"]] = [98.0, 99.0]  # high < low, close > high
    df.loc[6, "close"] = 98.5  # close < low
    df.loc[8, "volume"] = 0
    df = df.drop(index=7)  # 결측 거래일
    report = run_checks(normalize_prices(df), sessions)

    row = report.summary.loc["AAA"]
    assert row["missing_sessions"] == 1 and report.missing_sessions["AAA"] == [days[7]]
    assert row["return_outliers"] == 1
    assert set(report.bad_ranges["issue"]) == {"high<low close>high", "close<low"}
    assert row["zero_volume"] == 1
    text = report.to_text()
    assert "결측 거래일" in text and "AAA" in text


def test_nyse_calendar_knows_special_closures() -> None:
    s = nyse_sessions(date(2012, 10, 26), date(2012, 11, 2))
    assert pd.Timestamp("2012-10-29") not in s and pd.Timestamp("2012-10-30") not in s  # 허리케인 샌디
    assert pd.Timestamp("2012-10-31") in s


def test_last_completed_session_excludes_open_market() -> None:
    # 2024-01-08(월) 10:00 ET 장중 → 직전 금요일
    assert last_completed_session(datetime(2024, 1, 8, 15, 0, tzinfo=timezone.utc)) == date(2024, 1, 5)
    # 같은 날 17:00 ET(마감 + 1시간) → 당일
    assert last_completed_session(datetime(2024, 1, 8, 22, 0, tzinfo=timezone.utc)) == date(2024, 1, 8)
    # 토요일 → 금요일
    assert last_completed_session(datetime(2024, 1, 6, 18, 0, tzinfo=timezone.utc)) == date(2024, 1, 5)


def test_yfinance_source_normalizes_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="Date")
    good = pd.DataFrame(
        {"Open": [1.0, 2.0], "High": [1.5, 2.5], "Low": [0.5, 1.5], "Close": [1.2, 2.2], "Volume": [10, 20]}, index=idx
    )
    responses = [RuntimeError("timeout"), pd.DataFrame(), good]
    calls: list[dict] = []

    def fake_download(ticker: str, **kwargs):
        calls.append(kwargs)
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(yfinance_source.yf, "download", fake_download)
    src = YFinanceSource(backoff_seconds=0, pause_seconds=0)
    df = src.fetch(["SPY"], date(2024, 1, 2), date(2024, 1, 3))
    assert len(calls) == 3  # 예외, 빈 응답 후 성공
    assert calls[0]["auto_adjust"] is True and calls[0]["end"] == "2024-01-04"  # end 포함 처리
    assert calls[0]["timeout"] == src.timeout_seconds
    assert list(df.columns) == ["date", "ticker", "open", "high", "low", "close", "volume", "source"]
    assert df["source"].unique().tolist() == ["yfinance"]


def test_yfinance_source_raises_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yfinance_source.yf, "download", lambda *a, **k: pd.DataFrame())
    with pytest.raises(FetchError):
        YFinanceSource(backoff_seconds=0, pause_seconds=0).fetch(["XXX"], date(2024, 1, 2), date(2024, 1, 3))


def test_project_configs_match_spec() -> None:
    universe = load_universe(PROJECT / "config")
    assert len(universe.tickers) == 20
    assert universe.benchmark == "SPY" and len(universe.tradable) == 19
    cfg = load_backtest_config(PROJECT / "config")
    assert cfg.cost_bps == 5 and cfg.max_positions == 5 and cfg.cost_sensitivity_bps == [0, 5, 10]
