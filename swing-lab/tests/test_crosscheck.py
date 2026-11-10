"""외부 교차 확인의 갭 보정 계산 검증 (합성 데이터)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from conftest import random_walk_prices

from swinglab.backtest.crosscheck import PublishedReturn, buy_hold_crosscheck


@pytest.fixture
def prices() -> pd.DataFrame:
    # 시가에 갭이 있는 SPY 1300거래일 (2020-01-01부터 약 5년)
    return random_walk_prices(["SPY"], 1300, seed=5)


def _close_to_close_pct(prices: pd.DataFrame, as_of: str, years: int) -> float:
    s = prices.set_index("date")["close"]
    end = s[s.index <= pd.Timestamp(as_of)].iloc[-1]
    start = s[s.index <= pd.Timestamp(as_of) - pd.DateOffset(years=years)].iloc[-1]
    return ((end / start) ** (1 / years) - 1) * 100


def test_gap_adjusted_engine_equals_close_to_close(prices: pd.DataFrame) -> None:
    published = [PublishedReturn(date(2024, 6, 30), y, 0.0) for y in (1, 3)]
    table = buy_hold_crosscheck(prices, "SPY", published)
    for _, row in table.iterrows():
        assert row["engine_adjusted_pct"] == pytest.approx(row["close_to_close_pct"], rel=1e-10)
        # 갭이 0이 아니면 보정 전 엔진 값은 종가→종가와 다르다
        assert row["first_day_gap_pct"] != 0
        assert row["engine_pct"] != pytest.approx(row["close_to_close_pct"], rel=1e-10)


def test_window_uses_last_trading_day_on_or_before_dates(prices: pd.DataFrame) -> None:
    # 2024-06-30은 일요일 → 끝은 6-28(금), 1년 전 2023-06-30(금)이 시작 종가일
    table = buy_hold_crosscheck(prices, "SPY", [PublishedReturn(date(2024, 6, 30), 1, 0.0)])
    row = table.iloc[0]
    assert row["end_date"] == date(2024, 6, 28)
    assert row["start_close_date"] == date(2023, 6, 30)


def test_diff_against_published(prices: pd.DataFrame) -> None:
    truth = _close_to_close_pct(prices, "2024-06-30", 3)
    table = buy_hold_crosscheck(prices, "SPY", [PublishedReturn(date(2024, 6, 30), 3, round(truth + 0.05, 2))])
    assert table.iloc[0]["diff_adjusted_pp"] == pytest.approx(truth - round(truth + 0.05, 2), abs=1e-9)
    assert abs(table.iloc[0]["diff_adjusted_pp"]) <= 0.1
