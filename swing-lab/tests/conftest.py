"""테스트용 합성 데이터와 고정 신호 전략."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from swinglab.config import BacktestConfig


def trading_dates(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n).astype("datetime64[ns]")


def make_prices(opens: pd.DataFrame, closes: pd.DataFrame | None = None, volume: int = 1_000) -> pd.DataFrame:
    """date x ticker 시가·종가 표로 스키마 형태의 long DataFrame을 만든다. NaN 칸은 행이 없는 날(결측)."""
    closes = opens if closes is None else closes
    rows = []
    for ticker in opens.columns:
        for d in opens.index:
            o, c = opens.at[d, ticker], closes.at[d, ticker]
            if np.isnan(o) or np.isnan(c):
                continue
            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "open": float(o),
                    "high": float(max(o, c)),
                    "low": float(min(o, c)),
                    "close": float(c),
                    "volume": volume,
                    "source": "test",
                }
            )
    return pd.DataFrame(rows)


def random_walk_prices(
    tickers: list[str], n: int, seed: int = 0, drift: float = 0.0004, vol: float = 0.015
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = trading_dates(n)
    closes, opens = {}, {}
    for t in tickers:
        rets = rng.normal(drift, vol, n)
        close = 100 * np.exp(np.cumsum(rets))
        gap = rng.normal(0, vol / 3, n)
        opens[t] = np.r_[close[0], close[:-1]] * np.exp(gap)
        closes[t] = close
    return make_prices(pd.DataFrame(opens, index=dates), pd.DataFrame(closes, index=dates))


@dataclass
class FixedStrategy:
    """미리 정한 목표 상태·강도를 그대로 돌려주는 테스트용 전략."""

    target: pd.DataFrame
    strength_table: pd.DataFrame | None = None
    name: str = "fixed"
    warmup_days: int = 0
    max_positions: int | None = None
    max_hold_days: int | None = None

    def generate(self, prices: pd.DataFrame) -> pd.DataFrame:
        return self.target

    def strength(self, prices: pd.DataFrame) -> pd.DataFrame:
        if self.strength_table is not None:
            return self.strength_table
        return self.target.where(self.target == 1.0)


@pytest.fixture
def config() -> BacktestConfig:
    return BacktestConfig(initial_capital=100_000.0, cost_bps=5.0, max_positions=5, start=None, end=None)
