"""A 계열: RSI(2) 개선판 (평균회귀, 거래 빈도 줄이기).

- 진입: 종가 > SMA(200) 이고 RSI(2) < entry_th
- 청산: (종가 > SMA(5) 또는 RSI(2) > 70) 이고 보유일 ≥ min_hold, 또는 보유 10거래일 경과
- 신호 강도: RSI(2) 낮은 순, 최대 5개

목표 상태는 '청산 조건이 성립한 날 0, 그 외 1'이다. 그래서 청산은 조건이 성립한 그날
보유일이 min_hold 이상일 때만 일어난다 (엔진의 min_hold_days). 신규 진입은 강도를
진입 조건이 성립한 날에만 줘서 그날만 일어난다. 두 조건이 같은 날이면 청산이 우선한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from swinglab.indicators import rsi_wilder, sma
from swinglab.strategies.base import price_panel


@dataclass
class Rsi2Swing:
    tickers: list[str] = field(default_factory=list)
    entry_th: float = 10.0
    min_hold: int = 1
    exit_rsi: float = 70.0
    rsi_period: int = 2
    trend_window: int = 200
    exit_window: int = 5
    name: str = "rsi2_v2"
    warmup_days: int = 200
    max_positions: int | None = 5
    max_hold_days: int | None = 10
    rebalance: str = "daily"

    @property
    def min_hold_days(self) -> int:
        return self.min_hold

    def _components(self, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        close = price_panel(prices, "close", self.tickers)
        rsi = close.apply(lambda c: rsi_wilder(c, self.rsi_period))
        trend = close.apply(lambda c: sma(c, self.trend_window))
        exit_ma = close.apply(lambda c: sma(c, self.exit_window))
        valid = close.notna() & rsi.notna() & trend.notna() & exit_ma.notna()
        exit_ = valid & ((close > exit_ma) | (rsi > self.exit_rsi))
        entry = valid & (close > trend) & (rsi < self.entry_th) & ~exit_
        return rsi, entry, exit_, valid

    def generate(self, prices: pd.DataFrame) -> pd.DataFrame:
        _, _, exit_, valid = self._components(prices)
        return (~exit_).astype(float).where(valid)

    def strength(self, prices: pd.DataFrame) -> pd.DataFrame:
        rsi, entry, _, _ = self._components(prices)
        return (-rsi).where(entry)
