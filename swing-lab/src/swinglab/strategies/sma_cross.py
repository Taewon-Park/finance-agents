"""이동평균 교차 (추세 추종).

- 보유 조건: SMA(20) > SMA(60). 조건이 깨지면 익일 시가 청산.
- 신호 강도: SMA(20) / SMA(60) − 1, 큰 순으로 최대 max_positions개 보유.
- 강도가 조건 성립 기간 내내 있으므로, 빈 자리가 생기면 이미 추세 중인 종목도 진입한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from swinglab.indicators import sma
from swinglab.strategies.base import price_panel


@dataclass
class SmaCross:
    tickers: list[str] = field(default_factory=list)
    fast: int = 20
    slow: int = 60
    name: str = "sma_cross"
    warmup_days: int = 60
    max_positions: int | None = None
    max_hold_days: int | None = None

    def _ratio(self, prices: pd.DataFrame) -> pd.DataFrame:
        close = price_panel(prices, "close", self.tickers)
        return close.apply(lambda c: sma(c, self.fast) / sma(c, self.slow) - 1.0)

    def generate(self, prices: pd.DataFrame) -> pd.DataFrame:
        ratio = self._ratio(prices)
        return (ratio > 0).astype(float).where(ratio.notna())

    def strength(self, prices: pd.DataFrame) -> pd.DataFrame:
        return self._ratio(prices)
