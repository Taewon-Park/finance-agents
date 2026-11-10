"""C 계열: 돌파 (추세, 일간).

- 진입: 종가가 직전 entry_n일 최고 종가를 상향 돌파 (돌파한 날에만 진입 후보)
- 청산: 종가가 직전 exit_n일 최저 종가를 하향 이탈
- 신호 강도: 63일 수익률 큰 순, 최대 5개
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from swinglab.indicators import period_return, prior_high, prior_low
from swinglab.strategies.base import hold_state, price_panel


@dataclass
class Breakout:
    tickers: list[str] = field(default_factory=list)
    entry_n: int = 20
    exit_n: int = 10
    strength_window: int = 63
    name: str = "breakout"
    max_positions: int | None = 5
    max_hold_days: int | None = None
    min_hold_days: int | None = None
    rebalance: str = "daily"

    @property
    def warmup_days(self) -> int:
        return max(self.entry_n, self.exit_n, self.strength_window) + 1

    def _components(self, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        close = price_panel(prices, "close", self.tickers)
        high = close.apply(lambda c: prior_high(c, self.entry_n))
        low = close.apply(lambda c: prior_low(c, self.exit_n))
        ret = close.apply(lambda c: period_return(c, self.strength_window))
        valid = close.notna() & high.notna() & low.notna()
        entry = valid & (close > high)
        exit_ = valid & (close < low)
        return ret, entry, exit_, valid

    def generate(self, prices: pd.DataFrame) -> pd.DataFrame:
        _, entry, exit_, valid = self._components(prices)
        out = pd.DataFrame(np.nan, index=entry.index, columns=entry.columns)
        for ticker in entry.columns:
            out[ticker] = hold_state(entry[ticker].to_numpy(), exit_[ticker].to_numpy(), valid[ticker].to_numpy())
        return out

    def strength(self, prices: pd.DataFrame) -> pd.DataFrame:
        ret, entry, _, _ = self._components(prices)
        return ret.where(entry)
