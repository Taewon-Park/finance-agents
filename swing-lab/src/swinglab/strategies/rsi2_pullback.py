"""RSI(2) 눌림목 (스윙 평균회귀).

- 진입: 종가 > SMA(200) 이고 RSI(2) < 10 (RSI는 Wilder 평활)
- 청산: 종가 > SMA(5), 또는 보유 10거래일 경과 (보유일 제한은 실제 진입일 기준으로 엔진이 처리)
- 신호 강도: RSI(2)가 낮은 순. 진입 조건이 성립한 날에만 값을 줘서 늦은 진입을 막는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from swinglab.indicators import rsi_wilder, sma
from swinglab.strategies.base import hold_state as _hold_state
from swinglab.strategies.base import price_panel


@dataclass
class Rsi2Pullback:
    tickers: list[str] = field(default_factory=list)
    rsi_period: int = 2
    entry_rsi: float = 10.0
    trend_window: int = 200
    exit_window: int = 5
    name: str = "rsi2_pullback"
    warmup_days: int = 200
    max_positions: int | None = None
    max_hold_days: int | None = 10

    def _components(self, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        close = price_panel(prices, "close", self.tickers)
        rsi = close.apply(lambda c: rsi_wilder(c, self.rsi_period))
        trend = close.apply(lambda c: sma(c, self.trend_window))
        exit_ma = close.apply(lambda c: sma(c, self.exit_window))
        valid = close.notna() & rsi.notna() & trend.notna() & exit_ma.notna()
        entry = valid & (close > trend) & (rsi < self.entry_rsi)
        exit_ = valid & (close > exit_ma)
        return rsi, entry, exit_, valid

    def generate(self, prices: pd.DataFrame) -> pd.DataFrame:
        _, entry, exit_, valid = self._components(prices)
        out = pd.DataFrame(np.nan, index=entry.index, columns=entry.columns)
        for ticker in entry.columns:
            out[ticker] = _hold_state(entry[ticker].to_numpy(), exit_[ticker].to_numpy(), valid[ticker].to_numpy())
        return out

    def strength(self, prices: pd.DataFrame) -> pd.DataFrame:
        rsi, entry, _, _ = self._components(prices)
        return (-rsi).where(entry)
