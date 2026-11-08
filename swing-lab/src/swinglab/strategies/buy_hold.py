"""바이앤홀드 (기준선): 한 종목을 백테스트 첫 거래일 시가에 100% 매수해 끝까지 보유."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from swinglab.strategies.base import price_panel


@dataclass
class BuyAndHold:
    ticker: str = "SPY"
    name: str = "buy_hold"
    warmup_days: int = 0
    max_positions: int | None = 1
    max_hold_days: int | None = None

    def generate(self, prices: pd.DataFrame) -> pd.DataFrame:
        close = price_panel(prices, "close", [self.ticker])
        return close.notna().astype(float).where(close.notna())

    def strength(self, prices: pd.DataFrame) -> pd.DataFrame:
        return self.generate(prices)
