"""시장 필터 래퍼: t일 SPY 종가가 SPY SMA(200) 아래면 위험자산 목표 상태를 모두 0으로 만든다.

- 채권·금(기본 TLT, IEF, GLD)과 그 별칭(IEF#2 등)은 필터 대상에서 뺀다.
- SMA(200)를 계산할 수 없는 날(데이터 200일 미만)은 필터를 적용하지 않는다.
  워밍업을 max(감싼 전략, 200)으로 잡아 백테스트 구간에서는 항상 필터가 정의되게 한다.
- 신호가 없는 칸(NaN)은 그대로 둔다. 강도는 감싼 전략 것을 그대로 쓴다 (목표 0이면 진입 후보가 아니다).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from swinglab.indicators import sma
from swinglab.strategies.base import Strategy, price_panel

SAFE_ASSETS: tuple[str, ...] = ("TLT", "IEF", "GLD")


@dataclass
class MarketFilter:
    inner: Strategy
    benchmark: str = "SPY"
    window: int = 200
    exempt: tuple[str, ...] = SAFE_ASSETS
    name: str = field(init=False)
    warmup_days: int = field(init=False)
    max_positions: int | None = field(init=False)
    max_hold_days: int | None = field(init=False)
    min_hold_days: int | None = field(init=False)
    rebalance: str = field(init=False)
    price_aliases: dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        self.name = f"{self.inner.name}+market_filter"
        self.warmup_days = max(self.inner.warmup_days, self.window)
        self.max_positions = self.inner.max_positions
        self.max_hold_days = self.inner.max_hold_days
        self.min_hold_days = getattr(self.inner, "min_hold_days", None)
        self.rebalance = getattr(self.inner, "rebalance", "daily")
        self.price_aliases = dict(getattr(self.inner, "price_aliases", None) or {})

    def risk_off(self, prices: pd.DataFrame) -> pd.Series:
        """날짜별로 필터가 켜졌는지 (SPY 종가 < SPY SMA). 계산할 수 없는 날은 False."""
        spy = price_panel(prices, "close", [self.benchmark])[self.benchmark]
        return (spy < sma(spy, self.window)).fillna(False).astype(bool)

    def generate(self, prices: pd.DataFrame) -> pd.DataFrame:
        target = self.inner.generate(prices).copy()
        off = self.risk_off(prices).reindex(target.index, fill_value=False).to_numpy()
        for col in target.columns:
            if self.price_aliases.get(col, col) in self.exempt:
                continue
            values = target[col].to_numpy(copy=True)
            mask = off & ~pd.isna(values)
            values[mask] = 0.0
            target[col] = values
        return target

    def strength(self, prices: pd.DataFrame) -> pd.DataFrame:
        return self.inner.strength(prices)
