"""B 계열: 듀얼 모멘텀 (추세, 월간 리밸런싱).

- 월 마지막 거래일에 lookback개월 수익률(월말 종가 대비 월말 종가, 최근 1개월 제외 안 함)로 순위를 매겨
  상위 top_n 섹터 ETF를 고른다. 그 기간 데이터가 없는 종목(상장 전 등)은 순위에서 뺀다.
- 절대 모멘텀: 고른 종목의 수익률이 같은 기간 안전자산(IEF) 수익률보다 낮으면 그 자리는 IEF로 대체한다.
  대체된 자리마다 별도 포지션(IEF#1, IEF#2, ...)으로 들어 IEF 비중 = 대체된 자리 수 / top_n.
- 월중에는 직전 월말 목표 상태를 유지한다 (엔진도 monthly면 월말 행만 쓴다).
- 순위 동점이면 티커 알파벳순.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from swinglab.strategies.base import price_panel

SECTOR_ETFS: list[str] = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]


@dataclass
class DualMomentum:
    tickers: list[str] = field(default_factory=lambda: list(SECTOR_ETFS))
    lookback: int = 12  # 개월
    top_n: int = 2
    safe: str = "IEF"
    name: str = "dual_momentum"
    max_hold_days: int | None = None
    min_hold_days: int | None = None
    rebalance: str = "monthly"

    @property
    def max_positions(self) -> int:
        return self.top_n

    @property
    def warmup_days(self) -> int:
        # lookback개월 전 월말 종가가 처음 생기는 시점 (대략 월 21거래일)
        return 21 * (self.lookback + 1)

    @property
    def safe_slots(self) -> list[str]:
        return [f"{self.safe}#{k}" for k in range(1, self.top_n + 1)]

    @property
    def price_aliases(self) -> dict[str, str]:
        return {slot: self.safe for slot in self.safe_slots}

    def momentum(self, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """월말 행별 lookback개월 수익률 (섹터 표, 안전자산 시리즈). 월말이 아니거나 계산 불가면 NaN."""
        close = price_panel(prices, "close", [*self.tickers, self.safe])
        dates = close.index
        is_month_end = np.r_[dates.month[1:] != dates.month[:-1], False]  # 마지막 날은 월말인지 알 수 없다
        month_ends = pd.Series(dates[is_month_end], index=dates[is_month_end].to_period("M"))

        rets = pd.DataFrame(np.nan, index=dates, columns=[*self.tickers, self.safe])
        for t in dates[is_month_end]:
            ref = month_ends.get(t.to_period("M") - self.lookback)
            if ref is None:
                continue
            rets.loc[t] = close.loc[t] / close.loc[ref] - 1.0
        return rets[self.tickers], rets[self.safe]

    def _targets(self, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        sector_rets, safe_ret = self.momentum(prices)
        cols = [*self.tickers, *self.safe_slots]
        target = pd.DataFrame(np.nan, index=sector_rets.index, columns=cols)
        strength = pd.DataFrame(np.nan, index=sector_rets.index, columns=cols)
        for t in sector_rets.index:
            row = sector_rets.loc[t].dropna()
            if row.empty or np.isnan(safe_ret.loc[t]):
                continue
            ranked = sorted(row.items(), key=lambda kv: (-kv[1], kv[0]))[: self.top_n]
            kept = [(tk, r) for tk, r in ranked if r >= safe_ret.loc[t]]
            n_safe = len(ranked) - len(kept)
            target.loc[t] = 0.0
            for tk, r in kept:
                target.loc[t, tk] = 1.0
                strength.loc[t, tk] = r
            for slot in self.safe_slots[:n_safe]:
                target.loc[t, slot] = 1.0
                strength.loc[t, slot] = safe_ret.loc[t]
        return target.ffill(), strength

    def generate(self, prices: pd.DataFrame) -> pd.DataFrame:
        return self._targets(prices)[0]

    def strength(self, prices: pd.DataFrame) -> pd.DataFrame:
        return self._targets(prices)[1]
