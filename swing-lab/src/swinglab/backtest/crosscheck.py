"""바이앤홀드를 공개 총수익률과 비교 (0단계 완료 기준: 외부 교차 확인).

공개 총수익률은 시작일 종가 → 끝날 종가 기준이다. 엔진은 시작일 종가의 신호로
다음 날 시가에 사므로, 시작일 종가 → 다음 날 시가 사이의 움직임(첫날 갭)만큼 다르다.
엔진 결과에 갭(다음 날 시가 / 시작일 종가)을 곱해 되돌린 값을 공개 값과 비교한다.
비용은 0bp로 둔다 (공개 값에도 거래 비용이 없다).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from swinglab.backtest.engine import run_backtest
from swinglab.config import BacktestConfig
from swinglab.strategies.buy_hold import BuyAndHold


@dataclass(frozen=True)
class PublishedReturn:
    as_of: date
    years: int
    annualized_pct: float  # 연환산 총수익률(%), 예: 15.18


def buy_hold_crosscheck(
    prices: pd.DataFrame, ticker: str, published: list[PublishedReturn], initial_capital: float = 100_000.0
) -> pd.DataFrame:
    """공개 구간마다 종가→종가, 엔진, 갭 보정 엔진 연환산 수익률(%)과 차이(%p)를 계산한다."""
    series = prices[prices["ticker"] == ticker].set_index("date").sort_index()
    rows = []
    for p in published:
        end = series.index[series.index <= pd.Timestamp(p.as_of)][-1]
        start = series.index[series.index <= pd.Timestamp(p.as_of) - pd.DateOffset(years=p.years)][-1]
        first_trade = series.index[series.index > start][0]

        window = BacktestConfig(
            initial_capital=initial_capital, cost_bps=0.0, start=first_trade.date(), end=end.date()
        )
        result = run_backtest(prices, BuyAndHold(ticker=ticker), window)
        engine_ratio = result.equity.iloc[-1] / result.equity.iloc[0]
        gap = series.at[first_trade, "open"] / series.at[start, "close"]
        close_ratio = series.at[end, "close"] / series.at[start, "close"]

        close_to_close = _annualize(close_ratio, p.years)
        engine = _annualize(engine_ratio, p.years)
        adjusted = _annualize(engine_ratio * gap, p.years)
        rows.append(
            {
                "as_of": p.as_of,
                "years": p.years,
                "start_close_date": start.date(),
                "end_date": end.date(),
                "published_pct": p.annualized_pct,
                "close_to_close_pct": close_to_close,
                "engine_pct": engine,
                "first_day_gap_pct": (gap - 1.0) * 100.0,
                "engine_adjusted_pct": adjusted,
                "diff_adjusted_pp": adjusted - p.annualized_pct,
                "diff_unadjusted_pp": engine - p.annualized_pct,
            }
        )
    return pd.DataFrame(rows)


def _annualize(ratio: float, years: int) -> float:
    return (ratio ** (1.0 / years) - 1.0) * 100.0
