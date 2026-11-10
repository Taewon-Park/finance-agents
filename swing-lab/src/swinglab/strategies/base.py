"""전략 인터페이스.

스펙의 Strategy(name, warmup_days, generate)에 엔진이 순위·청산에 필요한 항목을 더했다.

- generate(prices): date × ticker 목표 상태. 1 = 보유, 0 = 미보유,
  NaN = 그날 신호 없음 (상장 전·결측일·워밍업). t일 행은 t일 종가까지의 정보만 쓴다.
- strength(prices): date × ticker 신호 강도. 클수록 우선 진입.
  엔진은 '목표 1이고 강도 값이 있는' 종목만 신규 진입 후보로 본다.
  그래서 강도를 진입 조건이 성립한 날에만 주면 늦은 진입이 생기지 않는다 (RSI 전략).
- max_positions: 동시 보유 수. None이면 백테스트 설정값을 쓴다 (바이앤홀드 = 1).
- max_hold_days: 실제 진입일부터 센 최대 보유 거래일. 엔진이 강제 청산한다. None이면 제한 없음.

선택 속성 (없으면 엔진 기본값, 1단계에서 추가):
- rebalance: "daily"(기본) 또는 "monthly". monthly면 월 마지막 거래일 신호만 반영한다.
- min_hold_days: 목표 0이어도 보유일이 이보다 짧으면 청산하지 않는다. None이면 제한 없음.
- price_aliases: {"IEF#2": "IEF"}처럼 같은 종목을 여러 자리로 들 때의 별칭. 기본은 없음.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class Strategy(Protocol):
    name: str
    warmup_days: int
    max_positions: int | None
    max_hold_days: int | None

    def generate(self, prices: pd.DataFrame) -> pd.DataFrame:
        """date x ticker 목표 상태 반환. 1 = 보유, 0 = 미보유, NaN = 신호 없음.
        t일 행은 t일 종가까지의 정보만 사용해야 한다."""
        ...

    def strength(self, prices: pd.DataFrame) -> pd.DataFrame:
        """date x ticker 신호 강도 (클수록 우선). 신규 진입 후보가 아니면 NaN."""
        ...


def price_panel(prices: pd.DataFrame, field: str, tickers: list[str] | None = None) -> pd.DataFrame:
    """long-format 가격을 date × ticker 표로 바꾼다.

    인덱스는 prices에 있는 모든 날짜(전체 종목 합집합), 데이터가 없는 칸은 NaN.
    tickers를 주면 그 순서로 열을 맞추고, 데이터가 아예 없는 종목은 빈 열로 둔다.
    """
    panel = prices.pivot(index="date", columns="ticker", values=field).sort_index()
    panel.columns.name = None
    if tickers is not None:
        panel = panel.reindex(columns=tickers)
    return panel


def hold_state(entry: np.ndarray, exit_: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """청산 조건이면 0, 아니고 진입 조건이면 1, 둘 다 아니면 이전 상태 유지.

    두 조건이 같은 날 성립하면 청산이 우선한다 (보유 중이든 아니든 같은 규칙).
    데이터가 없는 날은 NaN이고 상태는 유지한다.
    """
    state = 0.0
    out = np.full(len(entry), np.nan)
    for i in range(len(entry)):
        if not valid[i]:
            continue
        if exit_[i]:
            state = 0.0
        elif entry[i]:
            state = 1.0
        out[i] = state
    return out
