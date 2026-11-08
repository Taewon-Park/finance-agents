"""일 단위 이벤트 루프 백테스트 엔진.

체결 규칙
- t일 종가까지의 신호로 주문을 정하고 t+1일 시가에 체결한다 (같은 날 종가 체결 없음).
- 롱 온리, 레버리지 없음, 소수 주식 허용.
- 시가에는 청산을 먼저 하고 진입한다. 그날 청산한 종목은 같은 시가에 다시 사지 않는다.
- 진입 예산(비용 포함) = min(그날 시가 기준 총자산 / max_positions, 보유 현금). 리밸런싱 없음.
- 신규 진입 후보: 미보유 & 목표 1 & 강도 값 있음. 강도 큰 순, 동점이면 티커 알파벳순.
- 보유 종목: 목표 0이면 청산, 목표 NaN(신호 없음)이면 유지, max_hold_days에 닿으면 청산.
- 체결할 날 시가가 없으면: 청산은 다음으로 시가가 있는 날로 미루고, 진입은 취소한다
  (다음 신호로 다시 판단).
- 평가: 종가 기준. 결측일에는 직전 종가로 평가한다.

백테스트 구간
- 첫 체결일 = max(설정 start 이후 첫 거래일, warmup_days번째 거래일, 둘째 거래일).
- 자산곡선의 첫 행은 첫 체결일 전날 종가(초기자본, 전액 현금)이고, 그날 신호로 첫 체결을 한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from swinglab.backtest.costs import CostModel
from swinglab.config import BacktestConfig
from swinglab.strategies.base import Strategy, price_panel
from swinglab.strategies.buy_hold import BuyAndHold

TRADE_COLUMNS: list[str] = [
    "ticker",
    "entry_date",
    "exit_date",
    "entry_price",
    "exit_price",
    "shares",
    "return",
    "gross_return",
    "holding_days",
    "entry_cost",
    "exit_cost",
    "exit_reason",
]
OPEN_POSITION_COLUMNS: list[str] = [
    "ticker",
    "entry_date",
    "entry_price",
    "shares",
    "mark_date",
    "mark_price",
    "unrealized_return",
    "holding_days",
    "entry_cost",
]


@dataclass
class BacktestResult:
    strategy_name: str
    cost_bps: float
    max_positions: int
    equity: pd.Series  # 일별 자산가치 (종가). 첫 행 = 시작 기준일, 초기자본
    cash: pd.Series
    positions: pd.DataFrame  # 일별 보유 수량, date x ticker
    trades: pd.DataFrame  # 청산 완료 거래 (TRADE_COLUMNS)
    open_positions: pd.DataFrame  # 기말 미청산 포지션, 종가 평가 (OPEN_POSITION_COLUMNS)
    total_cost: float

    @property
    def start_date(self) -> pd.Timestamp:
        """첫 체결일 (백테스트 구간의 첫 거래일)."""
        return self.equity.index[1]

    @property
    def end_date(self) -> pd.Timestamp:
        return self.equity.index[-1]


def run_backtest(prices: pd.DataFrame, strategy: Strategy, config: BacktestConfig) -> BacktestResult:
    cost = CostModel.from_bps(config.cost_bps)
    max_positions = strategy.max_positions or config.max_positions
    max_hold = strategy.max_hold_days

    calendar = pd.DatetimeIndex(np.sort(prices["date"].unique()))
    target = strategy.generate(prices).reindex(calendar)
    tickers = [str(t) for t in target.columns]
    strength = strategy.strength(prices).reindex(index=calendar, columns=tickers)
    _validate_target(target)

    opens = price_panel(prices, "open", tickers).reindex(calendar).to_numpy(dtype=float)
    closes_ff = price_panel(prices, "close", tickers).reindex(calendar).ffill().to_numpy(dtype=float)
    tgt = target.to_numpy(dtype=float)
    stg = strength.to_numpy(dtype=float)

    first_idx, last_idx = _window(calendar, strategy.warmup_days, config)
    anchor_idx = first_idx - 1

    k = len(tickers)
    cash = float(config.initial_capital)
    shares = np.zeros(k)
    entry_idx = np.full(k, -1)
    entry_price = np.zeros(k)
    entry_cost = np.zeros(k)
    total_cost = 0.0
    pending_exits: dict[int, str] = {}  # 종목 인덱스 → 청산 사유
    pending_entries: list[int] = []
    trades: list[dict] = []

    def decide(i: int) -> list[int]:
        """i일 종가 기준으로 다음 시가의 청산(pending_exits 갱신)과 진입 후보(반환)를 정한다."""
        for j in map(int, np.flatnonzero(shares > 0)):
            if j in pending_exits:
                continue
            if tgt[i, j] == 0.0:
                pending_exits[j] = "signal"
            elif max_hold is not None and i - entry_idx[j] + 1 >= max_hold:
                pending_exits[j] = "max_hold"
        candidates = [
            j
            for j in range(k)
            if shares[j] == 0 and tgt[i, j] == 1.0 and not np.isnan(stg[i, j])
        ]
        candidates.sort(key=lambda j: (-stg[i, j], tickers[j]))
        return candidates

    eq_rows = [cash]
    cash_rows = [cash]
    pos_rows = [shares.copy()]
    pending_entries = decide(anchor_idx)

    for i in range(first_idx, last_idx + 1):
        # --- 시가: 청산 먼저
        for j in sorted(pending_exits):
            price = opens[i, j]
            if np.isnan(price):
                continue  # 시가가 없으면 다음 거래일로 미룬다
            proceeds, exit_cost = cost.sell(shares[j] * price)
            cash += proceeds
            total_cost += exit_cost
            trades.append(
                {
                    "ticker": tickers[j],
                    "entry_date": calendar[entry_idx[j]],
                    "exit_date": calendar[i],
                    "entry_price": entry_price[j],
                    "exit_price": price,
                    "shares": shares[j],
                    "return": cost.trade_return(entry_price[j], price),
                    "gross_return": price / entry_price[j] - 1.0,
                    "holding_days": i - entry_idx[j],
                    "entry_cost": entry_cost[j],
                    "exit_cost": exit_cost,
                    "exit_reason": pending_exits.pop(j),
                }
            )
            shares[j] = 0.0
            entry_idx[j] = -1

        # --- 시가: 진입
        if pending_entries:
            held = int((shares > 0).sum())
            mark = np.where(np.isnan(opens[i]), closes_ff[i - 1], opens[i])
            equity_open = cash + float(np.nansum(shares * mark))
            for j in pending_entries:
                if held >= max_positions:
                    break
                price = opens[i, j]
                if np.isnan(price):
                    continue  # 시가가 없으면 진입 취소
                budget = min(equity_open / max_positions, cash)
                if budget <= 0:
                    break
                notional, buy_cost = cost.buy(budget)
                cash -= budget
                total_cost += buy_cost
                shares[j] = notional / price
                entry_idx[j] = i
                entry_price[j] = price
                entry_cost[j] = buy_cost
                held += 1
        pending_entries = []

        # --- 종가: 평가와 다음 날 주문 결정
        held_mask = shares > 0
        eq_rows.append(cash + float(np.sum(shares[held_mask] * closes_ff[i, held_mask])))
        cash_rows.append(cash)
        pos_rows.append(shares.copy())
        if i < last_idx:
            pending_entries = decide(i)

    index = calendar[anchor_idx : last_idx + 1]
    open_positions = [
        {
            "ticker": tickers[j],
            "entry_date": calendar[entry_idx[j]],
            "entry_price": entry_price[j],
            "shares": shares[j],
            "mark_date": calendar[last_idx],
            "mark_price": closes_ff[last_idx, j],
            "unrealized_return": closes_ff[last_idx, j] / (entry_price[j] * (1.0 + cost.rate)) - 1.0,
            "holding_days": last_idx - entry_idx[j] + 1,
            "entry_cost": entry_cost[j],
        }
        for j in np.flatnonzero(shares > 0)
    ]
    return BacktestResult(
        strategy_name=strategy.name,
        cost_bps=config.cost_bps,
        max_positions=max_positions,
        equity=pd.Series(eq_rows, index=index, name="equity"),
        cash=pd.Series(cash_rows, index=index, name="cash"),
        positions=pd.DataFrame(pos_rows, index=index, columns=tickers),
        trades=pd.DataFrame(trades, columns=TRADE_COLUMNS),
        open_positions=pd.DataFrame(open_positions, columns=OPEN_POSITION_COLUMNS),
        total_cost=total_cost,
    )


def run_benchmark(
    prices: pd.DataFrame, ticker: str, config: BacktestConfig, like: BacktestResult
) -> BacktestResult:
    """like와 같은 구간(같은 첫 체결일·마지막 날)의 바이앤홀드."""
    window = config.model_copy(update={"start": like.start_date.date(), "end": like.end_date.date()})
    return run_backtest(prices, BuyAndHold(ticker=ticker, name=f"{ticker} buy&hold"), window)


def _window(calendar: pd.DatetimeIndex, warmup_days: int, config: BacktestConfig) -> tuple[int, int]:
    if len(calendar) < 2:
        raise ValueError("백테스트에는 최소 2거래일이 필요합니다")
    start_idx = 0
    if config.start is not None:
        start_idx = int(calendar.searchsorted(pd.Timestamp(config.start), side="left"))
    last_idx = len(calendar) - 1
    if config.end is not None:
        last_idx = int(calendar.searchsorted(pd.Timestamp(config.end), side="right")) - 1
    first_idx = max(start_idx, warmup_days, 1)
    if first_idx > last_idx:
        raise ValueError(
            f"백테스트 구간이 비었습니다 (첫 체결일 인덱스 {first_idx}, 마지막 {last_idx}). "
            "start/end와 데이터 기간, 워밍업 일수를 확인하세요."
        )
    return first_idx, last_idx


def _validate_target(target: pd.DataFrame) -> None:
    values = target.to_numpy(dtype=float)
    bad = ~(np.isnan(values) | (values == 0.0) | (values == 1.0))
    if bad.any():
        raise ValueError("generate()는 1, 0, NaN만 반환해야 합니다")
