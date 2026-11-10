"""페이퍼 계정의 하루 처리.

체결 규칙은 0단계 엔진(backtest/engine.py)과 같다. 엔진은 한 번에 전 구간을 돌리지만
여기서는 처리할 거래일 하나씩 진행하고 상태(AccountState)를 남긴다.

하루(d) 처리 순서
1. 수정주가 재조정: 보유 종목의 직전 평가가(mark_price)가 이번 가격 스냅샷의 같은 날 종가와 다르면
   (새 배당·분할로 과거 수정주가가 바뀜) 평가액이 같도록 수량과 진입가를 맞춘다. 배당 수익이 빠지지 않게 하려고.
2. d 시가: 전날 정한 청산 먼저, 그다음 진입 (엔진과 같은 예산·순서·결측 처리)
3. d 종가: 평가
4. d 종가 신호로 다음 거래일 시가 주문을 정한다

엔진에 있는 rebalance="monthly"와 price_aliases는 페이퍼 대상 전략(A, C)이 쓰지 않으므로 지원하지 않는다.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from swinglab.backtest.costs import CostModel
from swinglab.pipeline.ingest import ADJUSTMENT_RTOL
from swinglab.strategies.base import Strategy, price_panel


@dataclass
class Position:
    ticker: str
    shares: float
    entry_date: str
    entry_price: float  # 현재 가격 스냅샷 기준 (재조정 때 함께 바뀜)
    entry_cost: float
    mark_date: str
    mark_price: float


@dataclass
class AccountState:
    paper_id: str
    account: str
    start_date: str
    last_processed: str
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    pending_exits: dict[str, str] = field(default_factory=dict)  # 티커 → 사유 (signal, max_hold)
    pending_entries: list[str] = field(default_factory=list)  # 우선순위 순

    def to_dict(self) -> dict:
        d = asdict(self)
        d["positions"] = {t: asdict(p) for t, p in self.positions.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> AccountState:
        d = dict(d)
        d["positions"] = {t: Position(**p) for t, p in d.get("positions", {}).items()}
        return cls(**d)


@dataclass
class DayRecord:
    """하루 처리 결과. 저장소(store)에 행으로 들어간다."""

    date: str
    equity: float
    cash: float
    fills: list[dict] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)
    orders: list[dict] = field(default_factory=list)  # 이날 신호로 새로 정한 주문
    order_updates: list[dict] = field(default_factory=list)  # 기존 주문의 체결·취소 상태
    positions: list[dict] = field(default_factory=list)


@dataclass
class MarketView:
    """한 가격 스냅샷의 date × ticker 표 (엔진과 같은 방식으로 만든다)."""

    calendar: pd.DatetimeIndex
    tickers: list[str]
    opens: pd.DataFrame
    closes_ff: pd.DataFrame
    target: pd.DataFrame
    strength: pd.DataFrame

    @classmethod
    def build(cls, prices: pd.DataFrame, strategy: Strategy) -> MarketView:
        if getattr(strategy, "rebalance", "daily") != "daily" or getattr(strategy, "price_aliases", None):
            raise ValueError("페이퍼 계정은 일간 신호·별칭 없는 전략만 지원합니다")
        calendar = pd.DatetimeIndex(np.sort(prices["date"].unique()))
        target = strategy.generate(prices).reindex(calendar)
        tickers = [str(t) for t in target.columns]
        values = target.to_numpy(dtype=float)
        if (~(np.isnan(values) | (values == 0.0) | (values == 1.0))).any():
            raise ValueError("generate()는 1, 0, NaN만 반환해야 합니다")
        return cls(
            calendar=calendar,
            tickers=tickers,
            opens=price_panel(prices, "open", tickers).reindex(calendar),
            closes_ff=price_panel(prices, "close", tickers).reindex(calendar).ffill(),
            target=target,
            strength=strategy.strength(prices).reindex(index=calendar, columns=tickers),
        )

    def complete_until(self, prices: pd.DataFrame) -> pd.Timestamp:
        """대상 종목 모두의 데이터가 있는 마지막 날. 수집이 일부 실패한 날을 처리하지 않기 위해 쓴다."""
        last = prices[prices["ticker"].isin(self.tickers)].groupby("ticker")["date"].max()
        missing = sorted(set(self.tickers) - set(last.index))
        if missing:
            raise ValueError(f"가격 데이터에 없는 종목: {', '.join(missing)}")
        return pd.Timestamp(last.min())


@dataclass(frozen=True)
class Rules:
    cost: CostModel
    max_positions: int
    min_hold: int | None
    max_hold: int | None

    @classmethod
    def from_strategy(cls, strategy: Strategy, cost_bps: float, default_max_positions: int) -> Rules:
        return cls(
            cost=CostModel.from_bps(cost_bps),
            max_positions=strategy.max_positions or default_max_positions,
            min_hold=getattr(strategy, "min_hold_days", None),
            max_hold=strategy.max_hold_days,
        )


def open_account(
    paper_id: str, account: str, start: pd.Timestamp, capital: float, view: MarketView, rules: Rules
) -> tuple[AccountState, DayRecord]:
    """시작일 종가: 전액 현금으로 평가하고 그날 신호로 첫 주문을 정한다 (엔진의 자산곡선 첫 행과 같음)."""
    day = _day(start)
    state = AccountState(paper_id=paper_id, account=account, start_date=day, last_processed=day, cash=capital)
    record = DayRecord(date=day, equity=capital, cash=capital)
    record.orders = _decide(state, view, start, rules)
    return state, record


def advance(state: AccountState, view: MarketView, d: pd.Timestamp, rules: Rules) -> DayRecord:
    """state.last_processed 다음 거래일 d 하나를 처리한다. state를 바꾼다."""
    i = view.calendar.get_loc(d)
    prev = view.calendar[i - 1]
    if _day(prev) != state.last_processed:
        raise ValueError(f"{_day(d)} 처리 전 날짜가 {state.last_processed}가 아닙니다 ({_day(prev)})")
    record = DayRecord(date=_day(d), equity=0.0, cash=0.0)
    cost = rules.cost

    # 1. 수정주가 재조정
    for pos in state.positions.values():
        snapshot_mark = float(view.closes_ff.at[pd.Timestamp(pos.mark_date), pos.ticker])
        if not math.isclose(snapshot_mark, pos.mark_price, rel_tol=ADJUSTMENT_RTOL):
            factor = snapshot_mark / pos.mark_price
            pos.shares /= factor
            pos.entry_price *= factor
            pos.mark_price = snapshot_mark
            record.fills.append(_fill(state, d, pos.ticker, "adjust", pos.shares, snapshot_mark, 0.0, 0.0, f"x{factor:.6f}"))

    opens = view.opens.loc[d]
    # 2-1. 시가: 청산
    for ticker in sorted(state.pending_exits):
        price = float(opens[ticker])
        if math.isnan(price):
            continue  # 시가가 없으면 다음 거래일로 미룬다
        pos = state.positions.pop(ticker)
        reason = state.pending_exits.pop(ticker)
        notional = pos.shares * price
        proceeds, exit_cost = cost.sell(notional)
        state.cash += proceeds
        record.fills.append(_fill(state, d, ticker, "sell", pos.shares, price, notional, exit_cost, reason))
        record.trades.append(
            {
                "paper_id": state.paper_id,
                "account": state.account,
                "ticker": ticker,
                "entry_date": pos.entry_date,
                "exit_date": _day(d),
                "entry_price": pos.entry_price,
                "exit_price": price,
                "shares": pos.shares,
                "return": cost.trade_return(pos.entry_price, price),
                "gross_return": price / pos.entry_price - 1.0,
                "holding_days": int(i - view.calendar.get_loc(pd.Timestamp(pos.entry_date))),
                "entry_cost": pos.entry_cost,
                "exit_cost": exit_cost,
                "exit_reason": reason,
            }
        )
        record.order_updates.append(_update(state, ticker, "sell", "filled", d))

    # 2-2. 시가: 진입 (예산 = min(시가 기준 총자산 / max_positions, 현금))
    if state.pending_entries:
        held = len(state.positions)
        prev_close = view.closes_ff.loc[prev]
        mark_value = sum(
            p.shares * (float(opens[t]) if not math.isnan(float(opens[t])) else float(prev_close[t]))
            for t, p in state.positions.items()
        )
        equity_open = state.cash + mark_value
        for ticker in state.pending_entries:
            if held >= rules.max_positions:
                record.order_updates.append(_update(state, ticker, "buy", "skipped", d))
                continue
            price = float(opens[ticker])
            if math.isnan(price):
                record.order_updates.append(_update(state, ticker, "buy", "canceled", d))
                continue  # 시가가 없으면 진입 취소
            budget = min(equity_open / rules.max_positions, state.cash)
            if budget <= 0:
                record.order_updates.append(_update(state, ticker, "buy", "skipped", d))
                continue
            notional, buy_cost = cost.buy(budget)
            state.cash -= budget
            shares = notional / price
            state.positions[ticker] = Position(
                ticker=ticker,
                shares=shares,
                entry_date=_day(d),
                entry_price=price,
                entry_cost=buy_cost,
                mark_date=_day(d),
                mark_price=price,
            )
            record.fills.append(_fill(state, d, ticker, "buy", shares, price, notional, buy_cost, "entry"))
            record.order_updates.append(_update(state, ticker, "buy", "filled", d))
            held += 1
        state.pending_entries = []

    # 3. 종가: 평가
    closes = view.closes_ff.loc[d]
    for pos in state.positions.values():
        pos.mark_price = float(closes[pos.ticker])
        pos.mark_date = _day(d)
    state.last_processed = _day(d)
    record.cash = state.cash
    record.equity = state.cash + sum(p.shares * p.mark_price for p in state.positions.values())
    record.positions = [
        {
            "paper_id": state.paper_id,
            "account": state.account,
            "date": _day(d),
            "ticker": p.ticker,
            "shares": p.shares,
            "entry_date": p.entry_date,
            "entry_price": p.entry_price,
            "mark_price": p.mark_price,
            "value": p.shares * p.mark_price,
            "holding_days": int(i - view.calendar.get_loc(pd.Timestamp(p.entry_date)) + 1),
        }
        for p in sorted(state.positions.values(), key=lambda p: p.ticker)
    ]

    # 4. 다음 거래일 주문
    record.orders = _decide(state, view, d, rules)
    return record


def _decide(state: AccountState, view: MarketView, d: pd.Timestamp, rules: Rules) -> list[dict]:
    """d 종가 기준 다음 시가 주문 (엔진의 decide와 같은 규칙). 새로 정한 주문 행을 반환한다."""
    i = view.calendar.get_loc(d)
    tgt = view.target.loc[d]
    stg = view.strength.loc[d]
    orders = []
    for ticker in sorted(state.positions):
        if ticker in state.pending_exits:
            continue  # 시가가 없어 미뤄진 청산
        held_days = i - view.calendar.get_loc(pd.Timestamp(state.positions[ticker].entry_date)) + 1
        if tgt[ticker] == 0.0 and (rules.min_hold is None or held_days >= rules.min_hold):
            state.pending_exits[ticker] = "signal"
        elif rules.max_hold is not None and held_days >= rules.max_hold:
            state.pending_exits[ticker] = "max_hold"
        else:
            continue
        orders.append(_order(state, d, ticker, "sell", state.pending_exits[ticker], None))
    candidates = [
        t for t in view.tickers if t not in state.positions and tgt[t] == 1.0 and not math.isnan(float(stg[t]))
    ]
    candidates.sort(key=lambda t: (-float(stg[t]), t))
    state.pending_entries = candidates
    orders += [_order(state, d, t, "buy", "entry", rank) for rank, t in enumerate(candidates, start=1)]
    return orders


def _order(state: AccountState, d: pd.Timestamp, ticker: str, side: str, reason: str, rank: int | None) -> dict:
    return {
        "paper_id": state.paper_id,
        "account": state.account,
        "signal_date": _day(d),
        "ticker": ticker,
        "side": side,
        "reason": reason,
        "rank": rank,
        "status": "pending",
        "fill_date": None,
    }


def _update(state: AccountState, ticker: str, side: str, status: str, d: pd.Timestamp) -> dict:
    return {"paper_id": state.paper_id, "account": state.account, "ticker": ticker, "side": side,
            "status": status, "fill_date": _day(d)}


def _fill(
    state: AccountState, d: pd.Timestamp, ticker: str, side: str, shares: float, price: float,
    notional: float, cost: float, reason: str,
) -> dict:
    return {
        "paper_id": state.paper_id,
        "account": state.account,
        "date": _day(d),
        "ticker": ticker,
        "side": side,
        "shares": shares,
        "price": price,
        "notional": notional,
        "cost": cost,
        "cash_after": state.cash,
        "reason": reason,
    }


def _day(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).date().isoformat()
