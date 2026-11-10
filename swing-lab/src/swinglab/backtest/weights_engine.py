"""목표 비중 엔진 (2단계 트랙 2). 0단계 엔진(보유/미보유, 동일 비중)과 별도.

체결·비용 규칙은 0단계와 같다
- 신호일 종가 기준으로 정한 목표를 다음 거래일 시가에 체결한다. 롱 온리, 소수 주식.
- 비용: 체결금액 × 편도 비용률을 매수·매도 각각에. 매도 먼저, 그다음 매수.
- 평가: 종가 (결측일은 직전 종가). 자산곡선 첫 행 = 첫 체결일 전날 (초기자본, 전액 현금).

목표 비중 규칙
- 신호(TargetSignal)마다 목표 비중(합 1)을 받는다. 남는 비중은 현금 대용(BIL)이 받는다.
- full=True(분기 재조정): 현재 비중이 목표에서 band(절대값, 기본 5%p) 이상 벗어난 종목만 목표로 맞춘다.
- forced: 허용 범위와 무관하게 목표로 맞출 종목 (P3 추세 필터의 이동). full=False면 forced만 매매한다.
- 현금 대용(BIL)은 허용 범위 점검 대상이 아니다. 매매 후 남는 현금으로 사고(sweep),
  매수 자금이 모자라면 팔아서 채운다(fund). 그래도 모자라면 매수를 같은 비율로 줄인다.
  남는 현금이 총자산의 sweep_min_fraction 미만이면 현금(수익률 0)으로 둔다.
- 체결일 시가가 없는 종목은 그 신호에서 매매하지 않는다 (다음 신호에서 다시 판단).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from swinglab.backtest.costs import CostModel
from swinglab.strategies.base import price_panel

BAND_TOLERANCE = 1e-9  # 5%p 경계의 부동소수 오차


@dataclass(frozen=True)
class TargetSignal:
    date: pd.Timestamp  # 신호일 (그날 종가까지의 정보로 정함)
    weights: dict[str, float]
    kind: str  # initial | quarterly | trend
    forced: frozenset[str] = frozenset()
    full: bool = True


@dataclass(frozen=True)
class WeightsConfig:
    initial_capital: float = 100_000.0
    cost_bps: float = 5.0
    band: float = 0.05
    cash_proxy: str | None = "BIL"
    sweep_min_fraction: float = 0.001
    start: date | None = None
    end: date | None = None


TRADE_COLUMNS = ["date", "signal_date", "kind", "ticker", "side", "shares", "price", "notional", "cost", "reason"]


@dataclass
class WeightsResult:
    name: str
    cost_bps: float
    equity: pd.Series
    cash: pd.Series
    shares: pd.DataFrame
    weights: pd.DataFrame  # 종가 기준 비중
    trades: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=TRADE_COLUMNS))
    executions: pd.DataFrame = field(default_factory=pd.DataFrame)  # 신호별 체결 요약

    @property
    def start_date(self) -> pd.Timestamp:
        return self.equity.index[1]

    @property
    def end_date(self) -> pd.Timestamp:
        return self.equity.index[-1]


def run_weights(
    prices: pd.DataFrame,
    tickers: list[str],
    schedule: dict[pd.Timestamp, TargetSignal],
    config: WeightsConfig,
    name: str = "weights",
) -> WeightsResult:
    """schedule = 신호일 → 목표. 첫 체결일 전날(anchor)의 신호가 초기 배분이다."""
    cost = CostModel.from_bps(config.cost_bps)
    prices = prices[prices["ticker"].isin(tickers)]
    calendar = pd.DatetimeIndex(np.sort(prices["date"].unique()))
    opens = price_panel(prices, "open", tickers).reindex(calendar).to_numpy(dtype=float)
    closes_ff = price_panel(prices, "close", tickers).reindex(calendar).ffill().to_numpy(dtype=float)
    first_idx, last_idx = _window(calendar, config)
    anchor_idx = first_idx - 1
    proxy = tickers.index(config.cash_proxy) if config.cash_proxy in tickers else None

    k = len(tickers)
    cash = float(config.initial_capital)
    shares = np.zeros(k)
    trades: list[dict] = []
    executions: list[dict] = []

    def execute(sig: TargetSignal, i: int) -> None:
        nonlocal cash
        day = calendar[i]
        o = opens[i]
        tradable = ~np.isnan(o)
        mark = np.where(tradable, o, closes_ff[i - 1])
        values = shares * np.nan_to_num(mark)
        total = cash + float(values.sum())
        current = values / total
        target = np.array([float(sig.weights.get(t, 0.0)) for t in tickers])
        n_before = len(trades)

        def record(j: int, side: str, n_shares: float, notional: float, fee: float, reason: str) -> None:
            trades.append({"date": day, "signal_date": sig.date, "kind": sig.kind, "ticker": tickers[j], "side": side,
                           "shares": n_shares, "price": float(o[j]), "notional": notional, "cost": fee,
                           "reason": reason})

        chosen = []
        for j in range(k):
            if j == proxy or not tradable[j]:
                continue
            breach = sig.full and abs(current[j] - target[j]) >= config.band - BAND_TOLERANCE
            selected = sig.kind == "initial" or tickers[j] in sig.forced or breach
            if selected and abs(target[j] * total - values[j]) > 1e-9 * total:
                chosen.append(j)

        # 매도 먼저
        for j in chosen:
            excess = values[j] - target[j] * total
            if excess <= 0:
                continue
            sold = shares[j] if target[j] == 0.0 else excess / o[j]
            notional = sold * o[j]
            proceeds, fee = cost.sell(notional)
            shares[j] -= sold
            if target[j] == 0.0:
                shares[j] = 0.0
            cash += proceeds
            record(j, "sell", sold, notional, fee, sig.kind)

        buys = {j: target[j] * total - values[j] for j in chosen if target[j] * total > values[j]}
        need = sum(buys.values()) * (1.0 + cost.rate)
        if need > cash and proxy is not None and shares[proxy] > 0 and tradable[proxy]:
            proxy_value = shares[proxy] * o[proxy]
            notional = min(proxy_value, (need - cash) / (1.0 - cost.rate))
            sold = shares[proxy] if notional >= proxy_value else notional / o[proxy]
            proceeds, fee = cost.sell(sold * o[proxy])
            shares[proxy] -= sold
            cash += proceeds
            record(proxy, "sell", sold, sold * o[proxy], fee, "fund")
        scale = min(1.0, cash / need) if need > 0 else 1.0
        for j, amount in buys.items():
            notional = amount * scale
            fee = notional * cost.rate
            cash -= notional + fee
            shares[j] += notional / o[j]
            record(j, "buy", notional / o[j], notional, fee, sig.kind)
        cash = max(cash, 0.0) if abs(cash) < 1e-9 * total else cash

        if proxy is not None and tradable[proxy] and cash > config.sweep_min_fraction * total:
            notional, fee = cost.buy(cash)
            shares[proxy] += notional / o[proxy]
            cash = 0.0
            record(proxy, "buy", notional / o[proxy], notional, fee, "sweep")

        executions.append({"signal_date": sig.date, "date": day, "kind": sig.kind, "n_trades": len(trades) - n_before})

    eq_rows, cash_rows, share_rows = [cash], [cash], [shares.copy()]
    pending = schedule.get(calendar[anchor_idx])
    if pending is None:
        raise ValueError(f"첫 체결일 전날({calendar[anchor_idx].date()})의 초기 배분 신호가 없습니다")
    for i in range(first_idx, last_idx + 1):
        if pending is not None:
            execute(pending, i)
            pending = None
        eq_rows.append(cash + float(np.sum(shares * closes_ff[i])))
        cash_rows.append(cash)
        share_rows.append(shares.copy())
        if i < last_idx:
            pending = schedule.get(calendar[i])

    index = calendar[anchor_idx : last_idx + 1]
    equity = pd.Series(eq_rows, index=index, name="equity")
    share_df = pd.DataFrame(share_rows, index=index, columns=tickers)
    close_df = pd.DataFrame(closes_ff[anchor_idx : last_idx + 1], index=index, columns=tickers)
    weights = (share_df * close_df.fillna(0.0)).div(equity, axis=0)
    return WeightsResult(
        name=name,
        cost_bps=config.cost_bps,
        equity=equity,
        cash=pd.Series(cash_rows, index=index, name="cash"),
        shares=share_df,
        weights=weights,
        trades=pd.DataFrame(trades, columns=TRADE_COLUMNS),
        executions=pd.DataFrame(executions, columns=["signal_date", "date", "kind", "n_trades"]),
    )


def _window(calendar: pd.DatetimeIndex, config: WeightsConfig) -> tuple[int, int]:
    if len(calendar) < 2:
        raise ValueError("백테스트에는 최소 2거래일이 필요합니다")
    start_idx = int(calendar.searchsorted(pd.Timestamp(config.start), side="left")) if config.start else 0
    last_idx = len(calendar) - 1
    if config.end is not None:
        last_idx = int(calendar.searchsorted(pd.Timestamp(config.end), side="right")) - 1
    first_idx = max(start_idx, 1)
    if first_idx > last_idx:
        raise ValueError("백테스트 구간이 비었습니다")
    return first_idx, last_idx
