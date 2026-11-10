"""배분 정책 P0~P3 → 목표 비중 신호 일정 (backtest/weights_engine.py 입력).

- 분기 재조정: 분기 마지막 거래일(3·6·9·12월 말) 신호 → 다음 거래일 시가 체결, 허용 범위 적용
- P3 추세 필터: 매월 마지막 거래일에 위험자산(SPY, EFA, EEM, GLD)의 월말 종가를 최근 10개 월말 종가 평균과 비교.
  아래(종가 < 평균)면 그 자산 목표를 0으로 하고 비중을 현금 대용(BIL)으로 옮긴다. 다시 위로 올라온 월말에 복귀.
  상태가 바뀐 자산은 허용 범위와 무관하게 다음 거래일 시가에 매매한다 (분기말이 아닌 달에는 그 자산과 BIL만).
  평균을 낼 월말이 모자라면(상장 초기) 필터를 적용하지 않는다 (위로 본다).
- 월말 판정: 다음 거래일의 달이 다른 날. 데이터 마지막 날은 다음 날을 모르므로 신호를 내지 않는다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from swinglab.alloc.config import AllocConfig, GlidePath
from swinglab.backtest.weights_engine import TargetSignal
from swinglab.strategies.base import price_panel

EQUITY_ASSETS = ("SPY", "EFA", "EEM")  # 목표 시점 조절에서 '주식'
GLIDE_DESTINATIONS = ("IEF", "BIL")


@dataclass(frozen=True)
class Policy:
    name: str
    weights: dict[str, float]
    trend_filter: bool
    trend_assets: tuple[str, ...]
    trend_months: int
    cash_proxy: str

    @classmethod
    def from_config(cls, name: str, cfg: AllocConfig) -> Policy:
        p = cfg.policies[name]
        return cls(name, dict(p.weights), p.trend_filter, tuple(cfg.trend.assets), cfg.trend.months, cfg.cash_proxy)

    @property
    def tickers(self) -> list[str]:
        out = list(self.weights)
        if self.cash_proxy not in out:
            out.append(self.cash_proxy)
        return out

    def params_json(self) -> str:
        """실험 로그의 params (정책 정의가 바뀌면 새 시도가 된다)."""
        params: dict = {"weights": self.weights}
        if self.trend_filter:
            params["trend"] = {"assets": list(self.trend_assets), "months": self.trend_months}
        return json.dumps(params, sort_keys=True, separators=(",", ":"))

    def targets(self, below: dict[str, bool]) -> dict[str, float]:
        """추세 상태(below[자산] = 10개월선 아래)를 반영한 목표 비중."""
        out = dict(self.weights)
        if self.trend_filter:
            for asset in self.trend_assets:
                if below.get(asset) and out.get(asset, 0.0) > 0:
                    out[self.cash_proxy] = out.get(self.cash_proxy, 0.0) + out[asset]
                    out[asset] = 0.0
        return out


def month_ends(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """월 마지막 거래일 (마지막 날 제외: 다음 거래일을 알아야 판정할 수 있다)."""
    if len(calendar) < 2:
        return pd.DatetimeIndex([])
    mask = np.r_[calendar.month[1:] != calendar.month[:-1], False]
    return calendar[mask]


def quarter_ends(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    me = month_ends(calendar)
    return me[me.month.isin([3, 6, 9, 12])]


def trend_below(prices: pd.DataFrame, policy: Policy, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """월말 × 위험자산: 월말 종가 < 최근 trend_months개 월말 종가 평균이면 True. 평균이 없으면 False."""
    assets = [a for a in policy.trend_assets if a in policy.weights]
    me = month_ends(calendar)
    closes = price_panel(prices, "close", assets).reindex(calendar).ffill().reindex(me)
    sma = closes.rolling(policy.trend_months, min_periods=policy.trend_months).mean()
    return (closes < sma) & sma.notna()


def build_schedule(
    policy: Policy, prices: pd.DataFrame, calendar: pd.DatetimeIndex, anchor: pd.Timestamp
) -> dict[pd.Timestamp, TargetSignal]:
    """anchor(첫 체결일 전날)의 초기 배분 + 이후 분기·월간 신호."""
    me = month_ends(calendar)
    qe = set(quarter_ends(calendar))
    below = trend_below(prices, policy, calendar) if policy.trend_filter else pd.DataFrame(index=me)

    def state_at(d: pd.Timestamp) -> dict[str, bool]:
        """d 이전(포함) 마지막 월말의 추세 상태."""
        if not policy.trend_filter:
            return {}
        past = below.index[below.index <= d]
        return {} if len(past) == 0 else {a: bool(v) for a, v in below.loc[past[-1]].items()}

    schedule = {anchor: TargetSignal(anchor, policy.targets(state_at(anchor)), "initial")}
    prev = state_at(anchor)
    for m in me[me > anchor]:
        state = state_at(m)
        flipped = frozenset(a for a in state if state[a] != prev.get(a, False))
        weights = policy.targets(state)
        if m in qe:
            schedule[m] = TargetSignal(m, weights, "quarterly", forced=flipped, full=True)
        elif flipped:
            schedule[m] = TargetSignal(m, weights, "trend", forced=flipped, full=False)
        prev = state
    return schedule


def trend_status(prices: pd.DataFrame, policy: Policy, as_of: pd.Timestamp) -> pd.DataFrame:
    """as_of 이전 마지막 월말 기준 자산별 월말 종가, 10개월 평균, 아래 여부 (실사용 문서용).

    as_of가 그달 마지막 거래일이면 그날도 월말로 친다 (다음 거래일 시가 주문을 위해 장 마감 후 실행).
    """
    from swinglab.pipeline.calendar import nyse_sessions

    calendar = pd.DatetimeIndex(np.sort(prices.loc[prices["date"] <= as_of, "date"].unique()))
    upcoming = nyse_sessions(as_of.date(), (as_of + pd.Timedelta(days=10)).date())
    nxt = upcoming[upcoming > as_of]
    extended = calendar.append(pd.DatetimeIndex([nxt[0]])) if len(nxt) else calendar
    me = month_ends(extended)
    assets = [a for a in policy.trend_assets if a in policy.weights]
    closes = price_panel(prices, "close", assets).reindex(calendar).ffill().reindex(me)
    sma = closes.rolling(policy.trend_months, min_periods=policy.trend_months).mean()
    last = me[-1]
    return pd.DataFrame(
        {
            "asset": assets,
            "month_end": last.date().isoformat(),
            "close": [float(closes.at[last, a]) for a in assets],
            "sma": [float(sma.at[last, a]) for a in assets],
            "below": [bool(closes.at[last, a] < sma.at[last, a]) for a in assets],
        }
    )


def apply_glide(weights: dict[str, float], as_of: date, glide: GlidePath, target_date: date | None) -> dict[str, float]:
    """목표 시점 대비 위험 조절. 꺼져 있으면 그대로.

    주식(SPY, EFA, EEM) 합계 비중을 start_equity → end_equity로 (start_date → target_date) 선형으로 바꾸고,
    주식끼리는 원래 비율을 유지한다. 줄어든 만큼은 IEF·BIL에 정책 비중 비율대로 (둘 다 0이면 반반) 나눈다.
    """
    if not glide.enabled or target_date is None:
        return dict(weights)
    span = (target_date - glide.start_date).days
    f = min(max((as_of - glide.start_date).days / span, 0.0), 1.0)
    equity_goal = glide.start_equity + f * (glide.end_equity - glide.start_equity)
    out = dict(weights)
    equity_now = sum(out.get(a, 0.0) for a in EQUITY_ASSETS)
    if equity_now <= 0:
        return out
    scale = min(equity_goal / equity_now, 1.0)  # 주식 비중을 늘리지는 않는다
    freed = equity_now * (1.0 - scale)
    for a in EQUITY_ASSETS:
        if a in out:
            out[a] *= scale
    dest = {d: weights.get(d, 0.0) for d in GLIDE_DESTINATIONS}
    total = sum(dest.values())
    for d in GLIDE_DESTINATIONS:
        out[d] = out.get(d, 0.0) + freed * (dest[d] / total if total > 0 else 1.0 / len(GLIDE_DESTINATIONS))
    return out
