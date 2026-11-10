"""해외주식 양도소득세 추정 (참고용, 실제 신고와 다를 수 있다).

- 매도 시 실현 차익(원) = 매도 수량 × 매도가 × 매도일 환율 − 매도 수량 × 평균 취득가 × 평균 취득 환율
- 평균 취득가: 이동평균법 (매수마다 수량 가중 평균, 매도해도 바뀌지 않음)
- 평균 취득 환율: 달러 취득금액 가중 평균. 그래서 수량 × 평균 취득가 × 평균 취득 환율 = 실제로 낸 원화 합계
- 연간: 실현 차익 합계(손실 상계) − 기본공제 250만 원, 남는 금액 × 22%
- 반올림: 매도 1건의 원화 양도가액·취득가액은 각각 원 미만 버림, 세액도 원 미만 버림
- 수수료(필요경비)는 빼지 않는다 (스펙의 계산식 그대로, 세액을 약간 크게 잡는 쪽)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_FLOOR, Decimal

import pandas as pd


def _d(x: float | str) -> Decimal:
    return Decimal(str(x))


def _won(x: Decimal) -> int:
    return int(x.to_integral_value(rounding=ROUND_FLOOR))


@dataclass
class Lot:
    """보유분. 평균 취득가·평균 취득 환율은 합계에서 계산한다 (나눗셈 반올림이 원 단위에 끼지 않게)."""

    shares: Decimal = Decimal(0)
    cost_usd: Decimal = Decimal(0)  # 취득 달러 합계
    cost_krw: Decimal = Decimal(0)  # 취득 원화 합계 (= Σ 수량 × 매수가 × 매수일 환율)

    @property
    def avg_cost_usd(self) -> Decimal:
        return self.cost_usd / self.shares if self.shares else Decimal(0)

    @property
    def avg_fx(self) -> Decimal:
        return self.cost_krw / self.cost_usd if self.cost_usd else Decimal(0)


@dataclass
class Realized:
    date: pd.Timestamp
    ticker: str
    shares: float
    proceeds_krw: int
    cost_krw: int

    @property
    def gain_krw(self) -> int:
        return self.proceeds_krw - self.cost_krw


@dataclass
class TaxLedger:
    lots: dict[str, Lot] = field(default_factory=dict)
    realized: list[Realized] = field(default_factory=list)

    def set_holding(self, ticker: str, shares: float, avg_cost_usd: float, avg_fx: float) -> None:
        """이미 가진 보유분 (실사용 모드의 holdings.csv)."""
        q, p = _d(shares), _d(avg_cost_usd)
        self.lots[ticker] = Lot(q, q * p, q * p * _d(avg_fx))

    def buy(self, ticker: str, shares: float, price: float, fx: float) -> None:
        if shares <= 0:
            return
        lot = self.lots.setdefault(ticker, Lot())
        usd = _d(shares) * _d(price)
        lot.shares += _d(shares)
        lot.cost_usd += usd
        lot.cost_krw += usd * _d(fx)

    def sell(self, date: pd.Timestamp, ticker: str, shares: float, price: float, fx: float) -> Realized:
        lot = self.lots.get(ticker)
        q = _d(shares)
        if lot is None or q > lot.shares * (1 + Decimal("1e-9")):
            raise ValueError(f"{ticker}: 보유 {lot.shares if lot else 0}보다 많이 팔 수 없습니다 ({shares})")
        q = min(q, lot.shares)
        sold_krw = lot.cost_krw * q / lot.shares  # = 수량 × 평균 취득가 × 평균 취득 환율 (곱한 뒤 나눈다)
        sold_usd = lot.cost_usd * q / lot.shares
        realized = Realized(
            date=pd.Timestamp(date),
            ticker=ticker,
            shares=shares,
            proceeds_krw=_won(q * _d(price) * _d(fx)),
            cost_krw=_won(sold_krw),
        )
        lot.cost_usd -= sold_usd
        lot.cost_krw -= sold_krw
        lot.shares -= q
        self.realized.append(realized)
        return realized


def tax_due(gain_krw: int, rate: float, deduction_krw: int) -> int:
    """연간 실현 차익 합계 → 세액 (원 미만 버림)."""
    taxable = max(gain_krw - deduction_krw, 0)
    return _won(_d(taxable) * _d(rate))


def yearly_tax(realized: list[Realized], rate: float, deduction_krw: int) -> pd.DataFrame:
    """연도별 실현 차익, 과세표준, 세액."""
    rows: dict[int, int] = {}
    for r in realized:
        rows[r.date.year] = rows.get(r.date.year, 0) + r.gain_krw
    out = [
        {"year": y, "gain_krw": g, "taxable_krw": max(g - deduction_krw, 0), "tax_krw": tax_due(g, rate, deduction_krw)}
        for y, g in sorted(rows.items())
    ]
    return pd.DataFrame(out, columns=["year", "gain_krw", "taxable_krw", "tax_krw"])


def ledger_from_trades(trades: pd.DataFrame, fx: pd.Series) -> TaxLedger:
    """백테스트 체결 기록(날짜순)으로 원장을 만든다. fx = 체결일 환율 (date 인덱스)."""
    ledger = TaxLedger()
    for t in trades.sort_values("date", kind="stable").itertuples(index=False):
        rate = float(fx.loc[pd.Timestamp(t.date)])
        if t.side == "buy":
            ledger.buy(t.ticker, float(t.shares), float(t.price), rate)
        else:
            ledger.sell(pd.Timestamp(t.date), t.ticker, float(t.shares), float(t.price), rate)
    return ledger


def after_tax_curve(equity: pd.Series, taxes: pd.DataFrame, fx: pd.Series) -> pd.Series:
    """세금을 낸 것으로 본 자산곡선 (근사).

    그해 세액을 그해 마지막 거래일 종가에 원화→달러로 바꿔 빼고, 그 뒤의 수익률은 그대로 받는다
    (포트폴리오를 그 비율만큼 줄인 것과 같다. 납부 재원을 마련하는 매도의 추가 세금은 무시).
    실제 납부는 다음 해 5월이다.
    """
    out = equity.astype(float).copy()
    factor = pd.Series(1.0, index=equity.index)
    for row in taxes.itertuples(index=False):
        if row.tax_krw <= 0:
            continue
        days = equity.index[equity.index.year == row.year]
        if len(days) == 0:
            continue
        pay_day = days[-1]
        pre = float(equity.loc[pay_day] * factor.loc[pay_day])
        tax_usd = row.tax_krw / float(fx.loc[pay_day])
        factor.loc[equity.index >= pay_day] *= 1.0 - tax_usd / pre
    return out * factor
