"""거래 비용 모델: 체결금액 × 편도 비용률을 매수·매도에 각각 적용한다."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    rate: float  # 편도 비용률 (5bp = 0.0005)

    @classmethod
    def from_bps(cls, bps: float) -> CostModel:
        if bps < 0:
            raise ValueError("비용은 0 이상이어야 합니다")
        return cls(rate=bps / 10_000.0)

    def buy(self, budget: float) -> tuple[float, float]:
        """비용 포함 예산으로 매수. (체결금액, 비용) 반환. 체결금액 + 비용 = 예산."""
        notional = budget / (1.0 + self.rate)
        return notional, notional * self.rate

    def sell(self, notional: float) -> tuple[float, float]:
        """체결금액만큼 매도. (현금 유입액, 비용) 반환."""
        cost = notional * self.rate
        return notional - cost, cost

    def trade_return(self, entry_price: float, exit_price: float) -> float:
        """비용 반영 거래 수익률 = 청산가(1 − c) / 진입가(1 + c) − 1."""
        return exit_price * (1.0 - self.rate) / (entry_price * (1.0 + self.rate)) - 1.0
