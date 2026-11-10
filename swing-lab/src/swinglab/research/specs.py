"""실험 한 건을 정의하는 전략 명세 (계열, 전략 이름, 파라미터, 시장 필터 여부, 대상 종목)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from swinglab.config import UniverseConfig
from swinglab.strategies import Strategy, build_parametrized

FILTER_SUFFIX = "+market_filter"


@dataclass(frozen=True)
class StrategySpec:
    family: str
    strategy: str
    params: dict[str, Any] = field(default_factory=dict)
    market_filter: bool = False
    tickers: tuple[str, ...] | None = None  # None이면 벤치마크를 뺀 유니버스 전체

    @property
    def display_name(self) -> str:
        """실험 로그의 strategy 컬럼 값. 필터 버전은 이름 뒤에 +market_filter."""
        return self.strategy + (FILTER_SUFFIX if self.market_filter else "")

    @property
    def params_json(self) -> str:
        return canonical_json(self.params)

    def build(self, universe: UniverseConfig) -> Strategy:
        return build_parametrized(self.strategy, dict(self.params), universe, self.tickers, self.market_filter)

    def with_filter(self, family: str) -> StrategySpec:
        return StrategySpec(family, self.strategy, dict(self.params), True, self.tickers)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "family": self.family,
            "strategy": self.strategy,
            "params": dict(self.params),
            "market_filter": self.market_filter,
        }
        if self.tickers is not None:
            out["tickers"] = list(self.tickers)
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StrategySpec:
        tickers = d.get("tickers")
        return cls(
            family=str(d["family"]),
            strategy=str(d["strategy"]),
            params=dict(d.get("params") or {}),
            market_filter=bool(d.get("market_filter", False)),
            tickers=tuple(tickers) if tickers else None,
        )


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def specs_hash(specs: list[StrategySpec]) -> str:
    """후보 설정 해시. 순서와 무관하게 같은 후보 집합이면 같은 값."""
    payload = sorted(canonical_json(s.to_dict()) for s in specs)
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
