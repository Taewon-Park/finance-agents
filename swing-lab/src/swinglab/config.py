"""config/*.yaml 로딩과 검증."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class UniverseConfig(BaseModel):
    benchmark: str
    start: date
    groups: dict[str, list[str]]

    @property
    def tickers(self) -> list[str]:
        """그룹 순서를 유지한 전체 티커 목록 (중복 제거)."""
        seen: dict[str, None] = {}
        for members in self.groups.values():
            for ticker in members:
                seen.setdefault(ticker, None)
        return list(seen)

    @property
    def tradable(self) -> list[str]:
        """매매 전략 대상: 벤치마크를 뺀 나머지."""
        return [t for t in self.tickers if t != self.benchmark]

    @model_validator(mode="after")
    def _benchmark_in_universe(self) -> UniverseConfig:
        if self.benchmark not in self.tickers:
            raise ValueError(f"benchmark {self.benchmark}가 groups에 없습니다")
        return self


class BacktestConfig(BaseModel):
    initial_capital: float = Field(default=100_000.0, gt=0)
    cost_bps: float = Field(default=5.0, ge=0)
    cost_sensitivity_bps: list[float] = Field(default_factory=lambda: [0.0, 5.0, 10.0])
    max_positions: int = Field(default=5, ge=1)
    start: date | None = None
    end: date | None = None

    @property
    def cost_rate(self) -> float:
        """편도 비용률 (5bp → 0.0005)."""
        return self.cost_bps / 10_000.0


def load_universe(config_dir: Path) -> UniverseConfig:
    return UniverseConfig.model_validate(_read_yaml(config_dir / "universe.yaml"))


def load_backtest_config(config_dir: Path) -> BacktestConfig:
    return BacktestConfig.model_validate(_read_yaml(config_dir / "backtest.yaml"))


def _read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: YAML 최상위는 매핑이어야 합니다")
    return data
