"""config/alloc.yaml 로딩과 검증."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

WEIGHT_TOLERANCE = 1e-9


class PolicyConfig(BaseModel):
    weights: dict[str, float]
    trend_filter: bool = False

    @model_validator(mode="after")
    def _sum_to_one(self) -> PolicyConfig:
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("비중은 0 이상이어야 합니다")
        if abs(sum(self.weights.values()) - 1.0) > WEIGHT_TOLERANCE:
            raise ValueError(f"비중 합이 1이 아닙니다: {sum(self.weights.values())}")
        return self


class TrendConfig(BaseModel):
    assets: list[str]
    months: int = Field(ge=2)


class DrawdownPeriod(BaseModel):
    name: str
    start: date
    end: date


class BacktestSettings(BaseModel):
    start: date
    end: date | None = None
    initial_capital: float = Field(gt=0)
    cost_bps: list[float]
    report_cost_bps: float
    drawdown_periods: list[DrawdownPeriod] = Field(default_factory=list)


class GoalConfig(BaseModel):
    policy: str
    max_drawdown: float = Field(lt=0)
    cost_bps: float


class TaxConfig(BaseModel):
    rate: float = Field(ge=0, le=1)
    deduction_krw: int = Field(ge=0)
    fx_pair: str


class GlidePath(BaseModel):
    enabled: bool = False
    start_date: date | None = None
    start_equity: float | None = None
    end_equity: float | None = None

    @model_validator(mode="after")
    def _complete_when_enabled(self) -> GlidePath:
        if self.enabled:
            if self.start_date is None or self.start_equity is None or self.end_equity is None:
                raise ValueError("glide_path를 켜면 start_date, start_equity, end_equity를 모두 정해야 합니다")
            if not (0 <= self.end_equity <= 1 and 0 <= self.start_equity <= 1):
                raise ValueError("주식 비중은 0~1 사이여야 합니다")
        return self


class LiveConfig(BaseModel):
    policy: str
    holdings: Path
    output_dir: Path
    cost_bps: float = Field(ge=0)
    target_date: date | None = None
    glide_path: GlidePath = Field(default_factory=GlidePath)

    @model_validator(mode="after")
    def _target_date_for_glide(self) -> LiveConfig:
        if self.glide_path.enabled:
            if self.target_date is None:
                raise ValueError("glide_path를 켜면 target_date가 필요합니다")
            if self.glide_path.start_date >= self.target_date:
                raise ValueError("glide_path.start_date는 target_date보다 앞이어야 합니다")
        return self


class LlmConfig(BaseModel):
    enabled: bool = False
    endpoint: str
    model: str
    timeout_seconds: float = Field(gt=0)
    max_retries: int = Field(ge=1)


class AllocConfig(BaseModel):
    band: float = Field(gt=0, lt=1)
    cash_proxy: str
    trend: TrendConfig
    policies: dict[str, PolicyConfig]
    max_trials: int = Field(ge=1)
    backtest: BacktestSettings
    goal: GoalConfig
    tax: TaxConfig
    live: LiveConfig
    llm: LlmConfig

    @model_validator(mode="after")
    def _check(self) -> AllocConfig:
        if len(self.policies) > self.max_trials:
            raise ValueError(f"정책 수 {len(self.policies)}가 max_trials {self.max_trials}를 넘습니다")
        for name in (self.goal.policy, self.live.policy):
            if name not in self.policies:
                raise ValueError(f"정책 {name}이 policies에 없습니다")
        if self.backtest.report_cost_bps not in self.backtest.cost_bps:
            raise ValueError("report_cost_bps는 cost_bps 안에 있어야 합니다")
        return self

    @property
    def tickers(self) -> list[str]:
        """정책이 쓰는 종목 전체 (현금 대용 포함)."""
        seen: dict[str, None] = {}
        for p in self.policies.values():
            for t in p.weights:
                seen.setdefault(t, None)
        seen.setdefault(self.cash_proxy, None)
        return list(seen)


def load_alloc_config(config_dir: Path) -> AllocConfig:
    with (config_dir / "alloc.yaml").open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{config_dir / 'alloc.yaml'}: YAML 최상위는 매핑이어야 합니다")
    return AllocConfig.model_validate(data)
