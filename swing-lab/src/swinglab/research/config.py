"""config/stage1.yaml 로딩: 구간, 비용, 예산, 선정 규칙, 계열별 파라미터 격자."""

from __future__ import annotations

from datetime import date
from itertools import product
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

GridIndex = tuple[int, ...]


class Period(BaseModel):
    start: date
    end: date | None = None
    evaluate_from: date | None = None  # 성과 집계 시작일 (모든 전략 공통). 없으면 전략별 워밍업 이후부터

    @property
    def evaluation_start(self) -> date:
        return self.evaluate_from or self.start


class Budget(BaseModel):
    per_family: int = Field(gt=0)
    total: int = Field(gt=0)


class SelectionRules(BaseModel):
    min_trades: int
    min_trades_exempt: list[str] = Field(default_factory=list)
    isolation_ratio: float
    filter_min_sharpe_gain: float


class FamilyConfig(BaseModel):
    strategy: str | None = None
    universe: list[str] | None = None  # None이면 벤치마크를 뺀 유니버스 전체
    fixed: dict[str, Any] = Field(default_factory=dict)
    axes: dict[str, list[Any]] = Field(default_factory=dict)
    wraps: list[str] = Field(default_factory=list)  # 필터 계열: 감쌀 계열의 후보
    extra: list[str] = Field(default_factory=list)  # 필터 계열: 기본 파라미터로 감쌀 전략

    @property
    def is_filter_family(self) -> bool:
        return bool(self.wraps or self.extra)

    def grid(self) -> list[tuple[GridIndex, dict[str, Any]]]:
        """(축별 인덱스, 파라미터) 목록. 축 값이 매핑이면 그 키들을 한 번에 넣는다."""
        names = list(self.axes)
        combos = []
        for idx in product(*(range(len(self.axes[n])) for n in names)):
            params = dict(self.fixed)
            for name, i in zip(names, idx):
                value = self.axes[name][i]
                if isinstance(value, dict):
                    params.update(value)
                else:
                    params[name] = value
            combos.append((tuple(idx), params))
        return combos

    @staticmethod
    def neighbors(idx: GridIndex, all_idx: list[GridIndex]) -> list[GridIndex]:
        """축 하나만 한 칸 다른 조합."""
        return [o for o in all_idx if sum(abs(a - b) for a, b in zip(idx, o)) == 1]

    def planned_trials(self) -> int:
        return len(self.wraps) + len(self.extra) if self.is_filter_family else len(self.grid())


class Stage1Config(BaseModel):
    periods: dict[str, Period]
    cost_bps: list[float]
    selection_cost_bps: float
    budget: Budget
    selection: SelectionRules
    families: dict[str, FamilyConfig]

    @property
    def train(self) -> Period:
        return self.periods["train"]

    @property
    def holdout(self) -> Period:
        return self.periods["holdout"]

    @model_validator(mode="after")
    def _check(self) -> Stage1Config:
        if set(self.periods) != {"train", "holdout"}:
            raise ValueError("periods에는 train, holdout만 있어야 합니다")
        if self.train.end is None or self.train.end >= self.holdout.start:
            raise ValueError("학습 구간은 검증 구간 시작 전에 끝나야 합니다")
        ev = self.train.evaluate_from
        if ev is not None and not (self.train.start < ev <= self.train.end):
            raise ValueError("evaluate_from은 학습 구간 안에 있어야 합니다")
        if self.selection_cost_bps not in self.cost_bps:
            raise ValueError("selection_cost_bps는 cost_bps 안에 있어야 합니다")
        for name, fam in self.families.items():
            if fam.planned_trials() > self.budget.per_family:
                raise ValueError(f"{name} 계열 조합 수가 계열 예산 {self.budget.per_family}을 넘습니다")
            for w in fam.wraps:
                if w not in self.families:
                    raise ValueError(f"{name}.wraps의 {w} 계열이 없습니다")
        total = sum(f.planned_trials() for f in self.families.values())
        if total > self.budget.total:
            raise ValueError(f"전체 조합 수 {total}이 예산 {self.budget.total}을 넘습니다")
        return self


def load_stage1(config_dir: Path) -> Stage1Config:
    with (config_dir / "stage1.yaml").open(encoding="utf-8") as f:
        return Stage1Config.model_validate(yaml.safe_load(f))
