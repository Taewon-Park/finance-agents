"""성과 지표를 손계산 값과 비교한다."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from swinglab.backtest.engine import (
    OPEN_POSITION_COLUMNS,
    TRADE_COLUMNS,
    BacktestResult,
)
from swinglab.backtest.metrics import (
    cagr,
    max_drawdown,
    performance_metrics,
    sharpe_ratio,
)


def test_cagr_two_calendar_years() -> None:
    idx = pd.to_datetime(["2020-01-01", "2022-01-01"])  # 731일
    eq = pd.Series([100.0, 121.0], index=idx)
    assert cagr(eq) == pytest.approx(1.21 ** (365.25 / 731) - 1)


def test_max_drawdown() -> None:
    eq = pd.Series([100.0, 120.0, 90.0, 130.0, 104.0])
    assert max_drawdown(eq) == pytest.approx(90 / 120 - 1)  # -25%


def test_sharpe_from_daily_returns() -> None:
    rets = pd.Series([0.01, -0.02, 0.03, 0.0])
    mean = 0.005
    std = math.sqrt(((0.005) ** 2 + (0.025) ** 2 + (0.025) ** 2 + (0.005) ** 2) / 3)
    assert sharpe_ratio(rets) == pytest.approx(mean / std * math.sqrt(252))


def test_performance_metrics_trade_stats() -> None:
    idx = pd.bdate_range("2021-01-01", periods=5)
    equity = pd.Series([100.0, 101.0, 102.0, 101.0, 103.0], index=idx)
    positions = pd.DataFrame({"A": [0, 1, 1, 0, 0]}, index=idx, dtype=float)
    trades = pd.DataFrame(
        [
            ["A", idx[0], idx[1], 1, 1, 1, r, r, h, 0, 0, "signal"]
            for r, h in [(0.1, 2), (-0.05, 4), (0.2, 3), (-0.1, 1)]
        ],
        columns=TRADE_COLUMNS,
    )
    result = BacktestResult(
        strategy_name="t",
        cost_bps=0,
        max_positions=5,
        equity=equity,
        cash=equity,
        positions=positions,
        trades=trades,
        open_positions=pd.DataFrame(columns=OPEN_POSITION_COLUMNS),
        total_cost=0.0,
    )
    m = performance_metrics(result)
    assert m["win_rate"] == 0.5
    assert m["payoff_ratio"] == pytest.approx(0.15 / 0.075)  # 평균 수익 / |평균 손실|
    assert m["avg_holding_days"] == 2.5
    assert m["exposure"] == 0.5  # 첫 행 제외 4일 중 2일 보유
    years = (idx[-1] - idx[0]).days / 365.25
    assert m["trades_per_year"] == pytest.approx(4 / years)
    assert m["total_return"] == pytest.approx(0.03)


def test_metrics_without_trades_are_nan() -> None:
    idx = pd.bdate_range("2021-01-01", periods=3)
    eq = pd.Series([100.0, 100.0, 100.0], index=idx)
    result = BacktestResult(
        "t", 0, 5, eq, eq, pd.DataFrame({"A": [0.0] * 3}, index=idx),
        pd.DataFrame(columns=TRADE_COLUMNS), pd.DataFrame(columns=OPEN_POSITION_COLUMNS), 0.0,
    )
    m = performance_metrics(result)
    assert np.isnan(m["win_rate"]) and np.isnan(m["payoff_ratio"]) and np.isnan(m["sharpe"])
    assert m["exposure"] == 0.0
