"""누수 테스트: t+1일 이후 가격을 무작위로 바꿔도 t일까지의 신호·강도·자산가치는 변하지 않아야 한다."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import random_walk_prices

from swinglab.backtest import run_backtest
from swinglab.config import BacktestConfig
from swinglab.strategies import BuyAndHold, Rsi2Pullback, SmaCross

TICKERS = ["AAA", "BBB", "CCC", "DDD", "SPY"]
N = 600


def perturb_after(prices: pd.DataFrame, cutoff: pd.Timestamp, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = prices.copy()
    future = out["date"] > cutoff
    factors = rng.uniform(0.5, 1.5, size=(int(future.sum()), 1))
    cols = ["open", "high", "low", "close"]
    out.loc[future, cols] = out.loc[future, cols].to_numpy() * factors
    out.loc[future, "high"] = out.loc[future, cols].max(axis=1)
    out.loc[future, "low"] = out.loc[future, cols].min(axis=1)
    return out


STRATEGIES = [
    BuyAndHold("SPY"),
    SmaCross(tickers=TICKERS[:-1]),
    Rsi2Pullback(tickers=TICKERS[:-1]),
]


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return random_walk_prices(TICKERS, N, seed=7, drift=0.0006, vol=0.02)


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.name)
@pytest.mark.parametrize("cut", [250, 380, 520])
def test_signals_do_not_depend_on_future_prices(prices: pd.DataFrame, strategy, cut: int) -> None:
    dates = np.sort(prices["date"].unique())
    cutoff = pd.Timestamp(dates[cut])
    changed = perturb_after(prices, cutoff, seed=cut)

    pd.testing.assert_frame_equal(
        strategy.generate(prices).loc[:cutoff], strategy.generate(changed).loc[:cutoff]
    )
    pd.testing.assert_frame_equal(
        strategy.strength(prices).loc[:cutoff], strategy.strength(changed).loc[:cutoff]
    )


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.name)
def test_engine_equity_until_t_does_not_depend_on_future(prices: pd.DataFrame, strategy) -> None:
    dates = np.sort(prices["date"].unique())
    cutoff = pd.Timestamp(dates[450])
    cfg = BacktestConfig(initial_capital=100_000, cost_bps=5)
    base = run_backtest(prices, strategy, cfg)
    alt = run_backtest(perturb_after(prices, cutoff, seed=1), strategy, cfg)

    pd.testing.assert_series_equal(base.equity.loc[:cutoff], alt.equity.loc[:cutoff])
    pd.testing.assert_frame_equal(base.positions.loc[:cutoff], alt.positions.loc[:cutoff])
    closed_before = base.trades[base.trades["exit_date"] <= cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(closed_before, alt.trades[alt.trades["exit_date"] <= cutoff].reset_index(drop=True))


def test_leak_test_is_not_vacuous(prices: pd.DataFrame) -> None:
    """누수 테스트 데이터에서 실제로 거래가 일어나는지 확인 (신호가 없으면 테스트가 무의미)."""
    cfg = BacktestConfig(initial_capital=100_000, cost_bps=5)
    for strategy in STRATEGIES[1:]:
        assert len(run_backtest(prices, strategy, cfg).trades) >= 5, strategy.name
