"""전략 신호 로직."""

from __future__ import annotations

import numpy as np
import pandas as pd

from conftest import make_prices, trading_dates
from swinglab.config import UniverseConfig
from swinglab.strategies import BuyAndHold, Rsi2Pullback, SmaCross, build_strategy
from swinglab.strategies.rsi2_pullback import _hold_state


def test_hold_state_machine() -> None:
    entry = np.array([0, 1, 0, 1, 0, 0, 1, 0], dtype=bool)
    exit_ = np.array([1, 1, 0, 0, 1, 0, 0, 1], dtype=bool)
    valid = np.array([1, 1, 1, 1, 1, 0, 1, 1], dtype=bool)
    # 1일은 진입·청산 동시 성립 → 청산 우선(0). 3일 진입 → 4일 청산 → 5일 결측(NaN) → 6일 진입 → 7일 청산
    expected = [0, 0, 0, 1, 0, np.nan, 1, 0]
    np.testing.assert_array_equal(_hold_state(entry, exit_, valid), expected)


def test_sma_cross_target_follows_trend() -> None:
    n = 120
    dates = trading_dates(n)
    up = np.linspace(100, 200, n)
    down = np.linspace(200, 100, n)
    prices = make_prices(pd.DataFrame({"UP": up, "DN": down}, index=dates))
    strat = SmaCross(tickers=["UP", "DN"])
    target = strat.generate(prices)
    assert target["UP"].iloc[:59].isna().all()  # SMA(60) 이전은 신호 없음
    assert (target["UP"].iloc[59:] == 1).all()
    assert (target["DN"].iloc[59:] == 0).all()
    strength = strat.strength(prices)
    assert (strength["UP"].iloc[59:] > 0).all()


def test_rsi2_strength_only_on_entry_days() -> None:
    n = 260
    dates = trading_dates(n)
    close = np.linspace(100, 150, n)  # 꾸준한 상승: 종가 > SMA(200)
    close[230:233] = close[229] * np.array([0.97, 0.94, 0.92])  # 3일 급락 → RSI(2) 급락
    prices = make_prices(pd.DataFrame({"A": close}, index=dates))
    strat = Rsi2Pullback(tickers=["A"])
    strength = strat.strength(prices)["A"]
    target = strat.generate(prices)["A"]

    entry_days = strength.dropna().index
    assert len(entry_days) > 0
    assert set(entry_days) <= set(dates[230:233])
    first = entry_days[0]
    assert target.loc[first] == 1
    assert target.iloc[:199].isna().all()  # SMA(200) 이전은 신호 없음
    # 반등해서 종가 > SMA(5)가 되면 상태 0
    assert target.iloc[-1] == 0


def test_strategies_emit_nan_on_missing_days() -> None:
    n = 300
    dates = trading_dates(n)
    a = 100 + np.sin(np.arange(n) / 5) * 5 + np.arange(n) * 0.1
    a_missing = a.copy()
    a_missing[250] = np.nan
    prices = make_prices(pd.DataFrame({"A": a_missing, "SPY": a}, index=dates))
    for strat in (SmaCross(tickers=["A"]), Rsi2Pullback(tickers=["A"])):
        assert np.isnan(strat.generate(prices).at[dates[250], "A"])
    assert np.isnan(BuyAndHold("A").generate(prices).at[dates[250], "A"])


def test_build_strategy_uses_universe_without_benchmark() -> None:
    universe = UniverseConfig(benchmark="SPY", start="2010-01-01", groups={"b": ["SPY"], "x": ["XLK", "XLF"]})
    assert build_strategy("sma_cross", universe).tickers == ["XLK", "XLF"]
    assert build_strategy("rsi2_pullback", universe).tickers == ["XLK", "XLF"]
    bh = build_strategy("buy_hold", universe)
    assert bh.ticker == "SPY" and bh.max_positions == 1
