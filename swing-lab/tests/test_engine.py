"""엔진 검증: 합성 데이터, 체결 시점, 비용, 엣지 케이스."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import FixedStrategy, make_prices, random_walk_prices, trading_dates

from swinglab.backtest import run_backtest, run_benchmark
from swinglab.config import BacktestConfig
from swinglab.strategies import BuyAndHold

C = 100_000.0


def frame(values: dict[str, list[float]], dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(values, index=dates, dtype=float)


# ---------------------------------------------------------------- 합성 데이터


@pytest.mark.parametrize("cost_bps", [0.0, 5.0, 10.0])
def test_buy_hold_matches_analytic_on_one_percent_daily_growth(cost_bps: float) -> None:
    n = 40
    dates = trading_dates(n)
    p = 100 * 1.01 ** np.arange(n)
    prices = make_prices(frame({"SPY": p}, dates))
    cfg = BacktestConfig(initial_capital=C, cost_bps=cost_bps)
    result = run_backtest(prices, BuyAndHold("SPY"), cfg)

    c = cost_bps / 10_000
    # 기준일(0) 전액 현금 → 1일 시가 매수 → i일 종가 평가: C × 1.01^(i-1) / (1+c)
    expected = np.r_[C, C * 1.01 ** np.arange(n - 1) / (1 + c)]
    np.testing.assert_allclose(result.equity.to_numpy(), expected, rtol=1e-12)
    assert result.start_date == dates[1]


def test_buy_hold_within_one_cost_of_simple_cumulative_return() -> None:
    prices = random_walk_prices(["SPY"], 300, seed=3)
    cfg = BacktestConfig(initial_capital=C, cost_bps=5.0)
    result = run_backtest(prices, BuyAndHold("SPY"), cfg)

    spy = prices.set_index("date")
    simple = spy["close"].iloc[-1] / spy.loc[result.start_date, "open"] - 1
    engine = result.equity.iloc[-1] / C - 1
    assert engine == pytest.approx((1 + simple) / (1 + 0.0005) - 1, rel=1e-12)
    assert abs(engine - simple) <= 0.0005 * (1 + simple)


# ---------------------------------------------------------------- 체결 시점


def test_orders_fill_at_next_day_open() -> None:
    n = 15
    dates = trading_dates(n)
    opens = frame({"A": 100 + np.arange(n)}, dates)
    closes = opens + 0.5
    target = frame({"A": [0] * 5 + [1] * 5 + [0] * 5}, dates)  # 5일 종가에 진입 신호, 10일 종가에 청산 신호
    result = run_backtest(make_prices(opens, closes), FixedStrategy(target), BacktestConfig(initial_capital=C, cost_bps=0))

    trade = result.trades.iloc[0]
    assert len(result.trades) == 1
    assert trade["entry_date"] == dates[6]
    assert trade["entry_price"] == opens.at[dates[6], "A"]
    assert trade["exit_date"] == dates[11]
    assert trade["exit_price"] == opens.at[dates[11], "A"]
    assert trade["holding_days"] == 5
    assert (result.positions.loc[: dates[5], "A"] == 0).all()
    assert (result.positions.loc[dates[6] : dates[10], "A"] > 0).all()
    assert (result.positions.loc[dates[11] :, "A"] == 0).all()


def test_no_same_day_close_fill() -> None:
    """신호 당일 종가가 아니라 다음 날 시가로 산다: 신호일 종가가 크게 바뀌어도 진입가는 그대로."""
    n = 8
    dates = trading_dates(n)
    opens = frame({"A": [100.0] * n}, dates)
    closes_a = opens.copy()
    closes_b = opens.copy()
    closes_b.iloc[3] = 150.0  # 신호일(3일) 종가만 다르게
    target = frame({"A": [0, 0, 0, 1, 1, 1, 1, 1]}, dates)
    cfg = BacktestConfig(initial_capital=C, cost_bps=0)
    ra = run_backtest(make_prices(opens, closes_a), FixedStrategy(target, max_positions=1), cfg)
    rb = run_backtest(make_prices(opens, closes_b), FixedStrategy(target, max_positions=1), cfg)
    assert ra.positions.iloc[-1, 0] == rb.positions.iloc[-1, 0] == pytest.approx(C / 100.0)


# ---------------------------------------------------------------- 비용


@pytest.mark.parametrize("cost_bps", [5.0, 10.0])
def test_round_trip_costs_exactly_twice_one_way(cost_bps: float) -> None:
    n = 10
    dates = trading_dates(n)
    prices = make_prices(frame({"A": [100.0] * n}, dates))  # 가격이 변하지 않으면 차이는 비용뿐
    target = frame({"A": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]}, dates)
    strat = FixedStrategy(target, max_positions=1)
    result = run_backtest(prices, strat, BacktestConfig(initial_capital=C, cost_bps=cost_bps))

    c = cost_bps / 10_000
    trade = result.trades.iloc[0]
    notional = trade["shares"] * trade["entry_price"]
    one_way = notional * c
    assert trade["entry_cost"] == pytest.approx(one_way, rel=1e-12)
    assert trade["exit_cost"] == pytest.approx(one_way, rel=1e-12)
    assert result.total_cost == pytest.approx(2 * one_way, rel=1e-12)
    assert C - result.equity.iloc[-1] == pytest.approx(2 * one_way, rel=1e-9)
    assert trade["return"] == pytest.approx((1 - c) / (1 + c) - 1, rel=1e-12)


def test_zero_cost_has_no_drag() -> None:
    n = 10
    dates = trading_dates(n)
    prices = make_prices(frame({"A": [100.0] * n}, dates))
    target = frame({"A": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]}, dates)
    result = run_backtest(prices, FixedStrategy(target, max_positions=1), BacktestConfig(initial_capital=C, cost_bps=0))
    assert result.equity.iloc[-1] == pytest.approx(C, rel=1e-12)
    assert result.total_cost == 0


# ---------------------------------------------------------------- 엣지 케이스


def test_tickers_with_different_listing_dates() -> None:
    n = 20
    dates = trading_dates(n)
    opens = frame({"A": 100 + np.arange(n), "B": [np.nan] * 10 + list(50 + np.arange(10))}, dates)
    prices = make_prices(opens)
    target = opens.notna().astype(float).where(opens.notna())  # 데이터가 있는 날만 신호
    result = run_backtest(prices, FixedStrategy(target, max_positions=2), BacktestConfig(initial_capital=C, cost_bps=0))

    assert result.equity.notna().all()
    assert (result.positions.loc[: dates[10], "B"] == 0).all()
    assert result.positions.at[dates[11], "B"] > 0  # 10일(상장일) 신호 → 11일 시가 진입
    assert result.positions.at[dates[1], "A"] > 0


def test_missing_day_keeps_position_and_defers_exit() -> None:
    n = 12
    dates = trading_dates(n)
    a = 100 + np.arange(n, dtype=float)
    a[7] = np.nan  # 7일 결측 (REF는 매일 거래되어 7일이 거래일임을 보여준다)
    opens = frame({"A": a, "REF": [10.0] * n}, dates)
    target = frame({"A": [1, 1, 1, 1, 1, 1, 0, np.nan, 0, 0, 0, 0]}, dates)  # 6일 청산 신호
    result = run_backtest(make_prices(opens), FixedStrategy(target), BacktestConfig(initial_capital=C, cost_bps=0))

    assert result.equity.notna().all()
    # 7일 평가는 6일 종가로
    shares = result.positions.at[dates[6], "A"]
    assert result.equity.at[dates[7]] == pytest.approx(result.cash.at[dates[7]] + shares * a[6])
    trade = result.trades.iloc[0]
    assert trade["exit_date"] == dates[8]  # 7일 시가가 없어 8일 시가로 미룸
    assert trade["exit_price"] == a[8]


def test_entry_cancelled_when_next_open_missing() -> None:
    n = 10
    dates = trading_dates(n)
    b = 50 + np.arange(n, dtype=float)
    b[4] = np.nan
    opens = frame({"B": b, "REF": [10.0] * n}, dates)
    target = frame({"B": [0, 0, 0, 1, np.nan, 1, 1, 1, 1, 1]}, dates)
    result = run_backtest(make_prices(opens), FixedStrategy(target), BacktestConfig(initial_capital=C, cost_bps=0))
    # 3일 신호 → 4일 시가 없음 → 취소. 4일은 신호 없음. 5일 신호 → 6일 진입
    assert (result.positions.loc[: dates[5], "B"] == 0).all()
    assert result.positions.at[dates[6], "B"] > 0


def test_more_signals_than_max_positions_ranked_by_strength_then_ticker() -> None:
    n = 6
    dates = trading_dates(n)
    tickers = list("ABCDEFG")
    opens = frame({t: [100.0] * n for t in tickers}, dates)
    target = frame({t: [0, 1, 1, 1, 1, 1] for t in tickers}, dates)
    s = {"A": 0.1, "B": 0.5, "C": 0.3, "D": 0.5, "E": 0.2, "F": 0.05, "G": 0.3}
    strength = frame({t: [s[t]] * n for t in tickers}, dates)
    result = run_backtest(
        make_prices(opens), FixedStrategy(target, strength), BacktestConfig(initial_capital=C, cost_bps=0, max_positions=5)
    )
    held = result.positions.loc[dates[2]]
    assert sorted(held[held > 0].index) == ["B", "C", "D", "E", "G"]  # B=D, C=G 동점은 알파벳순
    # 동일 비중 1/5
    np.testing.assert_allclose(held[held > 0].to_numpy() * 100.0, C / 5)


def test_tie_break_is_alphabetical_when_slots_run_out() -> None:
    n = 5
    dates = trading_dates(n)
    tickers = ["ZZ", "AA", "MM"]
    opens = frame({t: [10.0] * n for t in tickers}, dates)
    target = frame({t: [1] * n for t in tickers}, dates)
    strength = frame({t: [1.0] * n for t in tickers}, dates)
    result = run_backtest(
        make_prices(opens), FixedStrategy(target, strength), BacktestConfig(initial_capital=C, cost_bps=0, max_positions=2)
    )
    held = result.positions.loc[dates[1]]
    assert sorted(held[held > 0].index) == ["AA", "MM"]


def test_max_hold_days_exits_and_no_same_open_reentry() -> None:
    n = 10
    dates = trading_dates(n)
    opens = frame({"A": [100.0] * n}, dates)
    target = frame({"A": [1] * n}, dates)
    result = run_backtest(
        make_prices(opens), FixedStrategy(target, max_hold_days=3), BacktestConfig(initial_capital=C, cost_bps=0)
    )
    first, second = result.trades.iloc[0], result.trades.iloc[1]
    assert first["entry_date"] == dates[1]
    assert first["exit_date"] == dates[4]
    assert first["holding_days"] == 3
    assert first["exit_reason"] == "max_hold"
    assert second["entry_date"] == dates[5]  # 4일 시가에 판 종목을 같은 시가에 다시 사지 않는다


def test_nan_signal_keeps_existing_position() -> None:
    n = 8
    dates = trading_dates(n)
    opens = frame({"A": [100.0] * n}, dates)
    target = frame({"A": [1, 1, np.nan, np.nan, np.nan, 1, 1, 1]}, dates)
    result = run_backtest(make_prices(opens), FixedStrategy(target), BacktestConfig(initial_capital=C, cost_bps=0))
    assert result.trades.empty
    assert (result.positions.loc[dates[1] :, "A"] > 0).all()


def test_position_size_is_equal_weight_of_equity() -> None:
    n = 5
    dates = trading_dates(n)
    opens = frame({"A": [100.0] * n, "B": [100.0] * n}, dates)
    target = frame({"A": [1] * n, "B": [0] * n}, dates)
    result = run_backtest(make_prices(opens), FixedStrategy(target), BacktestConfig(initial_capital=C, cost_bps=0, max_positions=4))
    assert result.positions.at[dates[1], "A"] * 100.0 == pytest.approx(C / 4)
    assert result.cash.iloc[-1] == pytest.approx(C * 3 / 4)


def test_window_respects_warmup_and_start(config: BacktestConfig) -> None:
    prices = random_walk_prices(["SPY"], 50)
    dates = trading_dates(50)
    strat = BuyAndHold("SPY")
    strat.warmup_days = 10
    assert run_backtest(prices, strat, config).start_date == dates[10]
    late = config.model_copy(update={"start": dates[20].date(), "end": dates[40].date()})
    r = run_backtest(prices, strat, late)
    assert r.start_date == dates[20]
    assert r.end_date == dates[40]
    assert r.equity.index[0] == dates[19]


def test_benchmark_uses_same_window(config: BacktestConfig) -> None:
    prices = random_walk_prices(["SPY", "A"], 80)
    target = pd.DataFrame({"A": 1.0}, index=trading_dates(80))
    strat = FixedStrategy(target, warmup_days=30)
    result = run_backtest(prices, strat, config)
    bench = run_benchmark(prices, "SPY", config, result)
    assert bench.equity.index.equals(result.equity.index)


def test_rejects_invalid_target(config: BacktestConfig) -> None:
    dates = trading_dates(5)
    prices = make_prices(frame({"A": [1.0] * 5}, dates))
    with pytest.raises(ValueError):
        run_backtest(prices, FixedStrategy(frame({"A": [0.5] * 5}, dates)), config)
