"""1단계 엔진 확장: 전략별 최대 보유 수, 월간 리밸런싱, 최소 보유일, 가격 별칭, 시장 필터 (합성 데이터)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import FixedStrategy, make_prices, trading_dates

from swinglab.backtest import run_backtest
from swinglab.config import BacktestConfig
from swinglab.strategies import MarketFilter

C = 100_000.0
CFG = BacktestConfig(initial_capital=C, cost_bps=0, max_positions=5)


def frame(values: dict[str, list[float]], dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(values, index=dates, dtype=float)


# ---------------------------------------------------------------- 전략별 최대 보유 수


def test_strategy_max_positions_overrides_config_and_leaves_cash() -> None:
    dates = trading_dates(6)
    tickers = ["A", "B", "C"]
    opens = frame({t: [100.0] * 6 for t in tickers}, dates)
    target = frame({"A": [1] * 6, "B": [0] * 6, "C": [0] * 6}, dates)
    result = run_backtest(make_prices(opens), FixedStrategy(target, max_positions=2), CFG)
    assert result.max_positions == 2
    assert result.positions.at[dates[1], "A"] * 100 == pytest.approx(C / 2)  # 비중 1/2
    assert result.cash.iloc[-1] == pytest.approx(C / 2)  # 나머지는 현금

    target_all = frame({t: [1] * 6 for t in tickers}, dates)
    held = run_backtest(make_prices(opens), FixedStrategy(target_all, max_positions=2), CFG).positions.iloc[-1]
    assert (held > 0).sum() == 2


# ---------------------------------------------------------------- 월간 리밸런싱


@pytest.fixture
def month_dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2021-01-04", "2021-04-30").astype("datetime64[ns]")


def _monthly(target: pd.DataFrame, **kw) -> FixedStrategy:
    s = FixedStrategy(target, **kw)
    s.rebalance = "monthly"
    return s


def test_monthly_fills_next_month_first_open_and_ignores_mid_month(month_dates: pd.DatetimeIndex) -> None:
    n = len(month_dates)
    opens = frame({"A": 100 + np.arange(n), "B": 50 + np.arange(n)}, month_dates)
    tgt = frame({"A": [0.0] * n, "B": [0.0] * n}, month_dates)
    jan_end = month_dates[month_dates.month == 1][-1]
    feb_first = month_dates[month_dates.month == 2][0]
    feb_mid = month_dates[month_dates.month == 2][9]
    feb_end = month_dates[month_dates.month == 2][-1]
    mar_first = month_dates[month_dates.month == 3][0]
    tgt.loc[jan_end:, "A"] = 1.0  # 1월 말 신호 → 2월 첫 거래일 진입
    tgt.loc[feb_mid:, "B"] = 1.0  # 2월 중순 신호는 월말까지 반영 안 됨 → 3월 첫 거래일 진입
    tgt.loc[feb_mid : feb_mid, "A"] = 0.0  # 월중 하루 0이어도 청산 안 함

    result = run_backtest(make_prices(opens), _monthly(tgt, max_positions=2), CFG)
    trades_open = result.positions
    assert (trades_open.loc[:jan_end, "A"] == 0).all()
    assert trades_open.at[feb_first, "A"] > 0
    assert (trades_open.loc[feb_first:, "A"] > 0).all()  # 월중 0 신호 무시
    assert (trades_open.loc[:feb_end, "B"] == 0).all()
    assert trades_open.at[mar_first, "B"] > 0
    assert result.trades.empty


def test_monthly_exit_on_month_end_signal(month_dates: pd.DatetimeIndex) -> None:
    n = len(month_dates)
    opens = frame({"A": 100 + np.arange(n)}, month_dates)
    tgt = frame({"A": [1.0] * n}, month_dates)
    mar_end = month_dates[month_dates.month == 3][-1]
    apr_first = month_dates[month_dates.month == 4][0]
    tgt.loc[month_dates[month_dates.month == 3][5] :, "A"] = 0.0  # 3월 초부터 0이지만 3월 말에야 반영

    result = run_backtest(make_prices(opens), _monthly(tgt, max_positions=1), CFG)
    trade = result.trades.iloc[0]
    assert trade["exit_date"] == apr_first
    assert trade["exit_price"] == opens.at[apr_first, "A"]
    assert result.positions.at[mar_end, "A"] > 0


def test_monthly_first_trade_waits_for_month_end(month_dates: pd.DatetimeIndex) -> None:
    n = len(month_dates)
    opens = frame({"A": [100.0] * n}, month_dates)
    tgt = frame({"A": [1.0] * n}, month_dates)
    cfg = CFG.model_copy(update={"start": month_dates[10].date()})  # 1월 중순 시작
    result = run_backtest(make_prices(opens), _monthly(tgt, max_positions=1), cfg)
    feb_first = month_dates[month_dates.month == 2][0]
    assert (result.positions.loc[:feb_first, "A"].iloc[:-1] == 0).all()
    assert result.positions.at[feb_first, "A"] > 0


# ---------------------------------------------------------------- 최소 보유일


def test_min_hold_exits_only_when_signal_and_min_hold_on_same_day() -> None:
    dates = trading_dates(12)
    opens = frame({"A": [100.0] * 12}, dates)
    # 0일 신호 → 1일 진입. 2일(보유 2일) 청산 신호는 min_hold 3 미만이라 무시, 3일엔 신호 1,
    # 5일(보유 5일) 청산 신호 → 6일 청산
    tgt = frame({"A": [1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1]}, dates)
    strength = frame({"A": [1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]}, dates)
    s = FixedStrategy(tgt, strength, max_positions=1)
    s.min_hold_days = 3
    trade = run_backtest(make_prices(opens), s, CFG).trades.iloc[0]
    assert trade["entry_date"] == dates[1]
    assert trade["exit_date"] == dates[6]
    assert trade["exit_reason"] == "signal"


def test_max_hold_still_applies_with_min_hold() -> None:
    dates = trading_dates(15)
    opens = frame({"A": [100.0] * 15}, dates)
    tgt = frame({"A": [1.0] * 15}, dates)
    strength = frame({"A": [1.0] + [np.nan] * 14}, dates)
    s = FixedStrategy(tgt, strength, max_positions=1, max_hold_days=10)
    s.min_hold_days = 5
    trade = run_backtest(make_prices(opens), s, CFG).trades.iloc[0]
    assert trade["holding_days"] == 10 and trade["exit_reason"] == "max_hold"


# ---------------------------------------------------------------- 가격 별칭 (같은 종목 여러 자리)


def test_price_aliases_hold_separate_slots_of_same_asset() -> None:
    dates = trading_dates(6)
    opens = frame({"IEF": [100.0, 100.0, 102.0, 104.0, 104.0, 104.0], "X": [10.0] * 6}, dates)
    tgt = frame({"X": [1] * 6, "IEF#1": [1] * 6, "IEF#2": [0, 0, 1, 1, 1, 1]}, dates)
    s = FixedStrategy(tgt, max_positions=3)
    s.price_aliases = {"IEF#1": "IEF", "IEF#2": "IEF"}
    result = run_backtest(make_prices(opens), s, CFG)
    assert result.positions.at[dates[1], "IEF#1"] * 100 == pytest.approx(C / 3)
    assert result.positions.at[dates[3], "IEF#2"] > 0
    trade_free_equity = result.cash.iloc[-1] + result.positions.iloc[-1] @ pd.Series(
        {"X": 10.0, "IEF#1": 104.0, "IEF#2": 104.0}
    )
    assert result.equity.iloc[-1] == pytest.approx(trade_free_equity)


# ---------------------------------------------------------------- 시장 필터


@pytest.fixture
def filter_prices() -> pd.DataFrame:
    n = 260
    dates = trading_dates(n)
    spy = np.r_[np.linspace(100, 150, 220), np.linspace(150, 90, 40)]  # 마지막 40일 급락 → SMA(200) 아래로
    data = {"SPY": spy, "XLK": np.full(n, 50.0), "TLT": np.full(n, 80.0), "GLD": np.full(n, 120.0)}
    return make_prices(frame(data, dates))


def test_market_filter_zeroes_risk_assets_and_keeps_safe(filter_prices: pd.DataFrame) -> None:
    dates = np.sort(filter_prices["date"].unique())
    inner_target = frame({"XLK": [1.0] * len(dates), "TLT": [1.0] * len(dates), "GLD": [1.0] * len(dates)}, dates)
    inner_target.iloc[5, 0] = np.nan  # 신호 없는 칸은 그대로
    f = MarketFilter(FixedStrategy(inner_target))
    off = f.risk_off(filter_prices)
    target = f.generate(filter_prices)

    assert off.iloc[:199].sum() == 0  # SMA(200) 계산 전에는 필터 없음
    assert off.iloc[-1] and off.sum() > 0
    assert (target.loc[off.to_numpy(), "XLK"] == 0).all()
    assert (target.loc[~off.to_numpy(), "XLK"].dropna() == 1).all()
    assert np.isnan(target.iloc[5, 0])
    assert (target["TLT"] == 1).all() and (target["GLD"] == 1).all()
    assert f.warmup_days == 200 and f.name == "fixed+market_filter"


def test_market_filter_exempts_aliases_of_safe_assets(filter_prices: pd.DataFrame) -> None:
    dates = np.sort(filter_prices["date"].unique())
    inner = FixedStrategy(frame({"XLK": [1.0] * len(dates), "IEF#1": [1.0] * len(dates)}, dates))
    inner.price_aliases = {"IEF#1": "IEF"}
    target = MarketFilter(inner).generate(filter_prices)
    assert (target["IEF#1"] == 1).all()
    assert target["XLK"].iloc[-1] == 0


def test_market_filter_passes_engine_attributes() -> None:
    inner = FixedStrategy(pd.DataFrame(), max_positions=3, max_hold_days=7, warmup_days=250)
    inner.rebalance = "monthly"
    inner.min_hold_days = 2
    f = MarketFilter(inner)
    assert (f.max_positions, f.max_hold_days, f.min_hold_days, f.rebalance, f.warmup_days) == (3, 7, 2, "monthly", 250)
