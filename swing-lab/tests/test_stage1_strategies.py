"""1단계 전략과 지표: 손계산 예제, 듀얼 모멘텀 순위·IEF 대체, 돌파, 누수 (합성 데이터)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_prices, random_walk_prices
from swinglab.backtest import run_backtest
from swinglab.config import BacktestConfig
from swinglab.indicators import period_return, prior_high, prior_low
from swinglab.strategies import Breakout, DualMomentum, MarketFilter, Rsi2Swing

# ---------------------------------------------------------------- 지표


def test_prior_high_low_exclude_today() -> None:
    s = pd.Series([1.0, 3.0, 2.0, 5.0, 4.0])
    # 직전 2개 최고: idx2 = max(1,3)=3, idx3 = max(3,2)=3, idx4 = max(2,5)=5
    np.testing.assert_allclose(prior_high(s, 2).to_numpy(), [np.nan, np.nan, 3.0, 3.0, 5.0])
    np.testing.assert_allclose(prior_low(s, 2).to_numpy(), [np.nan, np.nan, 1.0, 2.0, 2.0])


def test_period_return_and_missing_days() -> None:
    s = pd.Series([100.0, 110.0, np.nan, 121.0])
    np.testing.assert_allclose(period_return(s, 1).to_numpy(), [np.nan, 0.1, np.nan, 0.1])
    np.testing.assert_allclose(period_return(s, 2).to_numpy(), [np.nan, np.nan, np.nan, 0.21])


# ---------------------------------------------------------------- A: RSI(2) 개선판


def test_rsi2_swing_target_and_strength() -> None:
    prices = random_walk_prices(["AAA", "BBB"], 500, seed=3, drift=0.0008, vol=0.02)
    s = Rsi2Swing(tickers=["AAA", "BBB"], entry_th=10, min_hold=3)
    rsi, entry, exit_, valid = s._components(prices)
    target, strength = s.generate(prices), s.strength(prices)
    assert s.min_hold_days == 3 and s.max_hold_days == 10
    t = target.to_numpy()
    assert (t[exit_.to_numpy()] == 0).all()
    assert (t[(valid & ~exit_).to_numpy()] == 1).all()
    assert np.isnan(t[~valid.to_numpy()]).all()
    assert strength.notna().equals(entry)  # 강도는 진입 조건이 성립한 날만
    assert entry.to_numpy().sum() > 0
    assert not (entry & exit_).to_numpy().any()  # 같은 날이면 청산 우선


# ---------------------------------------------------------------- B: 듀얼 모멘텀


def _monthly_prices(returns: dict[str, list[float]], months: int, listed_from: dict[str, int] | None = None) -> pd.DataFrame:
    """월마다 지정 수익률로 오르는 일봉 (월 안에서는 일정). listed_from: 상장 월 인덱스."""
    dates = pd.bdate_range("2015-01-01", periods=months * 21 + 30).astype("datetime64[ns]")
    month_idx = (dates.year - dates[0].year) * 12 + dates.month - dates[0].month
    data = {}
    for t, rets in returns.items():
        monthly_growth = np.cumprod([1.0] + [1 + r for r in rets])
        price = 100 * monthly_growth[np.minimum(month_idx, len(rets))]
        if listed_from and t in listed_from:
            price = np.where(month_idx >= listed_from[t], price, np.nan)
        data[t] = price
    return make_prices(pd.DataFrame(data, index=dates))


def test_dual_momentum_ranks_top_n_and_replaces_with_ief() -> None:
    months = 14
    rets = {
        "XLK": [0.03] * months,   # 가장 강함
        "XLF": [0.02] * months,
        "XLE": [-0.01] * months,  # IEF보다 약함
        "XLV": [0.005] * months,
        "IEF": [0.01] * months,
    }
    prices = _monthly_prices(rets, months)
    s = DualMomentum(tickers=["XLK", "XLF", "XLE", "XLV"], lookback=6, top_n=3)
    target, strength = s.generate(prices), s.strength(prices)
    sector_rets, safe_ret = s.momentum(prices)
    first = sector_rets.dropna(how="all").index[0]

    # 6개월 수익률 순위: XLK > XLF > XLV > XLE. 3위 XLV(0.5%/월)가 IEF(1%/월)보다 낮아 IEF#1로 대체
    row = target.loc[first]
    assert row["XLK"] == 1 and row["XLF"] == 1 and row["XLV"] == 0 and row["XLE"] == 0
    assert row["IEF#1"] == 1 and row["IEF#2"] == 0 and row["IEF#3"] == 0
    assert strength.at[first, "IEF#1"] == pytest.approx(safe_ret[first])
    assert s.max_positions == 3 and s.rebalance == "monthly"
    assert s.price_aliases == {"IEF#1": "IEF", "IEF#2": "IEF", "IEF#3": "IEF"}


def test_dual_momentum_multiple_slots_go_to_ief() -> None:
    months = 14
    rets = {"XLK": [0.005] * months, "XLF": [0.004] * months, "XLE": [-0.02] * months, "IEF": [0.01] * months}
    prices = _monthly_prices(rets, months)
    s = DualMomentum(tickers=["XLK", "XLF", "XLE"], lookback=6, top_n=2)
    target = s.generate(prices)
    first = s.momentum(prices)[0].dropna(how="all").index[0]
    assert target.loc[first, ["XLK", "XLF", "XLE"]].sum() == 0
    assert target.at[first, "IEF#1"] == 1 and target.at[first, "IEF#2"] == 1  # 두 자리 모두 IEF → 비중 100%


def test_dual_momentum_excludes_unlisted() -> None:
    months = 20
    rets = {"XLK": [0.01] * months, "XLC": [0.10] * months, "IEF": [0.0] * months}
    prices = _monthly_prices(rets, months, listed_from={"XLC": 8})
    s = DualMomentum(tickers=["XLK", "XLC"], lookback=6, top_n=1)
    sector_rets, _ = s.momentum(prices)
    target = s.generate(prices)
    month_ends = sector_rets.dropna(how="all").index
    # XLC는 상장 후 6개월 월말 종가가 생기기 전까지 순위에서 빠진다
    early = [d for d in month_ends if np.isnan(sector_rets.at[d, "XLC"])]
    late = [d for d in month_ends if not np.isnan(sector_rets.at[d, "XLC"])]
    assert early and late
    assert all(target.at[d, "XLK"] == 1 for d in early)
    assert all(target.at[d, "XLC"] == 1 and target.at[d, "XLK"] == 0 for d in late)


def test_dual_momentum_backtest_trades_only_at_month_turn() -> None:
    months = 30
    rng = np.random.default_rng(0)
    rets = {t: list(rng.normal(0.005, 0.04, months)) for t in ["XLK", "XLF", "XLE", "XLV", "IEF"]}
    prices = _monthly_prices(rets, months)
    s = DualMomentum(tickers=["XLK", "XLF", "XLE", "XLV"], lookback=6, top_n=2)
    result = run_backtest(prices, s, BacktestConfig(initial_capital=100_000, cost_bps=5))
    dates = pd.DatetimeIndex(np.sort(prices["date"].unique()))
    first_of_month = set(dates[np.r_[True, dates.month[1:] != dates.month[:-1]]])
    trade_days = set(result.trades["entry_date"]) | set(result.trades["exit_date"])
    assert len(result.trades) > 0
    assert trade_days <= first_of_month
    assert (result.positions.gt(0).sum(axis=1) <= 2).all()


# ---------------------------------------------------------------- C: 돌파


def test_breakout_enters_on_breakout_day_and_exits_on_breakdown() -> None:
    n = 120
    close = np.r_[np.full(70, 100.0), np.linspace(100, 130, 20), np.linspace(130, 90, 30)]
    close = close + np.sin(np.arange(n)) * 0.01  # 동점 방지용 아주 작은 흔들림
    dates = pd.bdate_range("2020-01-01", periods=n).astype("datetime64[ns]")
    prices = make_prices(pd.DataFrame({"AAA": close}, index=dates))
    s = Breakout(tickers=["AAA"], entry_n=20, exit_n=10)
    target, strength = s.generate(prices)["AAA"], s.strength(prices)["AAA"]
    entry_days = strength.dropna().index
    assert len(entry_days) > 0
    first_entry = entry_days[0]
    assert target.loc[first_entry] == 1
    assert close[dates.get_loc(first_entry)] > close[dates.get_loc(first_entry) - 20 : dates.get_loc(first_entry)].max()
    assert target.iloc[-1] == 0  # 급락 구간에서 이탈
    assert s.warmup_days == 64


# ---------------------------------------------------------------- 누수


TICKERS = ["XLK", "XLF", "XLE", "XLV", "IEF", "TLT", "SPY"]


@pytest.fixture(scope="module")
def walk() -> pd.DataFrame:
    return random_walk_prices(TICKERS, 700, seed=21, drift=0.0005, vol=0.018)


def _perturb_after(prices: pd.DataFrame, cutoff: pd.Timestamp, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = prices.copy()
    future = out["date"] > cutoff
    cols = ["open", "high", "low", "close"]
    out.loc[future, cols] = out.loc[future, cols].to_numpy() * rng.uniform(0.6, 1.4, size=(int(future.sum()), 1))
    return out


STAGE1 = [
    Rsi2Swing(tickers=TICKERS[:4], entry_th=10, min_hold=3),
    DualMomentum(tickers=TICKERS[:4], lookback=6, top_n=2),
    Breakout(tickers=TICKERS[:4], entry_n=20, exit_n=10),
    MarketFilter(Breakout(tickers=TICKERS[:4], entry_n=20, exit_n=10)),
]


@pytest.mark.parametrize("strategy", STAGE1, ids=lambda s: s.name)
def test_stage1_signals_do_not_depend_on_future(walk: pd.DataFrame, strategy) -> None:
    dates = np.sort(walk["date"].unique())
    cutoff = pd.Timestamp(dates[450])
    changed = _perturb_after(walk, cutoff, seed=4)
    pd.testing.assert_frame_equal(strategy.generate(walk).loc[:cutoff], strategy.generate(changed).loc[:cutoff])
    pd.testing.assert_frame_equal(strategy.strength(walk).loc[:cutoff], strategy.strength(changed).loc[:cutoff])
