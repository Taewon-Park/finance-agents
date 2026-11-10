"""2단계 트랙 2: 목표 비중 엔진, P3 추세 필터, 세금, 납입금 우선 배분, LLM 요약 검증."""

from __future__ import annotations

import json
import urllib.error
from datetime import date
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import pytest
from conftest import make_prices

from swinglab.alloc.backtest import (
    AllocRefused,
    check_alloc_budget,
    mdd_window,
    period_stats,
    run_all,
    yearly_returns,
)
from swinglab.alloc.config import GlidePath, load_alloc_config
from swinglab.alloc.policies import Policy, apply_glide, build_schedule, quarter_ends
from swinglab.alloc.rebalance import (
    load_holdings,
    plan_rebalance,
    render_plan,
    write_plan,
)
from swinglab.alloc.summary import checked_summary, request_summary, unknown_numbers
from swinglab.alloc.tax import TaxLedger, tax_due, yearly_tax
from swinglab.backtest.weights_engine import WeightsConfig, run_weights
from swinglab.research.log import LOG_COLUMNS, BudgetExceeded

ROOT = Path(__file__).resolve().parents[1]
CFG = load_alloc_config(ROOT / "config")


def _bdays(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end).astype("datetime64[ns]")


def _frame(paths: dict[str, np.ndarray], dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(paths, index=dates)


def _run(prices: pd.DataFrame, policy: Policy, cost_bps: float = 5.0, band: float = 0.05):
    dates = pd.DatetimeIndex(np.sort(prices["date"].unique()))
    schedule = build_schedule(policy, prices, dates, dates[0])
    cfg = WeightsConfig(initial_capital=100_000.0, cost_bps=cost_bps, band=band, cash_proxy="BIL", start=dates[1].date())
    return run_weights(prices, policy.tickers, schedule, cfg, name=policy.name), dates


def _policy(weights: dict[str, float], trend: bool = False) -> Policy:
    return Policy("T", weights, trend, ("SPY", "EFA", "EEM", "GLD"), 10, "BIL")


# --- 설정 (스펙 값) -------------------------------------------------------------------


def test_config_matches_spec() -> None:
    assert CFG.policies["P0"].weights == {"SPY": 1.0}
    assert CFG.policies["P1"].weights == {"SPY": 0.60, "IEF": 0.40}
    assert CFG.policies["P2"].weights == {"SPY": 0.55, "EFA": 0.10, "EEM": 0.05, "IEF": 0.15, "GLD": 0.10, "BIL": 0.05}
    assert "TLT" not in CFG.policies["P2"].weights
    assert CFG.policies["P3"].weights == CFG.policies["P2"].weights and CFG.policies["P3"].trend_filter
    assert not any(CFG.policies[p].trend_filter for p in ("P0", "P1", "P2"))
    assert (CFG.goal.policy, CFG.goal.max_drawdown, CFG.goal.cost_bps) == ("P3", -0.15, 5)
    assert CFG.band == 0.05 and CFG.trend.months == 10 and CFG.trend.assets == ["SPY", "EFA", "EEM", "GLD"]
    assert str(CFG.backtest.start) == "2011-03-01" and CFG.max_trials == 4
    assert CFG.live.glide_path.enabled is False and CFG.llm.enabled is False


# --- 목표 비중 엔진 ---------------------------------------------------------------------


def test_quarterly_rebalance_signal_then_next_open() -> None:
    dates = _bdays("2023-01-02", "2023-10-31")
    a = np.interp(np.arange(len(dates)), [0, 6, dates.get_loc(pd.Timestamp("2023-03-31"))], [100, 100, 125])
    prices = make_prices(_frame({"A": a, "B": np.full(len(dates), 100.0), "BIL": np.full(len(dates), 100.0)}, dates))

    result, dates = _run(prices, _policy({"A": 0.5, "B": 0.5}))

    ex = result.executions.set_index("signal_date")
    assert list(ex["kind"]) == ["initial", "quarterly", "quarterly", "quarterly"]
    assert [d.date().isoformat() for d in ex.index[1:]] == ["2023-03-31", "2023-06-30", "2023-09-29"]
    # 3월 말 A 비중 55.6% (5%p 이상) → 다음 거래일 4/3 시가에 매매, 그 뒤 분기는 범위 안이라 매매 없음
    assert ex.loc[pd.Timestamp("2023-03-31"), "date"] == pd.Timestamp("2023-04-03")
    assert list(ex["n_trades"]) == [2, 2, 0, 0]
    trades = result.trades
    assert set(trades["date"]) == {dates[1], pd.Timestamp("2023-04-03")}
    q1 = trades[trades["date"] == pd.Timestamp("2023-04-03")].set_index("ticker")
    assert q1.loc["A", "side"] == "sell" and q1.loc["B", "side"] == "buy"
    assert q1.loc["A", "price"] == 125.0  # 체결가 = 다음 거래일 시가
    assert result.weights.loc[pd.Timestamp("2023-04-03"), "A"] == pytest.approx(0.5, abs=1e-3)


def test_assets_within_band_are_not_traded() -> None:
    dates = _bdays("2023-01-02", "2023-05-31")
    qe = dates.get_loc(pd.Timestamp("2023-03-31"))
    a = np.interp(np.arange(len(dates)), [0, 5, qe], [100, 100, 127.78])
    flat = np.full(len(dates), 100.0)
    prices = make_prices(_frame({"A": a, "B": flat, "C": flat, "BIL": flat}, dates))

    result, _ = _run(prices, _policy({"A": 0.4, "B": 0.4, "C": 0.2}))

    w = result.weights.loc[pd.Timestamp("2023-03-31")]
    assert w["A"] - 0.4 >= 0.05 and abs(w["B"] - 0.4) < 0.05 and abs(w["C"] - 0.2) < 0.05
    day = result.trades[result.trades["date"] == pd.Timestamp("2023-04-03")]
    assert set(zip(day["ticker"], day["side"], day["reason"])) == {("A", "sell", "quarterly"), ("BIL", "buy", "sweep")}


def test_costs_deducted_exactly() -> None:
    dates = _bdays("2023-01-02", "2023-02-28")
    opens = _frame({"SPY": np.full(len(dates), 50.0), "BIL": np.full(len(dates), 100.0)}, dates)
    closes = opens.copy()
    closes.loc[dates[1], "SPY"] = 55.0
    prices = make_prices(opens, closes)

    result, _ = _run(prices, _policy({"SPY": 1.0}), cost_bps=10.0)

    shares = 100_000 / 1.001 / 50.0  # 비용 포함 예산 = 초기자본 (체결금액 + 비용)
    assert result.shares.loc[dates[1], "SPY"] == pytest.approx(shares, rel=1e-12)
    assert result.equity.loc[dates[1]] == pytest.approx(shares * 55.0, rel=1e-12)
    assert result.trades["cost"].sum() == pytest.approx(100_000 - 100_000 / 1.001, rel=1e-12)
    assert result.cash.loc[dates[1]] == pytest.approx(0.0, abs=1e-6)


def test_rebalance_cost_identity() -> None:
    """매매 당일 시가 기준 자산 감소분 = 그날 비용 합, 각 체결 비용 = 체결금액 × 편도 비용률."""
    dates = _bdays("2023-01-02", "2023-05-31")
    a = np.interp(np.arange(len(dates)), [0, 5, dates.get_loc(pd.Timestamp("2023-03-31"))], [100, 100, 200])
    flat = np.full(len(dates), 100.0)
    prices = make_prices(_frame({"A": a, "B": flat, "BIL": flat}, dates))
    result, _ = _run(prices, _policy({"A": 0.5, "B": 0.5}), cost_bps=7.0)

    day = pd.Timestamp("2023-04-03")
    t = result.trades[result.trades["date"] == day]
    np.testing.assert_allclose(t["cost"], t["notional"] * 0.0007, rtol=1e-12)
    prev = result.shares.index[result.shares.index.get_loc(day) - 1]
    before = (result.shares.loc[prev] * prices[prices["date"] == day].set_index("ticker")["open"]).sum() + result.cash.loc[prev]
    after = (result.shares.loc[day] * prices[prices["date"] == day].set_index("ticker")["open"]).sum() + result.cash.loc[day]
    assert before - after == pytest.approx(t["cost"].sum(), rel=1e-9)


# --- P3 추세 필터 ---------------------------------------------------------------------


def _trend_prices() -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    dates = _bdays("2022-01-03", "2023-09-29")
    idx = np.arange(len(dates))
    jan31 = dates.get_loc(pd.Timestamp("2023-01-31"))
    feb1 = dates.get_loc(pd.Timestamp("2023-02-01"))
    may1 = dates.get_loc(pd.Timestamp("2023-05-01"))
    spy = np.interp(idx, [0, jan31, feb1, may1 - 1, may1], [100, 150, 120, 120, 170])
    flat = np.full(len(dates), 100.0)
    paths = {"SPY": spy, "EFA": flat, "EEM": flat, "IEF": flat, "GLD": flat, "BIL": flat}
    return make_prices(_frame(paths, dates)), dates


def test_p3_moves_to_bil_after_month_end_below_and_returns() -> None:
    prices, dates = _trend_prices()
    p3 = Policy.from_config("P3", CFG)
    # 직접 계산: 월말 종가 < 최근 10개 월말 평균
    closes = prices[prices["ticker"] == "SPY"].set_index("date")["close"]
    me = closes.groupby(closes.index.to_period("M")).tail(1)
    sma = me.rolling(10).mean()
    assert me.loc["2023-02-28"] < sma.loc["2023-02-28"] and me.loc["2023-05-31"] > sma.loc["2023-05-31"]

    result, _ = _run(prices, p3)

    out_day, back_day = pd.Timestamp("2023-03-01"), pd.Timestamp("2023-06-01")  # 월말 다음 거래일
    t = result.trades
    exits = t[t["date"] == out_day]
    assert set(zip(exits["ticker"], exits["side"])) == {("SPY", "sell"), ("BIL", "buy")}  # 분기말이 아닌 달: SPY와 BIL만
    assert exits["kind"].unique().tolist() == ["trend"]
    assert result.shares.loc[pd.Timestamp("2023-02-28"), "SPY"] > 0
    between = result.shares.loc[out_day:pd.Timestamp("2023-05-31"), "SPY"]
    assert (between == 0).all()
    # SPY 매도 대금이 비용만 빼고 전부 BIL로 간다 (다른 자산은 그대로)
    spy_sell = exits[exits["ticker"] == "SPY"].iloc[0]
    bil_buy = exits[exits["ticker"] == "BIL"].iloc[0]
    assert bil_buy["notional"] == pytest.approx(spy_sell["notional"] * (1 - 0.0005) / (1 + 0.0005), rel=1e-9)
    prev = pd.Timestamp("2023-02-28")
    for other in ("EFA", "EEM", "IEF", "GLD"):
        assert result.shares.loc[out_day, other] == result.shares.loc[prev, other]
    back = t[t["date"] == back_day]
    assert set(zip(back["ticker"], back["side"])) == {("SPY", "buy"), ("BIL", "sell")}
    assert result.weights.loc[back_day, "SPY"] == pytest.approx(0.55, abs=0.01)
    # 추세 이동 외의 SPY 매매는 초기 배분과 분기 재조정(허용 범위)뿐이고, 이탈~복귀 사이에는 없다
    spy = t[t["ticker"] == "SPY"]
    others = spy[~spy["date"].isin([out_day, back_day])]
    assert set(others["kind"]) <= {"initial", "quarterly"} and dates[1] in set(others["date"])
    assert not ((spy["date"] > out_day) & (spy["date"] < back_day)).any()
    assert set(spy.loc[spy["kind"] == "trend", "date"]) == {out_day, back_day}


def test_quarter_ends_are_last_trading_days() -> None:
    dates = _bdays("2023-01-02", "2023-12-29")
    assert [d.date().isoformat() for d in quarter_ends(dates)] == ["2023-03-31", "2023-06-30", "2023-09-29"]


# --- 세금 -----------------------------------------------------------------------------


def test_tax_matches_hand_calculation_to_the_won() -> None:
    """손계산
    SPY 취득: 10주 × 400 × 1,300 + 10주 × 420 × 1,350 = 5,200,000 + 5,670,000 = 10,870,000원 (20주, 주당 543,500원)
    매도 1 (2023-06-01) SPY 5주 × 450 × 1,380 = 3,105,000, 취득 5 × 543,500 = 2,717,500 → 차익 387,500
    QQQ 취득: 7 × 333.33 × 1,312.5 = 3,062,469.375원
    매도 2 (2023-09-01) QQQ 7주 × 300.01 × 1,290.7 = 2,710,560.349 → 2,710,560, 취득 3,062,469 → 차익 −351,909
    매도 3 (2024-03-01) SPY 5주 × 900 × 1,400 = 6,300,000, 취득 5 × 543,500 = 2,717,500 → 차익 3,582,500
    2023년 합계 35,591 → 공제 후 0 → 세액 0
    2024년 합계 3,582,500 → 과세표준 1,082,500 → × 22% = 238,150
    """
    ledger = TaxLedger()
    ledger.buy("SPY", 10, 400, 1300)
    ledger.buy("SPY", 10, 420, 1350)
    ledger.buy("QQQ", 7, 333.33, 1312.5)
    r1 = ledger.sell(pd.Timestamp("2023-06-01"), "SPY", 5, 450, 1380)
    r2 = ledger.sell(pd.Timestamp("2023-09-01"), "QQQ", 7, 300.01, 1290.7)
    r3 = ledger.sell(pd.Timestamp("2024-03-01"), "SPY", 5, 900, 1400)

    assert (r1.proceeds_krw, r1.cost_krw, r1.gain_krw) == (3_105_000, 2_717_500, 387_500)
    assert (r2.proceeds_krw, r2.cost_krw, r2.gain_krw) == (2_710_560, 3_062_469, -351_909)
    assert (r3.proceeds_krw, r3.cost_krw, r3.gain_krw) == (6_300_000, 2_717_500, 3_582_500)
    years = yearly_tax(ledger.realized, 0.22, 2_500_000).set_index("year")
    assert years.loc[2023].tolist() == [35_591, 0, 0]
    assert years.loc[2024].tolist() == [3_582_500, 1_082_500, 238_150]
    assert float(ledger.lots["SPY"].avg_cost_usd) == pytest.approx(410.0)  # 매도해도 평균가는 그대로


def test_tax_deduction_boundary() -> None:
    assert tax_due(2_500_000, 0.22, 2_500_000) == 0
    assert tax_due(2_500_004, 0.22, 2_500_000) == 0  # 0.88원 → 버림
    assert tax_due(2_500_005, 0.22, 2_500_000) == 1  # 1.1원
    assert tax_due(2_500_010, 0.22, 2_500_000) == 2
    assert tax_due(3_000_000, 0.22, 2_500_000) == 110_000
    assert tax_due(-1_000_000, 0.22, 2_500_000) == 0


def test_holdings_basis_from_avg_cost_and_fx() -> None:
    ledger = TaxLedger()
    ledger.set_holding("SPY", 3, 500.0, 1200.0)  # 취득 원화 = 3 × 500 × 1,200 = 1,800,000
    r = ledger.sell(pd.Timestamp("2026-01-05"), "SPY", 1, 600.0, 1300.0)
    assert (r.proceeds_krw, r.cost_krw, r.gain_krw) == (780_000, 600_000, 180_000)


# --- 실사용 모드: 납입금 우선 배분 --------------------------------------------------------


def _flat_market(tickers: list[str], end: str = "2026-09-30") -> pd.DataFrame:
    dates = _bdays("2026-08-03", end)
    return make_prices(_frame({t: np.full(len(dates), 100.0) for t in tickers}, dates))


def _holdings(tmp_path: Path, rows: list[tuple]) -> pd.DataFrame:
    path = tmp_path / "holdings.csv"
    lines = ["ticker,shares,avg_cost_usd,avg_fx", *[",".join(map(str, r)) for r in rows]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return load_holdings(path)


FX = pd.Series([1300.0], index=pd.DatetimeIndex([pd.Timestamp("2026-08-03")]))
P2_TICKERS = ["SPY", "EFA", "EEM", "IEF", "GLD", "BIL"]
OFF_TARGET = [("SPY", 50, 90, 1250), ("EFA", 10, 90, 1250), ("EEM", 5, 90, 1250),
              ("IEF", 15, 90, 1250), ("GLD", 4, 90, 1250), ("BIL", 5, 90, 1250)]


def test_contribution_alone_fixes_weights_without_selling(tmp_path: Path) -> None:
    holdings = _holdings(tmp_path, OFF_TARGET)  # 8,900달러, GLD 4.5% (목표 10%, 5.5%p 부족)
    plan = plan_rebalance(holdings, _flat_market(P2_TICKERS), FX, CFG, policy_name="P2", contribution_krw=1_560_000)

    assert all(o.side == "buy" for o in plan.orders) and plan.orders
    assert {o.reason for o in plan.orders} == {"납입금"}
    after = plan.table.set_index("ticker")
    assert ((after["weight_after"] - after["target"]).abs() < CFG.band).all()
    assert {o.ticker: o.shares for o in plan.orders} == {"SPY": 5, "GLD": 6}  # 부족분 555달러, 610달러


def test_without_contribution_band_breach_needs_sell(tmp_path: Path) -> None:
    holdings = _holdings(tmp_path, OFF_TARGET)
    plan = plan_rebalance(holdings, _flat_market(P2_TICKERS), FX, CFG, policy_name="P2", contribution_krw=0)
    sides = {(o.ticker, o.side) for o in plan.orders}
    assert ("GLD", "buy") in sides and ("BIL", "sell") in sides  # 매수 자금은 현금 대용을 팔아서


def test_p3_trend_exit_sells_regardless_of_band_and_renders(tmp_path: Path) -> None:
    dates = _bdays("2025-06-02", "2026-09-30")
    spy = np.interp(np.arange(len(dates)), [0, len(dates) - 60, len(dates) - 1], [100, 140, 100])
    paths = {t: np.full(len(dates), 100.0) for t in P2_TICKERS}
    paths["SPY"] = spy
    prices = make_prices(_frame(paths, dates))
    holdings = _holdings(tmp_path, [("SPY", 2, 90, 1250), ("EFA", 10, 90, 1250), ("EEM", 5, 90, 1250),
                                    ("IEF", 15, 90, 1250), ("GLD", 10, 90, 1250), ("BIL", 50, 90, 1250)])
    plan = plan_rebalance(holdings, prices, FX, CFG, policy_name="P3")

    trend = plan.trend.set_index("asset")
    assert bool(trend.loc["SPY", "below"]) and trend.loc["SPY", "month_end"] == "2026-09-30"
    spy_orders = [o for o in plan.orders if o.ticker == "SPY"]
    assert len(spy_orders) == 1 and spy_orders[0].side == "sell" and spy_orders[0].shares == 2
    assert spy_orders[0].reason == "추세 필터"  # 비중 2% 남짓이라 허용 범위로는 팔지 않을 크기
    assert plan.tax["plan_gain_krw"] == 2 * 100 * 1300 - 2 * 90 * 1250

    path = write_plan(render_plan(plan), plan, tmp_path / "out")
    assert path.name == "2026-Q3.md"
    text = path.read_text(encoding="utf-8")
    for section in ("## 현재 비중 vs 목표 비중", "## 주문 목록", "## 추세 필터 상태", "## 예상 세액"):
        assert section in text


def test_glide_path_linear() -> None:
    w = CFG.policies["P2"].weights
    glide = GlidePath(enabled=True, start_date=date(2026, 1, 1), start_equity=0.70, end_equity=0.30)
    out = apply_glide(w, date(2031, 1, 1), glide, date(2036, 1, 1))  # 정확히 중간 → 주식 50%
    assert out["SPY"] + out["EFA"] + out["EEM"] == pytest.approx(0.50)
    assert out["SPY"] / out["EFA"] == pytest.approx(5.5)  # 주식끼리 비율 유지
    assert out["IEF"] == pytest.approx(0.15 + 0.20 * 0.75) and out["BIL"] == pytest.approx(0.05 + 0.20 * 0.25)
    assert sum(out.values()) == pytest.approx(1.0)
    assert apply_glide(w, date(2031, 1, 1), GlidePath(), None) == w  # 기본은 꺼짐


# --- LLM 요약 검증 ----------------------------------------------------------------------

DOC = "| SPY | 매도 | 12 | 5,432.10 | 7,390,000 |\n차이 −3.25%p, 10개월 평균"


def test_summary_with_unknown_number_is_discarded() -> None:
    good = "SPY를 12주 매도한다. 금액은 약 5432.1달러다. 비중 차이가 3.25%p였고 10개월 평균 아래다."
    assert unknown_numbers(good, DOC) == []
    assert checked_summary(good, DOC)[0] == good
    bad = "SPY를 15주 매도한다. 금액은 5,432.10달러다."
    summary, reason = checked_summary(bad, DOC)
    assert summary is None and "15" in reason
    assert checked_summary("   ", DOC)[0] is None


class _Resp:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return self.payload

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *a) -> None:
        return None


def test_request_summary_timeout_retry_and_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = []

    def failing(req, timeout):
        calls.append(timeout)
        raise urllib.error.URLError("connection refused")

    text, err = request_summary(DOC, CFG.llm, opener=failing)
    assert text is None and "connection refused" in err and calls == [CFG.llm.timeout_seconds] * CFG.llm.max_retries

    def ok(req, timeout):
        body = json.loads(req.data)
        assert body["model"] == CFG.llm.model and body["stream"] is False and DOC in body["prompt"]
        return _Resp(json.dumps({"response": "요약"}).encode())

    assert request_summary(DOC, CFG.llm, opener=ok) == ("요약", "")


# --- 백테스트 실행 조건과 기록 ------------------------------------------------------------


def test_alloc_budget_limits_trials() -> None:
    policies = [Policy.from_config(n, CFG) for n in CFG.policies]
    rows = [{c: None for c in LOG_COLUMNS} | {"family": "alloc", "strategy": p.name, "params": p.params_json()}
            for p in policies]
    log = pd.DataFrame(rows, columns=LOG_COLUMNS)
    check_alloc_budget(log, policies, 4)  # 같은 정책 다시 실행은 새 시도가 아니다
    changed = Policy("P2", {"SPY": 0.5, "IEF": 0.5}, False, (), 10, "BIL")
    with pytest.raises(BudgetExceeded):
        check_alloc_budget(log, [changed], 4)


def test_run_all_requires_committed_config(tmp_path: Path) -> None:
    prices = _flat_market(P2_TICKERS)
    with pytest.raises(AllocRefused, match="커밋"):
        run_all(prices, FX, CFG, tmp_path / "results", tmp_path, "test")
    assert not (tmp_path / "results").exists()


def test_yearly_returns_and_period_stats() -> None:
    idx = pd.DatetimeIndex(["2019-12-31", "2020-01-31", "2020-02-28", "2020-03-31", "2020-12-31", "2021-06-30"])
    eq = pd.Series([100.0, 110.0, 99.0, 88.0, 121.0, 133.1], index=idx)
    y = yearly_returns(eq)
    assert y.loc[2020] == pytest.approx(0.21) and y.loc[2021] == pytest.approx(0.10)
    p = period_stats(eq, [("crash", date(2020, 2, 1), date(2020, 3, 31))]).iloc[0]
    assert p["return"] == pytest.approx(88 / 110 - 1) and p["mdd"] == pytest.approx(88 / 110 - 1)
    assert mdd_window(eq) == ("2020-01-31", "2020-03-31")
