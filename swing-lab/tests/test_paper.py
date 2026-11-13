"""2단계 트랙 1: 페이퍼 트레이딩."""

from __future__ import annotations

import fcntl
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from conftest import random_walk_prices
from typer.testing import CliRunner

from swinglab.backtest.engine import run_backtest
from swinglab.backtest.metrics import annual_volatility, daily_returns
from swinglab.cli import app
from swinglab.config import BacktestConfig, UniverseConfig
from swinglab.paper.account import MarketView, Rules, advance, open_account
from swinglab.paper.config import (
    PaperConfig,
    PaperRefused,
    load_paper_config,
    read_frozen,
)
from swinglab.paper.report import (
    report_text,
    summarize_accounts,
    weekly_line,
    write_weekly,
)
from swinglab.paper.runner import run_paper
from swinglab.paper.store import TABLE_KEYS, read_state, read_table, run_lock
from swinglab.research.selection import read_candidates
from swinglab.strategies import Breakout, BuyAndHold, Rsi2Swing

ROOT = Path(__file__).resolve().parents[1]
TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
UNIVERSE = UniverseConfig(benchmark="SPY", start=date(2020, 1, 1), groups={"b": ["SPY"], "t": TICKERS})


def _prices(n: int = 330, seed: int = 7) -> pd.DataFrame:
    return random_walk_prices([*TICKERS, "SPY"], n, seed=seed, drift=0.0006, vol=0.02)


def _cfg(**overrides) -> PaperConfig:
    data = yaml.safe_load((ROOT / "config" / "paper.yaml").read_text(encoding="utf-8"))
    data.update(tickers=TICKERS, **overrides)
    return PaperConfig.model_validate(data)


def _upto(prices: pd.DataFrame, d: pd.Timestamp) -> pd.DataFrame:
    return prices[prices["date"] <= d]


def _dates(prices: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(np.sort(prices["date"].unique()))


def _run(cfg: PaperConfig, prices: pd.DataFrame, paper_dir: Path, repo: Path = ROOT):
    return run_paper(cfg, prices, UNIVERSE, paper_dir, repo, "test")


# --- 고정 설정과 해시 ---------------------------------------------------------------


def test_paper_yaml_matches_stage1_candidates() -> None:
    cfg = load_paper_config(ROOT / "config")
    cands = {c.family: c for c in read_candidates(ROOT / "config" / "candidates.yaml")}
    for fam in ("A", "C"):
        assert cfg.strategies[fam].strategy == cands[fam].strategy
        assert cfg.strategies[fam].params == cands[fam].params
    universe = UniverseConfig.model_validate(yaml.safe_load((ROOT / "config" / "universe.yaml").read_text()))
    assert cfg.tickers == universe.tradable and len(cfg.tickers) == 19
    assert cfg.cost_bps == 5 and cfg.max_positions == 5


def test_hash_mismatch_refused_and_new_paper_id_allowed(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    cfg = _cfg()
    for rel in cfg.source_files:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, repo / rel)
    paper_dir = tmp_path / "paper"
    prices = _prices()
    _run(cfg, _upto(prices, _dates(prices)[250]), paper_dir, repo)
    frozen_before = read_frozen(paper_dir)

    # 전략 소스가 바뀌면 거부
    src = repo / "src/swinglab/strategies/breakout.py"
    src.write_text(src.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    with pytest.raises(PaperRefused, match="breakout.py"):
        _run(cfg, prices, paper_dir, repo)
    # 설정(파라미터)이 바뀌어도 거부 (원래 소스로 되돌린 뒤)
    shutil.copy(ROOT / "src/swinglab/strategies/breakout.py", src)
    changed = _cfg(strategies={"C": {"strategy": "breakout", "params": {"entry_n": 55, "exit_n": 20}},
                               "A": {"strategy": "rsi2_v2", "params": {"entry_th": 5, "min_hold": 5}}})
    with pytest.raises(PaperRefused, match="paper.yaml"):
        _run(changed, prices, paper_dir, repo)
    # 거부된 실행은 아무것도 바꾸지 않는다
    assert read_frozen(paper_dir) == frozen_before
    assert read_state(paper_dir, "p1", "C").last_processed == _dates(prices)[250].date().isoformat()

    # 새 paper_id는 새 계정으로 시작하고 기존 기록은 그대로
    eq_before = read_table(paper_dir, "equity")
    _run(changed.model_copy(update={"paper_id": "p2"}), prices, paper_dir, repo)
    frozen = read_frozen(paper_dir)
    assert set(frozen) == {"p1", "p2"} and frozen["p1"] == frozen_before["p1"]
    eq = read_table(paper_dir, "equity")
    pd.testing.assert_frame_equal(eq[eq["paper_id"] == "p1"].reset_index(drop=True), eq_before)
    assert frozen["p2"]["start_date"] == _dates(prices)[-1].date().isoformat()


# --- 날짜 처리 ---------------------------------------------------------------------


def test_same_day_rerun_adds_nothing(tmp_path: Path) -> None:
    cfg, prices = _cfg(), _prices()
    dates = _dates(prices)
    _run(cfg, _upto(prices, dates[240]), tmp_path)
    _run(cfg, _upto(prices, dates[300]), tmp_path)
    tables = {n: read_table(tmp_path, n) for n in ("orders", "fills", "positions", "equity", "trades")}
    state = (tmp_path / "state.json").read_text(encoding="utf-8")

    report = _run(cfg, _upto(prices, dates[300]), tmp_path)

    assert all(not a.processed for a in report.accounts)
    for name, before in tables.items():
        pd.testing.assert_frame_equal(read_table(tmp_path, name), before, obj=name)
    assert (tmp_path / "state.json").read_text(encoding="utf-8") == state
    eq = read_table(tmp_path, "equity")
    assert not eq.duplicated(["paper_id", "account", "date"]).any()


def test_missed_days_processed_in_order(tmp_path: Path) -> None:
    cfg, prices = _cfg(), _prices()
    dates = _dates(prices)
    _run(cfg, _upto(prices, dates[250]), tmp_path)

    report = _run(cfg, _upto(prices, dates[257]), tmp_path)  # 7거래일 동안 실행 못 함

    expected = [d.date().isoformat() for d in dates[251:258]]
    for acct in report.accounts:
        assert acct.processed == expected
    eq = read_table(tmp_path, "equity")
    for acct in cfg.strategies:
        days = eq.loc[eq["account"] == acct, "date"].tolist()
        assert days == [d.date().isoformat() for d in dates[250:258]]

    # 한 번에 따라잡은 결과 = 매일 실행한 결과
    daily = tmp_path / "daily"
    for i in range(250, 258):
        _run(cfg, _upto(prices, dates[i]), daily)
    for name in ("equity", "fills", "orders", "trades"):  # 행 순서만 다를 수 있다 (계정별로 따라잡음)
        a, b = read_table(daily, name), read_table(tmp_path, name)
        assert a.empty == b.empty, name
        if a.empty:
            continue
        keys = TABLE_KEYS[name]
        a = a.sort_values(keys).reset_index(drop=True)
        b = b.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, obj=name)


def test_partial_ingest_day_not_processed(tmp_path: Path) -> None:
    cfg, prices = _cfg(), _prices()
    dates = _dates(prices)
    _run(cfg, _upto(prices, dates[250]), tmp_path)
    partial = _upto(prices, dates[252])
    partial = partial[~((partial["ticker"] == "CCC") & (partial["date"] == dates[252]))]  # CCC만 마지막 날 수집 실패

    report = _run(cfg, partial, tmp_path)

    assert all(a.last_processed == dates[251].date().isoformat() for a in report.accounts)


def _sorted(paper_dir: Path, name: str) -> pd.DataFrame:
    df = read_table(paper_dir, name)
    return df.sort_values(TABLE_KEYS[name]).reset_index(drop=True) if not df.empty else df


def test_catch_up_after_three_missed_days_matches_daily_runs(tmp_path: Path) -> None:
    """3거래일씩 실행을 못 하고 넷째 날 한 번에 따라잡기를 반복해도 매일 실행한 기록과 같다 (모든 표와 상태)."""
    cfg, prices = _cfg(), _prices()
    dates = _dates(prices)
    daily, catch_up = tmp_path / "daily", tmp_path / "catch_up"
    for i in range(240, 329):
        _run(cfg, _upto(prices, dates[i]), daily)
    reports = [_run(cfg, _upto(prices, dates[i]), catch_up) for i in range(240, 329, 4)]

    assert all(len(a.processed) == 4 for r in reports[1:] for a in r.accounts)  # 밀린 3일 + 그날
    fills = read_table(daily, "fills")
    assert {"buy", "sell"} <= set(fills["side"]) and len(read_table(daily, "trades")) >= 5  # 매매가 실제로 있었던 구간
    for name in ("orders", "fills", "positions", "equity", "trades"):
        pd.testing.assert_frame_equal(_sorted(catch_up, name), _sorted(daily, name), obj=name)
    for account in cfg.strategies:
        assert read_state(catch_up, "p1", account) == read_state(daily, "p1", account)


def test_paper_run_skips_when_another_run_holds_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """정시 실행과 로그인 시 실행이 겹치면 뒤의 실행은 아무것도 하지 않고 종료 코드 0."""
    monkeypatch.setattr("swinglab.cli.power_source", lambda: "ac")  # 전원 확인(a)을 통과한 상태
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    with (paper_dir / ".lock").open("w") as other:  # 다른 프로세스가 잡은 잠금과 같은 상태
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = CliRunner().invoke(app, ["paper", "run", "--no-ingest", "--paper-dir", str(paper_dir)])
        fcntl.flock(other, fcntl.LOCK_UN)
    assert result.exit_code == 0, result.output
    assert "이미 실행 중, 건너뜀" in result.output
    assert sorted(p.name for p in paper_dir.iterdir()) == [".lock"]  # 기록·상태를 쓰지 않았다


def test_run_lock_is_reentrant_within_process(tmp_path: Path) -> None:
    """CLI가 수집 전에 잡은 잠금 안에서 run_paper가 다시 잡아도 된다. 다른 프로세스에게는 계속 잠겨 있다."""
    cfg, prices = _cfg(), _prices()
    with run_lock(tmp_path):
        _run(cfg, _upto(prices, _dates(prices)[250]), tmp_path)
        with (tmp_path / ".lock").open("w") as other, pytest.raises(BlockingIOError):
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with (tmp_path / ".lock").open("w") as other:  # 빠져나오면 풀린다
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(other, fcntl.LOCK_UN)


def test_orders_fill_at_next_session_open(tmp_path: Path) -> None:
    cfg, prices = _cfg(), _prices()
    dates = _dates(prices)
    for i in range(240, 330):
        _run(cfg, _upto(prices, dates[i]), tmp_path)
    fills = read_table(tmp_path, "fills")
    orders = read_table(tmp_path, "orders")
    filled = orders[orders["status"] == "filled"]
    assert len(filled) > 10 and (fills["side"] == "sell").any()

    opens = prices.pivot(index="date", columns="ticker", values="open")
    for _, o in filled.iterrows():
        nxt = dates[dates.get_loc(pd.Timestamp(o["signal_date"])) + 1]
        f = fills[(fills["account"] == o["account"]) & (fills["ticker"] == o["ticker"])
                  & (fills["side"] == o["side"]) & (fills["date"] == o["fill_date"])]
        assert len(f) == 1
        if o["side"] == "buy":  # 진입은 신호 다음 거래일 시가
            assert o["fill_date"] == nxt.date().isoformat()
        assert f["price"].iloc[0] == pytest.approx(opens.at[pd.Timestamp(o["fill_date"]), o["ticker"]])
        assert pd.Timestamp(o["fill_date"]) > pd.Timestamp(o["signal_date"])
    # 시작일 이전 날짜의 기록은 없다
    start = read_frozen(tmp_path)["p1"]["start_date"]
    assert fills["date"].min() > start and read_table(tmp_path, "equity")["date"].min() == start


# --- 엔진과 같은 규칙 ------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy",
    [Breakout(tickers=TICKERS, entry_n=20, exit_n=10), Rsi2Swing(tickers=TICKERS, entry_th=10, min_hold=3)],
    ids=["breakout", "rsi2_v2"],
)
def test_day_by_day_matches_engine(strategy) -> None:
    prices = _prices(n=360, seed=11)
    # 결측 시가 (청산 연기·진입 취소 경로)
    drop = (prices["ticker"] == "BBB") & prices["date"].isin(_dates(prices)[[262, 263, 290]])
    prices = prices[~drop]
    dates = _dates(prices)
    t0 = dates[230]
    rules = Rules.from_strategy(strategy, 5.0, 5)

    state, _ = open_account("p", "x", t0, 100_000.0, MarketView.build(_upto(prices, t0), strategy), rules)
    equity = [100_000.0]
    trades = []
    for d in dates[231:]:
        record = advance(state, MarketView.build(_upto(prices, d), strategy), d, rules)
        equity.append(record.equity)
        trades += record.trades

    bt = run_backtest(prices, strategy, BacktestConfig(initial_capital=100_000.0, cost_bps=5.0, start=dates[231].date()))
    np.testing.assert_allclose(equity, bt.equity.to_numpy(), rtol=1e-12)
    assert len(trades) == len(bt.trades) > 5
    np.testing.assert_allclose([t["return"] for t in trades], bt.trades["return"].to_numpy(), rtol=1e-12)


def test_dividend_readjustment_keeps_total_return() -> None:
    """보유 중 과거 수정주가가 바뀌어도(배당) 한 스냅샷으로 계속 돌린 결과와 같다."""
    strategy = Breakout(tickers=TICKERS, entry_n=20, exit_n=10)
    prices = _prices(n=360, seed=5)
    dates = _dates(prices)
    t0, k = dates[230], dates[280]
    rules = Rules.from_strategy(strategy, 5.0, 5)

    def run(snapshot_for):
        state, _ = open_account("p", "x", t0, 100_000.0, MarketView.build(snapshot_for(t0), strategy), rules)
        out = [100_000.0]
        for d in dates[231:]:
            out.append(advance(state, MarketView.build(snapshot_for(d), strategy), d, rules).equity)
        return out

    reference = run(lambda d: _upto(prices, d))
    # k일까지는 배당 반영 전 스냅샷: 모든 종목의 과거 가격이 1/0.97배 (k+1일부터 새 기준)
    old = prices.copy()
    old[["open", "high", "low", "close"]] = old[["open", "high", "low", "close"]] / 0.97
    adjusted = run(lambda d: _upto(old if d <= k else prices, d))

    np.testing.assert_allclose(np.array(adjusted[51:]) / np.array(reference[51:]), 1.0, rtol=1e-9)


# --- 리포트 ----------------------------------------------------------------------------


def test_report_and_weekly_line(tmp_path: Path) -> None:
    cfg, prices = _cfg(), _prices()
    dates = _dates(prices)
    for i in (240, 300, 329):
        _run(cfg, _upto(prices, dates[i]), tmp_path)
    summaries = summarize_accounts(cfg, tmp_path, prices)
    assert {s.account for s in summaries} == {"C", "A"}
    s = summaries[0]
    assert s.sessions == 89 and not np.isnan(s.sharpe)
    # 비교 대상: 시작일 다음 거래일 시가에 산 SPY, 혼합 비중 = 계정 변동성 / SPY 변동성
    spy = run_backtest(prices, BuyAndHold(ticker="SPY"), BacktestConfig(
        initial_capital=cfg.initial_capital, cost_bps=5.0, start=dates[241].date(), end=dates[329].date()))
    assert s.spy_total_return == pytest.approx(spy.equity.iloc[-1] / spy.equity.iloc[0] - 1.0)
    assert s.mix_weight == pytest.approx(s.vol / annual_volatility(daily_returns(spy.equity)))
    key, line = weekly_line(cfg, summaries)
    write_weekly(tmp_path, key, line)
    write_weekly(tmp_path, key, line.replace("|", "¦"))  # 같은 주는 한 줄로 바뀐다
    text = (tmp_path / "weekly.md").read_text(encoding="utf-8")
    assert text.count(key) == 1 and "¦" in text


def test_report_hides_annualized_metrics_before_60_sessions(tmp_path: Path) -> None:
    cfg, prices = _cfg(), _prices()
    dates = _dates(prices)
    _run(cfg, _upto(prices, dates[240]), tmp_path)
    _run(cfg, _upto(prices, dates[299]), tmp_path)  # 59거래일

    short = summarize_accounts(cfg, tmp_path, prices)
    assert all(s.sessions == 59 for s in short)
    for s in short:
        assert all(np.isnan(v) for v in (s.cagr, s.sharpe, s.spy_cagr, s.spy_sharpe, s.mix_cagr))
        assert not np.isnan(s.total_return) and not np.isnan(s.spy_total_return) and not np.isnan(s.vol)
    text = report_text(cfg, short, dates[240].date().isoformat())
    row = next(line for line in text.splitlines() if line.startswith("| C "))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[4] == "–" and cells[6] == "–" and cells[3] != "–"  # CAGR, 샤프는 빈칸, 누적은 표시
    assert "60거래일 전에는 CAGR·샤프를" in text

    _run(cfg, _upto(prices, dates[300]), tmp_path)  # 60거래일부터 표시
    full = summarize_accounts(cfg, tmp_path, prices)
    assert all(s.sessions == 60 and not np.isnan(s.cagr) and not np.isnan(s.sharpe) for s in full)
    assert "60거래일 전에는" not in report_text(cfg, full, dates[240].date().isoformat())
