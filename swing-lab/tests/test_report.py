"""리포트 생성 스모크 테스트."""

from __future__ import annotations

from pathlib import Path

from conftest import random_walk_prices
from swinglab.config import BacktestConfig
from swinglab.reports import metrics_table, run_cost_scenarios, write_report
from swinglab.strategies import SmaCross


def test_report_has_cost_scenarios_and_charts(tmp_path: Path) -> None:
    prices = random_walk_prices(["SPY", "AAA", "BBB", "CCC"], 300, seed=11)
    config = BacktestConfig(initial_capital=100_000, cost_bps=5, cost_sensitivity_bps=[0, 5, 10])
    strategy = SmaCross(tickers=["AAA", "BBB", "CCC"])

    scenarios = run_cost_scenarios(prices, strategy, config, "SPY")
    assert [s.cost_bps for s in scenarios] == [0, 5, 10]
    # 비용이 클수록 최종 자산이 작다 (같은 신호, 같은 거래)
    finals = [s.result.equity.iloc[-1] for s in scenarios]
    assert finals[0] > finals[1] > finals[2]
    assert all(s.benchmark.equity.index.equals(s.result.equity.index) for s in scenarios)

    table = metrics_table(scenarios, 5, "SPY")
    assert list(table.columns) == ["전략 0bp", "전략 5bp", "전략 10bp", "SPY 바이앤홀드 5bp"]
    assert {"CAGR", "샤프비율", "MDD", "SPY 대비 CAGR 차이"} <= set(table.index)

    path = write_report(scenarios, strategy, config, "SPY", tmp_path)
    html = path.read_text(encoding="utf-8")
    assert path.name == "sma_cross.html"
    assert html.count('class="plotly-graph-div"') == 2  # 자산곡선, 드로다운 (제목은 JSON 안에 이스케이프됨)
    assert "성과 지표" in html and "거래 내역" in html
