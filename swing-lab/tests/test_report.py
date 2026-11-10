"""리포트 생성 스모크 테스트."""

from __future__ import annotations

import re
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


CDN_SCRIPT = re.compile(r'<script[^>]*src="https://cdn\.plot\.ly/plotly-[^"]+\.min\.js"')


def test_report_loads_plotly_from_cdn_by_default_and_embeds_when_offline(tmp_path: Path) -> None:
    prices = random_walk_prices(["SPY", "AAA"], 150, seed=2)
    config = BacktestConfig(initial_capital=100_000, cost_bps=5, cost_sensitivity_bps=[5])
    strategy = SmaCross(tickers=["AAA"])
    scenarios = run_cost_scenarios(prices, strategy, config, "SPY")

    cdn_html = write_report(scenarios, strategy, config, "SPY", tmp_path / "cdn").read_text(encoding="utf-8")
    offline_html = write_report(scenarios, strategy, config, "SPY", tmp_path / "off", offline=True).read_text(
        encoding="utf-8"
    )
    assert len(CDN_SCRIPT.findall(cdn_html)) == 1
    assert not CDN_SCRIPT.search(offline_html)
    assert len(offline_html) > len(cdn_html) + 1_000_000  # plotly.js 본문(수 MB)이 들어감
