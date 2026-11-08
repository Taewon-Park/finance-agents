"""1단계 전체 흐름 (합성 데이터): 학습 실행 A·B·C → D → 후보 선정 → 리포트 → 검증 구간 → 리포트."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from conftest import random_walk_prices
from swinglab.config import BacktestConfig, UniverseConfig
from swinglab.reports.stage1 import write_stage1_report
from swinglab.research.config import Stage1Config
from swinglab.research.holdout import run_holdout
from swinglab.research.log import read_log
from swinglab.research.runner import run_train
from swinglab.research.selection import candidates_document, read_candidates, select_all, write_candidates

UNIVERSE = UniverseConfig(
    benchmark="SPY", start="2016-01-01", groups={"b": ["SPY"], "x": ["AAA", "BBB", "CCC"], "safe": ["IEF", "TLT"]}
)
BT = BacktestConfig(initial_capital=100_000, cost_bps=5, max_positions=5)
CFG = Stage1Config.model_validate(
    {
        "periods": {"train": {"start": "2016-01-01", "end": "2018-12-31"}, "holdout": {"start": "2019-01-01", "end": None}},
        "cost_bps": [0, 5, 10],
        "selection_cost_bps": 5,
        "budget": {"per_family": 12, "total": 50},
        "selection": {"min_trades": 1, "min_trades_exempt": ["B"], "isolation_ratio": 0.7, "filter_min_sharpe_gain": -10},
        "families": {
            "A": {"strategy": "rsi2_v2", "axes": {"entry_th": [10], "min_hold": [1, 3]}},
            "B": {"strategy": "dual_momentum", "universe": ["AAA", "BBB", "CCC"], "fixed": {"safe": "IEF"},
                  "axes": {"lookback": [6], "top_n": [2]}},
            "C": {"strategy": "breakout", "axes": {"channel": [{"entry_n": 20, "exit_n": 10}]}},
            "D": {"wraps": ["A", "C"], "extra": ["sma_cross"]},
        },
    }
)


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return random_walk_prices(["SPY", "AAA", "BBB", "CCC", "IEF", "TLT"], 1300, seed=12, drift=0.0005, start="2016-01-01")


def test_full_stage1_flow_and_report(tmp_path: Path, prices: pd.DataFrame) -> None:
    results, reports = tmp_path / "results", tmp_path / "reports"
    for fam in ["A", "B", "C", "D"]:
        run_train(fam, prices, UNIVERSE, BT, CFG, results, tmp_path)
    log = read_log(results)
    assert set(log["period"]) == {"train"}
    assert set(log["family"]) == {"A", "B", "C", "D", "baseline"}
    assert len(log) == (2 + 1 + 1 + 3 + 1) * 3  # 조합(D 3개 + baseline 1개) × 비용 3

    selections = select_all(log, CFG)
    doc = candidates_document(selections, CFG, "test")
    cand = tmp_path / "candidates.yaml"
    write_candidates(cand, doc)
    specs = read_candidates(cand)
    assert {s.family for s in specs} == {"A", "B", "C", "D"}  # 필터 기준 -10이라 D도 후보가 생긴다
    assert next(s for s in specs if s.family == "B").tickers == ("AAA", "BBB", "CCC")

    html = write_stage1_report(CFG, results, cand, reports).read_text(encoding="utf-8")
    assert "미실행" in html and "검증 구간 · 후보와" not in html
    assert "상관행렬" in html and html.count('class="plotly-graph-div"') == 2

    run_holdout(cand, prices, UNIVERSE, BT, CFG, results, tmp_path, require_committed=False)
    html = write_stage1_report(CFG, results, cand, reports).read_text(encoding="utf-8")
    assert "검증 구간 · 후보와 SPY" in html and "학습 대비 검증 샤프" in html
    assert html.count('class="plotly-graph-div"') == 4
    assert html.count("cdn.plot.ly/plotly-") == 1  # plotly.js는 한 번만 불러온다
