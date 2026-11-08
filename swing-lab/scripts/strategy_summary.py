"""전략 비교 요약표 (마크다운).

전략 × 비용 시나리오(backtest.yaml의 cost_sensitivity_bps)별로
CAGR, MDD, 샤프비율, 노출도, 연 거래 횟수, 같은 기간·같은 비용의 SPY 바이앤홀드 CAGR을 한 표로 낸다.
값은 swinglab backtest 리포트와 같은 함수(run_cost_scenarios)로 계산한다.

실행: swing-lab 디렉터리에서 `uv run python scripts/strategy_summary.py`
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

from swinglab.config import load_backtest_config, load_universe
from swinglab.pipeline.storage import clean_path, read_prices
from swinglab.reports import run_cost_scenarios
from swinglab.strategies import STRATEGY_NAMES, build_strategy

ROOT = Path(__file__).resolve().parents[1]


def summary_table() -> tuple[pd.DataFrame, str]:
    universe = load_universe(ROOT / "config")
    config = load_backtest_config(ROOT / "config")
    prices = read_prices(clean_path(ROOT / "data"))
    bench = universe.benchmark

    rows = []
    end_date = None
    for name in STRATEGY_NAMES:
        for s in run_cost_scenarios(prices, build_strategy(name, universe), config, bench):
            m = s.metrics
            end_date = s.result.end_date.date()
            rows.append(
                {
                    "전략": name,
                    "비용": f"{s.cost_bps:g}bp",
                    "시작일": s.result.start_date.date().isoformat(),
                    "CAGR": _pct(m["cagr"]),
                    "MDD": _pct(m["mdd"]),
                    "샤프비율": _num(m["sharpe"], 2),
                    "노출도": _pct(m["exposure"]),
                    "연 거래 횟수": _num(m["trades_per_year"], 1),
                    f"같은 기간 {bench} CAGR": _pct(s.benchmark_metrics["cagr"]),
                }
            )
    note = f"마지막 날 {end_date}. {bench} CAGR은 전략과 같은 첫 체결일·마지막 날, 같은 비용의 바이앤홀드."
    return pd.DataFrame(rows), note


def to_markdown(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join("---" for _ in df.columns) + " |"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _pct(v: float) -> str:
    return "–" if math.isnan(v) else f"{v:.2%}"


def _num(v: float, digits: int) -> str:
    return "–" if math.isnan(v) else f"{v:.{digits}f}"


def main() -> int:
    if not clean_path(ROOT / "data").exists():
        print("data/clean/prices.parquet가 없습니다. 먼저 swinglab ingest를 실행하세요.", file=sys.stderr)
        return 1
    table, note = summary_table()
    print(to_markdown(table))
    print(f"\n{note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
