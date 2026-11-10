"""검증 구간 변동성 맞춤 벤치마크 (마크다운 표).

후보마다 SPY와 현금(수익률 0%)을 섞어 검증 구간 연 변동성을 후보와 같게 맞춘 포트폴리오를 만들고,
CAGR·샤프·MDD를 후보 옆에 나란히 낸다.

- SPY 비중 w = 후보 연 변동성 / SPY 연 변동성. 같은 구간·같은 비용의 일간 수익률로 사후에 정한다.
- 매일 비중 w로 다시 맞춘다고 본다 (일간 수익률 = w × SPY 일간 수익률). 재조정 비용은 넣지 않는다.
- 샤프는 무위험수익률 0으로 계산하므로 벤치마크 샤프는 비중과 상관없이 SPY 샤프와 같다.

새 전략 시도가 아니므로 실험 로그와 예산에 넣지 않는다. results/에 저장된 검증 구간 자산곡선만 읽고
백테스트를 다시 돌리지 않는다 (swinglab holdout 실행 후에만 쓸 수 있다).

실행: swing-lab 디렉터리에서 `uv run python scripts/vol_matched_benchmark.py`
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

from swinglab.backtest.metrics import (
    annual_volatility,
    cagr,
    daily_returns,
    max_drawdown,
    sharpe_ratio,
)
from swinglab.research.config import load_stage1
from swinglab.research.holdout import read_lock
from swinglab.research.log import latest_runs, read_log
from swinglab.research.runner import read_curves
from swinglab.research.selection import read_candidates

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CANDIDATES = ROOT / "config" / "candidates.yaml"


class MissingResults(Exception):
    pass


def vol_matched_table() -> pd.DataFrame:
    cfg = load_stage1(ROOT / "config")
    if not read_lock(RESULTS):
        raise MissingResults("검증 구간 실행 기록(results/holdout.lock)이 없습니다.")
    log = read_log(RESULTS)
    curves = read_curves(RESULTS)
    specs = read_candidates(CANDIDATES)

    rows = []
    for cost in cfg.cost_bps:
        spy = _curve(curves, f"benchmark:holdout:{cost:g}")
        spy_rets = daily_returns(spy)
        spy_vol = annual_volatility(spy_rets)
        runs = latest_runs(log, "holdout", cost)
        for spec in specs:
            match = runs[
                (runs["family"] == spec.family)
                & (runs["strategy"] == spec.display_name)
                & (runs["params"] == spec.params_json)
            ]
            if match.empty:
                raise MissingResults(f"검증 구간 실행 없음: {spec.family} {spec.display_name} {cost:g}bp")
            run = match.iloc[0]
            equity = _curve(curves, str(run["run_id"]))
            if not equity.index.equals(spy.index):
                raise MissingResults(f"{spec.family} {cost:g}bp: 후보와 SPY 자산곡선의 날짜가 다릅니다")

            vol = annual_volatility(daily_returns(equity))
            if not math.isclose(vol, float(run["vol"]), rel_tol=1e-9):
                raise MissingResults(f"{spec.family} {cost:g}bp: 다시 계산한 변동성이 실험 로그와 다릅니다")
            weight = vol / spy_vol
            mix = _mixed_curve(spy_rets, weight, spy.index[0])
            mix_rets = daily_returns(mix)
            assert math.isclose(annual_volatility(mix_rets), vol, rel_tol=1e-9)

            rows.append(
                {
                    "비용": f"{cost:g}bp",
                    "후보": f"{spec.family} {spec.display_name}",
                    "연 변동성": _pct(vol),
                    "SPY 비중": _pct(weight),
                    "CAGR 후보": _pct(float(run["cagr"])),
                    "CAGR 벤치": _pct(cagr(mix)),
                    "샤프 후보": _num(float(run["sharpe"])),
                    "샤프 벤치": _num(sharpe_ratio(mix_rets)),
                    "MDD 후보": _pct(float(run["mdd"])),
                    "MDD 벤치": _pct(max_drawdown(mix)),
                }
            )
    return pd.DataFrame(rows)


def _curve(curves: pd.DataFrame, run_id: str) -> pd.Series:
    c = curves[curves["run_id"] == run_id]
    if c.empty:
        raise MissingResults(f"자산곡선 없음: {run_id}")
    return c.set_index("date")["equity"].sort_index()


def _mixed_curve(spy_rets: pd.Series, weight: float, start: pd.Timestamp) -> pd.Series:
    """SPY 비중 weight, 나머지 현금(0%)을 매일 다시 맞춘 자산곡선 (시작값 1)."""
    growth = (1.0 + weight * spy_rets).cumprod()
    return pd.concat([pd.Series([1.0], index=[start]), growth])


def to_markdown(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join("---" for _ in df.columns) + " |"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _pct(v: float) -> str:
    return "–" if math.isnan(v) else f"{v:.2%}"


def _num(v: float) -> str:
    return "–" if math.isnan(v) else f"{v:.2f}"


def main() -> int:
    try:
        table = vol_matched_table()
    except MissingResults as exc:
        print(exc, file=sys.stderr)
        return 1
    print(to_markdown(table))
    return 0


if __name__ == "__main__":
    sys.exit(main())
