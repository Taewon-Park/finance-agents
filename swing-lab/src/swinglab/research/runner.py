"""학습 구간 실험 실행.

학습 구간 실행은 가격 데이터를 학습 구간 끝날까지 잘라서 넘긴다. 그래서 이 경로에는 검증 구간
가격이 아예 들어오지 않는다 (신호·체결·성과 모두). 검증 구간은 research.holdout만 실행한다.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from swinglab.backtest.engine import BacktestResult, run_backtest
from swinglab.backtest.metrics import performance_metrics
from swinglab.config import BacktestConfig, UniverseConfig
from swinglab.research.config import Period, Stage1Config
from swinglab.research.log import BASELINE_FAMILY, write_atomic, append_rows, check_budget, git_commit, read_log
from swinglab.research.selection import filter_pairs, select_grid_family
from swinglab.research.specs import StrategySpec
from swinglab.strategies import BuyAndHold


def curves_path(results_dir: Path) -> Path:
    return results_dir / "equity_curves.parquet"


def benchmarks_path(results_dir: Path) -> Path:
    return results_dir / "benchmarks.parquet"


def train_prices(prices: pd.DataFrame, cfg: Stage1Config) -> pd.DataFrame:
    """학습 구간 끝날까지만 남긴 가격."""
    return prices[prices["date"] <= pd.Timestamp(cfg.train.end)].reset_index(drop=True)


def train_specs(family: str, cfg: Stage1Config, log: pd.DataFrame) -> list[StrategySpec]:
    fam = cfg.families[family]
    if not fam.is_filter_family:
        tickers = tuple(fam.universe) if fam.universe else None
        return [StrategySpec(family, fam.strategy or "", params, False, tickers) for _, params in fam.grid()]
    wrapped = {w: select_grid_family(log, w, cfg).candidate for w in fam.wraps}
    specs: list[StrategySpec] = []
    for base, filtered in filter_pairs(family, cfg, wrapped):
        if base.family == BASELINE_FAMILY:
            specs.append(base)  # 필터 전 비교 기준 (시도 수에는 넣지 않음)
        specs.append(filtered)
    return specs


def run_train(
    family: str,
    prices: pd.DataFrame,
    universe: UniverseConfig,
    bt_config: BacktestConfig,
    cfg: Stage1Config,
    results_dir: Path,
    repo_dir: Path,
) -> pd.DataFrame:
    """한 계열의 모든 조합을 학습 구간에서 비용 시나리오별로 실행하고 실험 로그에 기록한다."""
    if family not in cfg.families:
        raise ValueError(f"알 수 없는 계열: {family} (가능: {', '.join(cfg.families)})")
    log = read_log(results_dir)
    specs = train_specs(family, cfg, log)
    check_budget(log, specs, cfg)

    sliced = train_prices(prices, cfg)
    if sliced.empty:
        raise ValueError("학습 구간 가격 데이터가 없습니다")
    window = Period(start=cfg.train.evaluation_start, end=cfg.train.end)
    if cfg.train.evaluate_from is not None:
        _check_warmup(specs, sliced, window, universe)
    rows = run_specs(specs, sliced, window, "train", universe, bt_config, cfg, results_dir, git_commit(repo_dir))
    save_benchmark(sliced, "train", window, universe, bt_config, cfg, results_dir)
    return pd.DataFrame(rows)


def _check_warmup(specs: list[StrategySpec], prices: pd.DataFrame, window: Period, universe: UniverseConfig) -> None:
    """공통 평가 시작일 전에 모든 전략의 워밍업이 끝나는지. 아니면 전략마다 시작일이 달라지므로 거부한다."""
    calendar = pd.DatetimeIndex(np.sort(prices["date"].unique()))
    start_idx = int(calendar.searchsorted(pd.Timestamp(window.start), side="left"))
    late = [s.display_name for s in specs if s.build(universe).warmup_days > start_idx]
    if late:
        raise ValueError(f"평가 시작일 {window.start}이 워밍업보다 이릅니다: {', '.join(late)}")


def run_specs(
    specs: list[StrategySpec],
    prices: pd.DataFrame,
    window: Period,
    period: str,
    universe: UniverseConfig,
    bt_config: BacktestConfig,
    cfg: Stage1Config,
    results_dir: Path,
    commit: str,
) -> list[dict]:
    """명세 × 비용마다 백테스트 1회 → 실험 로그 1행 + 자산곡선 저장."""
    rows, curves = [], []
    for spec in specs:
        strategy = spec.build(universe)
        for cost in cfg.cost_bps:
            run_cfg = bt_config.model_copy(update={"cost_bps": cost, "start": window.start, "end": window.end})
            result = run_backtest(prices, strategy, run_cfg)
            run_id = str(uuid.uuid4())
            rows.append(_log_row(run_id, commit, spec, period, cost, result))
            curves.append(_curve(run_id, result))
    append_rows(results_dir, rows)
    _append_curves(results_dir, curves)
    return rows


def save_benchmark(
    prices: pd.DataFrame,
    period: str,
    window: Period,
    universe: UniverseConfig,
    bt_config: BacktestConfig,
    cfg: Stage1Config,
    results_dir: Path,
) -> pd.DataFrame:
    """벤치마크 바이앤홀드를 구간 전체로 비용별 실행해 저장한다 (실험이 아니므로 실험 로그에는 넣지 않는다)."""
    rows, curves = [], []
    for cost in cfg.cost_bps:
        run_cfg = bt_config.model_copy(update={"cost_bps": cost, "start": window.start, "end": window.end})
        result = run_backtest(prices, BuyAndHold(ticker=universe.benchmark), run_cfg)
        run_id = f"benchmark:{period}:{cost:g}"
        m = performance_metrics(result)
        rows.append(
            {"run_id": run_id, "period": period, "cost_bp": cost, "ticker": universe.benchmark,
             **_metric_columns(m), "window_start": result.start_date.date().isoformat(),
             "window_end": result.end_date.date().isoformat()}
        )
        curves.append(_curve(run_id, result))
    new = pd.DataFrame(rows)
    path = benchmarks_path(results_dir)
    if path.exists():
        old = pd.read_parquet(path)
        new = pd.concat([old[~old["run_id"].isin(new["run_id"])], new], ignore_index=True)
    write_atomic(new, path)
    _append_curves(results_dir, curves)
    return new


def read_curves(results_dir: Path) -> pd.DataFrame:
    path = curves_path(results_dir)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["run_id", "date", "equity"])


def _append_curves(results_dir: Path, curves: list[pd.DataFrame]) -> None:
    if not curves:
        return
    new = pd.concat(curves, ignore_index=True)
    old = read_curves(results_dir)
    if not old.empty:
        new = pd.concat([old[~old["run_id"].isin(new["run_id"])], new], ignore_index=True)
    write_atomic(new, curves_path(results_dir))


def _curve(run_id: str, result: BacktestResult) -> pd.DataFrame:
    return pd.DataFrame({"run_id": run_id, "date": result.equity.index, "equity": result.equity.to_numpy()})


def _metric_columns(m: dict[str, float]) -> dict[str, float]:
    return {
        "cagr": m["cagr"],
        "vol": m["annual_vol"],
        "sharpe": m["sharpe"],
        "mdd": m["mdd"],
        "win_rate": m["win_rate"],
        "payoff": m["payoff_ratio"],
        "avg_hold": m["avg_holding_days"],
        "trades_per_year": m["trades_per_year"],
        "exposure": m["exposure"],
        "n_trades": int(m["n_trades"]),
    }


def _log_row(run_id: str, commit: str, spec: StrategySpec, period: str, cost: float, result: BacktestResult) -> dict:
    return {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "git_commit": commit,
        "family": spec.family,
        "strategy": spec.display_name,
        "params": spec.params_json,
        "period": period,
        "cost_bp": cost,
        **_metric_columns(performance_metrics(result)),
        "window_start": result.start_date.date().isoformat(),
        "window_end": result.end_date.date().isoformat(),
    }
