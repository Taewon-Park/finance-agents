"""P0~P3 백테스트 (서술용, 검증 아님).

정책 × 비용마다 목표 비중 엔진을 돌리고, 실험 로그에 family=alloc, period=descriptive로 기록한다.
시도 수 = 서로 다른 (정책 이름, 정책 정의) 조합. max_trials(4)를 넘으면 거부한다.
"""

from __future__ import annotations

import subprocess
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from swinglab.alloc.config import AllocConfig
from swinglab.alloc.policies import Policy, build_schedule
from swinglab.alloc.tax import after_tax_curve, ledger_from_trades, yearly_tax
from swinglab.backtest.metrics import (
    annual_volatility,
    cagr,
    daily_returns,
    max_drawdown,
    sharpe_ratio,
    years_covered,
)
from swinglab.backtest.weights_engine import WeightsConfig, WeightsResult, run_weights
from swinglab.pipeline.fx import fx_on
from swinglab.research.log import (
    LOG_COLUMNS,
    BudgetExceeded,
    append_rows,
    read_log,
    write_atomic,
)

FAMILY = "alloc"
PERIOD = "descriptive"
ALLOC_CONFIG_PATH = "config/alloc.yaml"


class AllocRefused(RuntimeError):
    """사전 고정 조건(설정 커밋, 시도 수)을 지키지 않은 실행."""


@dataclass
class PolicyRun:
    policy: Policy
    cost_bps: float
    result: WeightsResult
    metrics: dict[str, float]
    yearly: pd.Series  # 연도 → 수익률 (첫 해·마지막 해는 부분 연도)
    periods: pd.DataFrame  # 하락장 구간별 수익률·MDD
    quarterly_trades: pd.Series  # "YYYY-Qn" → 매매 건수
    taxes: pd.DataFrame  # 연도별 실현 차익·세액 (원)
    after_tax_cagr: float
    mdd_window: tuple[str, str] = ("", "")  # MDD 고점일, 저점일
    avg_cash_proxy: float = float("nan")  # 현금 대용(BIL) 평균 비중 (종가 기준)

    @property
    def run_key(self) -> tuple[str, str, str]:
        return (FAMILY, self.policy.name, self.policy.params_json())


def run_policy(policy: Policy, prices: pd.DataFrame, fx: pd.Series, cfg: AllocConfig, cost_bps: float) -> PolicyRun:
    bt = cfg.backtest
    px = prices[prices["ticker"].isin(policy.tickers)]
    calendar = pd.DatetimeIndex(np.sort(px["date"].unique()))
    start_idx = int(calendar.searchsorted(pd.Timestamp(bt.start), side="left"))
    if start_idx < 1:
        raise ValueError("시작일 전 거래일(초기 배분 신호일)이 데이터에 없습니다")
    anchor = calendar[start_idx - 1]
    schedule = build_schedule(policy, px, calendar, anchor)
    wcfg = WeightsConfig(
        initial_capital=bt.initial_capital, cost_bps=cost_bps, band=cfg.band, cash_proxy=cfg.cash_proxy,
        start=bt.start, end=bt.end,
    )
    result = run_weights(px, policy.tickers, schedule, wcfg, name=policy.name)
    equity = result.equity

    rets = daily_returns(equity)
    trades = result.trades
    years = years_covered(equity)
    metrics = {
        "cagr": cagr(equity),
        "vol": annual_volatility(rets),
        "sharpe": sharpe_ratio(rets),
        "mdd": max_drawdown(equity),
        "n_trades": float(len(trades)),
        "trades_per_year": float(len(trades) / years) if years > 0 else np.nan,
        "exposure": float((1.0 - result.cash / equity).iloc[1:].mean()),
        "total_cost": float(trades["cost"].sum()) if len(trades) else 0.0,
    }

    fx_days = fx_on(fx, equity.index)
    ledger = ledger_from_trades(trades, fx_days)
    taxes = yearly_tax(ledger.realized, cfg.tax.rate, cfg.tax.deduction_krw)
    after_tax = after_tax_curve(equity, taxes, fx_days)

    quarters = pd.PeriodIndex(pd.to_datetime(trades["date"]), freq="Q") if len(trades) else pd.PeriodIndex([], freq="Q")
    quarterly = pd.Series(1, index=quarters.astype(str)).groupby(level=0).sum() if len(trades) else pd.Series(dtype=int)
    return PolicyRun(
        policy=policy,
        cost_bps=cost_bps,
        result=result,
        metrics=metrics,
        yearly=yearly_returns(equity),
        periods=period_stats(equity, [(p.name, p.start, p.end) for p in bt.drawdown_periods]),
        quarterly_trades=quarterly,
        taxes=taxes,
        after_tax_cagr=cagr(after_tax),
        mdd_window=mdd_window(equity),
        avg_cash_proxy=float(result.weights[cfg.cash_proxy].iloc[1:].mean()) if cfg.cash_proxy in result.weights else 0.0,
    )


def mdd_window(equity: pd.Series) -> tuple[str, str]:
    """최대 낙폭의 고점일과 저점일."""
    dd = equity / equity.cummax() - 1.0
    trough = dd.idxmin()
    peak = equity.loc[:trough].idxmax()
    return peak.date().isoformat(), trough.date().isoformat()


def yearly_returns(equity: pd.Series) -> pd.Series:
    """달력 연도별 수익률. 기준은 전년 마지막 거래일 종가 (첫 해는 자산곡선 첫 행)."""
    year_end = equity.groupby(equity.index.year).last()
    base = year_end.shift(1)
    base.iloc[0] = float(equity.iloc[0])
    return year_end / base - 1.0


def period_stats(equity: pd.Series, periods: list[tuple[str, object, object]]) -> pd.DataFrame:
    """구간별 수익률(구간 시작 전날 종가 기준)과 구간 안 MDD."""
    rows = []
    for name, start, end in periods:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        before = equity[equity.index < start]
        inside = equity[(equity.index >= start) & (equity.index <= end)]
        if before.empty or inside.empty:
            rows.append({"period": name, "start": start, "end": end, "return": np.nan, "mdd": np.nan})
            continue
        window = pd.concat([before.iloc[-1:], inside])
        rows.append({"period": name, "start": start, "end": end,
                     "return": float(window.iloc[-1] / window.iloc[0] - 1.0), "mdd": max_drawdown(window)})
    return pd.DataFrame(rows, columns=["period", "start", "end", "return", "mdd"])


def check_alloc_config_committed(repo_dir: Path, rel_path: str = ALLOC_CONFIG_PATH) -> None:
    """정책 설정이 커밋된 상태에서만 실행 (결과를 본 뒤 설정을 바꾸고 다시 돌리는 일을 막는다)."""
    try:
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel_path], cwd=repo_dir, check=False,
                                 capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", rel_path], cwd=repo_dir,
                               capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise AllocRefused(f"git으로 {rel_path} 커밋 여부를 확인할 수 없습니다: {exc}") from exc
    if tracked.returncode != 0 or dirty.stdout.strip():
        raise AllocRefused(f"{rel_path}를 먼저 커밋하세요 (정책은 백테스트 전에 고정)")


def check_alloc_budget(log: pd.DataFrame, policies: list[Policy], max_trials: int) -> None:
    rows = log[log["family"] == FAMILY]
    keys = set(zip(rows["family"], rows["strategy"], rows["params"]))
    keys |= {(FAMILY, p.name, p.params_json()) for p in policies}
    if len(keys) > max_trials:
        raise BudgetExceeded(
            f"alloc 시도 {len(keys)}개가 한도 {max_trials}개를 넘습니다. 정책을 바꾸려면 스펙을 먼저 고치세요."
        )


def run_all(
    prices: pd.DataFrame, fx: pd.Series, cfg: AllocConfig, results_dir: Path, repo_dir: Path, git_commit: str,
    require_committed: bool = True,
) -> list[PolicyRun]:
    if require_committed:
        check_alloc_config_committed(repo_dir)
    policies = [Policy.from_config(name, cfg) for name in cfg.policies]
    check_alloc_budget(read_log(results_dir), policies, cfg.max_trials)
    runs = [run_policy(p, prices, fx, cfg, c) for p in policies for c in cfg.backtest.cost_bps]
    _log(runs, results_dir, git_commit)
    return runs


def curves_path(results_dir: Path) -> Path:
    return results_dir / "alloc_curves.parquet"


def _log(runs: list[PolicyRun], results_dir: Path, git_commit: str) -> None:
    now = datetime.now(UTC).isoformat()
    rows, curves = [], []
    for r in runs:
        run_id = str(uuid.uuid4())
        m = r.metrics
        rows.append({
            "run_id": run_id, "timestamp": now, "git_commit": git_commit, "family": FAMILY,
            "strategy": r.policy.name, "params": r.policy.params_json(), "period": PERIOD, "cost_bp": r.cost_bps,
            "cagr": m["cagr"], "vol": m["vol"], "sharpe": m["sharpe"], "mdd": m["mdd"],
            "win_rate": np.nan, "payoff": np.nan, "avg_hold": np.nan,
            "trades_per_year": m["trades_per_year"], "exposure": m["exposure"], "n_trades": m["n_trades"],
            "window_start": r.result.start_date.date().isoformat(), "window_end": r.result.end_date.date().isoformat(),
        })
        curves.append(pd.DataFrame({"run_id": run_id, "policy": r.policy.name, "cost_bp": r.cost_bps,
                                    "date": r.result.equity.index, "equity": r.result.equity.to_numpy()}))
    append_rows(results_dir, [{c: row[c] for c in LOG_COLUMNS} for row in rows])
    path = curves_path(results_dir)
    new = pd.concat(curves, ignore_index=True)
    if path.exists():
        new = pd.concat([pd.read_parquet(path), new], ignore_index=True)
    write_atomic(new, path)
