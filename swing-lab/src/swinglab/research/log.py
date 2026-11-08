"""실험 로그 (results/experiments.parquet). 실행마다 행을 덧붙이기만 하고 지우지 않는다.

한 행 = 전략 명세 하나 × 비용 하나 × 구간 하나의 백테스트 1회.
시도 횟수는 계열별로 서로 다른 (전략, 파라미터) 조합 수로 센다. baseline(필터 비교용 원본)은 세지 않는다.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from swinglab.research.config import Stage1Config
from swinglab.research.specs import StrategySpec

LOG_COLUMNS: list[str] = [
    "run_id",
    "timestamp",
    "git_commit",
    "family",
    "strategy",
    "params",
    "period",
    "cost_bp",
    "cagr",
    "vol",
    "sharpe",
    "mdd",
    "win_rate",
    "payoff",
    "avg_hold",
    "trades_per_year",
    "exposure",
    "n_trades",
    "window_start",
    "window_end",
]
BASELINE_FAMILY = "baseline"
GIT_TIMEOUT_SECONDS = 10


class BudgetExceeded(RuntimeError):
    """시도 예산(계열당·전체)을 넘는 실행."""


def log_path(results_dir: Path) -> Path:
    return results_dir / "experiments.parquet"


def git_commit(repo_dir: Path) -> str:
    """현재 커밋 해시. 추적 중인 파일에 커밋 안 된 변경이 있으면 끝에 -dirty. git을 못 쓰면 unknown."""
    try:
        head = _git(["rev-parse", "HEAD"], repo_dir)
        dirty = _git(["status", "--porcelain", "--untracked-files=no"], repo_dir)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return head + ("-dirty" if dirty else "")


def _git(args: list[str], cwd: Path) -> str:
    out = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=GIT_TIMEOUT_SECONDS, check=True
    )
    return out.stdout.strip()


def read_log(results_dir: Path) -> pd.DataFrame:
    path = log_path(results_dir)
    if not path.exists():
        return pd.DataFrame({c: pd.Series(dtype=object) for c in LOG_COLUMNS})
    return pd.read_parquet(path)


def append_rows(results_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    new = pd.DataFrame(rows, columns=LOG_COLUMNS)
    old = read_log(results_dir)
    combined = new if old.empty else pd.concat([old, new], ignore_index=True)
    write_atomic(combined, log_path(results_dir))


def write_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def trial_keys(log: pd.DataFrame) -> set[tuple[str, str, str]]:
    """학습 구간에서 시도한 (계열, 전략, 파라미터) 조합. baseline 제외."""
    train = log[(log["period"] == "train") & (log["family"] != BASELINE_FAMILY)]
    return set(zip(train["family"], train["strategy"], train["params"]))


def check_budget(log: pd.DataFrame, specs: list[StrategySpec], cfg: Stage1Config) -> None:
    """이번 실행으로 예산을 넘으면 BudgetExceeded. 이미 시도한 조합을 다시 돌리는 것은 새 시도로 세지 않는다."""
    keys = trial_keys(log)
    keys |= {(s.family, s.display_name, s.params_json) for s in specs if s.family != BASELINE_FAMILY}
    per_family: dict[str, int] = {}
    for fam, _, _ in keys:
        per_family[fam] = per_family.get(fam, 0) + 1
    over = {f: n for f, n in per_family.items() if n > cfg.budget.per_family}
    if over:
        raise BudgetExceeded(f"계열 예산 {cfg.budget.per_family}개 초과: {over}. 스펙을 먼저 수정하세요.")
    if len(keys) > cfg.budget.total:
        raise BudgetExceeded(f"전체 예산 {cfg.budget.total}개 초과: {len(keys)}개. 스펙을 먼저 수정하세요.")


def latest_runs(log: pd.DataFrame, period: str, cost_bp: float | None = None) -> pd.DataFrame:
    """같은 (계열, 전략, 파라미터, 비용) 조합을 여러 번 돌렸으면 가장 최근 실행만 남긴다."""
    rows = log[log["period"] == period]
    if cost_bp is not None:
        rows = rows[rows["cost_bp"] == cost_bp]
    rows = rows.sort_values("timestamp", kind="stable")
    return rows.drop_duplicates(subset=["family", "strategy", "params", "cost_bp"], keep="last").reset_index(drop=True)


def summarize(log: pd.DataFrame, cfg: Stage1Config) -> pd.DataFrame:
    """계열별 시도 횟수, 실행 횟수, 선정 비용 기준 샤프 분포 (학습 구간만)."""
    train = log[log["period"] == "train"]
    latest = latest_runs(log, "train", cfg.selection_cost_bps)
    keys = trial_keys(log)
    out = []
    for fam in sorted(set(train["family"])):
        sharpe = latest.loc[latest["family"] == fam, "sharpe"].astype(float)
        out.append(
            {
                "family": fam,
                "trials": sum(1 for k in keys if k[0] == fam) if fam != BASELINE_FAMILY else 0,
                "runs": int((train["family"] == fam).sum()),
                f"sharpe_min@{cfg.selection_cost_bps:g}bp": float(sharpe.min()) if len(sharpe) else np.nan,
                f"sharpe_median@{cfg.selection_cost_bps:g}bp": float(sharpe.median()) if len(sharpe) else np.nan,
                f"sharpe_max@{cfg.selection_cost_bps:g}bp": float(sharpe.max()) if len(sharpe) else np.nan,
            }
        )
    table = pd.DataFrame(out)
    if not table.empty:
        table.attrs["total_trials"] = len(keys)
        table.attrs["budget"] = (cfg.budget.per_family, cfg.budget.total)
    return table
