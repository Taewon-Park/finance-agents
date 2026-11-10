"""검증 구간 실행과 잠금.

- 후보 파일(config/candidates.yaml)이 git에 커밋된 상태여야 실행한다.
- results/holdout.lock은 JSON Lines 기록이다. 실행할 때마다 후보 해시, 실행 시각, git 커밋,
  forced 여부를 한 줄 덧붙인다. 백테스트보다 먼저 기록하므로 도중에 실패해도 재실행에는 --force가 필요하다.
- 같은 후보 해시가 이미 있으면 거부한다. force=True면 실행하고 forced=true로 남긴다.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from swinglab.config import BacktestConfig, UniverseConfig
from swinglab.research.config import Period, Stage1Config
from swinglab.research.log import GIT_TIMEOUT_SECONDS, git_commit
from swinglab.research.runner import run_specs, save_benchmark
from swinglab.research.selection import read_candidates
from swinglab.research.specs import specs_hash


class HoldoutRefused(RuntimeError):
    """검증 구간 실행 거부 (재실행, 커밋 안 된 후보 파일 등)."""


def lock_path(results_dir: Path) -> Path:
    return results_dir / "holdout.lock"


def read_lock(results_dir: Path) -> list[dict]:
    path = lock_path(results_dir)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_committed(path: Path, repo_dir: Path) -> bool:
    """파일이 git에 추적되고 커밋 안 된 변경이 없는지."""
    try:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path.resolve())],
            cwd=repo_dir, capture_output=True, timeout=GIT_TIMEOUT_SECONDS, check=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", str(path.resolve())],
            cwd=repo_dir, capture_output=True, text=True, timeout=GIT_TIMEOUT_SECONDS, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return status.stdout.strip() == ""


def run_holdout(
    candidates_file: Path,
    prices: pd.DataFrame,
    universe: UniverseConfig,
    bt_config: BacktestConfig,
    cfg: Stage1Config,
    results_dir: Path,
    repo_dir: Path,
    force: bool = False,
    require_committed: bool = True,
) -> pd.DataFrame:
    specs = read_candidates(candidates_file)
    if not specs:
        raise HoldoutRefused(f"{candidates_file}에 후보가 없습니다")
    if require_committed and not is_committed(candidates_file, repo_dir):
        raise HoldoutRefused(f"{candidates_file}를 먼저 커밋하세요 (스펙: 후보 커밋 후 검증 구간 실행)")

    digest = specs_hash(specs)
    previous = [e for e in read_lock(results_dir) if e.get("candidates_hash") == digest]
    if previous and not force:
        raise HoldoutRefused(
            f"같은 후보로 이미 검증 구간을 실행했습니다 ({previous[-1]['timestamp']}). 다시 하려면 --force"
        )

    commit = git_commit(repo_dir)
    entry = {
        "candidates_hash": digest,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "forced": bool(force),
        "candidates_file": str(candidates_file),
        "n_candidates": len(specs),
    }
    path = lock_path(results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    window = Period(start=cfg.holdout.start, end=cfg.holdout.end)
    rows = run_specs(specs, prices, window, "holdout", universe, bt_config, cfg, results_dir, commit)
    save_benchmark(prices, "holdout", window, universe, bt_config, cfg, results_dir)
    return pd.DataFrame(rows)
