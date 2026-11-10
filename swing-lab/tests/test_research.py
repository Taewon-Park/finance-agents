"""1단계 연구 절차: 구간 분리, 실험 로그, 예산, 후보 선정 규칙, 검증 잠금 (합성 데이터, 네트워크 없음)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import random_walk_prices

from swinglab.config import BacktestConfig, UniverseConfig
from swinglab.research import runner as runner_module
from swinglab.research.config import Stage1Config, load_stage1
from swinglab.research.holdout import HoldoutRefused, read_lock, run_holdout
from swinglab.research.log import (
    LOG_COLUMNS,
    BudgetExceeded,
    check_budget,
    git_commit,
    read_log,
    summarize,
)
from swinglab.research.runner import run_specs, run_train
from swinglab.research.selection import (
    IncompleteExperiments,
    candidates_document,
    select_filter_family,
    select_grid_family,
    write_candidates,
)
from swinglab.research.specs import StrategySpec, specs_hash

PROJECT = Path(__file__).resolve().parents[1]
UNIVERSE = UniverseConfig(benchmark="SPY", start="2016-01-01", groups={"b": ["SPY"], "x": ["AAA", "BBB", "CCC"]})
BT = BacktestConfig(initial_capital=100_000, cost_bps=5, max_positions=5)


def small_cfg(**overrides) -> Stage1Config:
    data = {
        "periods": {"train": {"start": "2016-01-01", "end": "2018-12-31"}, "holdout": {"start": "2019-01-01", "end": None}},
        "cost_bps": [0, 5],
        "selection_cost_bps": 5,
        "budget": {"per_family": 12, "total": 50},
        "selection": {"min_trades": 1, "min_trades_exempt": ["B"], "isolation_ratio": 0.7, "filter_min_sharpe_gain": 0.05},
        "families": {
            "C": {"strategy": "breakout", "axes": {"channel": [{"entry_n": 20, "exit_n": 10}, {"entry_n": 55, "exit_n": 20}]}}
        },
    }
    data.update(overrides)
    return Stage1Config.model_validate(data)


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return random_walk_prices(["SPY", "AAA", "BBB", "CCC"], 1300, seed=8, drift=0.0005, vol=0.015, start="2016-01-01")


def perturb_after(prices: pd.DataFrame, cutoff: str, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = prices.copy()
    future = out["date"] > pd.Timestamp(cutoff)
    cols = ["open", "high", "low", "close"]
    out.loc[future, cols] = out.loc[future, cols].to_numpy() * rng.uniform(0.5, 1.5, size=(int(future.sum()), 1))
    return out


# ---------------------------------------------------------------- 설정


def test_project_stage1_config_matches_spec() -> None:
    cfg = load_stage1(PROJECT / "config")
    assert str(cfg.train.end) == "2018-12-31" and str(cfg.holdout.start) == "2019-01-01"
    assert [f.planned_trials() for f in cfg.families.values()] == [6, 4, 2, 3]
    assert cfg.cost_bps == [0, 5, 10] and cfg.selection_cost_bps == 5
    grid = cfg.families["A"].grid()
    idx = [i for i, _ in grid]
    # (entry_th=10, min_hold=3)의 인접 = (5,3), (10,1), (10,5)
    assert sorted(cfg.families["A"].neighbors((1, 1), idx)) == [(0, 1), (1, 0), (1, 2)]
    assert cfg.families["C"].grid()[1][1] == {"entry_n": 55, "exit_n": 20}
    assert cfg.families["B"].grid()[0][1] == {"safe": "IEF", "lookback": 6, "top_n": 2}


def test_config_rejects_overlap_and_over_budget() -> None:
    with pytest.raises(ValueError):
        small_cfg(periods={"train": {"start": "2016-01-01", "end": "2019-06-30"}, "holdout": {"start": "2019-01-01"}})
    with pytest.raises(ValueError):
        small_cfg(budget={"per_family": 1, "total": 50})


# ---------------------------------------------------------------- 구간 분리


def test_train_results_unchanged_when_holdout_prices_change(tmp_path: Path, prices: pd.DataFrame) -> None:
    cfg = small_cfg()
    a = run_train("C", prices, UNIVERSE, BT, cfg, tmp_path / "a", tmp_path)
    b = run_train("C", perturb_after(prices, "2018-12-31"), UNIVERSE, BT, cfg, tmp_path / "b", tmp_path)
    metrics = ["cagr", "vol", "sharpe", "mdd", "n_trades", "window_start", "window_end"]
    pd.testing.assert_frame_equal(a[metrics], b[metrics])
    assert (pd.to_datetime(a["window_end"]) <= pd.Timestamp("2018-12-31")).all()
    assert set(a["period"]) == {"train"}


def test_train_path_never_receives_holdout_prices(tmp_path: Path, prices: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[pd.Timestamp] = []
    real = runner_module.run_backtest

    def spy(p, strategy, config):
        seen.append(p["date"].max())
        return real(p, strategy, config)

    monkeypatch.setattr(runner_module, "run_backtest", spy)
    run_train("C", prices, UNIVERSE, BT, small_cfg(), tmp_path, tmp_path)
    assert seen and max(seen) <= pd.Timestamp("2018-12-31")


# ---------------------------------------------------------------- 실험 로그


def test_each_run_appends_exactly_one_row(tmp_path: Path, prices: pd.DataFrame) -> None:
    cfg = small_cfg(cost_bps=[5])
    spec = StrategySpec("C", "breakout", {"entry_n": 20, "exit_n": 10})
    window = cfg.train
    sliced = runner_module.train_prices(prices, cfg)
    run_specs([spec], sliced, window, "train", UNIVERSE, BT, cfg, tmp_path, "abc")
    log = read_log(tmp_path)
    assert len(log) == 1 and list(log.columns) == LOG_COLUMNS
    run_specs([spec], sliced, window, "train", UNIVERSE, BT, cfg, tmp_path, "abc")
    log = read_log(tmp_path)
    assert len(log) == 2 and log["run_id"].is_unique
    row = log.iloc[0]
    assert row["strategy"] == "breakout" and json.loads(row["params"]) == {"entry_n": 20, "exit_n": 10}
    assert row["cost_bp"] == 5 and row["git_commit"] == "abc"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args], cwd=repo, check=True,
                   capture_output=True, timeout=30)


def test_git_commit_marks_dirty_tree(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    (tmp_path / "f.txt").write_text("a")
    _git(tmp_path, "add", "f.txt")
    _git(tmp_path, "commit", "-q", "-m", "init")
    clean = git_commit(tmp_path)
    assert len(clean) == 40 and not clean.endswith("-dirty")
    (tmp_path / "new.txt").write_text("untracked")  # 추적 안 하는 파일은 dirty로 보지 않는다
    assert git_commit(tmp_path) == clean
    (tmp_path / "f.txt").write_text("b")
    assert git_commit(tmp_path) == clean + "-dirty"
    assert git_commit(tmp_path / "missing") == "unknown"


def test_summary_counts_trials_and_sharpe_distribution(tmp_path: Path, prices: pd.DataFrame) -> None:
    cfg = small_cfg()
    run_train("C", prices, UNIVERSE, BT, cfg, tmp_path, tmp_path)
    run_train("C", prices, UNIVERSE, BT, cfg, tmp_path, tmp_path)  # 같은 조합 재실행은 새 시도가 아님
    table = summarize(read_log(tmp_path), cfg).set_index("family")
    assert table.at["C", "trials"] == 2
    assert table.at["C", "runs"] == 8  # 2조합 × 2비용 × 2회
    s = read_log(tmp_path).query("cost_bp == 5")["sharpe"]
    assert table.at["C", "sharpe_max@5bp"] == pytest.approx(s.max())


def test_budget_blocks_new_combinations(tmp_path: Path, prices: pd.DataFrame) -> None:
    cfg = small_cfg(budget={"per_family": 2, "total": 50})
    run_train("C", prices, UNIVERSE, BT, cfg, tmp_path, tmp_path)
    log = read_log(tmp_path)
    check_budget(log, [StrategySpec("C", "breakout", {"entry_n": 20, "exit_n": 10})], cfg)  # 기존 조합 재실행은 허용
    with pytest.raises(BudgetExceeded):
        check_budget(log, [StrategySpec("C", "breakout", {"entry_n": 30, "exit_n": 15})], cfg)
    with pytest.raises(BudgetExceeded):
        check_budget(log, [StrategySpec("C", "breakout", {"entry_n": 30, "exit_n": 15})], small_cfg(budget={"per_family": 12, "total": 2}))


# ---------------------------------------------------------------- 후보 선정 규칙


def _row(family: str, strategy: str, params: dict, sharpe: float, n_trades: int = 500, cost: float = 5) -> dict:
    return {c: np.nan for c in LOG_COLUMNS} | {
        "run_id": f"{family}{strategy}{params}{cost}", "timestamp": "2026-11-08T00:00:00", "git_commit": "x",
        "family": family, "strategy": strategy, "params": json.dumps(params, sort_keys=True, separators=(",", ":")),
        "period": "train", "cost_bp": cost, "sharpe": sharpe, "n_trades": n_trades,
    }


A_CFG = {
    "A": {"strategy": "rsi2_v2", "axes": {"entry_th": [5, 10], "min_hold": [1, 3, 5]}},
}


def _a_log(sharpes: dict[tuple[int, int], float], trades: dict[tuple[int, int], int] | None = None) -> pd.DataFrame:
    rows = [
        _row("A", "rsi2_v2", {"entry_th": e, "min_hold": m}, s, (trades or {}).get((e, m), 500))
        for (e, m), s in sharpes.items()
    ]
    return pd.DataFrame(rows)


GRID_SHARPES = {(5, 1): 0.5, (5, 3): 0.2, (5, 5): 0.55, (10, 1): 0.3, (10, 3): 1.0, (10, 5): 0.4}


def test_isolated_peak_switches_to_best_neighborhood() -> None:
    cfg = small_cfg(families=A_CFG, selection={"min_trades": 100, "min_trades_exempt": [], "isolation_ratio": 0.7,
                                               "filter_min_sharpe_gain": 0.05})
    sel = select_grid_family(_a_log(GRID_SHARPES), "A", cfg)
    # 1순위 (10,3): 인접 (5,3)=0.2, (10,1)=0.3, (10,5)=0.4 평균 0.3 < 0.7 → 고립
    # 자기+인접 평균 최고: (10,5) = (0.4 + 0.55 + 1.0) / 3 = 0.65
    assert sel.candidate.params == {"entry_th": 10, "min_hold": 5}
    assert sel.table.loc[sel.table["isolated"], "params"].tolist() == ['{"entry_th":10,"min_hold":3}']
    assert any("고립" in n for n in sel.notes)


def test_non_isolated_best_is_kept() -> None:
    cfg = small_cfg(families=A_CFG)
    sharpes = GRID_SHARPES | {(5, 3): 0.9, (10, 1): 0.8, (10, 5): 0.85}
    sel = select_grid_family(_a_log(sharpes), "A", cfg)
    assert sel.candidate.params == {"entry_th": 10, "min_hold": 3}
    assert not sel.table["isolated"].any()


def test_min_trades_excludes_but_neighbors_still_count() -> None:
    cfg = small_cfg(families=A_CFG, selection={"min_trades": 100, "min_trades_exempt": [], "isolation_ratio": 0.7,
                                               "filter_min_sharpe_gain": 0.05})
    sharpes = GRID_SHARPES | {(5, 3): 0.9, (10, 1): 0.8, (10, 5): 0.85}
    sel = select_grid_family(_a_log(sharpes, trades={(10, 3): 50}), "A", cfg)
    assert sel.candidate.params == {"entry_th": 5, "min_hold": 3}  # 1.0은 거래 50건이라 제외 → 0.9
    assert not sel.table.set_index("params").at['{"entry_th":10,"min_hold":3}', "eligible"]


def test_monthly_family_is_exempt_from_min_trades() -> None:
    cfg = small_cfg(families={"B": {"strategy": "dual_momentum", "axes": {"lookback": [6, 12]}}},
                    selection={"min_trades": 100, "min_trades_exempt": ["B"], "isolation_ratio": 0.7,
                               "filter_min_sharpe_gain": 0.05})
    log = pd.DataFrame([_row("B", "dual_momentum", {"lookback": 6}, 0.4, 30), _row("B", "dual_momentum", {"lookback": 12}, 0.6, 20)])
    assert select_grid_family(log, "B", cfg).candidate.params == {"lookback": 12}


def test_isolation_rule_skipped_when_best_sharpe_not_positive() -> None:
    cfg = small_cfg(families=A_CFG)
    sharpes = {k: v - 2.0 for k, v in GRID_SHARPES.items()}
    sel = select_grid_family(_a_log(sharpes), "A", cfg)
    assert sel.candidate.params == {"entry_th": 10, "min_hold": 3}
    assert any("0 이하" in n for n in sel.notes)


def test_selection_requires_all_combinations() -> None:
    cfg = small_cfg(families=A_CFG)
    partial = {k: v for k, v in GRID_SHARPES.items() if k != (5, 5)}
    with pytest.raises(IncompleteExperiments):
        select_grid_family(_a_log(partial), "A", cfg)


def _d_cfg() -> Stage1Config:
    return small_cfg(
        families={
            "A": {"strategy": "rsi2_v2", "axes": {"entry_th": [10]}},
            "C": {"strategy": "breakout", "axes": {"channel": [{"entry_n": 20, "exit_n": 10}]}},
            "D": {"wraps": ["A", "C"], "extra": ["sma_cross"]},
        },
        selection={"min_trades": 100, "min_trades_exempt": [], "isolation_ratio": 0.7, "filter_min_sharpe_gain": 0.05},
    )


def _d_log(gains: dict[str, tuple[float, float]], filtered_trades: int = 500) -> pd.DataFrame:
    a, c = {"entry_th": 10}, {"entry_n": 20, "exit_n": 10}
    return pd.DataFrame([
        _row("A", "rsi2_v2", a, gains["A"][0]), _row("D", "rsi2_v2+market_filter", a, gains["A"][1], filtered_trades),
        _row("C", "breakout", c, gains["C"][0]), _row("D", "breakout+market_filter", c, gains["C"][1], filtered_trades),
        _row("baseline", "sma_cross", {}, gains["S"][0]), _row("D", "sma_cross+market_filter", {}, gains["S"][1], filtered_trades),
    ])


def _wrapped(cfg: Stage1Config, log: pd.DataFrame) -> dict:
    return {w: select_grid_family(log, w, cfg).candidate for w in ("A", "C")}


def test_filter_family_picks_best_passing_version() -> None:
    cfg = _d_cfg()
    log = _d_log({"A": (0.50, 0.53), "C": (0.50, 0.60), "S": (0.30, 0.50)})
    sel = select_filter_family(log, "D", cfg, _wrapped(cfg, log))
    # A는 +0.03이라 탈락, C(+0.10, 0.60)와 sma_cross(+0.20, 0.50) 통과 → 필터 버전 샤프가 높은 C
    assert sel.candidate == StrategySpec("D", "breakout", {"entry_n": 20, "exit_n": 10}, True)
    assert sel.table["passes"].tolist() == [False, True, True]


def test_filter_family_has_no_candidate_when_nothing_passes() -> None:
    cfg = _d_cfg()
    log = _d_log({"A": (0.5, 0.52), "C": (0.5, 0.4), "S": (0.3, 0.34)})
    assert select_filter_family(log, "D", cfg, _wrapped(cfg, log)).candidate is None
    log = _d_log({"A": (0.5, 0.9), "C": (0.5, 0.9), "S": (0.3, 0.9)}, filtered_trades=10)  # 거래 수 미달
    assert select_filter_family(log, "D", cfg, _wrapped(cfg, log)).candidate is None


# ---------------------------------------------------------------- 검증 구간 잠금


def _candidates_file(path: Path, specs: list[StrategySpec]) -> Path:
    from swinglab.research.selection import FamilySelection

    sels = {s.family: FamilySelection(s.family, pd.DataFrame(), s) for s in specs}
    write_candidates(path, candidates_document(sels, small_cfg(), "test"))
    return path


def test_holdout_refuses_rerun_and_logs_forced(tmp_path: Path, prices: pd.DataFrame) -> None:
    cfg = small_cfg()
    spec = StrategySpec("C", "breakout", {"entry_n": 20, "exit_n": 10})
    cand = _candidates_file(tmp_path / "candidates.yaml", [spec])
    results = tmp_path / "results"

    rows = run_holdout(cand, prices, UNIVERSE, BT, cfg, results, tmp_path, require_committed=False)
    assert set(rows["period"]) == {"holdout"} and len(rows) == len(cfg.cost_bps)
    assert (pd.to_datetime(rows["window_start"]) >= pd.Timestamp("2019-01-01")).all()
    lock = read_lock(results)
    assert len(lock) == 1 and lock[0]["forced"] is False
    assert lock[0]["candidates_hash"] == specs_hash([spec]) and lock[0]["git_commit"]

    with pytest.raises(HoldoutRefused):
        run_holdout(cand, prices, UNIVERSE, BT, cfg, results, tmp_path, require_committed=False)
    assert len(read_lock(results)) == 1  # 거부된 실행은 기록하지 않음
    assert (read_log(results)["period"] == "holdout").sum() == len(cfg.cost_bps)

    run_holdout(cand, prices, UNIVERSE, BT, cfg, results, tmp_path, force=True, require_committed=False)
    lock = read_lock(results)
    assert len(lock) == 2 and lock[1]["forced"] is True


def test_holdout_requires_committed_candidates(tmp_path: Path, prices: pd.DataFrame) -> None:
    _git(tmp_path, "init", "-q")
    cand = _candidates_file(tmp_path / "candidates.yaml", [StrategySpec("C", "breakout", {"entry_n": 20, "exit_n": 10})])
    with pytest.raises(HoldoutRefused, match="커밋"):
        run_holdout(cand, prices, UNIVERSE, BT, small_cfg(), tmp_path / "results", tmp_path)
    assert read_lock(tmp_path / "results") == []
    _git(tmp_path, "add", "candidates.yaml")
    _git(tmp_path, "commit", "-q", "-m", "candidates")
    rows = run_holdout(cand, prices, UNIVERSE, BT, small_cfg(), tmp_path / "results", tmp_path)
    assert len(rows) == 2


# ---------------------------------------------------------------- 공통 평가 시작일


def test_evaluate_from_aligns_all_windows(tmp_path: Path, prices: pd.DataFrame) -> None:
    cfg = small_cfg(periods={"train": {"start": "2016-01-01", "end": "2018-12-31", "evaluate_from": "2016-09-01"},
                             "holdout": {"start": "2019-01-01"}})
    rows = run_train("C", prices, UNIVERSE, BT, cfg, tmp_path, tmp_path)
    assert set(rows["window_start"]) == {"2016-09-01"}  # 2016-09-01(목) 첫 체결, 두 조합 같은 날
    bench = pd.read_parquet(tmp_path / "benchmarks.parquet")
    assert set(bench["window_start"]) == {"2016-09-01"}


def test_evaluate_from_before_warmup_is_refused(tmp_path: Path, prices: pd.DataFrame) -> None:
    cfg = small_cfg(periods={"train": {"start": "2016-01-01", "end": "2018-12-31", "evaluate_from": "2016-02-01"},
                             "holdout": {"start": "2019-01-01"}})
    with pytest.raises(ValueError, match="워밍업"):
        run_train("C", prices, UNIVERSE, BT, cfg, tmp_path, tmp_path)  # 55/20 조합 워밍업 64일 > 시작 인덱스
    assert read_log(tmp_path).empty


def test_candidates_document_records_procedure_notes(tmp_path: Path) -> None:
    from swinglab.research.selection import FamilySelection, read_candidates_document

    spec = StrategySpec("C", "breakout", {"entry_n": 20, "exit_n": 10})
    doc = candidates_document({"C": FamilySelection("C", pd.DataFrame(), spec)}, small_cfg(), "abc", ["재실행 사유"])
    write_candidates(tmp_path / "c.yaml", doc)
    back = read_candidates_document(tmp_path / "c.yaml")
    assert back["procedure_notes"] == ["재실행 사유"] and back["candidates_hash"] == specs_hash([spec])
