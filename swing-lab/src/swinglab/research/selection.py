"""후보 선정 규칙 (stage1-spec.md, 실험 전에 고정). 학습 구간 결과만 쓴다.

격자 계열 (A, B, C)
1. 선정 비용(5bp) 기준 샤프 최고 조합이 1순위. 동점이면 격자 순서가 앞선 조합.
2. 학습 구간 총 거래 수가 min_trades 미만이면 선택할 수 없다 (min_trades_exempt 계열은 예외).
3. 1순위 샤프가 0보다 크고, 1순위의 인접 조합(축 하나만 한 칸 다른 조합) 샤프 평균이
   1순위 × isolation_ratio 미만이면 '고립된 봉우리'다. 이때는 선택 가능한 조합 중
   '자기 + 인접 조합' 샤프 평균이 가장 높은 조합을 고른다 (동점이면 자기 샤프, 그다음 격자 순서).
   인접 평균에는 규칙 2로 제외된 조합도 넣는다.

필터 계열 (D)
- 감싼 계열의 후보와 extra 전략 각각에 대해, 필터 버전 샤프 − 필터 전 샤프 ≥ filter_min_sharpe_gain 이고
  필터 버전이 규칙 2를 통과하면 '통과'. 통과한 것 중 필터 버전 샤프가 가장 높은 1개가 후보. 없으면 후보 없음.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from swinglab.research.config import FamilyConfig, Stage1Config
from swinglab.research.log import BASELINE_FAMILY, latest_runs
from swinglab.research.specs import StrategySpec, specs_hash


class IncompleteExperiments(RuntimeError):
    """선정에 필요한 학습 구간 실행이 아직 없다."""


@dataclass
class FamilySelection:
    family: str
    table: pd.DataFrame  # 조합별 판정 근거
    candidate: StrategySpec | None
    notes: list[str] = field(default_factory=list)


def select_grid_family(log: pd.DataFrame, family: str, cfg: Stage1Config) -> FamilySelection:
    fam = cfg.families[family]
    if fam.is_filter_family:
        raise ValueError(f"{family}는 필터 계열입니다")
    tickers = tuple(fam.universe) if fam.universe else None
    specs = {idx: StrategySpec(family, fam.strategy or "", params, False, tickers) for idx, params in fam.grid()}
    rows = latest_runs(log, "train", cfg.selection_cost_bps)
    rows = rows[rows["family"] == family].set_index(["strategy", "params"])

    records = []
    for idx, spec in specs.items():
        key = (spec.display_name, spec.params_json)
        if key not in rows.index:
            raise IncompleteExperiments(f"{family} 계열 {spec.params_json} 학습 실행이 없습니다")
        r = rows.loc[key]
        records.append({"idx": idx, "params": spec.params_json, "sharpe": float(r["sharpe"]), "n_trades": int(r["n_trades"])})
    table = pd.DataFrame(records)
    exempt = family in cfg.selection.min_trades_exempt
    table["eligible"] = exempt | (table["n_trades"] >= cfg.selection.min_trades)
    all_idx = list(table["idx"])
    sharpe = dict(zip(table["idx"], table["sharpe"]))
    table["neighbor_mean"] = [
        np.mean([sharpe[n] for n in FamilyConfig.neighbors(i, all_idx)]) if FamilyConfig.neighbors(i, all_idx) else np.nan
        for i in all_idx
    ]
    table["smoothed"] = [
        np.mean([sharpe[i], *[sharpe[n] for n in FamilyConfig.neighbors(i, all_idx)]]) for i in all_idx
    ]
    table["order"] = range(len(table))
    table["isolated"] = False
    table["chosen"] = False

    notes: list[str] = []
    eligible = table[table["eligible"]]
    if eligible.empty:
        notes.append(f"모든 조합이 거래 수 {cfg.selection.min_trades}건 미만이라 후보 없음")
        return FamilySelection(family, table, None, notes)

    best = eligible.sort_values(["sharpe", "order"], ascending=[False, True]).iloc[0]
    chosen_idx = best["idx"]
    if best["sharpe"] > 0 and not np.isnan(best["neighbor_mean"]):
        if best["neighbor_mean"] < cfg.selection.isolation_ratio * best["sharpe"]:
            table.loc[table["idx"] == best["idx"], "isolated"] = True
            alt = eligible.sort_values(["smoothed", "sharpe", "order"], ascending=[False, False, True]).iloc[0]
            chosen_idx = alt["idx"]
            outcome = (
                f"자기+인접 평균 최고 {alt['params']} 선택"
                if alt["idx"] != best["idx"]
                else "자기+인접 평균 최고도 같은 조합이라 유지"
            )
            notes.append(
                f"1순위 {best['params']}가 고립된 봉우리 (인접 평균 {best['neighbor_mean']:.3f} < "
                f"{cfg.selection.isolation_ratio} × {best['sharpe']:.3f}) → {outcome}"
            )
    elif best["sharpe"] <= 0:
        notes.append("1순위 샤프가 0 이하라 고립 봉우리 규칙을 적용하지 않음")
    table.loc[table["idx"] == chosen_idx, "chosen"] = True
    excluded = table.loc[~table["eligible"], "params"].tolist()
    if excluded:
        notes.append(f"거래 수 {cfg.selection.min_trades}건 미만 제외: {', '.join(excluded)}")
    return FamilySelection(family, table, specs[chosen_idx], notes)


def filter_pairs(family: str, cfg: Stage1Config, wrapped: dict[str, StrategySpec | None]) -> list[tuple[StrategySpec, StrategySpec]]:
    """필터 계열의 (필터 전, 필터 버전) 명세 쌍."""
    fam = cfg.families[family]
    pairs = []
    for w in fam.wraps:
        base = wrapped.get(w)
        if base is not None:
            pairs.append((base, base.with_filter(family)))
    for name in fam.extra:
        base = StrategySpec(BASELINE_FAMILY, name, {}, False, None)
        pairs.append((base, base.with_filter(family)))
    return pairs


def select_filter_family(
    log: pd.DataFrame, family: str, cfg: Stage1Config, wrapped: dict[str, StrategySpec | None]
) -> FamilySelection:
    rows = latest_runs(log, "train", cfg.selection_cost_bps).set_index(["family", "strategy", "params"])
    records = []
    for base, filt in filter_pairs(family, cfg, wrapped):
        rb, rf = (base.family, base.display_name, base.params_json), (filt.family, filt.display_name, filt.params_json)
        for key in (rb, rf):
            if key not in rows.index:
                raise IncompleteExperiments(f"{family} 계열 비교에 필요한 학습 실행이 없습니다: {key}")
        sb, sf = float(rows.loc[rb, "sharpe"]), float(rows.loc[rf, "sharpe"])
        n_f = int(rows.loc[rf, "n_trades"])
        gain = sf - sb
        passes = gain >= cfg.selection.filter_min_sharpe_gain and (
            family in cfg.selection.min_trades_exempt or n_f >= cfg.selection.min_trades
        )
        records.append(
            {"base": base.display_name, "params": base.params_json, "sharpe_base": sb, "sharpe_filtered": sf,
             "gain": gain, "n_trades_filtered": n_f, "passes": passes, "spec": filt}
        )
    table = pd.DataFrame(records)
    notes: list[str] = []
    candidate = None
    if not table.empty and table["passes"].any():
        best = table[table["passes"]].sort_values("sharpe_filtered", ascending=False).iloc[0]
        candidate = best["spec"]
    else:
        notes.append(f"필터로 샤프가 {cfg.selection.filter_min_sharpe_gain} 이상 오른 조합이 없어 후보 없음")
    if not table.empty:
        table["chosen"] = [c is candidate for c in table["spec"]]
    return FamilySelection(family, table.drop(columns=["spec"]) if not table.empty else table, candidate, notes)


def select_all(log: pd.DataFrame, cfg: Stage1Config) -> dict[str, FamilySelection]:
    out: dict[str, FamilySelection] = {}
    for name, fam in cfg.families.items():
        if not fam.is_filter_family:
            out[name] = select_grid_family(log, name, cfg)
    for name, fam in cfg.families.items():
        if fam.is_filter_family:
            wrapped = {w: out[w].candidate for w in fam.wraps}
            out[name] = select_filter_family(log, name, cfg, wrapped)
    return out


# ---------------------------------------------------------------- candidates.yaml


def candidates_document(selections: dict[str, FamilySelection], cfg: Stage1Config, commit: str) -> dict[str, Any]:
    specs = [s.candidate for s in selections.values() if s.candidate is not None]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_from_commit": commit,
        "selection_cost_bps": cfg.selection_cost_bps,
        "candidates_hash": specs_hash(specs),
        "candidates": [s.to_dict() for s in specs],
        "notes": {fam: sel.notes for fam, sel in selections.items() if sel.notes},
    }


def write_candidates(path: Path, doc: dict[str, Any]) -> None:
    header = (
        "# swinglab experiments select가 학습 구간 결과만으로 생성 (stage1-spec.md 후보 선정 규칙).\n"
        "# 손으로 고치지 않는다. 커밋한 뒤 swinglab holdout으로 검증 구간을 실행한다.\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")


def read_candidates(path: Path) -> list[StrategySpec]:
    with path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    return [StrategySpec.from_dict(d) for d in doc.get("candidates") or []]


__all__ = [
    "FamilySelection",
    "IncompleteExperiments",
    "candidates_document",
    "filter_pairs",
    "read_candidates",
    "select_all",
    "select_filter_family",
    "select_grid_family",
    "write_candidates",
]
