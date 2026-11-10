"""config/paper.yaml 로딩과 고정(frozen.json) 해시.

설정은 파싱한 내용을 정렬된 JSON으로 만들어 해시하므로 주석·공백은 영향을 주지 않는다.
전략 소스는 파일 바이트를 그대로 해시한다.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class PaperRefused(RuntimeError):
    """고정된 설정·소스와 달라 페이퍼 실행을 거부할 때."""


class AccountConfig(BaseModel):
    strategy: str
    params: dict[str, Any] = Field(default_factory=dict)


class PaperConfig(BaseModel):
    paper_id: str
    initial_capital: float = Field(gt=0)
    cost_bps: float = Field(ge=0)
    max_positions: int = Field(ge=1)
    tickers: list[str]
    benchmark: str
    strategies: dict[str, AccountConfig]
    source_files: list[str]

    @field_validator("paper_id")
    @classmethod
    def _plain_id(cls, v: str) -> str:
        if not v or not all(ch.isalnum() or ch in "-_" for ch in v):
            raise ValueError("paper_id는 영문·숫자·-·_만 쓸 수 있습니다")
        return v


def load_paper_config(config_dir: Path) -> PaperConfig:
    return PaperConfig.model_validate(_read(config_dir / "paper.yaml"))


def _read(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path}: YAML 최상위는 매핑이어야 합니다")
    return data


def frozen_path(paper_dir: Path) -> Path:
    return paper_dir / "frozen.json"


def fingerprint(cfg: PaperConfig, repo_dir: Path) -> dict[str, Any]:
    """설정 해시, 소스 파일별 해시, 둘을 합친 해시."""
    config_hash = _sha256(json.dumps(cfg.model_dump(mode="json"), sort_keys=True, ensure_ascii=False).encode())
    files: dict[str, str] = {}
    for rel in cfg.source_files:
        path = repo_dir / rel
        if not path.is_file():
            raise PaperRefused(f"고정 대상 소스 파일이 없습니다: {rel}")
        files[rel] = _sha256(path.read_bytes())
    combined = _sha256(json.dumps({"config": config_hash, "files": files}, sort_keys=True).encode())
    return {"config_hash": config_hash, "files": files, "hash": combined}


def read_frozen(paper_dir: Path) -> dict[str, dict[str, Any]]:
    path = frozen_path(paper_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def freeze_or_verify(
    cfg: PaperConfig, repo_dir: Path, paper_dir: Path, start_date: str, git_commit: str
) -> tuple[dict[str, Any], bool]:
    """paper_id가 처음이면 고정 기록을 만들고 (기록, True), 있으면 해시를 비교해 다르면 PaperRefused."""
    current = fingerprint(cfg, repo_dir)
    frozen = read_frozen(paper_dir)
    entry = frozen.get(cfg.paper_id)
    if entry is not None:
        if entry["hash"] != current["hash"]:
            raise PaperRefused(_mismatch_message(cfg.paper_id, entry, current))
        return entry, False

    entry = {
        "paper_id": cfg.paper_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "start_date": start_date,
        "git_commit": git_commit,
        "config": cfg.model_dump(mode="json"),
        **current,
    }
    frozen[cfg.paper_id] = entry
    paper_dir.mkdir(parents=True, exist_ok=True)
    tmp = frozen_path(paper_dir).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(frozen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(frozen_path(paper_dir))
    return entry, True


def _mismatch_message(paper_id: str, entry: dict, current: dict) -> str:
    parts = []
    if entry["config_hash"] != current["config_hash"]:
        parts.append("config/paper.yaml")
    old_files, new_files = entry["files"], current["files"]
    parts += [f for f in sorted(set(old_files) | set(new_files)) if old_files.get(f) != new_files.get(f)]
    return (
        f"페이퍼 계정 {paper_id}의 고정 기록과 다릅니다: {', '.join(parts)}. "
        "설정이나 전략 코드를 바꿔야 하면 paper.yaml의 paper_id를 새로 정하세요 (기존 기록은 그대로 둔다)."
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
