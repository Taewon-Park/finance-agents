"""paper run 실행 전 확인 (잠금·수집 전에 한다).

a. 전원: 배터리 상태면 건너뛴다. 노트북이 잠든 채 배터리로 잠깐 깨어날 때(DarkWake) 실행되면
   몇 초씩 끊겨 진행되고 네트워크 요청이 실패하기 쉬워서다.
b. 처리할 날: 모든 계정의 마지막 처리일이 이미 마감된 최근 거래일(NYSE) 이상이면 수집 없이 건너뛴다.

건너뛴 기록은 skip 로그에 날짜·사유별 한 줄로만 남긴다 (1시간마다 실행돼도 실제 처리 로그가 묻히지 않게).
"""

from __future__ import annotations

import subprocess
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from swinglab.paper.config import PaperConfig
from swinglab.paper.store import read_state

PowerSource = Literal["ac", "battery", "unknown"]
PMSET_TIMEOUT_SECONDS = 5.0
SKIP_LOG_SEPARATOR = " | "


def power_source() -> PowerSource:
    """macOS pmset으로 전원 상태. 확인할 수 없으면(다른 OS, 명령 실패) unknown."""
    try:
        out = subprocess.run(
            ["pmset", "-g", "batt"], capture_output=True, text=True, timeout=PMSET_TIMEOUT_SECONDS, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    first = out.splitlines()[0] if out.strip() else ""
    if "'AC Power'" in first:
        return "ac"
    if "'Battery Power'" in first:
        return "battery"
    return "unknown"


def earliest_processed(paper_dir: Path, cfg: PaperConfig) -> date | None:
    """계정들의 마지막 처리일 중 가장 이른 날. 상태가 없는 계정이 있으면(첫 실행) None."""
    days = []
    for account in cfg.strategies:
        state = read_state(paper_dir, cfg.paper_id, account)
        if state is None:
            return None
        days.append(date.fromisoformat(state.last_processed))
    return min(days) if days else None


def nothing_to_process(paper_dir: Path, cfg: PaperConfig, last_session: date) -> bool:
    processed = earliest_processed(paper_dir, cfg)
    return processed is not None and processed >= last_session


def record_skip(path: Path, reason: str, now: datetime) -> None:
    """skip 로그: 날짜·사유별 한 줄 `YYYY-MM-DD | 사유 | N회 | 처음 HH:MM:SS | 마지막 HH:MM:SS`. 같은 날 같은 사유면 그 줄을 고친다."""
    day, clock = now.date().isoformat(), now.strftime("%H:%M:%S")
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    for i, line in enumerate(lines):
        parts = line.split(SKIP_LOG_SEPARATOR)
        if len(parts) == 5 and parts[0] == day and parts[1] == reason:
            count = int(parts[2].removesuffix("회")) + 1
            lines[i] = SKIP_LOG_SEPARATOR.join([day, reason, f"{count}회", parts[3], f"마지막 {clock}"])
            break
    else:
        lines.append(SKIP_LOG_SEPARATOR.join([day, reason, "1회", f"처음 {clock}", f"마지막 {clock}"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)
