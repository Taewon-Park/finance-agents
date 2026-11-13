"""paper run 실행 전 확인: 전원(a), 처리할 날(b), 그다음 잠금 → 수집. 건너뛴 기록은 날짜·사유별 한 줄."""

from __future__ import annotations

import fcntl
import subprocess
from datetime import date, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from swinglab.cli import app
from swinglab.paper.account import AccountState
from swinglab.paper.preflight import power_source, record_skip
from swinglab.paper.store import write_state

LAST_SESSION = date(2026, 11, 12)


class IngestCalled(Exception):
    """수집 단계까지 왔음을 알리고 실행을 멈춘다."""


@pytest.fixture
def ingest(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """수집 호출 기록. 호출되면 그 순간 잠금이 잡혀 있었는지를 남긴다."""
    calls: list[bool] = []

    def fake(tickers, source, start, end, data_dir, **kwargs):
        calls.append(_locked(Path(_PAPER_DIR[0])))
        raise IngestCalled

    monkeypatch.setattr("swinglab.cli.run_ingest", fake)
    monkeypatch.setattr("swinglab.cli.last_completed_session", lambda: LAST_SESSION)
    return calls


_PAPER_DIR: list[str] = [""]


def _locked(paper_dir: Path) -> bool:
    with (paper_dir / ".lock").open("w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(f, fcntl.LOCK_UN)
        return False


def _power(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setattr("swinglab.cli.power_source", lambda: value)


def _states(paper_dir: Path, last: dict[str, str]) -> None:
    for account, day in last.items():
        write_state(paper_dir, AccountState(paper_id="p1", account=account, start_date="2026-11-10",
                                            last_processed=day, cash=100_000.0))


def _invoke(paper_dir: Path, *extra: str):
    _PAPER_DIR[0] = str(paper_dir)
    return CliRunner().invoke(app, ["paper", "run", "--paper-dir", str(paper_dir), *extra])


# --- a. 전원 -----------------------------------------------------------------------------


def test_battery_skips_before_lock_and_ingest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ingest) -> None:
    _power(monkeypatch, "battery")
    paper_dir, log = tmp_path / "paper", tmp_path / "paper" / "logs" / "skip.log"
    for _ in range(2):
        result = _invoke(paper_dir, "--skip-log", str(log))
        assert result.exit_code == 0 and result.output == ""  # launchd.log에는 아무것도 남지 않는다
    assert ingest == []
    assert not (paper_dir / ".lock").exists() and not (paper_dir / "state.json").exists()
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and " | 배터리 상태, 건너뜀 | 2회 | " in lines[0]

    shown = _invoke(paper_dir)  # skip 로그 없이 직접 실행하면 화면에
    assert shown.exit_code == 0 and shown.output.strip() == "배터리 상태, 건너뜀"


def test_force_battery_skips_power_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ingest) -> None:
    _power(monkeypatch, "battery")
    result = _invoke(tmp_path, "--force-battery")
    assert isinstance(result.exception, IngestCalled) and ingest == [True]
    assert "배터리 상태지만 --force-battery로 진행" in result.output and "paper run" in result.output


def test_unknown_power_does_not_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ingest) -> None:
    _power(monkeypatch, "unknown")
    result = _invoke(tmp_path)
    assert isinstance(result.exception, IngestCalled) and "전원 상태를 확인하지 못함" in result.output


# --- b. 처리할 날 --------------------------------------------------------------------------


def test_nothing_to_process_skips_without_ingest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ingest) -> None:
    _power(monkeypatch, "ac")
    _states(tmp_path, {"C": "2026-11-12", "A": "2026-11-12"})  # 마감된 최근 거래일까지 처리됨
    log = tmp_path / "logs" / "skip.log"

    result = _invoke(tmp_path, "--skip-log", str(log))

    assert result.exit_code == 0 and result.output == ""
    assert ingest == []  # 수집을 호출하지 않는다
    assert " | 처리할 날 없음 | 1회 | " in log.read_text(encoding="utf-8")
    assert not (tmp_path / ".lock").exists()  # 잠금 전에 끝난다


def test_one_account_behind_goes_to_ingest_under_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ingest) -> None:
    _power(monkeypatch, "ac")
    _states(tmp_path, {"C": "2026-11-12", "A": "2026-11-11"})  # 가장 늦은 계정 기준
    result = _invoke(tmp_path, "--skip-log", str(tmp_path / "skip.log"))
    assert isinstance(result.exception, IngestCalled)
    assert ingest == [True]  # 수집은 잠금을 잡은 뒤에
    assert not (tmp_path / "skip.log").exists()


def test_first_run_without_state_goes_to_ingest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ingest) -> None:
    _power(monkeypatch, "ac")
    result = _invoke(tmp_path)
    assert isinstance(result.exception, IngestCalled) and ingest == [True]


# --- skip 로그와 전원 판정 -------------------------------------------------------------------


def test_skip_log_one_line_per_day_and_reason(tmp_path: Path) -> None:
    log = tmp_path / "skip.log"
    for h in (8, 9, 10):
        record_skip(log, "배터리 상태, 건너뜀", datetime(2026, 11, 13, h, 0, 5))
    record_skip(log, "처리할 날 없음", datetime(2026, 11, 13, 11, 0, 7))
    record_skip(log, "배터리 상태, 건너뜀", datetime(2026, 11, 14, 8, 0, 1))
    assert log.read_text(encoding="utf-8").splitlines() == [
        "2026-11-13 | 배터리 상태, 건너뜀 | 3회 | 처음 08:00:05 | 마지막 10:00:05",
        "2026-11-13 | 처리할 날 없음 | 1회 | 처음 11:00:07 | 마지막 11:00:07",
        "2026-11-14 | 배터리 상태, 건너뜀 | 1회 | 처음 08:00:01 | 마지막 08:00:01",
    ]


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ("Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t100%; charged; 0:00 remaining present: true\n", "ac"),
        ("Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t80%; discharging\n", "battery"),
        ("Now drawing from 'AC Power'\n", "ac"),  # 배터리 없는 맥
        ("", "unknown"),
    ],
)
def test_power_source_parsing(monkeypatch: pytest.MonkeyPatch, stdout: str, expected: str) -> None:
    monkeypatch.setattr(
        "swinglab.paper.preflight.subprocess.run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, stdout=stdout, stderr=""),
    )
    assert power_source() == expected


def test_power_source_unknown_when_pmset_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*a, **k):
        raise FileNotFoundError("pmset")

    monkeypatch.setattr("swinglab.paper.preflight.subprocess.run", missing)
    assert power_source() == "unknown"
