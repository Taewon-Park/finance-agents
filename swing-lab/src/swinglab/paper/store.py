"""페이퍼 기록 저장 (paper/ 아래, git 제외).

- state.json: 계정별 현재 상태 (마지막 처리일, 현금, 보유, 대기 주문)
- orders / fills / positions / equity / trades .parquet: 날짜별 기록. 키가 같은 행은 덮어써서
  같은 날을 다시 처리해도 중복되지 않는다 (기록을 먼저 쓰고 상태를 마지막에 바꾼다).
"""

from __future__ import annotations

import fcntl
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from swinglab.paper.account import AccountState, DayRecord

TABLE_KEYS: dict[str, list[str]] = {
    "orders": ["paper_id", "account", "signal_date", "ticker", "side"],
    "fills": ["paper_id", "account", "date", "ticker", "side"],
    "positions": ["paper_id", "account", "date", "ticker"],
    "equity": ["paper_id", "account", "date"],
    "trades": ["paper_id", "account", "ticker", "entry_date", "exit_date"],
}


class PaperBusy(RuntimeError):
    """다른 페이퍼 실행이 진행 중."""


def table_path(paper_dir: Path, name: str) -> Path:
    return paper_dir / f"{name}.parquet"


def read_table(paper_dir: Path, name: str) -> pd.DataFrame:
    path = table_path(paper_dir, name)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def upsert(paper_dir: Path, name: str, rows: list[dict]) -> None:
    if not rows:
        return
    keys = TABLE_KEYS[name]
    new = pd.DataFrame(rows)
    old = read_table(paper_dir, name)
    if not old.empty:
        key_new = set(map(tuple, new[keys].astype(str).to_numpy()))
        keep = [tuple(k) not in key_new for k in old[keys].astype(str).to_numpy()]
        new = pd.concat([old[keep], new], ignore_index=True)
    _write(new, table_path(paper_dir, name))


def apply_order_updates(paper_dir: Path, updates: list[dict]) -> None:
    """대기(pending) 주문의 상태를 체결·취소·건너뜀으로 바꾼다. 같은 종목·방향의 가장 최근 대기 주문 하나."""
    if not updates:
        return
    orders = read_table(paper_dir, "orders")
    if orders.empty:
        raise ValueError("갱신할 주문 기록이 없습니다")
    for u in updates:
        same = (
            (orders["paper_id"] == u["paper_id"])
            & (orders["account"] == u["account"])
            & (orders["ticker"] == u["ticker"])
            & (orders["side"] == u["side"])
        )
        mask = same & (orders["status"] == "pending")
        if not mask.any():
            # 같은 날을 다시 처리하는 경우 (기록 후 상태 저장 전에 중단됐던 실행): 이미 반영됨
            if (same & (orders["status"] == u["status"]) & (orders["fill_date"] == u["fill_date"])).any():
                continue
            raise ValueError(f"대기 주문이 없습니다: {u}")
        idx = orders.index[mask][orders.loc[mask, "signal_date"].argmax()]
        orders.loc[idx, ["status", "fill_date"]] = [u["status"], u["fill_date"]]
    _write(orders, table_path(paper_dir, "orders"))


def save_day(paper_dir: Path, state: AccountState, record: DayRecord) -> None:
    """하루 처리 결과를 기록하고 마지막에 상태를 바꾼다."""
    apply_order_updates(paper_dir, record.order_updates)
    upsert(paper_dir, "orders", record.orders)
    upsert(paper_dir, "fills", record.fills)
    upsert(paper_dir, "trades", record.trades)
    upsert(paper_dir, "positions", record.positions)
    upsert(
        paper_dir,
        "equity",
        [
            {
                "paper_id": state.paper_id,
                "account": state.account,
                "date": record.date,
                "equity": record.equity,
                "cash": record.cash,
                "n_positions": len(record.positions),
            }
        ],
    )
    write_state(paper_dir, state)


def state_path(paper_dir: Path) -> Path:
    return paper_dir / "state.json"


def read_states(paper_dir: Path) -> dict[str, dict]:
    path = state_path(paper_dir)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_state(paper_dir: Path, paper_id: str, account: str) -> AccountState | None:
    entry = read_states(paper_dir).get(f"{paper_id}/{account}")
    return AccountState.from_dict(entry) if entry else None


def write_state(paper_dir: Path, state: AccountState) -> None:
    states = read_states(paper_dir)
    states[f"{state.paper_id}/{state.account}"] = state.to_dict()
    paper_dir.mkdir(parents=True, exist_ok=True)
    tmp = state_path(paper_dir).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(states, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(state_path(paper_dir))


@contextmanager
def run_lock(paper_dir: Path) -> Iterator[None]:
    """launchd 실행과 수동 실행이 겹치지 않게 한다."""
    paper_dir.mkdir(parents=True, exist_ok=True)
    with (paper_dir / ".lock").open("w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PaperBusy("다른 페이퍼 실행이 진행 중입니다") from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
