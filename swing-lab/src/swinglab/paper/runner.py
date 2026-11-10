"""페이퍼 트레이딩 하루 실행 (swinglab paper run).

순서: 고정 확인 → 계정별로 마지막 처리일 다음 거래일부터 '대상 종목 모두 데이터가 있는 마지막 날'까지
하루씩 처리. 실패로 건너뛴 날은 다음 실행에서 순서대로 처리하고, 처리한 날은 다시 처리하지 않는다.
첫 실행이면 그날(데이터 마지막 날)이 시작일이고, 그 이전 성과는 기록하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from swinglab.config import UniverseConfig
from swinglab.paper.account import (
    AccountState,
    MarketView,
    Rules,
    advance,
    open_account,
)
from swinglab.paper.config import (
    PaperConfig,
    PaperRefused,
    freeze_or_verify,
    read_frozen,
)
from swinglab.paper.store import read_state, run_lock, save_day
from swinglab.strategies import build_parametrized


@dataclass
class AccountRun:
    account: str
    opened: bool
    processed: list[str] = field(default_factory=list)
    last_processed: str = ""
    equity: float = 0.0
    n_positions: int = 0
    pending_orders: int = 0


@dataclass
class PaperRunReport:
    paper_id: str
    frozen_created: bool
    start_date: str
    accounts: list[AccountRun] = field(default_factory=list)


def run_paper(
    cfg: PaperConfig,
    prices: pd.DataFrame,
    universe: UniverseConfig,
    paper_dir: Path,
    repo_dir: Path,
    git_commit: str,
) -> PaperRunReport:
    with run_lock(paper_dir):
        views, rules = {}, {}
        for name, acct in cfg.strategies.items():
            strategy = build_parametrized(acct.strategy, acct.params, universe, tickers=cfg.tickers)
            views[name] = MarketView.build(prices, strategy)
            rules[name] = Rules.from_strategy(strategy, cfg.cost_bps, cfg.max_positions)
        until = min(v.complete_until(prices) for v in views.values())

        existing = read_frozen(paper_dir).get(cfg.paper_id)
        start = existing["start_date"] if existing else until.date().isoformat()
        entry, created = freeze_or_verify(cfg, repo_dir, paper_dir, start, git_commit)
        report = PaperRunReport(paper_id=cfg.paper_id, frozen_created=created, start_date=entry["start_date"])

        for name in cfg.strategies:
            view, rule = views[name], rules[name]
            state = read_state(paper_dir, cfg.paper_id, name)
            run = AccountRun(account=name, opened=state is None)
            if state is None:
                state, record = open_account(
                    cfg.paper_id, name, pd.Timestamp(entry["start_date"]), cfg.initial_capital, view, rule
                )
                save_day(paper_dir, state, record)
                run.processed.append(record.date)
            for d in _days_to_process(state, view, until):
                record = advance(state, view, d, rule)
                save_day(paper_dir, state, record)
                run.processed.append(record.date)
            run.last_processed = state.last_processed
            run.equity = state.cash + sum(p.shares * p.mark_price for p in state.positions.values())
            run.n_positions = len(state.positions)
            run.pending_orders = len(state.pending_entries) + len(state.pending_exits)
            report.accounts.append(run)
        return report


def _days_to_process(state: AccountState, view: MarketView, until: pd.Timestamp) -> list[pd.Timestamp]:
    last = pd.Timestamp(state.last_processed)
    if last not in view.calendar:
        raise PaperRefused(f"마지막 처리일 {state.last_processed}가 가격 데이터에 없습니다")
    return [d for d in view.calendar if last < d <= until]
