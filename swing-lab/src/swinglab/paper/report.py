"""페이퍼 기록 리포트 (swinglab paper report)와 주간 요약 한 줄 (paper/weekly.md).

지표 정의는 0단계와 같다 (backtest/metrics.py). 비교 대상은 같은 기간의 SPY 바이앤홀드
(시작일 다음 거래일 시가 매수, 같은 비용)와 변동성을 계정에 맞춘 SPY+현금 혼합.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from swinglab.backtest.benchmarks import vol_matched_curve
from swinglab.backtest.engine import run_backtest
from swinglab.backtest.metrics import (
    annual_volatility,
    cagr,
    daily_returns,
    max_drawdown,
    sharpe_ratio,
)
from swinglab.config import BacktestConfig
from swinglab.paper.config import PaperConfig
from swinglab.paper.store import read_table
from swinglab.strategies.buy_hold import BuyAndHold

JUDGEMENT_MONTHS = 6
MIN_DAYS_FOR_METRICS = 3  # 자산가치 3행(수익률 2개) 미만이면 변동성·샤프를 내지 않는다


@dataclass
class AccountSummary:
    account: str
    start: str
    last: str
    sessions: int
    total_return: float
    cagr: float
    vol: float
    sharpe: float
    mdd: float
    n_trades: int
    win_rate: float
    exposure: float
    spy_total_return: float
    spy_cagr: float
    spy_sharpe: float
    spy_mdd: float
    mix_weight: float
    mix_total_return: float
    mix_cagr: float
    mix_mdd: float


def account_equity(paper_dir: Path, paper_id: str, account: str) -> pd.DataFrame:
    eq = read_table(paper_dir, "equity")
    if eq.empty:
        return eq
    eq = eq[(eq["paper_id"] == paper_id) & (eq["account"] == account)].copy()
    eq["date"] = pd.to_datetime(eq["date"])
    return eq.sort_values("date").set_index("date")


def summarize_accounts(cfg: PaperConfig, paper_dir: Path, prices: pd.DataFrame) -> list[AccountSummary]:
    trades = read_table(paper_dir, "trades")
    out = []
    for account in cfg.strategies:
        eq = account_equity(paper_dir, cfg.paper_id, account)
        if eq.empty:
            continue
        curve = eq["equity"].astype(float)
        enough = len(curve) >= MIN_DAYS_FOR_METRICS
        rets = daily_returns(curve)
        acct_trades = (
            trades[(trades["paper_id"] == cfg.paper_id) & (trades["account"] == account)]
            if not trades.empty
            else trades
        )
        n = len(acct_trades)
        spy = _spy_curve(cfg, prices, curve) if len(curve) >= 2 else None
        vol = annual_volatility(rets) if enough else math.nan
        mix, weight = (
            vol_matched_curve(spy, vol) if spy is not None and enough and vol > 0 else (None, math.nan)
        )
        out.append(
            AccountSummary(
                account=account,
                start=curve.index[0].date().isoformat(),
                last=curve.index[-1].date().isoformat(),
                sessions=len(curve) - 1,
                total_return=float(curve.iloc[-1] / curve.iloc[0] - 1.0),
                cagr=cagr(curve) if len(curve) >= 2 else math.nan,
                vol=vol,
                sharpe=sharpe_ratio(rets) if enough else math.nan,
                mdd=max_drawdown(curve),
                n_trades=n,
                win_rate=float((acct_trades["return"] > 0).mean()) if n else math.nan,
                exposure=float((eq["n_positions"].iloc[1:] > 0).mean()) if len(eq) > 1 else math.nan,
                spy_total_return=_total(spy),
                spy_cagr=cagr(spy) if spy is not None else math.nan,
                spy_sharpe=sharpe_ratio(daily_returns(spy)) if spy is not None and enough else math.nan,
                spy_mdd=max_drawdown(spy) if spy is not None else math.nan,
                mix_weight=weight,
                mix_total_return=_total(mix),
                mix_cagr=cagr(mix) if mix is not None else math.nan,
                mix_mdd=max_drawdown(mix) if mix is not None else math.nan,
            )
        )
    return out


def _spy_curve(cfg: PaperConfig, prices: pd.DataFrame, curve: pd.Series) -> pd.Series:
    """계정과 같은 기간: 시작일 다음 거래일 시가 매수, 시작일 종가 = 초기자본."""
    bt = BacktestConfig(
        initial_capital=cfg.initial_capital,
        cost_bps=cfg.cost_bps,
        start=curve.index[1].date(),
        end=curve.index[-1].date(),
    )
    result = run_backtest(prices, BuyAndHold(ticker=cfg.benchmark), bt)
    return result.equity.reindex(curve.index).ffill()


def _total(curve: pd.Series | None) -> float:
    return float(curve.iloc[-1] / curve.iloc[0] - 1.0) if curve is not None else math.nan


def report_text(cfg: PaperConfig, summaries: list[AccountSummary], start_date: str) -> str:
    judge = (pd.Timestamp(start_date) + pd.DateOffset(months=JUDGEMENT_MONTHS)).date().isoformat()
    lines = [
        f"# 페이퍼 트레이딩 {cfg.paper_id}",
        "",
        f"시작일 {start_date} (그 이전 성과는 기록에 없음). 판단은 {judge} 이후. 그 전에는 결과를 보고 설정을 바꾸지 않는다.",
        f"비용 편도 {cfg.cost_bps:g}bp. 비교: 같은 기간 {cfg.benchmark} 바이앤홀드, 변동성을 맞춘 {cfg.benchmark}+현금.",
        "",
        "| 계정 | 기간 | 거래일 | 누적 | CAGR | 변동성 | 샤프 | MDD | 청산 거래 | 승률 | 노출도 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in summaries:
        name = f"{s.account} {cfg.strategies[s.account].strategy}"
        lines.append(
            f"| {name} | {s.start} ~ {s.last} | {s.sessions} | {_ret(s.total_return)} | {_ret(s.cagr)} | "
            f"{_pct(s.vol)} | {_num(s.sharpe)} | {_ret(s.mdd)} | {s.n_trades} | {_pct(s.win_rate)} | {_pct(s.exposure)} |"
        )
    lines += [
        "",
        (
            f"| 계정 | {cfg.benchmark} 누적 | {cfg.benchmark} CAGR | {cfg.benchmark} 샤프 | {cfg.benchmark} MDD | "
            "혼합 비중 | 혼합 누적 | 혼합 CAGR | 혼합 MDD |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in summaries:
        lines.append(
            f"| {s.account} | {_ret(s.spy_total_return)} | {_ret(s.spy_cagr)} | {_num(s.spy_sharpe)} | "
            f"{_ret(s.spy_mdd)} | {_pct(s.mix_weight)} | {_ret(s.mix_total_return)} | {_ret(s.mix_cagr)} | "
            f"{_ret(s.mix_mdd)} |"
        )
    if any(s.sessions < MIN_DAYS_FOR_METRICS - 1 for s in summaries):
        lines += ["", "거래일이 적어 변동성·샤프·혼합 비교는 아직 계산하지 않는다."]
    return "\n".join(lines)


def weekly_line(cfg: PaperConfig, summaries: list[AccountSummary]) -> tuple[str, str]:
    """(주 키, 한 줄). 주 키 = paper_id + 마지막 처리일의 ISO 주."""
    if not summaries:
        return "", ""
    last = max(s.last for s in summaries)
    iso = pd.Timestamp(last).isocalendar()
    key = f"{cfg.paper_id} {iso.year}-W{iso.week:02d}"
    parts = [f"{s.account} {_ret(s.total_return)} (거래 {s.n_trades})" for s in summaries]
    spy = summaries[0].spy_total_return
    return key, f"{key} | 마지막 {last} | " + " | ".join(parts) + f" | {cfg.benchmark} {_ret(spy)}"


def write_weekly(paper_dir: Path, key: str, line: str) -> Path:
    """같은 주의 줄은 바꾸고, 새 주면 덧붙인다."""
    path = paper_dir / "weekly.md"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else ["# 페이퍼 주간 요약", ""]
    lines = [ln for ln in lines if not ln.startswith(key + " ")]
    lines.append(line)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _ret(v: float) -> str:
    return "–" if v is None or math.isnan(v) else f"{v:+.2%}"


def _pct(v: float) -> str:
    return "–" if v is None or math.isnan(v) else f"{v:.2%}"


def _num(v: float) -> str:
    return "–" if v is None or math.isnan(v) else f"{v:.2f}"
