"""swinglab 명령줄 도구.

    swinglab ingest                       # ETF 일봉 수집 (재실행 시 증분)
    swinglab check                        # 데이터 품질 점검 리포트
    swinglab backtest --strategy <이름>   # 백테스트 + HTML 리포트

경로 기본값은 swing-lab 디렉터리에서 실행한다고 보고 정한다.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import typer

from swinglab.alloc.backtest import AllocRefused, run_all
from swinglab.alloc.config import load_alloc_config
from swinglab.alloc.rebalance import (
    load_holdings,
    plan_rebalance,
    render_plan,
    write_plan,
)
from swinglab.alloc.summary import checked_summary, request_summary
from swinglab.config import load_backtest_config, load_universe
from swinglab.paper import PaperBusy, PaperRefused, load_paper_config, run_paper
from swinglab.paper.config import read_frozen
from swinglab.paper.report import (
    report_text,
    summarize_accounts,
    weekly_line,
    write_weekly,
)
from swinglab.paper.store import run_lock
from swinglab.pipeline.calendar import last_completed_session, nyse_sessions
from swinglab.pipeline.clean import run_checks
from swinglab.pipeline.fx import ingest_fx, read_fx
from swinglab.pipeline.ingest import ingest as run_ingest
from swinglab.pipeline.storage import clean_path, read_prices
from swinglab.reports import metrics_table, run_cost_scenarios, write_report
from swinglab.reports.alloc import goal_verdict, metrics_frame, write_alloc_report
from swinglab.reports.stage1 import write_stage1_report
from swinglab.research.config import load_stage1
from swinglab.research.holdout import HoldoutRefused, run_holdout
from swinglab.research.log import BudgetExceeded, git_commit, read_log, summarize
from swinglab.research.runner import run_train
from swinglab.research.selection import (
    IncompleteExperiments,
    candidates_document,
    select_all,
    write_candidates,
)
from swinglab.sources.fred_source import FredSource
from swinglab.sources.yfinance_source import YFinanceSource
from swinglab.strategies import STRATEGY_NAMES, build_strategy

app = typer.Typer(help="미국 섹터 ETF 스윙 백테스트", no_args_is_help=True, add_completion=False)
experiments_app = typer.Typer(help="1단계 학습 구간 실험 (검증 구간은 holdout 명령으로만)", no_args_is_help=True)
report_app = typer.Typer(help="단계별 리포트", no_args_is_help=True)
paper_app = typer.Typer(help="2단계 페이퍼 트레이딩 (설정 고정, 매일 실행)", no_args_is_help=True)
alloc_app = typer.Typer(help="2단계 자산배분 P0~P3 백테스트 (서술용)", no_args_is_help=True)
app.add_typer(experiments_app, name="experiments")
app.add_typer(report_app, name="report")
app.add_typer(paper_app, name="paper")
app.add_typer(alloc_app, name="alloc")
PaperDir = typer.Option(Path("paper"), "--paper-dir", help="페이퍼 기록 위치 (git 제외)")

ResultsDir = typer.Option(Path("results"), "--results-dir", help="실험 로그·자산곡선·검증 잠금 위치")
CandidatesFile = Path("config/candidates.yaml")

ConfigDir = typer.Option(Path("config"), "--config-dir", help="universe.yaml, backtest.yaml 위치")
DataDir = typer.Option(Path("data"), "--data-dir", help="raw/, clean/ 상위 디렉터리")
ReportsDir = typer.Option(Path("reports"), "--reports-dir", help="리포트 저장 위치")


@app.callback()
def _setup(verbose: bool = typer.Option(False, "--verbose", "-v", help="상세 로그")) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


@app.command()
def ingest(
    full: bool = typer.Option(False, "--full", help="증분 대신 전체 기간 다시 수집"),
    tickers: str | None = typer.Option(None, "--tickers", help="쉼표로 구분한 일부 티커만 (기본: 유니버스 전체)"),
    end: str | None = typer.Option(None, "--end", help="수집 마지막 날 YYYY-MM-DD (기본: 마감된 최근 거래일)"),
    config_dir: Path = ConfigDir,
    data_dir: Path = DataDir,
) -> None:
    """ETF 일봉과 환율을 수집해 data/raw, data/clean/(prices, fx).parquet에 저장한다."""
    universe = load_universe(config_dir)
    selected = [t.strip().upper() for t in tickers.split(",")] if tickers else universe.collect_tickers
    end_date = date.fromisoformat(end) if end else last_completed_session()
    typer.echo(f"수집: {len(selected)}종목, {universe.start} ~ {end_date}{' (전체 재수집)' if full else ''}")

    report = run_ingest(
        selected, YFinanceSource(), universe.start, end_date, data_dir, full=full,
        clean_tickers=universe.collect_tickers,
    )

    rows = [
        {
            "ticker": r.ticker,
            "mode": r.mode,
            "added": r.rows_added,
            "first": r.first_date,
            "last": r.last_date,
            "note": r.message,
        }
        for r in report.results
    ]
    typer.echo(pd.DataFrame(rows).to_string(index=False))
    typer.echo(f"\n정제본: {report.clean_file} ({report.clean_rows:,}행)")

    if report.clean_rows:
        quality = _quality_report(data_dir)
        typer.echo(f"품질 점검: 이슈 {quality.issue_count}건 (자세히: swinglab check)")

    fx_failed = []
    if universe.fx and not tickers:
        for r in ingest_fx(universe.fx, FredSource(), universe.start, end_date, data_dir):
            if r.error:
                fx_failed.append(r.pair)
                typer.echo(f"환율 {r.pair} ({r.series_id}) 실패: {r.error}", err=True)
            else:
                typer.echo(f"환율 {r.pair} ({r.series_id}): {r.rows:,}행, {r.first_date} ~ {r.last_date}")
    if report.failed or fx_failed:
        if report.failed:
            typer.echo(f"실패 {len(report.failed)}종목: {', '.join(r.ticker for r in report.failed)}", err=True)
        raise typer.Exit(code=1)


@app.command()
def check(
    data_dir: Path = DataDir,
    reports_dir: Path = ReportsDir,
    save: bool = typer.Option(True, "--save/--no-save", help="reports/data_quality.txt로도 저장"),
) -> None:
    """결측 거래일·수익률 이상치·가격 범위 오류·거래량 0을 점검해 출력한다."""
    report = _quality_report(data_dir)
    text = report.to_text()
    typer.echo(text)
    if save:
        reports_dir.mkdir(parents=True, exist_ok=True)
        path = reports_dir / "data_quality.txt"
        path.write_text(text, encoding="utf-8")
        typer.echo(f"\n저장: {path}")


@app.command()
def backtest(
    strategy: str = typer.Option(..., "--strategy", "-s", help=f"전략 이름: {', '.join(STRATEGY_NAMES)}"),
    config_dir: Path = ConfigDir,
    data_dir: Path = DataDir,
    reports_dir: Path = ReportsDir,
    offline: bool = typer.Option(
        False, "--offline", help="plotly.js를 HTML에 포함 (인터넷 없이 열람, 파일 약 5MB). 기본은 CDN 로드"
    ),
) -> None:
    """전략 백테스트를 비용 시나리오별로 돌리고 HTML 리포트를 만든다."""
    if strategy not in STRATEGY_NAMES:
        raise typer.BadParameter(f"가능한 전략: {', '.join(STRATEGY_NAMES)}", param_hint="--strategy")
    universe = load_universe(config_dir)
    config = load_backtest_config(config_dir)
    prices = _load_prices(data_dir)
    strat = build_strategy(strategy, universe)

    needed = set(getattr(strat, "tickers", [getattr(strat, "ticker", universe.benchmark)])) | {universe.benchmark}
    missing = sorted(needed - set(prices["ticker"].unique()))
    if missing:
        typer.echo(f"경고: 데이터에 없는 종목 {', '.join(missing)} (swinglab ingest로 수집 필요)", err=True)

    scenarios = run_cost_scenarios(prices, strat, config, universe.benchmark)
    path = write_report(scenarios, strat, config, universe.benchmark, reports_dir, offline=offline)
    base = scenarios[0].result
    typer.echo(f"{strategy}: {base.start_date.date()} ~ {base.end_date.date()}\n")
    typer.echo(metrics_table(scenarios, config.cost_bps, universe.benchmark).to_string())
    typer.echo(f"\n리포트: {path}")


@experiments_app.command("run")
def experiments_run(
    family: str = typer.Option(..., "--family", "-f", help="계열: A, B, C, D (D는 A·C 실행 후)"),
    config_dir: Path = ConfigDir,
    data_dir: Path = DataDir,
    results_dir: Path = ResultsDir,
) -> None:
    """계열의 모든 파라미터 조합을 학습 구간에서 비용별로 실행하고 실험 로그에 기록한다."""
    cfg = load_stage1(config_dir)
    try:
        rows = run_train(
            family.upper(), _load_prices(data_dir), load_universe(config_dir), load_backtest_config(config_dir),
            cfg, results_dir, Path.cwd(),
        )
    except (BudgetExceeded, IncompleteExperiments, ValueError, TypeError) as exc:
        typer.echo(f"실행 거부: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    cols = ["family", "strategy", "params", "cost_bp", "sharpe", "cagr", "mdd", "n_trades", "window_start", "window_end"]
    typer.echo(rows[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    typer.echo(f"\n{len(rows)}행 기록 → {results_dir / 'experiments.parquet'} (git {rows['git_commit'].iloc[0][:12]})")


@experiments_app.command("summary")
def experiments_summary(config_dir: Path = ConfigDir, results_dir: Path = ResultsDir) -> None:
    """계열별 시도 횟수와 샤프 분포 (학습 구간)."""
    cfg = load_stage1(config_dir)
    table = summarize(read_log(results_dir), cfg)
    if table.empty:
        typer.echo("실험 로그가 없습니다.")
        return
    typer.echo(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    typer.echo(f"\n전체 시도 {table.attrs['total_trials']} / 예산 {cfg.budget.total} (계열당 {cfg.budget.per_family})")


@experiments_app.command("select")
def experiments_select(
    config_dir: Path = ConfigDir,
    results_dir: Path = ResultsDir,
    output: Path = typer.Option(CandidatesFile, "--output", help="후보 파일 경로"),
    note: list[str] = typer.Option([], "--note", help="candidates.yaml에 남길 절차 메모 (여러 번 가능)"),
) -> None:
    """후보 선정 규칙으로 계열당 후보를 고르고 candidates.yaml을 쓴다 (학습 구간 결과만 사용)."""
    cfg = load_stage1(config_dir)
    try:
        selections = select_all(read_log(results_dir), cfg)
    except IncompleteExperiments as exc:
        typer.echo(f"선정 불가: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    doc = candidates_document(selections, cfg, git_commit(Path.cwd()), procedure_notes=note)
    write_candidates(output, doc)
    for fam, sel in selections.items():
        c = sel.candidate
        typer.echo(f"{fam}: {c.display_name + ' ' + c.params_json if c else '후보 없음'}")
        for line in sel.notes:
            typer.echo(f"    {line}")
    typer.echo(f"\n저장: {output}. 커밋한 뒤 swinglab holdout --candidates {output}")


@app.command()
def holdout(
    candidates: Path = typer.Option(..., "--candidates", help="커밋된 후보 파일 (config/candidates.yaml)"),
    force: bool = typer.Option(False, "--force", help="같은 후보로 다시 실행 (잠금 기록에 forced=true)"),
    config_dir: Path = ConfigDir,
    data_dir: Path = DataDir,
    results_dir: Path = ResultsDir,
) -> None:
    """검증 구간 실행. 같은 후보로는 한 번만 (results/holdout.lock)."""
    try:
        rows = run_holdout(
            candidates, _load_prices(data_dir), load_universe(config_dir), load_backtest_config(config_dir),
            load_stage1(config_dir), results_dir, Path.cwd(), force=force,
        )
    except HoldoutRefused as exc:
        typer.echo(f"실행 거부: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"검증 구간 {len(rows)}행 기록. 결과는 swinglab report stage1로 확인")


@report_app.command("stage1")
def report_stage1(
    config_dir: Path = ConfigDir,
    results_dir: Path = ResultsDir,
    reports_dir: Path = ReportsDir,
    candidates: Path = typer.Option(CandidatesFile, "--candidates", help="후보 파일"),
    offline: bool = typer.Option(False, "--offline", help="plotly.js를 HTML에 포함"),
) -> None:
    """1단계 리포트 HTML (results/ 파일만 읽음, 백테스트를 다시 돌리지 않음)."""
    try:
        path = write_stage1_report(load_stage1(config_dir), results_dir, candidates, reports_dir, offline=offline)
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"리포트: {path}")


@paper_app.command("run")
def paper_run(
    do_ingest: bool = typer.Option(True, "--ingest/--no-ingest", help="처리 전에 증분 수집 (launchd 실행은 기본값)"),
    config_dir: Path = ConfigDir,
    data_dir: Path = DataDir,
    paper_dir: Path = PaperDir,
) -> None:
    """증분 수집 → 고정 확인 → 놓친 거래일까지 하루씩 체결·평가·다음 주문 (같은 날은 다시 처리하지 않음).

    다른 paper run이 실행 중이면(정시 실행과 로그인 시 실행이 겹친 경우 등) 아무것도 하지 않고 정상 종료한다.
    """
    typer.echo(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] paper run")
    try:
        with run_lock(paper_dir):
            _paper_run(do_ingest, config_dir, data_dir, paper_dir)
    except PaperBusy:
        typer.echo("이미 실행 중, 건너뜀 (다른 paper run이 잠금을 잡고 있음)")


def _paper_run(do_ingest: bool, config_dir: Path, data_dir: Path, paper_dir: Path) -> None:
    """수집 전부터 잠금을 잡은 상태에서 실행한다 (두 실행이 데이터를 동시에 쓰지 않게)."""
    cfg = load_paper_config(config_dir)
    universe = load_universe(config_dir)
    failed: list[str] = []
    if do_ingest:
        # 페이퍼에 필요한 0단계 유니버스만 수집하고, 정제본은 수집 대상 전체로 만든다
        report = run_ingest(
            universe.tickers, YFinanceSource(), universe.start, last_completed_session(), data_dir,
            clean_tickers=universe.collect_tickers,
        )
        failed = [r.ticker for r in report.failed]
        typer.echo(f"수집: {len(report.results)}종목, 실패 {len(failed)}{' (' + ', '.join(failed) + ')' if failed else ''}")
    prices = _load_prices(data_dir)
    try:
        result = run_paper(cfg, prices, universe, paper_dir, Path.cwd(), git_commit(Path.cwd()))
    except (PaperRefused, ValueError) as exc:
        typer.echo(f"실행 거부: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if result.frozen_created:
        typer.echo(f"새 페이퍼 계정 {cfg.paper_id}: 시작일 {result.start_date}, 고정 기록 {paper_dir / 'frozen.json'}")
    for acct in result.accounts:
        days = f"{acct.processed[0]} ~ {acct.processed[-1]} ({len(acct.processed)}일)" if acct.processed else "없음"
        typer.echo(
            f"{acct.account}: 처리 {days}, 마지막 {acct.last_processed}, 자산 {acct.equity:,.2f}, "
            f"보유 {acct.n_positions}, 다음 시가 주문 후보 {acct.pending_orders}"
        )
    key, line = weekly_line(cfg, summarize_accounts(cfg, paper_dir, prices))
    if line:
        write_weekly(paper_dir, key, line)
        typer.echo(line)
    if failed:
        typer.echo("일부 종목 수집 실패: 모든 대상 종목의 데이터가 있는 날까지만 처리했습니다.", err=True)
        raise typer.Exit(code=1)


@paper_app.command("report")
def paper_report(config_dir: Path = ConfigDir, data_dir: Path = DataDir, paper_dir: Path = PaperDir) -> None:
    """시작일 이후 누적 지표, 같은 기간 SPY 바이앤홀드, 변동성 맞춤 SPY 혼합 비교."""
    cfg = load_paper_config(config_dir)
    entry = read_frozen(paper_dir).get(cfg.paper_id)
    if entry is None:
        typer.echo(f"페이퍼 계정 {cfg.paper_id}의 기록이 없습니다. 먼저 swinglab paper run을 실행하세요.", err=True)
        raise typer.Exit(code=1)
    summaries = summarize_accounts(cfg, paper_dir, _load_prices(data_dir))
    typer.echo(report_text(cfg, summaries, entry["start_date"]))
    key, line = weekly_line(cfg, summaries)
    if line:
        write_weekly(paper_dir, key, line)
        typer.echo(f"\n주간 요약: {line}")


@alloc_app.command("backtest")
def alloc_backtest(
    config_dir: Path = ConfigDir,
    data_dir: Path = DataDir,
    results_dir: Path = ResultsDir,
    reports_dir: Path = ReportsDir,
    offline: bool = typer.Option(False, "--offline", help="plotly.js를 HTML에 포함"),
) -> None:
    """P0~P3 × 비용 0/5/10bp 백테스트 → 실험 로그(family=alloc) + reports/stage2_alloc.html, .md."""
    cfg = load_alloc_config(config_dir)
    prices = _load_prices(data_dir)
    try:
        fx = read_fx(data_dir, cfg.tax.fx_pair)
        runs = run_all(prices, fx, cfg, results_dir, Path.cwd(), git_commit(Path.cwd()))
    except (AllocRefused, BudgetExceeded, FileNotFoundError, ValueError) as exc:
        typer.echo(f"실행 거부: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    html_path, md_path = write_alloc_report(
        runs, cfg, reports_dir, prices["date"].max().date(), fx.index.max().date(), offline=offline
    )
    rep = cfg.backtest.report_cost_bps
    typer.echo(md_path.read_text(encoding="utf-8").splitlines()[0])
    typer.echo(goal_verdict(runs, cfg))
    typer.echo(f"\n{rep:g}bp\n" + metrics_frame(runs, rep).to_string(index=False))
    typer.echo(f"\n리포트: {html_path}, {md_path}")


@app.command()
def rebalance(
    holdings: Path | None = typer.Option(None, "--holdings", help="보유 내역 CSV (기본: alloc.yaml live.holdings)"),
    contribution: int = typer.Option(0, "--contribution", min=0, help="이번 분기 신규 납입액 (원)"),
    realized_ytd: int = typer.Option(0, "--realized-ytd", help="올해 이미 실현한 해외주식 차익 (원, 세액 추정용)"),
    policy: str | None = typer.Option(None, "--policy", help="정책 (기본: alloc.yaml live.policy)"),
    output_dir: Path | None = typer.Option(None, "--output-dir", help="문서 저장 위치 (기본: live.output_dir)"),
    llm: bool | None = typer.Option(None, "--llm/--no-llm", help="로컬 LLM 요약 (기본: alloc.yaml llm.enabled)"),
    config_dir: Path = ConfigDir,
    data_dir: Path = DataDir,
) -> None:
    """분기 리밸런싱 주문 목록 문서 (<YYYY-Qn>.md). 계좌와 연결하지 않으며 주문은 사람이 직접 한다."""
    cfg = load_alloc_config(config_dir)
    holdings_path = holdings or cfg.live.holdings
    out_dir = output_dir or cfg.live.output_dir
    try:
        plan = plan_rebalance(
            load_holdings(holdings_path), _load_prices(data_dir), read_fx(data_dir, cfg.tax.fx_pair), cfg,
            policy_name=policy, contribution_krw=contribution, realized_ytd_krw=realized_ytd,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        typer.echo(f"실행 거부: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if plan.as_of.date() < last_completed_session():
        typer.echo(f"주의: 가격 기준일 {plan.as_of.date()}이 마감된 최근 거래일보다 앞입니다 (swinglab ingest 필요).", err=True)
    text = render_plan(plan)
    summary_note = ""
    summary = None
    if llm if llm is not None else cfg.llm.enabled:
        raw, err = request_summary(text, cfg.llm)
        summary, summary_note = checked_summary(raw, text) if raw is not None else (None, err)
    text = render_plan(plan, summary, summary_note)
    path = write_plan(text, plan, out_dir)
    typer.echo(f"{plan.quarter_label} 정책 {plan.policy}, 기준일 {plan.as_of.date()}: 주문 {len(plan.orders)}건, "
               f"예상 세액 증가 {plan.tax['tax_increase_krw']:,}원")
    typer.echo(f"문서: {path}")


def _load_prices(data_dir: Path) -> pd.DataFrame:
    path = clean_path(data_dir)
    if not path.exists():
        typer.echo(f"{path}가 없습니다. 먼저 swinglab ingest를 실행하세요.", err=True)
        raise typer.Exit(code=1)
    return read_prices(path)


def _quality_report(data_dir: Path):
    prices = _load_prices(data_dir)
    sessions = nyse_sessions(prices["date"].min().date(), prices["date"].max().date())
    return run_checks(prices, sessions)


if __name__ == "__main__":
    app()
