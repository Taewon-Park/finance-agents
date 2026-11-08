"""swinglab 명령줄 도구.

    swinglab ingest                       # ETF 일봉 수집 (재실행 시 증분)
    swinglab check                        # 데이터 품질 점검 리포트
    swinglab backtest --strategy <이름>   # 백테스트 + HTML 리포트

경로 기본값은 swing-lab 디렉터리에서 실행한다고 보고 정한다.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import typer

from swinglab.config import load_backtest_config, load_universe
from swinglab.pipeline.calendar import last_completed_session, nyse_sessions
from swinglab.pipeline.clean import run_checks
from swinglab.pipeline.ingest import ingest as run_ingest
from swinglab.pipeline.storage import clean_path, read_prices
from swinglab.reports import metrics_table, run_cost_scenarios, write_report
from swinglab.reports.stage1 import write_stage1_report
from swinglab.research.config import load_stage1
from swinglab.research.holdout import HoldoutRefused, run_holdout
from swinglab.research.log import BudgetExceeded, git_commit, read_log, summarize
from swinglab.research.runner import run_train
from swinglab.research.selection import IncompleteExperiments, candidates_document, select_all, write_candidates
from swinglab.sources.yfinance_source import YFinanceSource
from swinglab.strategies import STRATEGY_NAMES, build_strategy

app = typer.Typer(help="미국 섹터 ETF 스윙 백테스트", no_args_is_help=True, add_completion=False)
experiments_app = typer.Typer(help="1단계 학습 구간 실험 (검증 구간은 holdout 명령으로만)", no_args_is_help=True)
report_app = typer.Typer(help="단계별 리포트", no_args_is_help=True)
app.add_typer(experiments_app, name="experiments")
app.add_typer(report_app, name="report")

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
    """ETF 일봉을 수집해 data/raw, data/clean/prices.parquet에 저장한다."""
    universe = load_universe(config_dir)
    selected = [t.strip().upper() for t in tickers.split(",")] if tickers else universe.tickers
    end_date = date.fromisoformat(end) if end else last_completed_session()
    typer.echo(f"수집: {len(selected)}종목, {universe.start} ~ {end_date}{' (전체 재수집)' if full else ''}")

    report = run_ingest(selected, YFinanceSource(), universe.start, end_date, data_dir, full=full)

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
    except (BudgetExceeded, IncompleteExperiments, ValueError) as exc:
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
) -> None:
    """후보 선정 규칙으로 계열당 후보를 고르고 candidates.yaml을 쓴다 (학습 구간 결과만 사용)."""
    cfg = load_stage1(config_dir)
    try:
        selections = select_all(read_log(results_dir), cfg)
    except IncompleteExperiments as exc:
        typer.echo(f"선정 불가: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    doc = candidates_document(selections, cfg, git_commit(Path.cwd()))
    write_candidates(output, doc)
    for fam, sel in selections.items():
        c = sel.candidate
        typer.echo(f"{fam}: {c.display_name + ' ' + c.params_json if c else '후보 없음'}")
        for note in sel.notes:
            typer.echo(f"    {note}")
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
