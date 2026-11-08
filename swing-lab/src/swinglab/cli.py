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
from swinglab.sources.yfinance_source import YFinanceSource
from swinglab.strategies import STRATEGY_NAMES, build_strategy

app = typer.Typer(help="미국 섹터 ETF 스윙 백테스트", no_args_is_help=True, add_completion=False)

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
    path = write_report(scenarios, strat, config, universe.benchmark, reports_dir)
    base = scenarios[0].result
    typer.echo(f"{strategy}: {base.start_date.date()} ~ {base.end_date.date()}\n")
    typer.echo(metrics_table(scenarios, config.cost_bps, universe.benchmark).to_string())
    typer.echo(f"\n리포트: {path}")


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
