"""HTML 리포트: 비용 민감도 지표 표 + 자산곡선·드로다운 차트 + 거래 내역 (파일 한 장)."""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from swinglab.backtest.engine import BacktestResult, run_backtest, run_benchmark
from swinglab.backtest.metrics import drawdown, performance_metrics
from swinglab.config import BacktestConfig
from swinglab.strategies.base import Strategy

# 차트 색 (dataviz 기본 팔레트 라이트 모드): 범주 1·2번 슬롯, 잉크·격자
SERIES_STRATEGY = "#2a78d6"
SERIES_BENCHMARK = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

STRATEGY_DESCRIPTIONS: dict[str, str] = {
    "buy_hold": "벤치마크를 백테스트 첫 거래일 시가에 100% 매수해 끝까지 보유 (엔진 기준선)",
    "sma_cross": "SMA(20) > SMA(60)이면 보유, 강도 SMA20/SMA60−1 큰 순 최대 5개, 조건이 깨지면 익일 시가 청산",
    "rsi2_pullback": "종가 > SMA(200) & RSI(2) < 10 진입, 종가 > SMA(5) 또는 10거래일 경과 시 청산, RSI 낮은 순 최대 5개",
}


@dataclass
class CostScenario:
    cost_bps: float
    result: BacktestResult
    benchmark: BacktestResult
    metrics: dict[str, float]
    benchmark_metrics: dict[str, float]


def run_cost_scenarios(
    prices: pd.DataFrame, strategy: Strategy, config: BacktestConfig, benchmark: str
) -> list[CostScenario]:
    """비용 시나리오별로 전략과 같은 구간의 벤치마크 바이앤홀드를 돌린다."""
    scenarios = []
    for bps in config.cost_sensitivity_bps:
        cfg = config.model_copy(update={"cost_bps": bps})
        result = run_backtest(prices, strategy, cfg)
        bench = run_benchmark(prices, benchmark, cfg, like=result)
        scenarios.append(
            CostScenario(bps, result, bench, performance_metrics(result), performance_metrics(bench))
        )
    return scenarios


# (키, 라벨, 형식) — 형식: pct = 백분율, num = 소수 2자리, days = 소수 1자리, int, money
METRIC_ROWS: list[tuple[str, str, str]] = [
    ("total_return", "총수익률", "pct"),
    ("cagr", "CAGR", "pct"),
    ("annual_vol", "연 변동성", "pct"),
    ("sharpe", "샤프비율", "num"),
    ("mdd", "MDD", "pct"),
    ("win_rate", "승률", "pct"),
    ("payoff_ratio", "평균 손익비", "num"),
    ("avg_holding_days", "평균 보유일", "days"),
    ("trades_per_year", "연 거래 횟수 (왕복)", "days"),
    ("exposure", "노출도", "pct"),
    ("n_trades", "청산 거래 수", "int"),
    ("open_positions", "기말 미청산 포지션", "int"),
    ("total_cost", "총 비용", "money"),
]


def metrics_table(scenarios: list[CostScenario], default_bps: float, benchmark: str) -> pd.DataFrame:
    """행 = 지표, 열 = 비용 시나리오별 전략 + 기본 비용의 벤치마크. 아래에 SPY 대비 차이 행."""
    cols: dict[str, dict[str, str]] = {}
    for s in scenarios:
        cols[f"전략 {s.cost_bps:g}bp"] = {label: _fmt(s.metrics[k], f) for k, label, f in METRIC_ROWS}
    base = next((s for s in scenarios if s.cost_bps == default_bps), scenarios[0])
    cols[f"{benchmark} 바이앤홀드 {base.cost_bps:g}bp"] = {
        label: _fmt(base.benchmark_metrics[k], f) for k, label, f in METRIC_ROWS
    }
    table = pd.DataFrame(cols)

    diff_rows = {}
    for key, label, kind in (("cagr", "CAGR", "pp"), ("sharpe", "샤프비율", "num"), ("mdd", "MDD", "pp")):
        diff_rows[f"{benchmark} 대비 {label} 차이"] = {
            f"전략 {s.cost_bps:g}bp": _fmt(s.metrics[key] - s.benchmark_metrics[key], kind, signed=True)
            for s in scenarios
        }
    diff = pd.DataFrame(diff_rows).T.reindex(columns=table.columns, fill_value="")
    return pd.concat([table, diff])


def write_report(
    scenarios: list[CostScenario],
    strategy: Strategy,
    config: BacktestConfig,
    benchmark: str,
    out_dir: Path,
) -> Path:
    base = next((s for s in scenarios if s.cost_bps == config.cost_bps), scenarios[0])
    result, bench = base.result, base.benchmark
    table = metrics_table(scenarios, config.cost_bps, benchmark)

    fig_equity = _line_figure(
        {f"{strategy.name} ({base.cost_bps:g}bp)": result.equity, f"{benchmark} 바이앤홀드": bench.equity},
        title="자산곡선",
        y_title="자산가치 (USD, 로그 축)",
        log_y=True,
        number_format=",.0f",
    )
    fig_dd = _line_figure(
        {f"{strategy.name} ({base.cost_bps:g}bp)": drawdown(result.equity), f"{benchmark} 바이앤홀드": drawdown(bench.equity)},
        title="드로다운 (고점 대비 하락폭)",
        y_title="드로다운",
        log_y=False,
        number_format=".1%",
    )

    trades = result.trades.copy()
    for col in ("entry_date", "exit_date"):
        trades[col] = pd.to_datetime(trades[col]).dt.date
    open_pos = result.open_positions.copy()
    for col in ("entry_date", "mark_date"):
        open_pos[col] = pd.to_datetime(open_pos[col]).dt.date

    period = f"{result.start_date.date()} ~ {result.end_date.date()}"
    tickers = ", ".join(result.positions.columns)
    body = f"""
<header>
  <h1>{html.escape(strategy.name)} 백테스트</h1>
  <p class="sub">{html.escape(STRATEGY_DESCRIPTIONS.get(strategy.name, ""))}</p>
  <dl class="meta">
    <div><dt>기간</dt><dd>{period} (첫 체결일 ~ 마지막 평가일)</dd></div>
    <div><dt>초기자본</dt><dd>${config.initial_capital:,.0f}</dd></div>
    <div><dt>최대 보유</dt><dd>{result.max_positions}종목, 동일 비중</dd></div>
    <div><dt>기본 비용</dt><dd>편도 {config.cost_bps:g}bp</dd></div>
    <div><dt>대상</dt><dd>{html.escape(tickers)}</dd></div>
    <div><dt>생성</dt><dd>{datetime.now():%Y-%m-%d %H:%M}</dd></div>
  </dl>
</header>
<section>
  <h2>성과 지표 · 비용 민감도</h2>
  <p class="note">거래 통계(승률·손익비·보유일·연 거래 횟수)는 청산 완료 거래 기준. 샤프는 일간 수익률 기반, 무위험수익률 0.
  {html.escape(benchmark)} 바이앤홀드는 전략과 같은 첫 체결일·마지막 날로 계산.</p>
  {table.to_html(classes="metrics", border=0, escape=True)}
</section>
<section>{fig_equity.to_html(full_html=False, include_plotlyjs=True)}</section>
<section>{fig_dd.to_html(full_html=False, include_plotlyjs=False)}</section>
<section>
  <h2>기말 미청산 포지션 ({len(open_pos)}개, 마지막 종가 평가, 청산 비용 미반영)</h2>
  {_table_html(open_pos, {"entry_price": "{:.4f}", "mark_price": "{:.4f}", "shares": "{:.4f}", "unrealized_return": "{:.4%}", "entry_cost": "{:.2f}"})}
</section>
<section>
  <details>
    <summary>거래 내역 {len(trades)}건 ({base.cost_bps:g}bp, 수익률은 비용 반영: 청산가(1−c) / 진입가(1+c) − 1)</summary>
    {_table_html(trades, {"entry_price": "{:.4f}", "exit_price": "{:.4f}", "shares": "{:.4f}", "return": "{:.4%}", "gross_return": "{:.4%}", "entry_cost": "{:.2f}", "exit_cost": "{:.2f}"})}
  </details>
</section>
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{strategy.name}.html"
    path.write_text(_page(f"{strategy.name} 백테스트", body), encoding="utf-8")
    return path


def _line_figure(
    series: dict[str, pd.Series], title: str, y_title: str, log_y: bool, number_format: str
) -> go.Figure:
    fig = go.Figure()
    for (name, s), color in zip(series.items(), (SERIES_STRATEGY, SERIES_BENCHMARK)):
        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.to_numpy(),
                name=name,
                mode="lines",
                line={"color": color, "width": 2},
                hovertemplate=f"%{{y:{number_format}}}<extra>{html.escape(name)}</extra>",
            )
        )
    axis = {"gridcolor": GRID, "linecolor": GRID, "tickfont": {"color": INK_MUTED}, "zeroline": False}
    fig.update_layout(
        title={"text": title, "font": {"color": INK_PRIMARY, "size": 16}, "x": 0, "xanchor": "left"},
        height=420,
        margin={"l": 60, "r": 20, "t": 60, "b": 40},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"family": "system-ui, -apple-system, 'Segoe UI', sans-serif", "color": INK_SECONDARY},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1},
        xaxis={**axis, "showspikes": True, "spikecolor": INK_MUTED, "spikethickness": 1, "spikemode": "across"},
        yaxis={**axis, "title": {"text": y_title}, "tickformat": number_format, "type": "log" if log_y else "linear"},
    )
    return fig


def _table_html(df: pd.DataFrame, formats: dict[str, str]) -> str:
    if df.empty:
        return "<p class='note'>없음</p>"
    shown = df.copy()
    for col, fmt in formats.items():
        if col in shown:
            shown[col] = shown[col].map(lambda v, f=fmt: f.format(v))
    return shown.to_html(index=False, classes="data", border=0)


def _fmt(value: float, kind: str, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    sign = "+" if signed else ""
    if kind == "pct":
        return f"{value:{sign}.2%}"
    if kind == "pp":
        return f"{value * 100:{sign}.2f}%p"
    if kind == "num":
        return f"{value:{sign}.2f}"
    if kind == "days":
        return f"{value:.1f}"
    if kind == "int":
        return f"{value:.0f}"
    if kind == "money":
        return f"${value:,.0f}"
    raise ValueError(kind)


def _page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  :root {{
    --surface: {SURFACE}; --page: #f9f9f7; --ink: {INK_PRIMARY}; --ink-2: {INK_SECONDARY};
    --muted: {INK_MUTED}; --grid: {GRID}; --border: rgba(11,11,11,0.10);
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--page); color: var(--ink);
         font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }}
  main {{ max-width: 1100px; margin: 0 auto; padding: 24px 16px 64px; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  h2 {{ font-size: 16px; margin: 0 0 8px; }}
  .sub, .note {{ color: var(--ink-2); margin: 0 0 12px; }}
  .note {{ font-size: 12px; }}
  section, header {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
                     padding: 16px; margin-bottom: 16px; overflow-x: auto; }}
  .meta {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 4px 16px; margin: 12px 0 0; }}
  .meta div {{ display: flex; gap: 8px; }}
  .meta dt {{ color: var(--muted); min-width: 64px; }}
  .meta dd {{ margin: 0; }}
  table {{ border-collapse: collapse; font-variant-numeric: tabular-nums; }}
  th, td {{ padding: 6px 10px; border-bottom: 1px solid var(--grid); text-align: right; white-space: nowrap; }}
  th {{ color: var(--ink-2); font-weight: 600; }}
  table.metrics tbody th {{ text-align: left; }}
  table.data {{ font-size: 12px; }}
  summary {{ cursor: pointer; font-weight: 600; }}
</style>
</head>
<body><main>{body}</main></body>
</html>
"""
