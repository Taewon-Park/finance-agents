"""1단계 리포트 (HTML 한 장). 백테스트를 다시 돌리지 않고 results/ 파일만 읽는다.

- 학습 구간: 계열별 전체 조합 결과, 시도 횟수, 샤프 분포, 선정 근거
- 검증 구간: holdout.lock이 있을 때만 후보와 SPY 바이앤홀드 비교 (비용별), 학습 대비 샤프 하락폭
- 후보 + SPY 일간 수익률 상관행렬 (학습 구간, 선정 비용)
- 후보 자산곡선·드로다운
해석 문단은 쓰지 않는다.
"""

from __future__ import annotations

import html
import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from swinglab.backtest.metrics import drawdown
from swinglab.research.config import Stage1Config
from swinglab.research.holdout import read_lock
from swinglab.research.log import BASELINE_FAMILY, latest_runs, read_log, summarize, trial_keys
from swinglab.research.runner import benchmarks_path, read_curves
from swinglab.research.selection import IncompleteExperiments, read_candidates, select_all
from swinglab.research.specs import StrategySpec
from swinglab.reports.report import GRID, INK_MUTED, INK_PRIMARY, INK_SECONDARY, SURFACE, render_page

# dataviz 기본 팔레트 범주 슬롯 순서 (라이트). 벤치마크는 회색 잉크.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
BENCH_COLOR = INK_MUTED

METRICS = [("sharpe", "샤프", "num"), ("cagr", "CAGR", "pct"), ("vol", "변동성", "pct"), ("mdd", "MDD", "pct"),
           ("win_rate", "승률", "pct"), ("payoff", "손익비", "num"), ("avg_hold", "평균 보유일", "d1"),
           ("trades_per_year", "연 거래", "d1"), ("exposure", "노출도", "pct"), ("n_trades", "거래 수", "int")]


def write_stage1_report(
    cfg: Stage1Config, results_dir: Path, candidates_file: Path, out_dir: Path, offline: bool = False
) -> Path:
    log = read_log(results_dir)
    if log.empty:
        raise FileNotFoundError(f"{results_dir}에 실험 로그가 없습니다. swinglab experiments run부터 실행하세요.")
    curves = read_curves(results_dir)
    bench = pd.read_parquet(benchmarks_path(results_dir)) if benchmarks_path(results_dir).exists() else pd.DataFrame()
    candidates = read_candidates(candidates_file) if candidates_file.exists() else []
    lock = read_lock(results_dir)
    sel_bp = cfg.selection_cost_bps
    plotly_js: str | bool = True if offline else "cdn"

    sections = [_header(cfg, log, candidates_file, candidates, lock)]
    sections.append(_train_section(cfg, log, bench))
    if candidates:
        train_ids = {_spec_key(s): _run_id(log, s, "train", sel_bp) for s in candidates}
        sections.append(_correlation_section(candidates, train_ids, curves, f"benchmark:train:{sel_bp:g}", cfg))
        fig_html = _curve_section("학습 구간 자산곡선·드로다운", candidates, train_ids, curves,
                                  f"benchmark:train:{sel_bp:g}", sel_bp, plotly_js)
        plotly_js = False if fig_html else plotly_js
        sections.append(fig_html)
    sections.append(_holdout_section(cfg, log, bench, candidates, lock, curves, plotly_js))

    body = "\n".join(s for s in sections if s)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "stage1.html"
    path.write_text(render_page("1단계 전략 비교", body), encoding="utf-8")
    return path


# ---------------------------------------------------------------- 섹션


def _header(cfg: Stage1Config, log: pd.DataFrame, cand_file: Path, candidates: list[StrategySpec], lock: list[dict]) -> str:
    n_trials = len(trial_keys(log))
    holdout_state = (
        f"실행됨 {len(lock)}회 (마지막 {html.escape(lock[-1]['timestamp'])}, forced={lock[-1]['forced']})" if lock else "미실행"
    )
    return f"""
<header>
  <h1>1단계 전략 비교</h1>
  <dl class="meta">
    <div><dt>학습 구간</dt><dd>{cfg.train.start} ~ {cfg.train.end}</dd></div>
    <div><dt>검증 구간</dt><dd>{cfg.holdout.start} ~ {cfg.holdout.end or '최신'}</dd></div>
    <div><dt>비용</dt><dd>{', '.join(f'{c:g}bp' for c in cfg.cost_bps)} (선정 기준 {cfg.selection_cost_bps:g}bp)</dd></div>
    <div><dt>시도 수</dt><dd>{n_trials} / 전체 예산 {cfg.budget.total} (계열당 {cfg.budget.per_family})</dd></div>
    <div><dt>후보 파일</dt><dd>{html.escape(str(cand_file))} ({len(candidates)}개)</dd></div>
    <div><dt>검증 실행</dt><dd>{holdout_state}</dd></div>
    <div><dt>생성</dt><dd>{datetime.now():%Y-%m-%d %H:%M}</dd></div>
  </dl>
</header>"""


def _train_section(cfg: Stage1Config, log: pd.DataFrame, bench: pd.DataFrame) -> str:
    parts = ["<section><h2>학습 구간 · 계열별 시도 횟수와 샤프 분포</h2>"]
    summary = summarize(log, cfg)
    parts.append(summary.to_html(index=False, border=0, float_format=lambda v: f"{v:.2f}", classes="data"))
    if not bench.empty:
        b = bench[bench["period"] == "train"].sort_values("cost_bp")
        if not b.empty:
            parts.append(f"<h3>SPY 바이앤홀드 (학습 구간 {b['window_start'].iloc[0]} ~ {b['window_end'].iloc[0]})</h3>")
            parts.append(_metric_table(b, "cost_bp", lambda v: f"{v:g}bp"))
    parts.append("</section>")

    try:
        selections = select_all(log, cfg)
    except IncompleteExperiments as exc:
        selections = {}
        parts.append(f"<section><p class='note'>후보 선정 전: {html.escape(str(exc))}</p></section>")

    train = latest_runs(log, "train")
    for fam_name, fam in cfg.families.items():
        rows = train[train["family"] == fam_name]
        if fam.is_filter_family:
            rows = pd.concat([rows, train[train["family"] == BASELINE_FAMILY]])
        if rows.empty:
            continue
        n_trials = sum(1 for k in trial_keys(log) if k[0] == fam_name)
        parts.append(f"<section><h2>{fam_name} 계열 · 학습 구간 전체 조합 (시도 {n_trials}회)</h2>")
        parts.append(_combo_table(rows, cfg))
        sel = selections.get(fam_name)
        if sel is not None:
            if not sel.table.empty:
                shown = sel.table.drop(columns=[c for c in ("idx", "order") if c in sel.table.columns])
                parts.append(f"<h3>선정 근거 ({cfg.selection_cost_bps:g}bp)</h3>")
                parts.append(shown.to_html(index=False, border=0, float_format=lambda v: f"{v:.3f}", classes="data"))
            chosen = sel.candidate
            parts.append(
                f"<p class='note'>후보: {html.escape(_spec_label(chosen)) if chosen else '없음'}</p>"
                + "".join(f"<p class='note'>{html.escape(n)}</p>" for n in sel.notes)
            )
        parts.append("</section>")
    return "\n".join(parts)


def _combo_table(rows: pd.DataFrame, cfg: Stage1Config) -> str:
    """조합별 행, 비용별 샤프 + 선정 비용의 주요 지표."""
    sel_bp = cfg.selection_cost_bps
    out = []
    for (fam, strat, params), g in rows.groupby(["family", "strategy", "params"], sort=False):
        by_cost = g.set_index("cost_bp")
        rec = {"계열": fam, "전략": strat, "파라미터": _params_label(params)}
        for c in cfg.cost_bps:
            rec[f"샤프 {c:g}bp"] = _fmt(by_cost["sharpe"].get(c, math.nan), "num")
        s = by_cost.loc[sel_bp] if sel_bp in by_cost.index else None
        for key, label, kind in METRICS[1:]:
            rec[f"{label} ({sel_bp:g}bp)"] = _fmt(s[key], kind) if s is not None else "–"
        rec["구간"] = f"{g['window_start'].iloc[0]} ~ {g['window_end'].iloc[0]}"
        out.append(rec)
    return pd.DataFrame(out).to_html(index=False, border=0, classes="data")


def _correlation_section(
    candidates: list[StrategySpec], run_ids: dict, curves: pd.DataFrame, bench_id: str, cfg: Stage1Config
) -> str:
    rets = {}
    for spec in candidates:
        rid = run_ids.get(_spec_key(spec))
        if rid:
            rets[_spec_label(spec)] = _returns(curves, rid)
    if bench_id in set(curves["run_id"]):
        rets["SPY 바이앤홀드"] = _returns(curves, bench_id)
    if len(rets) < 2:
        return ""
    frame = pd.DataFrame(rets).dropna()
    corr = frame.corr()
    return (
        f"<section><h2>일간 수익률 상관행렬 (학습 구간, {cfg.selection_cost_bps:g}bp, 공통 {len(frame)}거래일)</h2>"
        + corr.to_html(border=0, float_format=lambda v: f"{v:.2f}", classes="data")
        + "</section>"
    )


def _curve_section(
    title: str, specs: list[StrategySpec], run_ids: dict, curves: pd.DataFrame, bench_id: str, cost: float,
    plotly_js: str | bool,
) -> str:
    series = {}
    for spec in specs:
        rid = run_ids.get(_spec_key(spec))
        if rid:
            series[_spec_label(spec)] = _curve(curves, rid)
    if bench_id in set(curves["run_id"]):
        series["SPY 바이앤홀드"] = _curve(curves, bench_id)
    if not series:
        return ""
    norm = {k: v / v.iloc[0] for k, v in series.items()}
    dd = {k: drawdown(v) for k, v in series.items()}
    eq_fig = _lines(norm, f"{title} ({cost:g}bp, 시작=1)", "자산가치 (시작=1, 로그 축)", True, ".2f")
    dd_fig = _lines(dd, "드로다운", "드로다운", False, ".1%")
    return (
        f"<section>{eq_fig.to_html(full_html=False, include_plotlyjs=plotly_js)}</section>"
        f"<section>{dd_fig.to_html(full_html=False, include_plotlyjs=False)}</section>"
    )


def _holdout_section(
    cfg: Stage1Config, log: pd.DataFrame, bench: pd.DataFrame, candidates: list[StrategySpec], lock: list[dict],
    curves: pd.DataFrame, plotly_js: str | bool,
) -> str:
    if not lock:
        return "<section><h2>검증 구간</h2><p class='note'>미실행. swinglab holdout으로만 실행한다.</p></section>"
    parts = [f"<section><h2>검증 구간 · 후보와 SPY 바이앤홀드 ({cfg.holdout.start} ~ {cfg.holdout.end or '최신'})</h2>"]
    parts.append(
        "<p class='note'>실행 기록: "
        + "; ".join(f"{html.escape(e['timestamp'])} forced={e['forced']} commit={html.escape(e['git_commit'][:12])}" for e in lock)
        + "</p>"
    )
    hold = latest_runs(log, "holdout")
    for c in cfg.cost_bps:
        rows = []
        for spec in candidates:
            r = hold[(hold["family"] == spec.family) & (hold["strategy"] == spec.display_name)
                     & (hold["params"] == spec.params_json) & (hold["cost_bp"] == c)]
            if not r.empty:
                rows.append(r.iloc[0].to_dict() | {"label": _spec_label(spec)})
        b = bench[(bench["period"] == "holdout") & (bench["cost_bp"] == c)] if not bench.empty else pd.DataFrame()
        if not b.empty:
            rows.append(b.iloc[0].to_dict() | {"label": "SPY 바이앤홀드"})
        if rows:
            parts.append(f"<h3>{c:g}bp</h3>")
            parts.append(_metric_table(pd.DataFrame(rows), "label", str))

    sel_bp = cfg.selection_cost_bps
    train = latest_runs(log, "train", sel_bp)
    drop_rows = []
    for spec in candidates:
        key = (train["family"] == spec.family) & (train["strategy"] == spec.display_name) & (train["params"] == spec.params_json)
        hkey = (hold["family"] == spec.family) & (hold["strategy"] == spec.display_name) & (hold["params"] == spec.params_json) & (hold["cost_bp"] == sel_bp)
        if key.any() and hkey.any():
            st, sh = float(train.loc[key, "sharpe"].iloc[0]), float(hold.loc[hkey, "sharpe"].iloc[0])
            drop_rows.append({"후보": _spec_label(spec), "학습 샤프": f"{st:.2f}", "검증 샤프": f"{sh:.2f}", "차이 (검증 − 학습)": f"{sh - st:+.2f}"})
    if drop_rows:
        parts.append(f"<h3>학습 대비 검증 샤프 ({sel_bp:g}bp)</h3>")
        parts.append(pd.DataFrame(drop_rows).to_html(index=False, border=0, classes="data"))
    parts.append("</section>")

    hold_ids = {_spec_key(s): _run_id(log, s, "holdout", sel_bp) for s in candidates}
    parts.append(_curve_section("검증 구간 자산곡선·드로다운", candidates, hold_ids, curves,
                                f"benchmark:holdout:{sel_bp:g}", sel_bp, plotly_js))
    return "\n".join(parts)


# ---------------------------------------------------------------- 도우미


def _metric_table(rows: pd.DataFrame, label_col: str, label_fmt) -> str:
    out = []
    for _, r in rows.iterrows():
        rec = {"": label_fmt(r[label_col])}
        for key, label, kind in METRICS:
            rec[label] = _fmt(r.get(key, math.nan), kind)
        out.append(rec)
    return pd.DataFrame(out).to_html(index=False, border=0, classes="data")


def _lines(series: dict[str, pd.Series], title: str, y_title: str, log_y: bool, fmt: str) -> go.Figure:
    fig = go.Figure()
    colors = iter(SERIES)
    for name, s in series.items():
        color = BENCH_COLOR if name.startswith("SPY") else next(colors)
        fig.add_trace(go.Scatter(x=s.index, y=s.to_numpy(), name=name, mode="lines", line={"color": color, "width": 2},
                                 hovertemplate=f"%{{y:{fmt}}}<extra>{html.escape(name)}</extra>"))
    axis = {"gridcolor": GRID, "linecolor": GRID, "tickfont": {"color": INK_MUTED}, "zeroline": False}
    fig.update_layout(
        title={"text": title, "font": {"color": INK_PRIMARY, "size": 16}, "x": 0, "xanchor": "left"},
        height=460, margin={"l": 60, "r": 20, "t": 50, "b": 90}, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font={"family": "system-ui, -apple-system, 'Segoe UI', sans-serif", "color": INK_SECONDARY},
        hovermode="x unified", legend={"orientation": "h", "yanchor": "top", "y": -0.12, "xanchor": "left", "x": 0},
        xaxis={**axis, "showspikes": True, "spikecolor": INK_MUTED, "spikethickness": 1, "spikemode": "across"},
        yaxis={**axis, "title": {"text": y_title}, "tickformat": fmt, "type": "log" if log_y else "linear"},
    )
    return fig


def _curve(curves: pd.DataFrame, run_id: str) -> pd.Series:
    c = curves[curves["run_id"] == run_id]
    return pd.Series(c["equity"].to_numpy(), index=pd.to_datetime(c["date"]))


def _returns(curves: pd.DataFrame, run_id: str) -> pd.Series:
    return _curve(curves, run_id).pct_change().iloc[1:]


def _run_id(log: pd.DataFrame, spec: StrategySpec, period: str, cost: float) -> str | None:
    rows = latest_runs(log, period, cost)
    r = rows[(rows["family"] == spec.family) & (rows["strategy"] == spec.display_name) & (rows["params"] == spec.params_json)]
    return str(r["run_id"].iloc[0]) if not r.empty else None


def _spec_key(spec: StrategySpec) -> tuple[str, str, str]:
    return (spec.family, spec.display_name, spec.params_json)


def _spec_label(spec: StrategySpec | None) -> str:
    if spec is None:
        return "없음"
    params = ", ".join(f"{k}={v}" for k, v in spec.params.items() if k != "safe")
    return f"{spec.family}: {spec.display_name}" + (f" ({params})" if params else "")


def _params_label(params_json: str) -> str:
    params = json.loads(params_json)
    return ", ".join(f"{k}={v}" for k, v in params.items() if k != "safe") or "기본값"


def _fmt(value, kind: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    return {"pct": f"{value:.2%}", "num": f"{value:.2f}", "d1": f"{value:.1f}", "int": f"{value:.0f}"}[kind]
