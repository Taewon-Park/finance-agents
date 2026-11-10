"""2단계 P0~P3 배분 리포트 (reports/stage2_alloc.html, .md). 첫 줄에 서술용임을 적는다. 해석 문단은 쓰지 않는다."""

from __future__ import annotations

import html
import math
from datetime import date
from pathlib import Path

import pandas as pd

from swinglab.alloc.backtest import PolicyRun
from swinglab.alloc.config import AllocConfig
from swinglab.backtest.metrics import drawdown
from swinglab.reports.report import render_page
from swinglab.reports.stage1 import _lines

DISCLAIMER = (
    "서술용 백테스트, 검증 아님. 2019년 이후 구간은 1단계에서 이미 소진됐고, "
    "정책도 시장 흐름을 아는 상태에서 정했다."
)


def goal_verdict(runs: list[PolicyRun], cfg: AllocConfig) -> str:
    g = cfg.goal
    run = _find(runs, g.policy, g.cost_bps)
    mdd = run.metrics["mdd"]
    ok = mdd >= g.max_drawdown
    return (
        f"사전 고정 목표 판정: {g.policy} ({g.cost_bps:g}bp) MDD {_pct(mdd)} → "
        f"목표 {_pct(g.max_drawdown)} 이내 {'달성' if ok else '미달'}"
    )


def _find(runs: list[PolicyRun], policy: str, cost: float) -> PolicyRun:
    for r in runs:
        if r.policy.name == policy and r.cost_bps == cost:
            return r
    raise KeyError(f"{policy} {cost:g}bp 결과가 없습니다")


def metrics_frame(runs: list[PolicyRun], cost: float) -> pd.DataFrame:
    rows = []
    for r in runs:
        if r.cost_bps != cost:
            continue
        m = r.metrics
        rows.append({
            "정책": r.policy.name, "CAGR": _pct(m["cagr"]), "변동성": _pct(m["vol"], signed=False),
            "샤프": _num(m["sharpe"]), "MDD": _pct(m["mdd"]), "MDD 고점→저점": f"{r.mdd_window[0]} → {r.mdd_window[1]}",
            "세후 CAGR(추정)": _pct(r.after_tax_cagr), "BIL 평균 비중": _pct(r.avg_cash_proxy, signed=False),
            "매매 건수": f"{m['n_trades']:.0f}", "연 매매": f"{m['trades_per_year']:.1f}",
            "총 비용": f"${m['total_cost']:,.0f}",
        })
    return pd.DataFrame(rows)


def yearly_frame(runs: list[PolicyRun], cost: float) -> pd.DataFrame:
    sel = {r.policy.name: r.yearly for r in runs if r.cost_bps == cost}
    df = pd.DataFrame(sel)
    first, last = df.index.min(), df.index.max()
    df.index = [f"{y}{' (3월~)' if y == first else ''}{' (연초 이후)' if y == last else ''}" for y in df.index]
    return df.map(_pct).rename_axis("연도").reset_index()


def periods_frame(runs: list[PolicyRun]) -> pd.DataFrame:
    rows = []
    for r in runs:
        for p in r.periods.to_dict("records"):
            rows.append({"구간": f"{p['period']} ({p['start'].date()} ~ {p['end'].date()})", "정책": r.policy.name,
                         "비용": f"{r.cost_bps:g}bp", "수익률": _pct(p["return"]), "구간 MDD": _pct(p["mdd"])})
    return pd.DataFrame(rows)


def quarterly_frame(runs: list[PolicyRun], cost: float) -> pd.DataFrame:
    sel = {r.policy.name: r.quarterly_trades for r in runs if r.cost_bps == cost}
    df = pd.DataFrame(sel).fillna(0).astype(int).sort_index()
    return df.rename_axis("분기").reset_index()


def quarterly_summary(runs: list[PolicyRun], cost: float, n_quarters: int) -> pd.DataFrame:
    rows = []
    for r in runs:
        if r.cost_bps != cost:
            continue
        q = r.quarterly_trades
        rows.append({"정책": r.policy.name, "매매한 분기": f"{len(q)} / {n_quarters}",
                     "분기당 평균": f"{q.sum() / n_quarters:.2f}", "최대": f"{q.max() if len(q) else 0}"})
    return pd.DataFrame(rows)


def tax_frame(runs: list[PolicyRun], cost: float) -> pd.DataFrame:
    rows = []
    for r in runs:
        if r.cost_bps != cost:
            continue
        t = r.taxes
        rows.append({
            "정책": r.policy.name, "실현 차익 합계(원)": f"{int(t['gain_krw'].sum()):,}" if len(t) else "0",
            "추정 세액 합계(원)": f"{int(t['tax_krw'].sum()):,}" if len(t) else "0",
            "세금 낸 해": f"{int((t['tax_krw'] > 0).sum())}" if len(t) else "0",
            "CAGR": _pct(r.metrics["cagr"]), "세후 CAGR(추정)": _pct(r.after_tax_cagr),
        })
    return pd.DataFrame(rows)


def tax_yearly_frame(runs: list[PolicyRun], cost: float) -> pd.DataFrame:
    frames = []
    for r in runs:
        if r.cost_bps != cost or r.taxes.empty:
            continue
        t = r.taxes.set_index("year")
        frames.append(pd.DataFrame({f"{r.policy.name} 차익": t["gain_krw"], f"{r.policy.name} 세액": t["tax_krw"]}))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1).fillna(0).astype(int).sort_index()
    return df.map(lambda v: f"{v:,}").rename_axis("연도").reset_index()


def policy_frame(cfg: AllocConfig) -> pd.DataFrame:
    rows = []
    for name, p in cfg.policies.items():
        w = " / ".join(f"{t} {v * 100:g}" for t, v in p.weights.items())
        trend = (f"+ 추세 필터 ({', '.join(cfg.trend.assets)}: 월말 종가 < {cfg.trend.months}개월 평균이면 BIL로)"
                 if p.trend_filter else "")
        rows.append({"정책": name, "비중(%)": f"{w} {trend}".strip()})
    return pd.DataFrame(rows)


def write_alloc_report(
    runs: list[PolicyRun], cfg: AllocConfig, out_dir: Path, data_end: date, fx_end: date, offline: bool = False
) -> tuple[Path, Path]:
    rep = cfg.backtest.report_cost_bps
    any_run = _find(runs, cfg.goal.policy, rep)
    start, end = any_run.result.start_date.date(), any_run.result.end_date.date()
    n_quarters = len(pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="Q"))
    verdict = goal_verdict(runs, cfg)

    intro = (
        f"첫 체결일 {start}, 마지막 날 {end} (가격 데이터 끝 {data_end}, 환율 끝 {fx_end}). "
        f"초기자본 ${cfg.backtest.initial_capital:,.0f}, 분기 재조정 허용 범위 {cfg.band * 100:g}%p, "
        f"남는 비중은 {cfg.cash_proxy}. 시도 {len(cfg.policies)}개 / 한도 {cfg.max_trials}개."
    )
    md = [DISCLAIMER, "", verdict, "", f"# 2단계 자산배분 P0~P3 ({start} ~ {end})", "", intro, "",
          "## 정책", "", _md(policy_frame(cfg))]
    for c in cfg.backtest.cost_bps:
        md += ["", f"## 지표 ({c:g}bp)", "", _md(metrics_frame(runs, c))]
    for c in cfg.backtest.cost_bps:
        md += ["", f"## 연도별 수익률 ({c:g}bp)", "", _md(yearly_frame(runs, c))]
    md += ["", "## 하락장 구간", "", _md(periods_frame(runs))]
    md += ["", f"## 분기별 실제 매매 횟수 요약 ({rep:g}bp, 전체 {n_quarters}개 분기)", "",
           _md(quarterly_summary(runs, rep, n_quarters))]
    md += ["", f"## 세금 추정 ({rep:g}bp)", "", _md(tax_frame(runs, rep)), "", _md(tax_yearly_frame(runs, rep))]
    md += ["", "## 가정과 한계", "", *[f"- {n}" for n in _notes(cfg)]]
    text = "\n".join(md) + "\n"

    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "stage2_alloc.md"
    md_path.write_text(text, encoding="utf-8")

    plotly_js: str | bool = True if offline else "cdn"
    sel = [r for r in runs if r.cost_bps == rep]
    norm = {r.policy.name: r.result.equity / r.result.equity.iloc[0] for r in sel}
    dd = {r.policy.name: drawdown(r.result.equity) for r in sel}
    eq_fig = _lines(norm, f"자산가치 ({rep:g}bp, 시작=1)", "자산가치 (로그 축)", True, ".2f")
    dd_fig = _lines(dd, f"드로다운 ({rep:g}bp)", "드로다운", False, ".1%")
    sections = [
        (
            f"<header><p><strong>{html.escape(DISCLAIMER)}</strong></p><p><strong>{html.escape(verdict)}</strong></p>"
            f"<h1>2단계 자산배분 P0~P3</h1><p class='sub'>{html.escape(intro)}</p></header>"
        ),
        _section("정책", policy_frame(cfg)),
        *[_section(f"지표 ({c:g}bp)", metrics_frame(runs, c)) for c in cfg.backtest.cost_bps],
        f"<section>{eq_fig.to_html(full_html=False, include_plotlyjs=plotly_js)}</section>",
        f"<section>{dd_fig.to_html(full_html=False, include_plotlyjs=False)}</section>",
        *[_section(f"연도별 수익률 ({c:g}bp)", yearly_frame(runs, c)) for c in cfg.backtest.cost_bps],
        _section("하락장 구간", periods_frame(runs)),
        _section(f"분기별 실제 매매 횟수 ({rep:g}bp)", quarterly_frame(runs, rep)),
        _section(f"세금 추정 ({rep:g}bp, 참고용)", tax_frame(runs, rep)),
        _section("연도별 실현 차익·추정 세액 (원)", tax_yearly_frame(runs, rep)),
        "<section><h2>가정과 한계</h2><ul>" + "".join(f"<li>{html.escape(n)}</li>" for n in _notes(cfg)) + "</ul></section>",
    ]
    html_path = out_dir / "stage2_alloc.html"
    html_path.write_text(render_page("2단계 자산배분", "".join(sections)), encoding="utf-8")
    return html_path, md_path


def _notes(cfg: AllocConfig) -> list[str]:
    return [
        "가격은 배당·분할을 반영한 수정주가(총수익). 배당소득세(미국 원천징수 15%)는 넣지 않았다.",
        "체결: 신호일 다음 거래일 시가, 비용은 체결금액 × 편도 비용률. 소수 주식 허용.",
        (
            f"현금 대용({cfg.cash_proxy})은 허용 범위 점검 대상이 아니고, 매매 후 남거나 모자란 금액을 흡수한다. "
            "그래서 범위 안 자산을 다시 사지 않는 만큼 목표(5%)보다 커질 수 있다 (표의 평균 비중). "
            "남는 현금이 총자산의 0.1% 미만이면 현금(수익률 0)으로 둔다."
        ),
        "P3 추세 필터: 월말 종가와 최근 10개 월말 종가 평균을 비교. 상태가 바뀐 자산만 허용 범위와 무관하게 매매.",
        (
            "세금 추정은 참고용이며 실제 신고와 다를 수 있다. 수정주가로 계산해 취득가가 실제보다 낮게 잡히므로 "
            "(배당이 차익에 섞임) 양도차익과 세액이 실제보다 크게 나올 수 있다. 수수료는 필요경비로 빼지 않았다. "
            f"기본공제(연 {cfg.tax.deduction_krw:,}원)가 고정이라 세액과 세후 CAGR은 계좌 크기에 따라 달라진다 "
            f"(여기서는 초기자본 ${cfg.backtest.initial_capital:,.0f})."
        ),
        "세후 CAGR: 그해 추정 세액을 그해 마지막 거래일에 포트폴리오에서 뺀 것으로 본 근사 (실제 납부는 다음 해 5월).",
        "연도별 수익률의 첫 해는 3월 이후, 마지막 해는 연초 이후 부분 연도.",
    ]


def _section(title: str, df: pd.DataFrame) -> str:
    table = df.to_html(index=False, classes="data", border=0, escape=True) if not df.empty else "<p class='note'>없음</p>"
    return f"<section><h2>{html.escape(title)}</h2>{table}</section>"


def _md(df: pd.DataFrame) -> str:
    if df.empty:
        return "(없음)"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _pct(v: float, signed: bool = True) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{v:+.2%}" if signed else f"{v:.2%}"


def _num(v: float) -> str:
    return "–" if v is None or math.isnan(v) else f"{v:.2f}"
