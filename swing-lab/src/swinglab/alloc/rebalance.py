"""실사용 모드: 분기 리밸런싱 주문 목록 (swinglab rebalance). 실제 계좌와 연결하지 않고 사람이 주문한다.

처리 순서 (stage2-spec.md)
1. 최신 종가로 현재 비중
2. 정책(기본 P3)의 이번 분기 목표 비중 (추세 필터는 마지막 월말 기준, 목표 시점 조절은 설정 시)
3. 납입금 우선 배분: 매도 없이, 가장 많이 모자란 자산부터 채운다 (남는 최대 부족분이 가장 작아지게)
4. 그래도 허용 범위(5%p)를 벗어난 자산만 매도·매수 (정수 주). 추세 필터로 목표 0이 된 보유 자산은 범위와 무관하게 매도
5. 매도분의 예상 실현 차익과 올해 누적 기준 예상 세액
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from swinglab.alloc.config import AllocConfig
from swinglab.alloc.policies import Policy, apply_glide, trend_status
from swinglab.alloc.tax import TaxLedger, tax_due
from swinglab.backtest.weights_engine import BAND_TOLERANCE
from swinglab.pipeline.fx import fx_on

HOLDINGS_COLUMNS = ["ticker", "shares", "avg_cost_usd", "avg_fx"]


@dataclass
class Order:
    ticker: str
    side: str  # buy | sell
    shares: int
    price: float
    reason: str

    @property
    def usd(self) -> float:
        return self.shares * self.price


@dataclass
class RebalancePlan:
    as_of: pd.Timestamp
    policy: str
    fx: float
    fx_date: pd.Timestamp
    contribution_krw: int
    cost_bps: float
    table: pd.DataFrame
    orders: list[Order]
    trend: pd.DataFrame
    tax: dict[str, int]
    cash_left_usd: float
    glide_applied: bool
    notes: list[str] = field(default_factory=list)

    @property
    def quarter_label(self) -> str:
        return f"{self.as_of.year}-Q{(self.as_of.month - 1) // 3 + 1}"


def load_holdings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"ticker": str})
    missing = [c for c in HOLDINGS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: 컬럼이 없습니다 {missing} (필요: {HOLDINGS_COLUMNS})")
    df = df[HOLDINGS_COLUMNS].copy()
    df["ticker"] = df["ticker"].str.strip().str.upper()
    for c in ("shares", "avg_cost_usd", "avg_fx"):
        df[c] = pd.to_numeric(df[c], errors="raise").astype(float)
    if (df[["shares", "avg_cost_usd", "avg_fx"]] < 0).any().any():
        raise ValueError(f"{path}: 음수 값이 있습니다")
    if df["ticker"].duplicated().any():
        raise ValueError(f"{path}: 같은 티커가 두 번 있습니다")
    return df


def plan_rebalance(
    holdings: pd.DataFrame,
    prices: pd.DataFrame,
    fx: pd.Series,
    cfg: AllocConfig,
    policy_name: str | None = None,
    contribution_krw: int = 0,
    realized_ytd_krw: int = 0,
) -> RebalancePlan:
    policy = Policy.from_config(policy_name or cfg.live.policy, cfg)
    tickers = list(dict.fromkeys([*policy.tickers, *holdings["ticker"]]))
    px = prices[prices["ticker"].isin(tickers)]
    last = px.groupby("ticker")["date"].max()
    missing = sorted(set(tickers) - set(last.index))
    if missing:
        raise ValueError(f"가격 데이터에 없는 종목: {', '.join(missing)}")
    as_of = pd.Timestamp(last.min())
    close = px[px["date"] == as_of].set_index("ticker")["close"].reindex(tickers)
    if close.isna().any():
        raise ValueError(f"{as_of.date()} 종가가 없는 종목: {', '.join(close[close.isna()].index)}")
    fx_rate = float(fx_on(fx, pd.DatetimeIndex([as_of])).iloc[0])
    fx_date = fx.index[fx.index <= as_of].max()
    cost = cfg.live.cost_bps / 10_000.0

    shares = holdings.set_index("ticker")["shares"].reindex(tickers).fillna(0.0)
    values = shares * close
    contribution_usd = contribution_krw / fx_rate
    total = float(values.sum()) + contribution_usd

    trend = trend_status(px, policy, as_of) if policy.trend_filter else pd.DataFrame()
    below = dict(zip(trend["asset"], trend["below"])) if not trend.empty else {}
    raw_target = policy.targets(below)
    target_w = apply_glide(raw_target, as_of.date(), cfg.live.glide_path, cfg.live.target_date)
    target = pd.Series({t: target_w.get(t, 0.0) for t in tickers})
    notes = []
    if target_w != raw_target:
        notes.append("목표 시점 대비 위험 조절(glide_path)이 켜져 있어 주식 비중을 줄인 목표를 썼다.")

    orders: dict[tuple[str, str], Order] = {}

    def add(ticker: str, side: str, n: int, reason: str) -> None:
        if n <= 0:
            return
        key = (ticker, side)
        if key in orders:
            orders[key].shares += n
            if reason not in orders[key].reason:
                orders[key].reason += f", {reason}"
        else:
            orders[key] = Order(ticker, side, n, float(close[ticker]), reason)

    # 3. 납입금 우선 배분 (매수만)
    buy_amount = _water_fill((target * total - values).clip(lower=0.0), contribution_usd, target)
    held = shares.copy()
    cash = contribution_usd
    unit = close * (1.0 + cost)
    for t, amount in buy_amount.items():
        n = math.floor(amount / unit[t] + 1e-9)
        if n > 0:
            add(t, "buy", n, "납입금")
            held[t] += n
            cash -= n * unit[t]
    while True:  # 남은 납입금으로 가장 모자란 자산을 한 주씩
        gap = target * total - held * close
        affordable = gap[(unit <= cash + 1e-9) & (gap > 0)]
        if affordable.empty:
            break
        t = affordable.idxmax()
        add(t, "buy", 1, "납입금")
        held[t] += 1
        cash -= unit[t]

    # 4. 허용 범위 밖 자산만 매매 (현금 대용은 남는 돈을 받는 쪽)
    proxy = cfg.cash_proxy
    total_after = float((held * close).sum()) + cash
    weight = held * close / total_after
    trend_exit = {a for a, b in below.items() if b and held.get(a, 0) > 0}
    chosen = [
        t for t in tickers
        if t != proxy and (abs(weight[t] - target[t]) >= cfg.band - BAND_TOLERANCE or t in trend_exit)
    ]
    for t in chosen:  # 매도 먼저
        excess = held[t] * close[t] - target[t] * total_after
        if excess <= 0:
            continue
        n = int(held[t]) if target[t] == 0 else min(math.floor(excess / close[t]), int(held[t]))
        reason = "추세 필터" if t in trend_exit else "허용 범위"
        add(t, "sell", n, reason)
        held[t] -= n
        cash += n * close[t] * (1.0 - cost)
    for t in chosen:
        need = target[t] * total_after - held[t] * close[t]
        if need <= 0:
            continue
        want = math.floor(need / unit[t] + 1e-9)
        if want * unit[t] > cash and proxy in tickers and held[proxy] > 0:  # 모자라면 현금 대용을 팔아서
            shortfall = want * unit[t] - cash
            n_proxy = min(math.ceil(shortfall / (close[proxy] * (1.0 - cost))), int(held[proxy]))
            add(proxy, "sell", n_proxy, "매수 자금")
            held[proxy] -= n_proxy
            cash += n_proxy * close[proxy] * (1.0 - cost)
        n = min(want, math.floor(cash / unit[t] + 1e-9))
        add(t, "buy", n, "허용 범위")
        held[t] += n
        cash -= n * unit[t]
    if chosen and proxy in tickers and cash >= unit[proxy]:  # 매매로 남은 돈은 현금 대용으로
        n = math.floor(cash / unit[proxy] + 1e-9)
        add(proxy, "buy", n, "남는 금액")
        held[proxy] += n
        cash -= n * unit[proxy]

    # 5. 예상 세액
    ledger = TaxLedger()
    for h in holdings.itertuples(index=False):
        ledger.set_holding(h.ticker, h.shares, h.avg_cost_usd, h.avg_fx)
    gain = 0
    for o in orders.values():
        if o.side == "sell":
            gain += ledger.sell(as_of, o.ticker, o.shares, o.price, fx_rate).gain_krw
    tax_before = tax_due(realized_ytd_krw, cfg.tax.rate, cfg.tax.deduction_krw)
    tax_after = tax_due(realized_ytd_krw + gain, cfg.tax.rate, cfg.tax.deduction_krw)

    final_total = float((held * close).sum()) + cash
    table = pd.DataFrame({
        "ticker": tickers,
        "shares": shares.to_numpy(),
        "close": close.to_numpy(),
        "value_usd": values.to_numpy(),
        "weight": (values / float(values.sum())).fillna(0.0).to_numpy() if values.sum() > 0 else 0.0,
        "target": target.to_numpy(),
        "shares_after": held.to_numpy(),
        "weight_after": (held * close / final_total).to_numpy(),
    })
    order_list = sorted(orders.values(), key=lambda o: (o.side != "sell", o.ticker))
    return RebalancePlan(
        as_of=as_of, policy=policy.name, fx=fx_rate, fx_date=pd.Timestamp(fx_date), contribution_krw=contribution_krw,
        cost_bps=cfg.live.cost_bps, table=table, orders=order_list, trend=trend,
        tax={"realized_ytd_krw": realized_ytd_krw, "plan_gain_krw": gain, "tax_before_krw": tax_before,
             "tax_after_krw": tax_after, "tax_increase_krw": tax_after - tax_before},
        cash_left_usd=cash, glide_applied=target_w != raw_target, notes=notes,
    )


def _water_fill(deficit: pd.Series, budget: float, target: pd.Series) -> pd.Series:
    """budget을 deficit이 큰 자산부터 채운다: 매수 = max(0, deficit − λ), 합 = budget.

    모든 부족분을 채우고도 남으면 남는 금액은 목표 비중대로 나눈다.
    """
    if budget <= 0:
        return deficit * 0.0
    total_deficit = float(deficit.sum())
    if budget >= total_deficit:
        return deficit + (budget - total_deficit) * target / float(target.sum())
    levels = sorted(deficit.to_numpy(), reverse=True)
    lam = 0.0
    for k in range(1, len(levels) + 1):
        lam = (sum(levels[:k]) - budget) / k
        if k == len(levels) or levels[k] <= lam:
            break
    return (deficit - lam).clip(lower=0.0)


def render_plan(plan: RebalancePlan, summary: str | None = None, summary_note: str = "") -> str:
    """분기 문서 (마크다운). summary는 검증을 통과한 LLM 요약만 넣는다."""
    lines = [
        f"# 리밸런싱 {plan.quarter_label} (정책 {plan.policy})",
        "",
        (
            f"기준일 {plan.as_of.date()} 종가, 환율 {plan.fx:,.2f}원/달러 ({plan.fx_date.date()} 값). "
            f"이번 분기 납입금 {plan.contribution_krw:,}원. 주문은 사람이 직접 한다 (실제 계좌와 연결하지 않음)."
        ),
        "",
        "## 현재 비중 vs 목표 비중",
        "",
        "| 티커 | 보유 수량 | 종가(USD) | 평가액(USD) | 현재 비중 | 목표 비중 | 차이(%p) | 주문 후 수량 | 주문 후 비중 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in plan.table.itertuples(index=False):
        lines.append(
            f"| {r.ticker} | {r.shares:g} | {r.close:,.2f} | {r.value_usd:,.2f} | {r.weight:.2%} | {r.target:.2%} | "
            f"{(r.weight - r.target) * 100:+.2f} | {r.shares_after:g} | {r.weight_after:.2%} |"
        )
    lines += ["", "## 주문 목록", ""]
    if plan.orders:
        lines += ["| 티커 | 매수/매도 | 수량 | 예상 금액(USD) | 예상 금액(KRW) | 사유 |", "| --- | --- | --- | --- | --- | --- |"]
        for o in plan.orders:
            lines.append(
                f"| {o.ticker} | {'매수' if o.side == 'buy' else '매도'} | {o.shares} | {o.usd:,.2f} | "
                f"{o.usd * plan.fx:,.0f} | {o.reason} |"
            )
    else:
        lines.append("주문 없음 (모든 자산이 허용 범위 안).")
    lines.append("")
    lines.append(f"주문 후 남는 현금 약 {plan.cash_left_usd:,.2f}달러 (예상 비용 편도 {plan.cost_bps:g}bp 포함 계산).")
    if not plan.trend.empty:
        lines += ["", "## 추세 필터 상태 (월말 종가 vs 10개월 평균)", "",
                  "| 자산 | 기준 월말 | 월말 종가 | 10개월 평균 | 위치 |", "| --- | --- | --- | --- | --- |"]
        for t in plan.trend.itertuples(index=False):
            lines.append(f"| {t.asset} | {t.month_end} | {t.close:,.2f} | {t.sma:,.2f} | {'아래 (BIL로)' if t.below else '위'} |")
    tx = plan.tax
    lines += [
        "", "## 예상 세액 (참고용, 실제 신고와 다를 수 있음)", "",
        "| 항목 | 금액(원) |", "| --- | --- |",
        f"| 올해 이미 실현한 차익 (입력값) | {tx['realized_ytd_krw']:,} |",
        f"| 이번 매도의 예상 실현 차익 | {tx['plan_gain_krw']:,} |",
        f"| 올해 예상 세액 (이번 매도 전) | {tx['tax_before_krw']:,} |",
        f"| 올해 예상 세액 (이번 매도 후) | {tx['tax_after_krw']:,} |",
        f"| 이번 매도로 늘어나는 세액 | {tx['tax_increase_krw']:,} |",
        "",
        (
            "계산: (매도 수량 × 기준일 종가 × 기준일 환율) − (매도 수량 × 평균 취득가 × 평균 취득 환율), "
            "연간 합계에서 250만 원 공제 후 22%. 수수료는 빼지 않았다."
        ),
    ]
    if plan.notes:
        lines += ["", *[f"- {n}" for n in plan.notes]]
    if summary:
        lines += ["", "## 요약 (로컬 LLM, 숫자는 위 표와 대조함)", "", summary]
    elif summary_note:
        lines += ["", f"요약 없음: {summary_note}"]
    return "\n".join(lines) + "\n"


def write_plan(text: str, plan: RebalancePlan, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{plan.quarter_label}.md"
    path.write_text(text, encoding="utf-8")
    return path
