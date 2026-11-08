"""성과 지표.

- CAGR: 자산곡선 첫 행(시작 기준일)부터 마지막 날까지, 달력일 / 365.25로 연환산
- 연 변동성: 일간 수익률 표준편차 × √252
- 샤프: 일간 평균 초과수익 / 일간 표준편차 × √252 (무위험수익률 0)
- MDD: 자산곡선 고점 대비 최대 하락폭 (음수)
- 거래 통계(승률, 손익비, 보유일, 연 거래 횟수)는 청산 완료 거래만 센다.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from swinglab.backtest.engine import BacktestResult

TRADING_DAYS_PER_YEAR = 252
DAYS_PER_YEAR = 365.25


def daily_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().iloc[1:]


def years_covered(equity: pd.Series) -> float:
    return (equity.index[-1] - equity.index[0]).days / DAYS_PER_YEAR


def cagr(equity: pd.Series) -> float:
    years = years_covered(equity)
    if years <= 0:
        return math.nan
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)


def annual_volatility(returns: pd.Series) -> float:
    if len(returns) < 2:
        return math.nan
    return float(returns.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(returns: pd.Series, risk_free_daily: float = 0.0) -> float:
    if len(returns) < 2:
        return math.nan
    excess = returns - risk_free_daily
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return math.nan
    return float(excess.mean() / std * math.sqrt(TRADING_DAYS_PER_YEAR))


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown(equity).min())


def exposure(positions: pd.DataFrame) -> float:
    """포지션을 하나라도 들고 종가를 맞은 날의 비율. 첫 행(시작 기준일)은 뺀다."""
    held = (positions.iloc[1:] > 0).any(axis=1)
    return float(held.mean()) if len(held) else math.nan


def performance_metrics(result: BacktestResult) -> dict[str, float]:
    equity = result.equity
    rets = daily_returns(equity)
    years = years_covered(equity)
    trades = result.trades
    wins = trades.loc[trades["return"] > 0, "return"]
    losses = trades.loc[trades["return"] < 0, "return"]
    n = len(trades)
    return {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "cagr": cagr(equity),
        "annual_vol": annual_volatility(rets),
        "sharpe": sharpe_ratio(rets),
        "mdd": max_drawdown(equity),
        "win_rate": float(len(wins) / n) if n else math.nan,
        "payoff_ratio": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else math.nan,
        "avg_holding_days": float(trades["holding_days"].mean()) if n else math.nan,
        "trades_per_year": float(n / years) if years > 0 else math.nan,
        "exposure": exposure(result.positions),
        "n_trades": float(n),
        "open_positions": float(len(result.open_positions)),
        "total_cost": float(result.total_cost),
    }
