"""비교용 벤치마크 곡선."""

from __future__ import annotations

import pandas as pd

from swinglab.backtest.metrics import annual_volatility, daily_returns


def vol_matched_curve(benchmark: pd.Series, target_vol: float) -> tuple[pd.Series, float]:
    """벤치마크와 현금(0%)을 매일 같은 비중으로 섞어 연 변동성을 target_vol에 맞춘 곡선 (시작값 = 벤치마크 시작값).

    비중 = target_vol / 벤치마크 연 변동성. 재조정 비용은 넣지 않는다. (곡선, 벤치마크 비중) 반환.
    """
    rets = daily_returns(benchmark)
    bench_vol = annual_volatility(rets)
    weight = target_vol / bench_vol
    growth = (1.0 + weight * rets).cumprod() * float(benchmark.iloc[0])
    curve = pd.concat([benchmark.iloc[:1].astype(float), growth])
    curve.name = "vol_matched"
    return curve, weight
