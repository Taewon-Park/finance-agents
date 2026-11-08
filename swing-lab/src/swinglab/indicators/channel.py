"""가격 채널과 기간 수익률 (순수 함수). NaN(결측일)은 봉이 없는 날로 보고 건너뛴다."""

from __future__ import annotations

import pandas as pd


def prior_high(values: pd.Series, window: int) -> pd.Series:
    """직전 window개 값의 최고값 (당일 제외). 직전 값이 window개 미만이면 NaN."""
    return _prior(values, window, "max")


def prior_low(values: pd.Series, window: int) -> pd.Series:
    """직전 window개 값의 최저값 (당일 제외). 직전 값이 window개 미만이면 NaN."""
    return _prior(values, window, "min")


def period_return(values: pd.Series, periods: int) -> pd.Series:
    """periods개 봉 전 대비 수익률: 값_t / 값_(t−periods) − 1."""
    if periods < 1:
        raise ValueError("periods는 1 이상이어야 합니다")
    present = values.dropna()
    return (present / present.shift(periods) - 1.0).reindex(values.index)


def _prior(values: pd.Series, window: int, how: str) -> pd.Series:
    if window < 1:
        raise ValueError("window는 1 이상이어야 합니다")
    present = values.dropna()
    rolled = present.shift(1).rolling(window=window, min_periods=window)
    result = rolled.max() if how == "max" else rolled.min()
    return result.reindex(values.index)
