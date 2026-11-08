"""이동평균 (순수 함수)."""

from __future__ import annotations

import pandas as pd


def sma(values: pd.Series, window: int) -> pd.Series:
    """단순이동평균. 마지막 window개 '실제 값'의 평균이며 값이 window개 미만이면 NaN.

    NaN(결측일)은 봉이 없는 날로 보고 건너뛴다. 결과는 입력과 같은 인덱스이고
    입력이 NaN인 자리는 NaN이다.
    """
    if window < 1:
        raise ValueError("window는 1 이상이어야 합니다")
    present = values.dropna()
    result = present.rolling(window=window, min_periods=window).mean()
    return result.reindex(values.index)
