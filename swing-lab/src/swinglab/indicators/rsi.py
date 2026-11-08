"""RSI, Wilder 평활 방식 (순수 함수)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi_wilder(close: pd.Series, period: int) -> pd.Series:
    """Wilder RSI.

    - 첫 평균 상승·하락폭: 처음 period개 변화량의 단순평균 (period번째 변화가 나온 봉에서 첫 값)
    - 이후: avg = (이전 avg × (period − 1) + 이번 값) / period
    - 평균 하락폭이 0이면 100, 상승·하락 모두 0이면 50

    pandas ewm(alpha=1/period)은 첫 값 시드가 달라 결과가 다르므로 쓰지 않는다.
    NaN(결측일)은 봉이 없는 날로 보고 건너뛰며, 그 자리는 NaN으로 둔다.
    """
    if period < 1:
        raise ValueError("period는 1 이상이어야 합니다")
    present = close.dropna()
    values = present.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    if len(values) > period:
        delta = np.diff(values)
        gains = np.clip(delta, 0.0, None)
        losses = np.clip(-delta, 0.0, None)
        avg_gain = gains[:period].mean()
        avg_loss = losses[:period].mean()
        out[period] = _rsi_value(avg_gain, avg_loss)
        for i in range(period, len(delta)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            out[i + 1] = _rsi_value(avg_gain, avg_loss)
    return pd.Series(out, index=present.index, name=close.name).reindex(close.index)


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0.0:
        return 50.0 if avg_gain == 0.0 else 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)
