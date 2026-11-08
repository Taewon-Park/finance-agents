"""지표 함수를 손으로 계산한 소형 예제와 비교한다."""

import numpy as np
import pandas as pd
import pytest

from swinglab.indicators import rsi_wilder, sma


def test_sma_hand_example() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    # (1+2+3)/3=2, (2+3+4)/3=3, ...
    expected = [np.nan, np.nan, 2.0, 3.0, 4.0, 5.0]
    np.testing.assert_allclose(sma(s, 3).to_numpy(), expected)


def test_sma_skips_missing_days() -> None:
    s = pd.Series([1.0, 2.0, np.nan, 4.0, 6.0])
    # 결측일은 봉이 없는 날: (1+2)/2, NaN, (2+4)/2, (4+6)/2
    expected = [np.nan, 1.5, np.nan, 3.0, 5.0]
    np.testing.assert_allclose(sma(s, 2).to_numpy(), expected)


def test_rsi_wilder_period2_hand_example() -> None:
    close = pd.Series([10.0, 11.0, 10.5, 11.5, 12.0, 11.0])
    # 변화: +1, -0.5, +1, +0.5, -1
    # idx2 시드: 평균상승 (1+0)/2=0.5, 평균하락 (0+0.5)/2=0.25 → RS 2 → 66.667
    # idx3: 상승 (0.5+1)/2=0.75, 하락 (0.25+0)/2=0.125 → RS 6 → 85.714
    # idx4: 상승 (0.75+0.5)/2=0.625, 하락 0.0625 → RS 10 → 90.909
    # idx5: 상승 0.3125, 하락 (0.0625+1)/2=0.53125 → RS 10/17 → 37.037
    expected = [np.nan, np.nan, 200 / 3, 600 / 7, 1000 / 11, 1000 / 27]
    np.testing.assert_allclose(rsi_wilder(close, 2).to_numpy(), expected, rtol=1e-12)


def test_rsi_wilder_period3_uses_sma_seed() -> None:
    close = pd.Series([44.0, 45.0, 44.0, 46.0, 47.0, 45.0, 48.0])
    # 변화: +1, -1, +2, +1, -2, +3
    # idx3 시드: 상승 (1+0+2)/3=1, 하락 (0+1+0)/3=1/3 → RS 3 → 75
    # idx4: 상승 (1*2+1)/3=1, 하락 (1/3*2+0)/3=2/9 → RS 4.5 → 81.818
    # idx5: 상승 (1*2+0)/3=2/3, 하락 (2/9*2+2)/3=22/27 → RS 9/11 → 45
    # idx6: 상승 (2/3*2+3)/3=13/9, 하락 (22/27*2)/3=44/81 → RS 117/44 → 72.671
    expected = [np.nan, np.nan, np.nan, 75.0, 900 / 11, 45.0, 11700 / 161]
    np.testing.assert_allclose(rsi_wilder(close, 3).to_numpy(), expected, rtol=1e-12)


def test_rsi_wilder_differs_from_ewm_seed() -> None:
    close = pd.Series([44.0, 45.0, 44.0, 46.0, 47.0, 45.0, 48.0])
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 3, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 3, adjust=False).mean()
    ewm_rsi = 100 - 100 / (1 + gain / loss)
    assert not np.isclose(rsi_wilder(close, 3).iloc[-1], ewm_rsi.iloc[-1])


def test_rsi_edge_values() -> None:
    assert rsi_wilder(pd.Series([1.0, 2.0, 3.0, 4.0]), 2).iloc[-1] == 100.0
    assert rsi_wilder(pd.Series([5.0, 5.0, 5.0, 5.0]), 2).iloc[-1] == 50.0
    assert rsi_wilder(pd.Series([1.0, 2.0]), 2).isna().all()


def test_rsi_skips_missing_days() -> None:
    full = pd.Series([10.0, 11.0, np.nan, 10.5, 11.5, 12.0, 11.0])
    compact = pd.Series([10.0, 11.0, 10.5, 11.5, 12.0, 11.0])
    result = rsi_wilder(full, 2)
    assert np.isnan(result.iloc[2])
    np.testing.assert_allclose(result.dropna().to_numpy(), rsi_wilder(compact, 2).dropna().to_numpy())


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_window(bad: int) -> None:
    with pytest.raises(ValueError):
        sma(pd.Series([1.0]), bad)
    with pytest.raises(ValueError):
        rsi_wilder(pd.Series([1.0]), bad)
