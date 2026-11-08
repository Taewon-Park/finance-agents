"""가격 데이터 소스 어댑터 인터페이스.

전략·엔진은 데이터 소스를 직접 호출하지 않는다. 새 소스(Tiingo 등)는
PriceSource를 구현하는 클래스 하나만 추가하면 된다.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol

import numpy as np
import pandas as pd

PRICE_COLUMNS: list[str] = ["date", "ticker", "open", "high", "low", "close", "volume", "source"]
OHLC_COLUMNS: list[str] = ["open", "high", "low", "close"]


class FetchError(RuntimeError):
    """소스에서 데이터를 받지 못했을 때 (재시도 후에도 실패)."""


class PriceSource(Protocol):
    name: str

    def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        """[start, end] 구간(양끝 포함)의 일봉을 long-format DataFrame으로 반환.

        컬럼은 PRICE_COLUMNS. 가격은 분할·배당이 OHLC 모두에 일관되게 반영된 수정주가.
        """
        ...


def empty_prices() -> pd.DataFrame:
    return normalize_prices(pd.DataFrame(columns=PRICE_COLUMNS))


def normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    """스키마 컬럼·타입을 맞추고 (ticker, date) 순으로 정렬한다.

    date는 메모리에서는 자정 기준 datetime64로 다룬다 (parquet에는 date32로 저장).
    """
    missing = [c for c in PRICE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"가격 데이터에 컬럼이 없습니다: {missing}")
    out = df.loc[:, PRICE_COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize().astype("datetime64[ns]")
    out["ticker"] = out["ticker"].astype(str)
    out["source"] = out["source"].astype(str)
    for col in OHLC_COLUMNS:
        out[col] = out[col].astype("float64")
    out["volume"] = out["volume"].fillna(0).astype(np.int64)
    out = out.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)
    return out
