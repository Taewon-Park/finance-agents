"""FRED 시계열 (API 키 없이 쓰는 공개 CSV). 원/달러 환율 DEXKOUS 등.

https://fred.stlouisfed.org/graph/fredgraph.csv?id=<시리즈>&cosd=<시작일>
응답은 "observation_date,<시리즈>" 헤더의 CSV이고, 값이 없는 날(휴일)은 빈 칸 또는 "."이다.
"""

from __future__ import annotations

import io
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date

import pandas as pd

from swinglab.sources.base import FetchError

logger = logging.getLogger(__name__)

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}&coed={end}"


@dataclass
class FredSource:
    name: str = "fred"
    timeout_seconds: float = 30.0
    max_retries: int = 3
    backoff_seconds: float = 2.0

    def fetch_series(self, series_id: str, start: date, end: date) -> pd.DataFrame:
        """[start, end] 구간의 (date, value). 값이 없는 날은 뺀다."""
        url = FRED_CSV_URL.format(series=series_id, start=start.isoformat(), end=end.isoformat())
        last_error = "알 수 없는 오류"
        for attempt in range(1, self.max_retries + 1):
            try:
                with urllib.request.urlopen(url, timeout=self.timeout_seconds) as resp:
                    body = resp.read().decode("utf-8")
                return parse_fred_csv(body, series_id, start, end)
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < self.max_retries:
                wait = self.backoff_seconds * 2 ** (attempt - 1)
                logger.warning("%s 수집 실패 (%d/%d): %s, %.1f초 후 재시도", series_id, attempt, self.max_retries, last_error, wait)
                time.sleep(wait)
        raise FetchError(f"{series_id}: {self.max_retries}회 시도 후 실패 ({last_error})")


def parse_fred_csv(body: str, series_id: str, start: date, end: date) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(body), dtype=str)
    date_col = next((c for c in df.columns if c.lower() in ("observation_date", "date")), None)
    if date_col is None or series_id not in df.columns:
        raise ValueError(f"FRED 응답 형식이 다릅니다: {list(df.columns)}")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col]).astype("datetime64[ns]"),
            "value": pd.to_numeric(df[series_id].replace(".", None), errors="coerce"),
        }
    ).dropna()
    out = out[(out["date"] >= pd.Timestamp(start)) & (out["date"] <= pd.Timestamp(end))]
    if out.empty:
        raise ValueError(f"{series_id}: 구간 안에 값이 없습니다")
    return out.sort_values("date").reset_index(drop=True)
