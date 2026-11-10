"""일봉 수집 (전체·증분) 후 정제본 생성.

수정주가는 새 배당·분할이 생기면 과거 값 전체가 바뀐다. 그래서 증분 수집 때
최근 OVERLAP_ROWS개 거래일을 겹쳐 받아 기존 값과 비교하고, 달라졌으면 그 종목만
전체를 다시 받는다. 기준이 다른 수정주가가 한 파일에 섞이지 않게 하기 위해서다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from swinglab.pipeline.clean import build_clean
from swinglab.pipeline.storage import raw_path, read_prices, write_prices
from swinglab.sources.base import OHLC_COLUMNS, FetchError, PriceSource

OVERLAP_ROWS = 5
# Yahoo 수정주가는 받을 때마다 행별로 상대 1e-6 수준의 반올림 잡음이 있다 (실측 최대 1.8e-6).
# 배당·분할은 최소 1e-3 수준으로 값을 바꾸므로 그 사이인 1e-4를 기준으로 쓴다.
ADJUSTMENT_RTOL = 1e-4

IngestMode = Literal["full", "incremental", "refetch", "up_to_date", "failed"]


@dataclass
class TickerIngestResult:
    ticker: str
    mode: IngestMode
    rows_added: int = 0
    first_date: date | None = None
    last_date: date | None = None
    message: str = ""


@dataclass
class IngestReport:
    results: list[TickerIngestResult] = field(default_factory=list)
    clean_rows: int = 0
    clean_file: Path | None = None

    @property
    def failed(self) -> list[TickerIngestResult]:
        return [r for r in self.results if r.mode == "failed"]


def ingest(
    tickers: list[str],
    source: PriceSource,
    start: date,
    end: date,
    data_dir: Path,
    full: bool = False,
    clean_tickers: list[str] | None = None,
) -> IngestReport:
    """tickers를 수집하고 정제본을 다시 만든다.

    정제본은 clean_tickers(없으면 tickers)의 원본 전체로 만든다. 일부 종목만 수집할 때
    정제본에서 나머지 종목이 빠지지 않도록 호출하는 쪽에서 수집 대상 전체를 넘긴다.
    """
    report = IngestReport()
    for ticker in tickers:
        try:
            result = _ingest_one(ticker, source, start, end, data_dir, full)
        except FetchError as exc:
            result = TickerIngestResult(ticker=ticker, mode="failed", message=str(exc))
        report.results.append(result)

    report.clean_file, report.clean_rows = build_clean(clean_tickers or tickers, source.name, data_dir)
    return report


def _ingest_one(
    ticker: str, source: PriceSource, start: date, end: date, data_dir: Path, full: bool
) -> TickerIngestResult:
    path = raw_path(data_dir, source.name, ticker)
    existing = read_prices(path) if path.exists() and not full else None

    if existing is None or existing.empty:
        fresh = source.fetch([ticker], start, end)
        write_prices(fresh, path)
        return _result(ticker, "full", fresh, rows_added=len(fresh))

    last_stored = existing["date"].max().date()
    if last_stored >= end:
        return _result(ticker, "up_to_date", existing)

    overlap = existing.tail(OVERLAP_ROWS)
    fetch_start = overlap["date"].min().date()
    new = source.fetch([ticker], fetch_start, end)

    if _adjustment_changed(overlap, new):
        fresh = source.fetch([ticker], start, end)
        write_prices(fresh, path)
        return _result(
            ticker,
            "refetch",
            fresh,
            rows_added=len(fresh) - len(existing),
            message="겹친 구간의 수정주가가 달라져 전체 재수집 (배당·분할 반영)",
        )

    kept = existing[existing["date"] < pd.Timestamp(fetch_start)]
    combined = pd.concat([kept, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last")
    write_prices(combined, path)
    return _result(ticker, "incremental", combined, rows_added=len(combined) - len(existing))


def _adjustment_changed(overlap: pd.DataFrame, new: pd.DataFrame) -> bool:
    merged = overlap.merge(new, on="date", suffixes=("_old", "_new"))
    if len(merged) < len(overlap):  # 겹친 날이 빠져 있으면 비교할 수 없으므로 재수집
        return True
    for col in OHLC_COLUMNS:
        if not np.allclose(merged[f"{col}_old"], merged[f"{col}_new"], rtol=ADJUSTMENT_RTOL, atol=0.0):
            return True
    return False


def _result(
    ticker: str, mode: IngestMode, df: pd.DataFrame, rows_added: int = 0, message: str = ""
) -> TickerIngestResult:
    return TickerIngestResult(
        ticker=ticker,
        mode=mode,
        rows_added=rows_added,
        first_date=df["date"].min().date() if not df.empty else None,
        last_date=df["date"].max().date() if not df.empty else None,
        message=message,
    )
