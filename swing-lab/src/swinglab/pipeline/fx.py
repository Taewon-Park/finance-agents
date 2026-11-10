"""환율 수집·저장 (data/raw/fred/<시리즈>.parquet, data/clean/fx.parquet).

환율은 수정 개념이 없고 행 수가 적어서(하루 1행) 매번 전체 구간을 다시 받는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from swinglab.sources.base import FetchError
from swinglab.sources.fred_source import FredSource


@dataclass
class FxIngestResult:
    pair: str
    series_id: str
    rows: int = 0
    first_date: date | None = None
    last_date: date | None = None
    error: str = ""


def fx_raw_path(data_dir: Path, series_id: str) -> Path:
    return data_dir / "raw" / "fred" / f"{series_id}.parquet"


def fx_clean_path(data_dir: Path) -> Path:
    return data_dir / "clean" / "fx.parquet"


def ingest_fx(series: dict[str, str], source: FredSource, start: date, end: date, data_dir: Path) -> list[FxIngestResult]:
    """series = {이름: FRED 시리즈 ID}. 실패한 시리즈는 기존 파일을 그대로 둔다."""
    results = []
    for pair, series_id in series.items():
        try:
            df = source.fetch_series(series_id, start, end)
        except FetchError as exc:
            results.append(FxIngestResult(pair, series_id, error=str(exc)))
            continue
        _write(df, fx_raw_path(data_dir, series_id))
        results.append(FxIngestResult(pair, series_id, len(df), df["date"].min().date(), df["date"].max().date()))

    frames = [
        pd.read_parquet(p).assign(pair=pair)
        for pair, series_id in series.items()
        if (p := fx_raw_path(data_dir, series_id)).exists()
    ]
    if frames:
        _write(pd.concat(frames, ignore_index=True)[["date", "pair", "value"]], fx_clean_path(data_dir))
    return results


def read_fx(data_dir: Path, pair: str) -> pd.Series:
    """date → 환율. 파일이 없으면 FileNotFoundError."""
    path = fx_clean_path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path}가 없습니다. 먼저 swinglab ingest로 환율을 수집하세요.")
    df = pd.read_parquet(path)
    s = df[df["pair"] == pair].set_index("date")["value"].sort_index()
    if s.empty:
        raise FileNotFoundError(f"{path}에 {pair} 환율이 없습니다")
    s.index = pd.to_datetime(s.index).astype("datetime64[ns]")
    return s.astype(float)


def fx_on(fx: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """각 날짜의 환율. 그날 값이 없으면(미국·한국 휴일 차이) 직전 값."""
    out = fx.reindex(fx.index.union(dates)).ffill().reindex(dates)
    if out.isna().any():
        raise ValueError(f"환율이 없는 날짜가 있습니다 (첫 환율 {fx.index[0].date()} 이전)")
    return out


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
