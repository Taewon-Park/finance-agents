"""가격 parquet 읽기·쓰기. 디스크에는 date를 date32로, 메모리에서는 datetime64로 다룬다."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from swinglab.sources.base import normalize_prices


def raw_path(data_dir: Path, source_name: str, ticker: str) -> Path:
    return data_dir / "raw" / source_name / f"{ticker}.parquet"


def clean_path(data_dir: Path) -> Path:
    return data_dir / "clean" / "prices.parquet"


def write_prices(df: pd.DataFrame, path: Path) -> None:
    df = normalize_prices(df)
    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.set_column(
        table.schema.get_field_index("date"), "date", table.column("date").cast(pa.date32())
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)  # 쓰는 도중 실패해도 기존 파일이 깨지지 않게


def read_prices(path: Path) -> pd.DataFrame:
    return normalize_prices(pd.read_parquet(path))
