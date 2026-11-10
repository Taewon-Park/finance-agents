"""2단계 트랙 2 데이터: 수집 전용 종목, FRED 환율, 정제본 보존."""

from __future__ import annotations

import urllib.error
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from conftest import random_walk_prices

from swinglab.config import load_universe
from swinglab.pipeline.fx import fx_on, ingest_fx, read_fx
from swinglab.pipeline.ingest import ingest
from swinglab.pipeline.storage import clean_path, read_prices
from swinglab.sources.base import FetchError
from swinglab.sources.fred_source import FredSource, parse_fred_csv

ROOT = Path(__file__).resolve().parents[1]


def test_collect_only_tickers_are_not_strategy_universe() -> None:
    u = load_universe(ROOT / "config")
    assert {"EFA", "EEM", "BIL"} <= set(u.collect_tickers)
    assert not {"EFA", "EEM", "BIL"} & set(u.tickers)
    assert len(u.tickers) == 20 and len(u.tradable) == 19  # 0·1단계 유니버스 그대로
    assert u.fx == {"USDKRW": "DEXKOUS"}


def test_parse_fred_csv_skips_missing_values() -> None:
    body = "observation_date,DEXKOUS\n2024-01-01,\n2024-01-02,1300.50\n2024-01-03,.\n2024-01-04,1310.25\n"
    df = parse_fred_csv(body, "DEXKOUS", date(2024, 1, 1), date(2024, 1, 31))
    assert df["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-02", "2024-01-04"]
    assert df["value"].tolist() == [1300.50, 1310.25]
    with pytest.raises(ValueError, match="형식"):
        parse_fred_csv("DATE,OTHER\n2024-01-02,1\n", "DEXKOUS", date(2024, 1, 1), date(2024, 1, 31))


def test_fred_source_retries_then_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def boom(url, timeout):
        calls.append(timeout)
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    src = FredSource(max_retries=3, backoff_seconds=0.0, timeout_seconds=7.0)
    with pytest.raises(FetchError, match="3회"):
        src.fetch_series("DEXKOUS", date(2024, 1, 1), date(2024, 1, 31))
    assert calls == [7.0, 7.0, 7.0]


@dataclass
class _FakeFred:
    name: str = "fred"

    def fetch_series(self, series_id: str, start: date, end: date) -> pd.DataFrame:
        if series_id == "BAD":
            raise FetchError("BAD: 실패")
        return pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-04"]), "value": [1300.0, 1310.0]})


def test_ingest_fx_and_ffill(tmp_path: Path) -> None:
    results = ingest_fx({"USDKRW": "DEXKOUS", "X": "BAD"}, _FakeFred(), date(2024, 1, 1), date(2024, 1, 31), tmp_path)
    assert [r.error == "" for r in results] == [True, False]
    fx = read_fx(tmp_path, "USDKRW")
    days = pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    assert fx_on(fx, days).tolist() == [1300.0, 1300.0, 1310.0]  # 1/3은 직전 값
    with pytest.raises(ValueError):
        fx_on(fx, pd.DatetimeIndex([pd.Timestamp("2024-01-01")]))


@dataclass
class _FakePrices:
    name: str = "test"

    def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        return random_walk_prices(tickers, 20, seed=len(tickers[0]))


def test_partial_ingest_keeps_other_tickers_in_clean(tmp_path: Path) -> None:
    src = _FakePrices()
    ingest(["AAA", "BBB"], src, date(2020, 1, 1), date(2020, 2, 1), tmp_path)
    ingest(["CCC"], src, date(2020, 1, 1), date(2020, 2, 1), tmp_path, clean_tickers=["AAA", "BBB", "CCC"])
    assert sorted(read_prices(clean_path(tmp_path))["ticker"].unique()) == ["AAA", "BBB", "CCC"]
