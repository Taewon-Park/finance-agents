"""yfinance 기반 PriceSource 구현."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from swinglab.sources.base import FetchError, empty_prices, normalize_prices

logger = logging.getLogger(__name__)

_COLUMN_MAP = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}


@dataclass
class YFinanceSource:
    """Yahoo Finance 일봉. auto_adjust=True로 OHLC 전체에 분할·배당을 반영한다.

    yfinance는 실패해도 예외 대신 빈 DataFrame을 돌려주는 경우가 많아서,
    빈 결과도 실패로 보고 재시도한다. 호출 구간은 항상 거래일을 포함하도록
    ingest 쪽에서 잡는다 (증분 수집도 최근 거래일을 겹쳐서 요청).
    """

    name: str = "yfinance"
    timeout_seconds: float = 30.0
    max_retries: int = 3
    backoff_seconds: float = 2.0
    pause_seconds: float = 0.5  # 종목 사이 대기 (요청 과다 방지)

    def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for i, ticker in enumerate(tickers):
            if i > 0 and self.pause_seconds > 0:
                time.sleep(self.pause_seconds)
            frames.append(self._fetch_one(ticker, start, end))
        if not frames:
            return empty_prices()
        return normalize_prices(pd.concat(frames, ignore_index=True))

    def _fetch_one(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        last_error: str = "알 수 없는 오류"
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = yf.download(
                    ticker,
                    start=start.isoformat(),
                    end=(end + timedelta(days=1)).isoformat(),  # yfinance의 end는 미포함
                    interval="1d",
                    auto_adjust=True,
                    actions=False,
                    progress=False,
                    threads=False,
                    timeout=self.timeout_seconds,
                    multi_level_index=False,
                )
            except Exception as exc:  # noqa: BLE001 - 네트워크·파싱 오류 모두 재시도 대상
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if raw is not None and not raw.empty:
                    frame = self._to_schema(raw, ticker, start, end)
                    if not frame.empty:
                        return frame
                last_error = "빈 응답"
            if attempt < self.max_retries:
                wait = self.backoff_seconds * 2 ** (attempt - 1)
                logger.warning("%s 수집 실패 (%d/%d): %s, %.1f초 후 재시도", ticker, attempt, self.max_retries, last_error, wait)
                time.sleep(wait)
        raise FetchError(f"{ticker}: {self.max_retries}회 시도 후 실패 ({last_error})")

    def _to_schema(self, raw: pd.DataFrame, ticker: str, start: date, end: date) -> pd.DataFrame:
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw.droplevel("Ticker", axis=1) if "Ticker" in raw.columns.names else raw.droplevel(-1, axis=1)
        missing = [c for c in _COLUMN_MAP if c not in raw.columns]
        if missing:
            raise FetchError(f"{ticker}: 응답에 컬럼이 없습니다 {missing}")

        index = pd.DatetimeIndex(raw.index)
        if index.tz is not None:  # 거래일은 미국 동부 기준 날짜로 맞춘다
            index = index.tz_convert("America/New_York").tz_localize(None)

        df = raw.loc[:, list(_COLUMN_MAP)].rename(columns=_COLUMN_MAP)
        df.index = index.normalize()
        df = df.dropna(subset=["open", "high", "low", "close"], how="all")
        df = df.reset_index(names="date")
        df["ticker"] = ticker
        df["source"] = self.name
        mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
        return normalize_prices(df.loc[mask])
