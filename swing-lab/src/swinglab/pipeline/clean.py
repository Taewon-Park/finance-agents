"""정제본 생성과 데이터 품질 점검.

점검 항목 (자동 수정은 하지 않고 목록만 보고한다):
- 종목별 결측 거래일 수 (NYSE 거래일 캘린더 대비)
- 일간 수익률 ±25% 초과 이상치
- high < low, close가 [low, high] 범위 밖인 행
- 거래량 0인 날
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from swinglab.pipeline.storage import clean_path, raw_path, read_prices, write_prices
from swinglab.sources.base import OHLC_COLUMNS, empty_prices

RETURN_OUTLIER_THRESHOLD = 0.25
# 수정주가는 부동소수 계산을 거치므로 범위 비교에 아주 작은 여유를 둔다.
RANGE_RTOL = 1e-6
DETAIL_LIMIT = 30


def build_clean(tickers: list[str], source_name: str, data_dir: Path) -> tuple[Path, int]:
    """원본(raw)을 합쳐 data/clean/prices.parquet을 만든다. OHLC가 비어 있는 행만 뺀다."""
    frames = [
        read_prices(p)
        for t in tickers
        if (p := raw_path(data_dir, source_name, t)).exists()
    ]
    df = pd.concat(frames, ignore_index=True) if frames else empty_prices()
    df = df.dropna(subset=OHLC_COLUMNS, how="any")
    df = df.drop_duplicates(subset=["ticker", "date"], keep="last")
    out = clean_path(data_dir)
    write_prices(df, out)
    return out, len(df)


@dataclass
class QualityReport:
    summary: pd.DataFrame
    missing_sessions: dict[str, list[pd.Timestamp]]
    extra_dates: dict[str, list[pd.Timestamp]]
    return_outliers: pd.DataFrame
    bad_ranges: pd.DataFrame
    zero_volume: pd.DataFrame

    @property
    def issue_count(self) -> int:
        return (
            sum(len(v) for v in self.missing_sessions.values())
            + sum(len(v) for v in self.extra_dates.values())
            + len(self.return_outliers)
            + len(self.bad_ranges)
            + len(self.zero_volume)
        )

    def to_text(self) -> str:
        lines = ["# 데이터 품질 점검", "", "## 종목별 요약", self.summary.to_string(), ""]

        lines.append("## 결측 거래일 (NYSE 거래일 대비, 종목 첫 거래일 이후)")
        lines += _date_lists(self.missing_sessions)
        lines.append("")
        lines.append("## NYSE 휴장일인데 데이터가 있는 날")
        lines += _date_lists(self.extra_dates)
        lines.append("")
        for title, frame in (
            (f"## 일간 수익률 ±{RETURN_OUTLIER_THRESHOLD:.0%} 초과", self.return_outliers),
            ("## 가격 범위 오류 (high < low, close가 범위 밖)", self.bad_ranges),
            ("## 거래량 0", self.zero_volume),
        ):
            lines.append(title)
            if frame.empty:
                lines.append("없음")
            else:
                lines.append(f"총 {len(frame)}건" + (f" (앞 {DETAIL_LIMIT}건만 표시)" if len(frame) > DETAIL_LIMIT else ""))
                lines.append(frame.head(DETAIL_LIMIT).to_string(index=False))
            lines.append("")
        return "\n".join(lines)


def run_checks(prices: pd.DataFrame, sessions: pd.DatetimeIndex) -> QualityReport:
    """prices: 스키마의 long-format. sessions: 점검 구간을 덮는 NYSE 거래일."""
    sessions = pd.DatetimeIndex(sessions)
    global_last = prices["date"].max() if not prices.empty else None

    missing: dict[str, list[pd.Timestamp]] = {}
    extra: dict[str, list[pd.Timestamp]] = {}
    rows = []
    outlier_frames, range_frames = [], []

    for ticker, g in prices.groupby("ticker", sort=True):
        g = g.sort_values("date")
        dates = pd.DatetimeIndex(g["date"])
        expected = sessions[(sessions >= dates.min()) & (sessions <= global_last)]
        missing[ticker] = list(expected.difference(dates))
        extra[ticker] = list(dates.difference(sessions))

        ret = g["close"].pct_change()
        out = g.loc[ret.abs() > RETURN_OUTLIER_THRESHOLD, ["date", "ticker", "close"]].copy()
        out["return"] = ret[out.index]
        outlier_frames.append(out)

        tol_hi = g["high"] * (1 + RANGE_RTOL)
        tol_lo = g["low"] * (1 - RANGE_RTOL)
        bad = pd.Series("", index=g.index)
        bad[g["high"] < tol_lo] += "high<low "
        bad[g["close"] > tol_hi] += "close>high "
        bad[g["close"] < tol_lo] += "close<low "
        bad_rows = g.loc[bad != "", ["date", "ticker", "open", "high", "low", "close"]].copy()
        bad_rows["issue"] = bad[bad != ""].str.strip()
        range_frames.append(bad_rows)

        rows.append(
            {
                "ticker": ticker,
                "first_date": dates.min().date(),
                "last_date": dates.max().date(),
                "rows": len(g),
                "missing_sessions": len(missing[ticker]),
                "extra_dates": len(extra[ticker]),
                "return_outliers": len(out),
                "bad_ranges": len(bad_rows),
                "zero_volume": int((g["volume"] == 0).sum()),
            }
        )

    zero_volume = prices.loc[prices["volume"] == 0, ["date", "ticker", "close", "volume"]]
    return QualityReport(
        summary=pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame(),
        missing_sessions=missing,
        extra_dates=extra,
        return_outliers=_concat(outlier_frames, ["date", "ticker", "close", "return"]),
        bad_ranges=_concat(range_frames, ["date", "ticker", "open", "high", "low", "close", "issue"]),
        zero_volume=zero_volume.reset_index(drop=True),
    )


def _concat(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)


def _date_lists(by_ticker: dict[str, list[pd.Timestamp]]) -> list[str]:
    lines = [
        f"- {t}: {len(ds)}일 "
        + ", ".join(d.date().isoformat() for d in ds[:DETAIL_LIMIT])
        + (" ..." if len(ds) > DETAIL_LIMIT else "")
        for t, ds in by_ticker.items()
        if ds
    ]
    return lines or ["없음"]
