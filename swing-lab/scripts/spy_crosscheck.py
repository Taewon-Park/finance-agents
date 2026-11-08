"""SPY 바이앤홀드 외부 교차 확인 재현 (0단계 완료 기준).

기준: 공개 총수익률과 체결 시점(익일 시가) 차이를 보정한 값 기준 연 0.1%p 이내.
실행: swing-lab 디렉터리에서 `uv run python scripts/spy_crosscheck.py`
data/clean/prices.parquet이 필요하다 (`swinglab ingest`). 통과하면 종료 코드 0, 아니면 1.

공개 값 출처: State Street(SSGA) SPY 펀드 성과 데이터, 2026-11-08 조회
https://www.ssga.com/bin/v1/ssmp/fund/fundfinder?country=us&language=en&role=intermediary&product=etfs&ui=fund-finder
시장가격(Market Value) 기준 연환산 총수익률(%). 새 기준일 값을 비교하려면 PUBLISHED에 추가한다.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from swinglab.backtest.crosscheck import PublishedReturn, buy_hold_crosscheck
from swinglab.pipeline.storage import clean_path, read_prices

ROOT = Path(__file__).resolve().parents[1]
TICKER = "SPY"
TOLERANCE_PP = 0.1

PUBLISHED: list[PublishedReturn] = [
    PublishedReturn(date(2026, 9, 30), 1, 15.66),
    PublishedReturn(date(2026, 9, 30), 3, 22.73),
    PublishedReturn(date(2026, 9, 30), 5, 13.66),
    PublishedReturn(date(2026, 9, 30), 10, 15.18),
    PublishedReturn(date(2026, 10, 31), 1, 11.97),
    PublishedReturn(date(2026, 10, 31), 3, 23.25),
    PublishedReturn(date(2026, 10, 31), 5, 11.93),
    PublishedReturn(date(2026, 10, 31), 10, 15.29),
]

LABELS = {
    "as_of": "기준일",
    "years": "기간(년)",
    "start_close_date": "시작 종가일",
    "end_date": "끝",
    "published_pct": "공개(%)",
    "close_to_close_pct": "종가→종가(%)",
    "engine_pct": "엔진(%)",
    "first_day_gap_pct": "첫날 갭(%)",
    "engine_adjusted_pct": "엔진 보정(%)",
    "diff_adjusted_pp": "보정 차이(%p)",
    "diff_unadjusted_pp": "보정 전 차이(%p)",
}


def main() -> int:
    path = clean_path(ROOT / "data")
    if not path.exists():
        print(f"{path}가 없습니다. 먼저 swinglab ingest를 실행하세요.", file=sys.stderr)
        return 1
    table = buy_hold_crosscheck(read_prices(path), TICKER, PUBLISHED)
    print(table.rename(columns=LABELS).round(3).to_string(index=False))

    worst = table["diff_adjusted_pp"].abs().max()
    passed = bool(worst <= TOLERANCE_PP)
    print(f"\n보정 값 기준 최대 차이 {worst:.3f}%p (기준 ±{TOLERANCE_PP}%p) → {'통과' if passed else '실패'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
