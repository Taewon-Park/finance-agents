# swing-lab

미국 섹터 ETF 일봉 스윙 매매 백테스트 엔진 (0단계). 스펙: `../stage0-spec.md`

## 실행

`swing-lab/` 디렉터리에서 실행한다. Python 3.11+, uv.

```bash
uv sync                                    # 가상환경 + 의존성
uv run swinglab ingest                     # ETF 20개 일봉 수집 (재실행 시 증분)
uv run swinglab check                      # 데이터 품질 점검 → reports/data_quality.txt
uv run swinglab backtest -s rsi2_pullback  # buy_hold | sma_cross | rsi2_pullback → reports/<전략>.html
uv run pytest                              # 자동 테스트
```

- `backtest`의 리포트는 plotly.js를 CDN에서 불러온다 (열 때 인터넷 필요, 약 0.9MB). `--offline`을 주면 파일에 포함한다 (약 5.7MB).
- `uv run python scripts/strategy_summary.py`: 전략 × 비용 0/5/10bp 비교 요약표(마크다운). CAGR, MDD, 샤프, 노출도, 연 거래 횟수, 같은 기간 SPY CAGR.
- `ingest --full`: 전체 재수집. `--tickers SPY,QQQ`: 일부만. 기본 마지막 날은 장 마감된 최근 거래일.
- 증분 수집은 최근 5거래일을 겹쳐 받아 비교하고, 수정주가가 달라졌으면(새 배당·분할) 그 종목만 전체를 다시 받는다.
- `data/`, `reports/` 결과물은 git에 넣지 않는다.

## 엔진 규칙 요약

- t일 종가까지의 신호 → t+1일 시가 체결. 롱 온리, 소수 주식, 동일 비중 `1 / max_positions`, 리밸런싱 없음.
- 신규 진입 후보: 목표 1이고 신호 강도 값이 있는 미보유 종목. 강도 큰 순, 동점이면 티커 알파벳순.
- 보유 종목: 목표 0이면 청산, 신호 없음(NaN)이면 유지, `max_hold_days`에 닿으면 청산.
- 비용: 체결금액 × 편도 bp를 매수·매도 각각. 거래 수익률 = 청산가(1−c) / 진입가(1+c) − 1.
- 시가가 없는 날: 청산은 다음 시가로 미루고 진입은 취소. 평가는 직전 종가.
- 백테스트 첫 체결일 = max(설정 start, 워밍업 일수번째 거래일, 둘째 거래일). 자산곡선 첫 행은 그 전날(초기자본).

## 전략 인터페이스

스펙의 `Strategy`에 엔진이 쓰는 항목을 더했다 (`src/swinglab/strategies/base.py`).

| 항목 | 의미 |
| --- | --- |
| `generate(prices)` | date × ticker 목표 상태. 1 보유, 0 미보유, NaN 신호 없음 |
| `strength(prices)` | 신호 강도 (클수록 우선). 신규 진입 후보가 아니면 NaN |
| `max_positions` | 동시 보유 수. None이면 `backtest.yaml` 값 (바이앤홀드 = 1) |
| `max_hold_days` | 실제 진입일부터 센 최대 보유 거래일 (RSI(2) = 10) |

## 검증 기록

- `notebooks/manual_check.ipynb`: RSI(2) 거래 무작위 5건을 엔진과 독립된 계산으로 대조
- `notebooks/external_check.ipynb`: SPY 수익률을 SSGA 공개 총수익률과 비교
- `uv run python scripts/spy_crosscheck.py`: 위 비교를 재현. 기준은 체결 시점(익일 시가) 차이를 보정한 값 기준 연 0.1%p 이내, 통과 시 종료 코드 0
