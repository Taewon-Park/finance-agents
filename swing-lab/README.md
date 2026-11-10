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

## 1단계: 전략 비교 (`../stage1-spec.md`)

```bash
uv run swinglab experiments run -f A        # A, B, C 순서로 학습 구간 실행 (비용 0/5/10bp, 실험 로그 자동 기록)
uv run swinglab experiments run -f D        # A·C 후보가 정해진 뒤 시장 필터 비교
uv run swinglab experiments summary         # 계열별 시도 횟수, 샤프 분포
uv run swinglab experiments select          # 후보 선정 규칙 → config/candidates.yaml (커밋할 것)
uv run swinglab holdout --candidates config/candidates.yaml   # 검증 구간, 같은 후보로는 1회
uv run swinglab report stage1               # reports/stage1.html (results/ 파일만 읽음)
```

- 구간·예산·파라미터 후보·선정 규칙은 `config/stage1.yaml`. 바꾸려면 스펙을 먼저 고친다.
- 학습 실행은 가격을 2018-12-31까지 잘라서 넘기므로 검증 구간 데이터가 코드 경로에 없다.
- 결과물: `results/experiments.parquet`(실험 로그), `equity_curves.parquet`, `benchmarks.parquet`, `holdout.lock`. git 제외.

스펙에서 정한 해석
- A: 청산은 조건이 성립한 그날 보유일 ≥ min_hold일 때만 (엔진 `min_hold_days`). 진입·청산 같은 날이면 청산 우선.
- B: 월말 종가 대비 lookback개월 수익률. IEF 대체는 자리마다 별도 포지션(IEF#1, IEF#2 …)이라 비중 = 대체 자리 수 / top_n.
- C: 돌파한 날에만 진입 후보.
- 규칙 3: 1순위의 인접 조합 샤프 평균 < 1순위 × 0.7이면 고립, '자기+인접' 평균 최고 조합 선택. 1순위 샤프 ≤ 0이면 생략.
- D: 필터 버전이 필터 전보다 샤프 0.05 이상 높은 것 중 최고 1개. 필터 전 sma_cross는 baseline으로 기록, 시도 수 제외.

## 2단계: 페이퍼 트레이딩 + 분기 리밸런싱 (`../stage2-spec.md`)

```bash
uv run swinglab ingest                      # 0단계 20종 + EFA·EEM·BIL(collect_only) + 원/달러 환율(FRED DEXKOUS)
uv run swinglab paper run                   # 매일 1회 (launchd). 수집 → 고정 확인 → 놓친 거래일까지 하루씩 처리
uv run swinglab paper report                # 시작일 이후 누적 지표, 같은 기간 SPY, 변동성 맞춤 SPY 혼합
uv run swinglab alloc backtest              # P0~P3 × 0/5/10bp (서술용) → reports/stage2_alloc.html, .md
uv run swinglab rebalance --contribution 3000000   # private/holdings.csv → private/rebalance/<YYYY-Qn>.md
```

- 페이퍼: `config/paper.yaml`(C breakout 20/10, A rsi2_v2 (5, 5))과 전략 소스 해시를 첫 실행 때 `paper/frozen.json`에 고정한다. 다르면 거부하고, 바꾸려면 `paper_id`를 새로 정한다. 첫 실행일이 시작일이다.
- 배분: `config/alloc.yaml`. 커밋된 상태에서만 `alloc backtest`가 돈다. 실험 로그에는 `family=alloc`, 시도는 정책 4개까지.
- `rebalance`는 실제 계좌와 연결하지 않는다. 보유 내역·결과 문서는 `private/` 아래에만 둔다 (git 제외).
- `paper/`, `private/`는 git에 넣지 않는다.

## 검증 기록

- `notebooks/manual_check.ipynb`: RSI(2) 거래 무작위 5건을 엔진과 독립된 계산으로 대조
- `notebooks/external_check.ipynb`: SPY 수익률을 SSGA 공개 총수익률과 비교
- `uv run python scripts/spy_crosscheck.py`: 위 비교를 재현. 기준은 체결 시점(익일 시가) 차이를 보정한 값 기준 연 0.1%p 이내, 통과 시 종료 코드 0
