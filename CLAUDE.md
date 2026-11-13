# finance-agents 작업 규칙

이 저장소에서 작업할 때 따른다. 개인 전역 규칙(`~/.claude/CLAUDE.md`)과 겹치면 이 파일을 따른다.
코드와 명령은 `swing-lab/`에서 실행한다 (`uv run ...`).

## Git

- 작업은 main에서 새 브랜치를 만들어 한다 (`git switch --no-track -c <브랜치명> main`).
- 브랜치 생성 시에는 upstream을 걸지 않는다. 첫 push 때 `git push -u origin <브랜치명>`으로 같은 이름의 원격 브랜치만 추적한다. 작업 브랜치가 origin/main을 추적하면 안 된다.
- 커밋마다 테스트와 ruff가 통과해야 한다 (`swing-lab/`에서 `uv run pytest`, `uv run ruff check .`).
  - 커밋 전에 `git diff --cached --stat`으로 스테이징된 파일을 확인한다.
- PR은 merge commit 방식으로 머지한다.
  - `gh pr merge <번호> --merge --match-head-commit <헤드 SHA>`
  - 머지 후 로컬 main fast-forward와 원격 브랜치 삭제까지를 기본으로 한다.
- `backup/*` 브랜치는 지우지 않는다.

관례
- 브랜치 이름: 기능은 `feature/<영문-슬러그>`, 경량 작업은 `fix/…`, `chore/…`
- 커밋 메시지: `type: 한국어 설명` (feat, fix, docs, chore, style, test)

## 데이터와 민감 정보

- `data/`, `results/`, `reports/`, `paper/`, `private/`는 커밋하지 않는다 (`.gitignore`에 있음, `.gitkeep`만 예외).
- `private/holdings.csv`는 읽거나 만들지 않는다. `swinglab rebalance` 확인은 합성 보유 내역으로만 한다.
- 실험 로그(`results/experiments.parquet`)의 행은 지우지 않는다. 덧붙이기만 한다.

## 먼저 확인받을 작업

- 검증 구간(holdout) 실행: `swinglab holdout`
- launchd 등록·해제: `launchctl bootstrap`, `launchctl bootout`
- 페이퍼 고정(frozen) 해시 범위 파일의 수정: 수정 전에 알린다. 바뀌면 `swinglab paper run`이 거부되고 새 `paper_id`가 필요하다.
  - `swing-lab/config/paper.yaml`
  - `config/paper.yaml`의 `source_files`에 적힌 전략·지표 파일: `strategies/base.py`, `strategies/breakout.py`,
    `strategies/rsi2_swing.py`, `indicators/channel.py`, `indicators/moving_average.py`, `indicators/rsi.py`

## 실험 원칙 (관례)

- 스펙에서 사전 고정한 값(구간, 선정 규칙, 정책 비중, 목표)은 결과를 보고 바꾸지 않는다. 바꾸려면 스펙을 먼저 고치고 그 사실을 기록한다.
- 실데이터 백테스트는 실험 로그를 거치는 명령으로만 돌린다 (로그에 남지 않는 시도를 만들지 않는다).
- 단계별 결과는 `swing-lab/docs/stageN-results.md`에 정리한다.
