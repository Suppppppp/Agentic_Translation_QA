# evaluation_v1 배치 1 severity 보충 검수

이 디렉터리는 이미 사람이 확정한 배치 1의 기존 라벨을 변경하지 않고
`manual_severity`만 추가로 검수하기 위한 별도 보충 배치다. 원본 배치 1의
`review_labels.csv`, `selection.json`, `ingestion.json`, `scores.json`과 기존
XLSX는 수정하거나 덮어쓰지 않는다.

## 입력 규칙

- `manual_initial_needs_revision=true`인 행만 사람이 `manual_severity`를
  `MAJOR` 또는 `MINOR`로 입력한다.
- `manual_severity`는 오류 유형으로부터 자동 추론하거나 자동 채우지 않는다.
- severity를 입력한 행은 `severity_reviewer`를 반드시 입력한다.
- `severity_note`는 선택 입력이며 판단 근거를 남길 때 사용한다.
- `manual_initial_needs_revision=false`인 행은 세 severity 입력 칸을 모두
  비워 둔다.
- `source_text`, `initial_translation`, 기존 오류 라벨과 키는 배치 1에서
  복사한 동결 증거이므로 수정하지 않는다.

판정할 때 `MAJOR`는 핵심 의미·주요 용어·개체·중요 정보 누락처럼 번역의
사용 가능성을 크게 해치거나 국소 수정만으로 해결하기 어려운 오류를 뜻한다.
`MINOR`는 핵심 의미를 유지하면서 제한된 문구·문법·유창성 수정으로 해결할
수 있는 오류를 뜻한다. 기존 오류 코드만으로 severity를 자동 대응시키지
말고 각 번역을 사람이 직접 판단한다.

이 실제 보충 배치는 기존 배치 1의 10행 모두
`manual_initial_needs_revision=true`이므로 10행 모두 사람이 severity를
판단해야 한다.

## 파일

- `selection.json`: 원본 배치 1 selection/review CSV와 benchmark·dataset
  provenance 해시를 고정한 manual-review schema v2 manifest
- `severity_labels.csv`: 보충 검수표의 Git 추적용 빈 템플릿. 새 severity,
  severity reviewer, severity note만 비어 있으며 나머지는 원본 배치 1에서
  그대로 복사했다.

## 생성 재현

새 경로에만 다음 명령을 실행한다. 생성기는 기존 출력을 덮어쓰지 않는다.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/prepare_severity_supplement.py \
  --base-selection data/manual_reviews/evaluation_v1_batch1/selection.json \
  --base-review-csv data/manual_reviews/evaluation_v1_batch1/review_labels.csv \
  --output-selection data/manual_reviews/evaluation_v1_batch1_severity_rebuild/selection.json \
  --output-csv data/manual_reviews/evaluation_v1_batch1_severity_rebuild/severity_labels.csv \
  --project-root .
```

## 검수 완료 후 ingest

XLSX의 `Severity Review` 시트 입력이 끝나면 아래와 같이 **새 출력 경로**로
ingest한다. `row_status` 값은 신뢰하지 않고 Python 검증기가 조건을 다시
검사한다. 성공 시 scorer가 읽을 수 있는 전체 v2 review CSV와 provenance
JSON을 만든다.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/ingest_manual_review_workbook.py \
  --kind severity-supplement \
  --sheet-name "Severity Review" \
  --workbook /path/to/completed_batch1_severity.xlsx \
  --selection-manifest data/manual_reviews/evaluation_v1_batch1_severity/selection.json \
  --baseline-csv data/manual_reviews/evaluation_v1_batch1_severity/severity_labels.csv \
  --output-csv data/manual_reviews/evaluation_v1_batch1_severity/review_labels_with_severity.csv \
  --provenance-output data/manual_reviews/evaluation_v1_batch1_severity/ingestion.json \
  --project-root .
```

ingest가 만든 전체 v2 CSV는 기존 배치 1과 같은 frozen artifact를 대상으로
아래처럼 채점한다. 이 명령은 모델이나 benchmark를 실행하지 않는다.

```bash
HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  scripts/score_manual_review.py \
  --artifact artifacts/benchmark-429d6c4a-a1b6-4514-bfd3-dab6966c4101.json \
  --dataset data/evaluation_v1.jsonl \
  --selection-manifest data/manual_reviews/evaluation_v1_batch1_severity/selection.json \
  --review-csv data/manual_reviews/evaluation_v1_batch1_severity/review_labels_with_severity.csv \
  --output data/manual_reviews/evaluation_v1_batch1_severity/scores.json \
  --project-root .
```

## 완료 상태

2026-08-21에 project owner `Sup`이 10행을 모두 검수했다. 모든 행은
`READY`, `manual_severity=MAJOR`이며 새 라벨은 완료 workbook에서만 ingest했다.

- `review_labels_with_severity.csv`: 기존 배치 1 전체 CSV를 보존하면서
  `manual_severity`만 추가한 schema v2 결과
- `ingestion.json`: 완료 workbook·빈 template·기존 batch 1 원본의 경로와
  SHA-256, reviewer별 severity annotation을 기록한 provenance
- `scores.json`: 기존 frozen benchmark artifact에 대한 offline score. 모델,
  번역 또는 benchmark를 다시 실행하지 않았다.

이 10행은 대표 표본이므로 전체 번역 품질을 주장하는 데 사용하지 않는다.
component failure 1행은 human severity와 correction 집계에는 남기되 Agent
판단 정확도 및 MAJOR false-pass 계산에서는 제외한다.

현재 offline score 요약:

- confirmed 10건, Agent 판단 가능 9건, component failure 1건
- Agent 판단 정확도 66.7% (`TP=6`, `FN=3`)
- `MAJOR=10`, `MINOR=0`, MAJOR false pass 3건
- successful correction 5/10, 50.0%
