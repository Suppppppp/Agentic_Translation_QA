# evaluation_v1 수동 검수 배치 1

이 디렉터리는 동결된 `evaluation_v1` 벤치마크에서 실행 trace 패턴만으로 고른 5개 사례 × 2개 모드(`agent`, `agent_rag`)의 수동 검수 기록을 Git으로 추적한다. 전체 평가셋을 대표하는 통계 표본이 아니라 검수·채점 흐름을 확인하기 위한 부분 표본이다.

## 작업 흐름

1. 제공된 XLSX 검수표에서 10개 행의 수동 필드를 사람이 직접 입력한다.
2. 다음 작업에서 XLSX에 입력된 값을 해석하거나 보정하지 않고 `review_labels.csv`의 같은 `review_key` 행으로 그대로 복사한다.
3. 오프라인 채점기가 확인된 CSV만 읽어 이 디렉터리의 Git 추적 파일 `scores.json`을 만든다.

원본 `data/evaluation_v1.jsonl`과 기존 benchmark artifact는 동결 상태로 유지한다. 이 흐름에서는 어느 파일도 수정하거나 다시 벤치마크하지 않는다. `selection.json`은 파일 SHA-256, run ID, config SHA-256, reference-review/source-feedback SHA-256과 행 순서를 고정한다. 생성기는 기존 `review_labels.csv`를 덮어쓰지 않으므로 입력된 라벨을 실수로 지우지 않는다.

## 사람이 입력하는 필드와 허용값

- `manual_initial_needs_revision`: `true` 또는 `false`. 초기 번역에 수정이 필요했는지를 판단한다.
- `manual_primary_error`: `term`, `meaning`, `omission_addition`, `entity_value`, `fluency_grammar`, `other` 중 하나. `false`이면 비워 두고, `true`이면 주 오류를 고른다.
- `manual_error_types`: 위 오류 유형의 JSON 배열(예: `["term","meaning"]`). `false`이면 비워 두고, `true`이면 하나 이상 입력한다.
- `pairwise_outcome`: 최종 번역을 초기 번역과 비교해 `improved`, `same`, `worse` 중 하나를 사람이 고른다. 번역 문자열이 같아도 자동으로 `same`을 넣지 않는다.
- `review_status`: 판단을 확정할 수 있으면 `confirmed`, 근거가 모호하면 `ambiguous`.
- `reviewer`: 검수자를 식별할 수 있는 비어 있지 않은 값.
- `note`: 선택 입력. `ambiguous` 또는 `other`를 사용한 경우에는 판단 근거를 반드시 기록한다.

나머지 열은 동결된 데이터셋과 benchmark trace에서 복사한 증거이므로 수정하지 않는다. 이 배치의 수동 필드는 reviewer `Sup`이 확인한 XLSX에서 그대로 ingest했으며 자동 생성·보정하지 않았다.

## 완료 상태

- `review_labels.csv`: 10행 모두 `confirmed`, reviewer `Sup`
- `ingestion.json`: 검수 XLSX·선정 명세·ingest 전후 CSV SHA-256과 증거 일치 여부
- `scores.json`: 기존 benchmark artifact만 사용한 오프라인 채점 결과
- 전체: 판단 가능 9건, Agent 판단 정확도 66.7%, `improved` 5건, 수정 성공률 50.0%, component failure 1건

이 수치는 trace 패턴으로 고른 10행의 부분 표본 결과이며 전체 평가셋 성능으로 해석하지 않는다.

## 빈 CSV 재생성

라벨 입력 전 새 경로에만 다음 명령을 실행할 수 있다.

```bash
python scripts/prepare_manual_review_batch.py \
  --selection data/manual_reviews/evaluation_v1_batch1/selection.json \
  --output /tmp/evaluation_v1_batch1_review_labels.csv \
  --project-root .
```

`review_labels.csv`는 사람이 입력할 추적 대상 원본이므로 위 명령으로 덮어쓸 수 없다.

## 오프라인 채점

XLSX의 10개 행을 모두 검수한 뒤, 입력값을 해석하거나 자동 보정하지 않고 같은 순서의 `review_labels.csv` 수동 필드에 옮긴다. 그 다음 아래 명령으로 기존 artifact와 동결 데이터셋의 해시·증거 열을 검증하고 별도 결과를 생성한다.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/score_manual_review.py \
  --artifact artifacts/benchmark-429d6c4a-a1b6-4514-bfd3-dab6966c4101.json \
  --dataset data/evaluation_v1.jsonl \
  --selection-manifest data/manual_reviews/evaluation_v1_batch1/selection.json \
  --review-csv data/manual_reviews/evaluation_v1_batch1/review_labels.csv \
  --output data/manual_reviews/evaluation_v1_batch1/scores.json \
  --project-root .
```

수동 필드가 하나라도 비어 있거나 일부만 채워졌으면 채점기는 중단하고 `scores.json`을 만들지 않는다. 결과에는 Agent 판정 혼동행렬·정확도, 수정 recall, 불필요 수정률, 수정 성공률, 오류 분포와 component failure 수가 전체 및 모드별로 기록된다. 이 10행은 워크플로우 확인용 부분 표본이므로 전체 benchmark 품질 수치로 해석하지 않는다.
