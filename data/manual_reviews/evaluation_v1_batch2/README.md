# evaluation_v1 수동 검수 배치 2

이 배치는 기존 동결 benchmark artifact에서 두 Agent 모드가 모두 첫 시도에 `passed=true`로 종료한 신규 5개 source를 짝지어 검수한다. `agent`와 `agent_rag` 각 5행, 총 10행이다.

목적은 Agent PASS가 사람 판단에서도 실제 PASS인지 확인하고, 배치 1에서 계산할 수 없었던 불필요 수정 판단의 기반을 만드는 것이다. 선정 과정은 실행 trace만 사용했으며 번역 품질 정답을 미리 생성하거나 추론하지 않았다.

## 선정 조건

- 배치 1의 5개 source와 중복 없음
- 두 모드 모두 첫 judgment가 `passed=true`, `error_types=[]`, `next_action=accept`
- attempt 1회, retry 0회, `stop_reason=passed`
- warning, revision call, component failure 없음
- 29개 적격 source 중 길이와 glossary trace가 다른 5개 선정
- 선택 ID: `evaluation-v1-008`, `012`, `013`, `015`, `038`

선정된 glossary trace는 서버, 배포, 소프트웨어, 오픈 소스, 클라우드 컴퓨팅, 데이터베이스를 포함한다. 이는 행동 기반 표본이며 전체 평가셋을 대표하는 통계 표본이 아니다.

## 사람이 입력하는 필드

- `manual_initial_needs_revision`: `true` 또는 `false`
- `manual_severity`: 초기 번역에 수정이 필요한 경우에만 `MAJOR` 또는 `MINOR`. `false`이면 반드시 비워 둔다.
- `manual_primary_error`: 수정이 필요할 때 `term`, `meaning`, `omission_addition`, `entity_value`, `fluency_grammar`, `other` 중 하나
- `manual_error_types`: 수정이 필요할 때 오류 유형 JSON 배열
- `pairwise_outcome`: `improved`, `same`, `worse`
- `review_status`: `confirmed` 또는 `ambiguous`
- `reviewer`: 검수자 식별값
- `note`: 판단 근거. `ambiguous` 또는 `other`일 때 필수

Agent가 PASS했다고 해서 `manual_initial_needs_revision=false`를 자동 입력하면 안 된다. source와 initial translation을 사람이 독립적으로 비교해 판단한다. `manual_severity`도 오류 유형이나 Agent 출력으로부터 자동 생성하지 않는다. `initial_translation`과 `final_translation`이 같아도 `pairwise_outcome=same`은 사람이 직접 입력한다.

## 현재 상태

- `selection.json`: `manual_review_schema_version=2`와 artifact·dataset·run/config·reference provenance, 행 순서를 고정
- `review_labels.csv`: 수동 필드가 모두 비어 있는 Git 추적용 원본
- `review_labels_reviewed.csv`: Sup이 확정한 10개 라벨을 별도 ingest한 파일
- `ingestion.json`: 완료 XLSX, 빈 템플릿, 선택 manifest와 출력 CSV의 SHA-256 및 immutable evidence 검증 결과
- `scores.json`: 기존 benchmark artifact만 사용해 생성한 offline score
- `verification.json`: 보호 입력 해시, 예상/실제 지표 일치, 테스트 실행 결과를 기록한 감사 자료
- `tests/test_manual_review_batch2_result.py`: 실제 배치2 산출물과 scorer 재현성을 검증하는 회귀 테스트

완료 XLSX는 `outputs/01a02074-03fa-7530-a959-1f4de7b5c360/evaluation_v1_manual_review_batch2_sup_reviewed.xlsx`이며 SHA-256은 `26fb34a945d88f31b0c06e66d0a1f0fbdb73a2600d6ba1076a7abbdf27084acb`이다. 10행 모두 `READY`, reviewer `Sup`, `review_status=confirmed`이며, ingest가 동결 evidence와 manifest 행 순서를 다시 검증했다. 빈 템플릿과 원본 XLSX, 기존 artifact·동결 데이터셋·배치 1 라벨과 점수는 수정하지 않았다.

## 오프라인 채점 결과

- 사람 판단: 수정 필요 4건(`MAJOR` 4), 수정 불필요 6건
- Agent 판단: TP 0 / TN 6 / FP 0 / FN 4, 정확도 60.0%, 수정 필요 recall 0.0%
- Agent와 Agent+RAG: 각각 TN 3 / FN 2, 정확도 60.0%
- pairwise outcome: `same` 10건. 수정 대상 4건 중 개선 0건으로 successful correction rate 0.0%
- component failure: 0건

이 수치는 Agent가 처음부터 PASS한 사례를 행동 기준으로 선별한 10행 표본에만 해당한다. 전체 benchmark 품질 수치로 일반화하지 않는다.

## 검수 완료 후 ingest와 채점

사람이 10행을 모두 작성한 별도 XLSX만 아래 명령에 넣는다. ingest는
`row_status`를 신뢰하지 않고 수동 필드와 동결 evidence를 다시 검증하며,
원본 빈 CSV를 덮어쓰지 않는다.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/ingest_manual_review_workbook.py \
  --kind full \
  --sheet-name Review \
  --workbook /path/to/completed_batch2.xlsx \
  --selection-manifest data/manual_reviews/evaluation_v1_batch2/selection.json \
  --baseline-csv data/manual_reviews/evaluation_v1_batch2/review_labels.csv \
  --output-csv data/manual_reviews/evaluation_v1_batch2/review_labels_reviewed.csv \
  --provenance-output data/manual_reviews/evaluation_v1_batch2/ingestion.json \
  --project-root .

HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  scripts/score_manual_review.py \
  --artifact artifacts/benchmark-429d6c4a-a1b6-4514-bfd3-dab6966c4101.json \
  --dataset data/evaluation_v1.jsonl \
  --selection-manifest data/manual_reviews/evaluation_v1_batch2/selection.json \
  --review-csv data/manual_reviews/evaluation_v1_batch2/review_labels_reviewed.csv \
  --output data/manual_reviews/evaluation_v1_batch2/scores.json \
  --project-root .
```
