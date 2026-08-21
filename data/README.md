# Data and Evaluation Records

이 디렉터리는 실행기가 읽는 평가 JSONL과 glossary CSV를 보존한다. 대용량 원본 snapshot, EDA 결과, 후보 pool, benchmark artifact, 검수 sheet는 `artifacts/`에 따로 둔다.

## 1. 현재 데이터 상태

| 파일 | 역할 | 상태 |
| --- | --- | --- |
| `pilot_v1.jsonl` | 파이프라인·API·실험용 합성 12문장 | 최종 과제 평가셋 아님 |
| `glossary_pilot.csv` | 합성 파일럿용 glossary와 치환 규칙 | 최종 수치에 직접 사용하지 않음 |
| `glossary_evaluation_v1.csv` | 공개 데이터 후보 검색 전에 고정한 초기 10용어 | 이력 보존용 |
| `glossary_evaluation_v2.csv` | source-only 커버리지 EDA 후 6용어를 더한 16용어 | 현재 후보 pool/materializer 기준; 최종 평가 출력으로 조정하지 않음 |
| `evaluation_selection_v1_draft.json` | AI 보조 선정 manifest | 40문장·8용어, `human_confirmed=false` |
| `reference_reviews/evaluation_v1_draft.json` | 공개 reference를 덮어쓰지 않는 교정 overlay | 외부 의견을 독립 검토한 AI 보조 초안, `human_confirmed=false` |
| `evaluation_v1_draft.jsonl` | 현재 manifest와 reference overlay로 materialize한 구조·사람 검수용 초안 | 40문장·8용어·교정 6건, `benchmark_allowed=false`; 최종 평가셋 아님 |
| `evaluation_selection_v1.json` | 사용자 승인 근거와 hash를 보존한 최종 manifest | 40문장·8용어, `human_confirmed=true` |
| `reference_reviews/evaluation_v1.json` | 선정 40건의 최종 결정 overlay | 원 reference 34건 유지·6건 교정 |
| `evaluation_v1.jsonl` | 확정한 공개 데이터 기반 평가셋 | 40문장·8용어, `benchmark_allowed=true` |

`glossary_evaluation_v2.csv`에 추가한 용어는 한국어 source 커버리지만 보고 정했다. 영어 reference와 모델 출력은 추가 용어 선정에 사용하지 않았다. 이는 누수 방지 조치이지, 각 항목의 최종 정답성을 증명한다는 뜻은 아니다.

현재 `artifacts/eda/evaluation_candidates.jsonl`에 source-only 방식으로 고정한 154개 후보가 있다. 선정 ID를 고정한 후에만 수동 정렬 검수를 위해 영어 reference를 join했다. `artifacts/reviews/evaluation_candidates_review_{a,b,c}.json`은 후보를 나눈 AI 보조 검수 초안이며 모두 `human_confirmed=false`이다. KEEP 표시도 gold 확정이 아니다.

최초 초안을 종합한 `evaluation_v1_draft.jsonl`은 materializer의 30~50문장·5용어 이상 구조 gate를 40문장·8용어로 통과했다. 그 후 외부 번역 의견을 독립 검토해 manifest의 5·35번 선정을 교체하고, 8·9·11·16·30·35번 reference 교정을 `reference_reviews/evaluation_v1_draft.json`에 overlay로 기록했다. 공개 후보 pool의 reference는 변경하지 않았다.

초안 XLSX의 행별 결정 셀은 저장된 입력 없이 비어 있었다. 이후 사용자가 이 대화에서 검수 완료를 명시적으로 확인했고, 시스템은 이를 현재 40개 선정과 effective reference의 일괄 승인으로 기록했다. 행별 셀을 사용자가 채웠다고 소급하지 않으며, 승인 문구·reviewer·UTC 시각·workbook hash를 최종 manifest와 overlay의 감사 이력에 남겼다. 그 결과 `evaluation_v1.jsonl`은 40문장·8용어, 원 reference 34건 유지·6건 교정 상태로 동결되었다.

## 2. 최종 평가셋 gate

최종 평가셋은 다음 gate를 통과해 동결했다.

1. 사람이 AI 검수 초안과 원문·reference를 직접 비교한다.
2. 한영 정렬, software 도메인 의미, glossary target 적합성, 중복, 난이도를 확인한다.
3. 30~50개 `source_record_id`와 서로 다른 용어 5개 이상을 담은 manifest를 확정한다.
4. 확정 시점의 candidate·glossary SHA-256을 manifest에 보존한다.
5. 사람 확정 manifest만 `evaluation_v1.jsonl`로 materialize한다.
6. 해당 파일에서만 최종 4-way benchmark를 실행한다.

선정 manifest의 필수 형태는 다음과 같다. hash는 코드상 선택 필드이지만, 최종 재현성을 위해 반드시 기록한다.

```json
{
  "human_confirmed": true,
  "candidate_sha256": "<sha256>",
  "glossary_sha256": "<sha256>",
  "reference_review_file": "reference_reviews/evaluation_v1.json",
  "reference_review_sha256": "<sha256>",
  "selected": [
    {
      "source_record_id": "<candidate source_record_id>",
      "selection_note": "<human-readable reason>"
    }
  ]
}
```

materializer는 선정 개수, 중복 ID, candidate/glossary/overlay hash, 선정 순서, 원 reference hash, 용어 커버리지를 검사한다. `human_confirmed=false`인 manifest는 기본적으로 거부된다. 최종 overlay는 선택된 모든 행의 결정을 1:1로 포함하고 reviewer와 검수 시각도 기록해야 한다.

구조 검사용 AI 초안을 명시적으로 만들 때만 다음과 같이 실행한다.

```bash
python scripts/materialize_evaluation_dataset.py \
  --candidates artifacts/eda/evaluation_candidates.jsonl \
  --manifest data/evaluation_selection_v1_draft.json \
  --glossary data/glossary_evaluation_v2.csv \
  --reference-review data/reference_reviews/evaluation_v1_draft.json \
  --output data/evaluation_v1_draft.jsonl \
  --summary-output artifacts/eda/evaluation_v1_draft_summary.json \
  --allow-unconfirmed-draft
```

이 결과의 summary는 `status=AI_ASSISTED_DRAFT_NOT_HUMAN_CONFIRMED`, `benchmark_allowed=false`이어야 한다. 최종 확정은 `scripts/finalize_evaluation_review.py`가 candidate·draft manifest·overlay·workbook hash와 대화 승인 근거를 검증한 뒤 별도 `evaluation_selection_v1.json`과 `reference_reviews/evaluation_v1.json`을 생성하는 방식으로 수행했다. 그 뒤 draft flag 없이 다음을 실행했다.

```bash
python scripts/materialize_evaluation_dataset.py \
  --candidates artifacts/eda/evaluation_candidates.jsonl \
  --manifest data/evaluation_selection_v1.json \
  --glossary data/glossary_evaluation_v2.csv \
  --reference-review data/reference_reviews/evaluation_v1.json \
  --output data/evaluation_v1.jsonl \
  --summary-output artifacts/eda/evaluation_v1_summary.json
```

boolean을 바꾸는 것만으로는 사람 검수가 된 것이 아니다. 이번 확정자·검수 시각·대화 승인 기준·workbook hash는 manifest 메타데이터와 별도 검수 기록에 보존되었다.

## 3. 평가 JSONL 스키마

`pilot_v1.jsonl`과 `evaluation_v1.jsonl`은 한 줄에 하나의 `EvaluationCase` JSON object를 저장한다.

| 필드 | 설명 |
| --- | --- |
| `case_id` | 데이터셋 내 고유 ID |
| `source_record_id` | 원 공개 데이터의 revision·shard·row를 보존한 ID |
| `source_text` | runtime에 전달할 한국어 원문 |
| `reference_text` | 오프라인 평가용 영어 reference |
| `reference_provenance` | 공개 원 reference와 hash, effective reference의 출처, 교정 상태·근거를 보존하는 오프라인 감사 이력 |
| `domain` | 현재 단일 도메인 `software` |
| `scenario_tags` | 용어·복수 용어 등 분석 태그 |
| `selection_note` | 선정 근거와 주의사항 |
| `expected_terms` | `source_term`과 `accepted_targets` 목록 |
| `manual_judgments` | Agent 조건별 첫 판정의 confirmed/ambiguous gold label |
| `manual_outcomes` | Agent 조건별 initial→final `IMPROVED/SAME/WORSE` label |

materializer는 candidate의 `hit_terms`와 고정 glossary를 사용해 `expected_terms`를 만든다. 최종 사람 검수에서 문맥상 해당 용어가 필수 대상인지도 다시 확인해야 한다.

## 4. Glossary CSV 스키마

```text
glossary_version
term_id
domain
source_term_ko
preferred_target_en
accepted_variants_json
disallowed_variants_json
replacement_rules_json
definition
source
notes
```

`source`에는 용어의 독립적인 근거를 기록한다. 최종 평가 reference나 최종 모델 출력을 보고 target, accepted variant, disallowed variant, replacement rule을 추가하지 않는다.

## 5. 누수 방지 경계

- 후보 추출 시 한국어 source와 미리 고정한 glossary의 `source_term_ko`만 선정에 사용한다.
- 영어 reference는 선정 ID가 고정된 후 사람의 bilingual alignment 검수용으로만 join한다.
- `EvaluationCase.to_translation_request()`는 `source_text`만 runtime 요청으로 보낸다.
- `reference_text`, `expected_terms`, `manual_judgments`, `manual_outcomes`는 Retriever·번역기·Agent 입력으로 보내지 않는다.
- 최종 실행 전 dataset, glossary, config·model version과 SHA-256을 고정한다.
- 최종 결과를 본 뒤 문장을 제외하거나 prompt·임계값·치환 규칙을 재조정하지 않는다.

## 6. Benchmark artifact와 수동 판정

benchmark runtime artifact에는 reference를 복사하지 않고, 다음을 추적한다.

- benchmark artifact의 dataset·config hash와 condition별 model version. glossary hash는 최종 manifest와 `artifacts/eda/evaluation_v1_summary.json`에 별도로 고정한다.
- source 입력, 검색 결과, 각 번역 후보, Agent 판정
- retry 횟수, 선택한 attempt, 중단 이유, rollback 여부
- retrieval·translation·Agent·전체 latency
- 자동 계산 가능한 용어 지표와 문장 변경률

`scripts/export_manual_review.py`는 runtime artifact와 dataset reference를 오프라인에서 join해 빈 CSV 검수 sheet를 만든다. 평가셋 자체의 한영 정렬·용어·선정 확정 근거는 final manifest·overlay·검수 workbook에 보존한다. 초안 workbook의 행별 결정 셀은 비어 있었고, 이후 대화에서 이루어진 일괄 승인을 별도로 기록했다. 어느 검수표도 사람 판정을 자동 생성하지 않는다.

첫 실제 4-way benchmark는 run ID `429d6c4a-a1b6-4514-bfd3-dab6966c4101`로 160결과를 생성했고 빈 번역은 없었다. 이는 품질 튜닝 전의 기능 기준선이다. Agent 출력 수동 라벨은 아직 없으므로 Agent 판단 정확도와 성공적 수정률은 `0`이 아니라 미산출이다.

confirmed `PASS/NEEDS_REVISION`이 있을 때만 Agent 판정 정확도·confusion counts를 계산한다. confirmed initial label과 `IMPROVED/SAME/WORSE` outcome이 함께 있을 때만 성공적 수정률을 계산한다. 라벨이 없으면 이 지표는 `0`이 아니라 미산출이며, 사유를 `unavailable_metrics`에 남긴다.

## 7. ID와 변경 규칙

- 모든 ID는 파일 간 join 후에도 유일해야 한다.
- JSONL은 UTF-8, 한 줄당 하나의 JSON object로 저장한다.
- glossary와 평가셋을 바꾸면 version을 올리고 변경 이유와 hash를 남긴다.
- 최종 평가 시작 후 기존 행을 덮어쓰지 않는다. 수정이 필요하면 새 version을 만들고 제외 전 결과와 사유를 보존한다.
- 설정을 바꾼 재실행은 새 run ID와 config hash를 사용한다.
