# Evaluation Contract

## 1. 목적과 범위

이 문서는 Baseline, RAG, Agent가 번역 품질과 응답 시간에 미치는 영향을 같은 기준으로 비교하기 위한 평가 계약이다.

- 번역 방향: 한국어 → 영어
- 최종 평가셋: 단일 도메인의 30~50문장
- 핵심 용어: 서로 다른 용어 5개 이상
- 파일럿: 최종 평가셋과 분리한 10~12문장
- 최종 결과는 비율뿐 아니라 항상 `분자/분모`를 함께 보고한다.

파일럿에서는 모델과 기준을 조정할 수 있지만, 최종 평가를 시작한 뒤에는 데이터, glossary, prompt, 임계값 및 디코딩 설정을 바꾸지 않는다.

## 2. 4-way ablation

| 조건 | 초기 번역 | RAG | Agent 판단 | 추가 시도 |
| --- | --- | --- | --- | --- |
| Baseline | NMT 1회 | 없음 | 없음 | 없음 |
| RAG only | 검색 결과를 반영한 NMT 1회 | 있음 | 없음 | 없음 |
| Agent only | Baseline 번역 | 없음 | 있음 | 최대 2회 |
| RAG + Agent | RAG-only 번역 | 있음 | 있음 | 최대 2회 |

비교 시 다음을 고정한다.

- 네 조건은 같은 평가 문장과 같은 기본 번역 모델·디코딩 설정을 사용한다.
- `Agent only`의 초기 후보는 해당 문장의 Baseline 결과와 같아야 한다.
- `RAG + Agent`의 초기 후보는 해당 문장의 RAG-only 결과와 같아야 한다.
- 두 Agent 조건은 같은 Agent 모델, 수정기, 종료 규칙과 재시도 한도를 사용한다.
- Agent-only에는 glossary나 검색 결과를 제공하지 않는다.
- 추가 시도 2회는 초기 번역 이후의 시도를 뜻하므로 후보는 최대 3개다.
- 최종 결과는 항상 마지막 후보가 아니라 rollback을 포함해 선택한 최선 후보로 한다.
- reference 번역은 어떤 조건에서도 실행 입력으로 제공하지 않는다.

## 3. 지표 정의

### 3.1 용어 정확도

평가 단위는 source에 표시된 필수 용어 occurrence다.

```text
Terminology Accuracy = 올바른 필수 용어 occurrence 수 / 전체 필수 용어 occurrence 수
```

preferred target 또는 사전에 등록한 accepted variant를 문맥에 맞게 사용하면 정답이다. 누락, 원문 방치, 금지 번역, 다른 의미로 사용한 경우는 오답이다. 문자열 규칙은 초벌 판정에만 사용하고 복수형·하이픈·문맥상 중의성은 사람이 확정한다.

다음 결과를 함께 보고한다.

- 전체 occurrence 기준 정확도
- 용어별 정확도
- 필수 용어를 모두 맞힌 문장 비율
- 조건 간 정답 occurrence 증감 수

문장 전체 의미 오류와 용어 오류는 별도로 센다. 예를 들어 문장의 의미가 틀렸어도 표시된 용어가 맞으면 해당 occurrence의 용어 점수는 정답이다.

### 3.2 문장 단위 수정률

Agent가 있는 두 조건에만 적용한다.

```text
Sentence Modification Rate = final과 initial이 다른 문장 수 / 전체 문장 수
```

비교 전 Unicode 정규화, 앞뒤 공백 제거, 연속 공백 축약만 수행한다. 대소문자와 문장부호 변화는 실제 수정으로 센다. Agent가 없는 조건은 `0`이 아니라 `N/A`다.

수정 자체를 개선으로 간주하지 않고 다음을 분리해서 보고한다.

- 재시도 발생률
- 사람이 판정한 `IMPROVED / SAME / WORSE` 수
- 수정된 문장 중 `IMPROVED` 비율
- 이전 후보를 선택한 rollback 비율

### 3.3 Agent 판단 정확도

수동 정답은 `PASS`와 `NEEDS_REVISION`으로 구분하며, positive class는 `NEEDS_REVISION`이다.

```text
Accuracy = (TP + TN) / N
Revision Recall = TP / (TP + FN)
Unnecessary Revision Rate = FP / (FP + TN)
```

`NEEDS_REVISION`은 다음 중 하나라도 해당할 때 부여한다.

- 필수 용어의 오역 또는 누락
- 의미 왜곡, 핵심 내용 누락 또는 추가
- 고유명사·숫자·날짜·단위 오류
- 이해나 의미에 영향을 주는 심한 문법 오류

단순한 문체 취향이나 더 자연스러운 대안이 있다는 이유만으로는 수정 대상으로 보지 않는다. 주 지표는 각 문장의 **첫 후보에 대한 첫 Agent 판단**이며, 모든 retry 판정의 정확도는 보조 지표다. Agent-only와 RAG + Agent의 초기 후보가 다르면 수동 정답도 각각 별도로 판정한다.

## 4. 오류 taxonomy

번역 결과의 오류와 파이프라인 원인을 분리한다. 한 결과에 여러 오류 유형을 허용하되 `primary_error` 하나와 `error_types` 목록을 함께 기록한다. 심각도는 `MINOR` 또는 `MAJOR`로 기록한다.

### 번역 품질 오류

| 코드 | 의미 |
| --- | --- |
| `NONE` | 수정이 필요한 오류 없음 |
| `TERM` | 필수 용어 오역·누락·불일치 |
| `MEANING` | 문맥 또는 중의성 해소 실패를 포함한 의미 왜곡 |
| `OMISSION_ADDITION` | 원문 정보 누락 또는 없는 정보 추가 |
| `ENTITY_VALUE` | 고유명사, 숫자, 날짜, 단위 오류 |
| `FLUENCY_GRAMMAR` | 유창성·문법 문제 |
| `OTHER` | 기타 오류이며 설명 필수 |

### 파이프라인 원인

`RETRIEVAL_MISS`, `RETRIEVAL_WRONG`, `TERM_NOT_APPLIED`, `AGENT_FALSE_PASS`, `AGENT_FALSE_REVISE`, `RETRY_NO_CHANGE`, `RETRY_REGRESSION`, `PARSE_OR_RUNTIME_ERROR`를 사용한다. 이 값은 사후 분석용이며 Agent 입력으로 제공하지 않는다.

## 5. 수동 판정 규칙

- source와 독립적으로 만든 glossary를 우선 기준으로 보고 reference는 판단 보조 자료로만 사용한다.
- 가능하면 condition 이름을 가린 상태에서 번역을 검토한다.
- `manual_initial_needs_revision=true`이면 검수자가 `manual_severity`를 정확히 `MAJOR` 또는 `MINOR`로 입력한다. `false`이면 severity는 반드시 비워 둔다.
- `MAJOR`는 핵심 의미·필수 용어·중요 정보·고유명사/수치 오류처럼 그대로 수용하기 어려운 오류, `MINOR`는 핵심 의미를 보존하지만 배포 전 국소 수정이 필요한 오류로 판정한다.
- severity는 Agent 판단, 오류 taxonomy, reference 또는 문자열 규칙에서 자동 생성하거나 대소문자를 정규화하지 않는다.
- initial과 final 비교는 `IMPROVED`, `SAME`, `WORSE` 중 하나로 판정한다.
- 애매한 사례는 `review_status=AMBIGUOUS`로 표시해 재검토하며, 확정 전 Agent 정확도 분모에 넣지 않는다.
- 한 문장을 여러 번 retry해도 첫 판단 정확도에서는 한 번만 센다.

## 6. latency 계약

주 latency는 모델과 인덱스를 이미 로드한 warm request 기준이다.

- batch size 1로 순차 실행한다.
- 조건별 warm-up 후 같은 설정으로 3회 측정한다.
- monotonic high-resolution timer를 사용한다.
- GPU 비동기 실행 시 측정 전후 synchronize한다.
- 모델 로드와 인덱스 구축 시간은 제외하고 `startup_ms`로 별도 기록한다.
- 전체 시간은 retrieval, translation, Agent 판단, retry와 orchestration을 모두 포함한다.
- 전체 시간과 단계별·attempt별 시간을 모두 저장한다.

평균, median, p95, Baseline 대비 추가 시간과 배수를 보고한다. 품질 판정은 고정된 출력 한 세트로 수행하고, latency 반복 실행 결과를 별도 품질 표본처럼 중복 집계하지 않는다.

## 7. 최종 평가셋 누수 방지

- `reference_en`과 정답 용어 occurrence는 평가 코드만 읽을 수 있는 데이터에 둔다.
- runtime에는 `sample_id`와 `source_ko`만 전달한다.
- glossary는 평가 reference 문장에서 추출하거나 복사하지 않고 독립 자료로 만든다.
- 파일럿과 최종 평가셋의 원문 중복·근접 중복을 확인한다.
- prompt, retrieval top-k, threshold, retry 규칙은 파일럿에서 결정한 후 최종 실행 전에 버전과 hash를 고정한다.
- 최종 결과를 본 뒤 일부 문장만 제외하거나 설정을 재조정하지 않는다. 제외가 불가피하면 사유와 제외 전 결과를 함께 남긴다.
- 실행마다 dataset, glossary, index, model 및 config 버전을 기록한다.

## 8. 파일럿 채택 기준

아래 값은 최종 통계적 주장보다는 다음 단계 진행 여부를 판단하는 단순 기준이다.

- RAG: 정답 용어 occurrence가 3개 이상 증가하거나 정확도가 10%p 이상 상승하며, 새 `MAJOR` 오류 문장이 1개 이하여야 한다.
- Agent: 첫 판단 정확도 80% 이상이고 `MAJOR` 오류 false-pass가 2건 이하여야 한다.
- Retry: `IMPROVED` 수가 `WORSE` 수보다 많고 `MAJOR` regression이 1건 이하여야 한다.
- 전체 시스템: RAG-only의 용어 정확도를 유지하면서 최종 수용 가능 문장 수를 늘려야 한다.

이 기준은 파일럿 종료 시 한 번 확정하고 최종 평가 중에는 변경하지 않는다.
