# 모델과 프레임워크 선정 이유 및 한계

## 1. 선정 원칙

과제 조건에 맞춰 모든 핵심 추론을 로컬에서 실행하고 외부 유료 API를 사용하지
않았다. 네 비교 조건은 같은 기본 번역기를 공유하며, 모델 선택은 16GB Apple
Silicon 환경에서 재현 가능한 크기와 다운로드·실행 비용을 우선했다.

## 2. 선택 요약

| 영역 | 선택 | 선정 이유 | 핵심 한계 |
|---|---|---|---|
| 번역 | `Helsinki-NLP/opus-mt-ko-en` | 공개 한→영 Marian, 로컬 CPU 실행, 조건 간 동일 기준선 | 도메인 용어와 주어·entity 누락, prompt-native glossary 부재 |
| Agent | Ollama `qwen3:1.7b` | 소형 로컬 multilingual instruct, 구조화 JSON, 16GB 환경 적합 | 약 11초 경로 latency, false PASS, 출력 불안정 |
| 임베딩 | `paraphrase-multilingual-MiniLM-L12-v2` | 한국어·영어 지원, 비교적 작고 로컬 실행 가능 | 작은 literal glossary에서 vector noise |
| 검색 | exact-first hybrid + in-memory cosine | exact의 높은 정밀도와 vector fallback, 작은 glossary에 충분 | 대규모 index·영속성·운영 확장성 부족 |
| API | FastAPI + Pydantic | 필수 프레임워크, 자동 스키마와 입력 검증 | 인증·rate limit·queue 미구현 |
| Workflow | 명시적 Python controller | 상태·retry·fallback·trace를 직접 감사 가능 | LangGraph류의 시각화·checkpoint 기능을 직접 구현해야 함 |

## 3. 번역 모델: Marian OPUS-MT

### 선택 이유

- `Helsinki-NLP/opus-mt-ko-en`은 한국어→영어 방향이 명확한 공개 모델이다.
- 유료 번역 API 없이 CPU에서 실행할 수 있다.
- Baseline, RAG, Agent, Agent+RAG가 같은 NMT를 사용하므로 개선 원인을 분리하기
  쉽다.
- 첫 실행 이후 warm inference가 가능하며 프로토타입 규모의 40문장 평가에
  충분했다.

### 검토·실험한 대안

- `force_words_ids`: glossary target을 강제했지만 반복 토큰과 퇴행 출력이 발생해
  채택하지 않았다.
- source augmentation: 한국어 source에 영어 용어 설명을 붙였으나 NMT가 이를
  안정적으로 반영하지 못했다.
- NMT fine-tuning: 과제 규모와 시간, 데이터 누수 위험에 비해 비용이 커서 첫
  구현 범위에서 제외했다.

### 실제 한계

- `배포`를 `distribution`으로 번역하는 도메인 용어 오류가 있었다.
- 문두의 `도커`, `로드 밸런싱`, `컨설` 같은 standalone 주어를 `It`으로 바꾸거나
  누락했다.
- 자연스러운 문장이라도 원문의 고유명사와 핵심 의미가 사라질 수 있다.
- seq2seq 번역 모델이라 LLM prompt처럼 arbitrary context를 안전하게 주입할 수
  없다.

## 4. Agent LLM: Qwen3 1.7B via Ollama

### 선택 이유

- 소형 모델이라 16GB 메모리의 로컬 macOS 환경에서 NMT와 순차 실행할 수 있다.
- 한국어 source와 영어 candidate를 함께 읽고 JSON 형태로 결과를 반환할 수 있다.
- Ollama로 모델 관리와 로컬 HTTP 호출을 단순화하며 유료 API 의존을 없앴다.
- Judge와 Reviser를 같은 배포 단위로 연결해 MVP를 빠르게 검증할 수 있었다.

### 실제 한계

- 동결 평가에서 Agent 경로 평균 latency는 약 11.2~11.8초로 Baseline 약
  0.44초보다 크게 느렸다.
- 수동 진단 표본에서 Agent 판단 정확도는 63.2%, 수정 필요 recall은 46.2%였다.
- 오류를 summary에서 언급하고도 구조화 결과는 PASS로 반환하는 내부 모순이
  관찰됐다.
- 계약에 없는 `pronoun` error type으로 component failure가 1건 발생했다.
- 같은 모델이 번역 수정과 품질 판정을 수행해 자기확증 편향이 생길 수 있다.
- 작은 모델은 긴 설명, 복잡한 JSON, 여러 제약을 동시에 만족시키는 데 불안정하다.

### 대안과 후속 방향

- 3B급 로컬 instruct 모델을 같은 고정 replay에서 비교할 수 있지만 latency와
  메모리 비용을 함께 측정해야 한다.
- Judge는 deterministic entity/term guard와 결합하고, semantic 판단만 LLM에
  맡기는 편이 안전하다.
- production에서는 Judge와 Reviser 역할 분리 또는 별도 verifier를 검토할 수 있다.

## 5. 임베딩과 검색

### 임베딩 모델 선정 이유

`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`는 한국어와 영어를
같은 공간에 임베딩할 수 있고 대형 생성 모델보다 작다. 첫 검색 시에만 lazy-load해
Baseline 경로의 시작 비용을 피했다.

### 검색 정책 선정 이유

파일럿 12문장에서 exact retrieval은 기대 용어를 12/12 찾았다. 순수 vector는
8/12 hit와 8개의 wrong hit으로 precision 50%, recall 66.7%에 그쳤다. 따라서
exact 결과를 우선하고 exact가 부족할 때만 vector를 보조하는 safe hybrid를
채택했다.

### 한계

- glossary가 작고 source term이 명시적일 때 vector 검색은 이득보다 noise가 클 수
  있다.
- in-memory cosine은 프로토타입에는 충분하지만 영속 index, 대규모 corpus,
  동시성에는 적합하지 않다.
- PDF의 FAISS/Chroma는 권장 예시다. 현재 구현은 vector 검색 요구는 충족하지만,
  운영형 backend까지 구현한 것은 아니다.
- 동의어·음역어는 독립적인 provenance가 없으면 검색 hit만으로 정답이라고 확정할
  수 없다.

## 6. FastAPI와 Pydantic

FastAPI는 과제에서 지정된 API 프레임워크이며 request/response 스키마와 OpenAPI
문서를 자동으로 제공한다. Pydantic은 1~5000자 입력, 실행 모드, retry 범위,
trace 일관성을 API 경계에서 검증한다.

한계는 현재 로컬 단일 사용자 프로토타입이라는 점이다. 인증, 요청 제한, 비동기
작업 큐, 모델 worker 분리, 장애 복구와 배포 설정은 제출 범위에 포함하지 않았다.

## 7. Workflow 프레임워크

LangChain/LangGraph를 의무적으로 사용하지 않고 작은 결정적 controller를 직접
구현했다. 단계가 `analyze → retrieve → translate → judge → conditional revise`로
짧고 retry 한도가 2회이므로, 직접 구현이 call count와 stop reason을 명확히
보여준다.

한편 단계가 늘거나 장기 실행·중단 복구가 필요해지면 LangGraph의 checkpoint,
상태 시각화, persistence가 유리할 수 있다. 현재 구현은 이런 운영 기능을 자체적으로
확장해야 한다.

## 8. 하드웨어·재현성 제약

- 개발 환경: Apple M1 Pro, RAM 16GB, macOS
- 모델은 lazy-load하며 여러 대형 모델을 동시에 상주시키지 않는다.
- 첫 모델 다운로드·load 시간은 warm request latency와 분리한다.
- 디스크와 메모리 여유에 따라 Ollama와 Hugging Face cache가 실행의 실질적
  제약이 될 수 있다.
- 모델 revision·Ollama build·Transformers 버전 차이가 결과를 바꿀 수 있으므로
  제출 artifact의 dataset/config/model ID와 hash를 함께 보존한다.

## 9. 결론

선택한 조합은 최고 품질 모델을 목표로 한 것이 아니라, 제한된 로컬 환경에서
RAG와 Agent의 기여를 분리해 재현 가능한 워크플로우를 만드는 선택이다. 자동 용어
정확도에서는 효과가 있었지만 Agent 판단 recall과 latency가 병목으로 확인됐다.
따라서 모델 교체보다 먼저 deterministic coverage, 구조화 판정 일관성, retry
regression guard를 작은 고정 replay로 검증하는 것이 합리적이다.

