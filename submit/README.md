# Agentic Translation QA 제출 문서 인덱스

## 목적

이 폴더는 `RAG + Agent 기반 Agentic Translation QA` 과제의 제출 문서 묶음이다.

**실제 제출본은 `FINAL_REPORT.pdf` 한 파일이며 총 4페이지다.** 흑백 보고서 형식으로
작성했고 표에만 옅은 회색을 사용했다. 1~8번 Markdown 문서는 수치와 근거를 확인하기
위한 선택적 참고자료로, 별도 요구가 없다면 제출 파일에 포함하지 않아도 된다.

- 제출용 문서와 시각화 파일만 둔다.
- 이미 구현된 코드, 평가 데이터셋, 수동 검수 자료와 벤치마크 결과는 복사하지 않고 근거 자료로 참조한다.
- 각 모델과 프레임워크의 **선정 이유와 한계**를 반드시 기록한다.

## 필수 API 확인 결과

필수 API 3개는 모두 `src/translation_qa/main.py`에 구현되어 있고 관련 API 테스트도 존재한다.

| 엔드포인트 | 코드 확인 | 제출 문서에 추가할 내용 |
|---|---|---|
| `POST /translate/baseline` | 완료 | Request/response, 예외 응답, 실행 예시 |
| `POST /translate/agent-rag` | 완료 | Request/response, 재시도 루프, trace 예시 |
| `POST /benchmark` | 완료 | 평가셋·지표·응답 결과 명세 |

따라서 API 코드를 새로 만들 필요는 없다. 최종 스모크 테스트와 예제 입·출력은 `04_API_SPECIFICATION.md`에 정리한다.

## 작성할 제출 문서

| 순서 | 파일 | 핵심 내용 | 상태 |
|---:|---|---|---|
| 1 | [`01_SYSTEM_ARCHITECTURE_AND_WORKFLOW.md`](01_SYSTEM_ARCHITECTURE_AND_WORKFLOW.md) | 아키텍처, Agent 워크플로우, 최대 2회 재시도, trace와 종료 조건 | 완료 |
| 2 | [`02_MODEL_AND_FRAMEWORK_SELECTION.md`](02_MODEL_AND_FRAMEWORK_SELECTION.md) | 번역 모델, Agent LLM, 임베딩, 벡터 검색, 워크플로우의 선정 이유와 한계 | 완료 |
| 3 | [`03_DATASET_AND_EDA.md`](03_DATASET_AND_EDA.md) | 공개 데이터셋, 30~50문장, 핵심 용어 5개 이상, reference provenance, EDA와 시각화 | 완료 |
| 4 | [`04_API_SPECIFICATION.md`](04_API_SPECIFICATION.md) | 필수 API 3개의 명세, 예제, 오류 처리, 스모크 테스트 결과 | 완료 |
| 5 | [`05_BENCHMARK_RESULTS.md`](05_BENCHMARK_RESULTS.md) | Baseline vs Agent-RAG, 4-way 보조 실험, 지표, latency, 재시도 분포 | 완료 |
| 6 | [`06_ERROR_AND_CASE_ANALYSIS.md`](06_ERROR_AND_CASE_ANALYSIS.md) | 오류 유형과 빈도, 근본 원인, 성공·실패 각 3건 이상의 심화 분석 | 완료 |
| 7 | [`07_LIMITATIONS_AND_IMPROVEMENTS.md`](07_LIMITATIONS_AND_IMPROVEMENTS.md) | 현재 구조의 한계, 일반화 제약, 개선 우선순위 | 완료 |
| 8 | [`08_REPRODUCIBILITY_AND_SUBMISSION.md`](08_REPRODUCIBILITY_AND_SUBMISSION.md) | 설치·실행·테스트·벤치마크 재현 명령과 제출 체크리스트 | 완료 |
| 9 | [`FINAL_REPORT.md`](FINAL_REPORT.md) / `FINAL_REPORT.pdf` | 1~8번 문서를 통합한 최종 리포트와 제출용 PDF | 완료 |

## 모델·프레임워크 문서 필수 항목

`02_MODEL_AND_FRAMEWORK_SELECTION.md`에는 다음 대상을 모두 포함한다.

1. 번역 모델
2. Agent 추론용 로컬 LLM
3. 임베딩 모델
4. 벡터 검색 백엔드
5. Agent 워크플로우 구현 방식

각 대상에는 최소한 다음 내용을 기록한다.

- 실제 선택과 사용 버전
- 과제 목표와 로컬 실행 환경에 적합한 이유
- 검토한 대안
- 외부 유료 API 미사용 여부
- 메모리, 속도 및 하드웨어 제약
- 한·영 언어 품질의 한계
- 실험에서 실제로 확인된 한계
- 남아 있는 위험과 향후 대안

## 기존 근거 자료

| 기존 파일 | 활용 목적 |
|---|---|
| `src/translation_qa/main.py` | 필수 API 라우트 근거 |
| `src/translation_qa/pipeline.py` | Agent-RAG 파이프라인과 trace 근거 |
| `src/translation_qa/config.py` | 모델·백엔드·실행 설정 |
| `src/translation_qa/benchmark.py` | 벤치마크 지표 산출 근거 |
| `docs/EVALUATION_CONTRACT.md` | 평가 계약과 지표 정의 |
| `docs/EXPERIMENT_LOG.md` | 실험 이력, 결과와 재현성 근거 |
| `docs/improvement_reports/01_initial_system_baseline.md` | 현재 구조·벤치마크·수동 검수 요약 |
| `data/evaluation_v1.jsonl` | 동결 평가셋 |
| `data/manual_reviews/` | Agent 수동 검수와 채점 근거 |
| `artifacts/` | 동결 벤치마크 결과와 EDA 산출물 |

## 제출 전 체크리스트

- [x] Python 3.11+와 FastAPI 사용 명시
- [x] 외부 유료 API 미사용 확인
- [x] 오픈소스 모델과 로컬 실행 방식 명시
- [x] 벡터 검색 기반 RAG 설명
- [x] 최대 2회 재시도와 판단 로그 설명
- [x] 30~50문장, 핵심 용어 5개 이상, reference 포함 확인
- [x] EDA 구조·분포·선정 근거와 시각화
- [x] 용어 정확도와 문장 단위 수정률
- [x] Agent 재시도 분포와 수동 검수 대비 판단 정확도
- [x] Baseline vs Proposed 평균 응답 시간
- [x] 오류 유형별 빈도와 근본 원인
- [x] 성공·실패 각 최소 3건의 심화 분석
- [x] 모델·프레임워크 선정 이유와 한계
- [x] 현재 구조의 한계와 개선 아이디어
- [x] 재현 명령, 파일 경로와 라이선스 확인

## 권장 열람 순서

1. `01_SYSTEM_ARCHITECTURE_AND_WORKFLOW.md`
2. `02_MODEL_AND_FRAMEWORK_SELECTION.md`
3. `03_DATASET_AND_EDA.md`
4. `04_API_SPECIFICATION.md`
5. `05_BENCHMARK_RESULTS.md`
6. `06_ERROR_AND_CASE_ANALYSIS.md`
7. `07_LIMITATIONS_AND_IMPROVEMENTS.md`
8. `08_REPRODUCIBILITY_AND_SUBMISSION.md`
9. `FINAL_REPORT.md`

빠른 검토는 `FINAL_REPORT.md`부터 시작하고, 평가 근거나 재현 세부사항이 필요할 때
1~8번 문서를 확인한다. `assets/`에는 보고서에서 사용하는 EDA·benchmark·manual
review·architecture 시각화가 들어 있다.
