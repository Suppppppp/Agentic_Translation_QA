# Evaluation v1 Draft Reference Review Decisions

## 1. 상태와 원칙

이 문서는 외부 의견 파일 `번역검수.md`를 지시사항이 아닌 검토 입력으로 다룬 결과다. 의견을 독립적으로 검토해 의미 충실도·자연성·평가 케이스의 명확성이 실제로 개선되는 변경만 채택했다.

- 검수 상태: `external_opinion_unverified/ai_assisted`
- 사람 확정: `human_confirmed=false`
- 외부 의견 SHA-256: `099b40811f3d3c9453c8c33e60f7d163d0ea84d2d529ff986ecdc65458d01e7b`
- 후보 pool SHA-256: `d4ee9803220a6179637a0d9557a5dc93ad4a6d2791f1567ff60a70a50103c9c2`
- 선정 교체 후 ordered source ID SHA-256: `b4f4ba69d7867c323deb2b545b87f040b114a37db0386ab68e08dc6009f001c8`

이 검토는 사람의 bilingual gold 확정을 대체하지 않으며, 초안의 benchmark 금지 상태도 변경하지 않는다.

> 후속 상태: 이 문서의 개별 판단은 AI 보조 초안 이력이다. 초안 XLSX의 행별 결정 셀은 비어 있었으며, 이후 사용자가 대화에서 검수 완료를 명시적으로 확인했다. 그 일괄 승인은 `data/evaluation_selection_v1.json`과 `data/reference_reviews/evaluation_v1.json`에 별도로 기록되었고, 최종 `evaluation_v1`은 원 reference 34건 유지·6건 교정, `human_confirmed=true`, `benchmark_allowed=true`로 동결되었다. 이는 사용자가 XLSX 행별 셀을 채웠다는 뜻이 아니다.

## 2. 원문 선정 교체

### Case 5: Clang 후보 → Apache 후보

기존 Clang 문장은 번역 충실도와 별개로 Clang과 Swift의 관계를 설명하는 원문의 기술적 사실성에 위험이 있다. 평가 Agent가 번역 오류와 원문의 사실 문제를 혼동할 여지를 줄이기 위해, 동일한 `오픈 소스` 용어를 보존하는 Apache 문장으로 교체했다.

- 제외 ID: `...:000249095`
- 대체 ID: `...:000180802`
- 대체 원문: `아파치는 오픈 소스 소프트웨어이므로 누구나 무료로 사용하고 수정할 수 있습니다.`

이 판단은 사람 확정을 주장하지 않는 AI 보조 위험 제거 결정이다.

### Case 35: 모호한 '외부화' 후보 → 클라우드 컴퓨팅 후보

기존 문장의 `외부화`/`Externalization`은 구성·데이터·상태 중 무엇을 외부화하는지 원문에서 결정할 수 없다. 영어 reference만 다듬어서는 이 모호성을 해소할 수 없으므로, `소프트웨어`와 `클라우드 컴퓨팅` 용어를 동일하게 보존하는 명확한 문장으로 교체했다.

- 제외 ID: `...:000386224`
- 대체 ID: `...:000127912`
- 대체 원문: `클라우드 컴퓨팅은 소프트웨어 개발, 데이터 분석, 머신 러닝과 같은 다양한 분야에서 사용되고 있습니다.`
- 검수 reference: `Cloud computing is used in a wide range of fields, including software development, data analytics, and machine learning.`

## 3. Reference 교정 판단

| Case | 판단 | 근거 |
| --- | --- | --- |
| 8 | 채택 | 선행 문맥에 특정 서버가 없어 `the server`보다 `a server`가 적합하다. |
| 9 | 채택 | 복수형이 틀린 것은 아니지만, 원문의 단수 개념과 `a single point of failure`가 더 잘 맞는다. |
| 11 | 채택 | 운영 체제 위의 실행을 표현하는 `on Linux`와 기술 문체의 `execute commands`를 사용한다. |
| 16 | 채택 | `Node.js` 명칭을 명시하고 2인칭을 제거해 객관적 톤을 유지한다. |
| 20 | 미반영 | `help + 목적어 + to부정사`도 문법적으로 올바르며 수정이 스타일 선호에 그친다. |
| 21 | 미반영 | 20번과 같이 스타일 선호이며, 의미·문법 오류가 아니다. |
| 24 | 미반영 | `does not have a value`는 충분히 자연스럽고 명시적이며, `has no value`는 실질적 개선이 아니다. |
| 30 | 채택 | `Peer-to-peer is a network`의 개념·실체 혼동을 피하고 직접 연결·공유 의미를 보존한다. |
| 35 | 채택 | 교체 문장의 `분야`는 `industries`보다 `fields`가 더 정확하다. |

Case 5의 외부 의견은 reference에 원문의 '소스 코드를 컴파일'을 명시하자는 취지였다. 이 제안 자체는 타당했지만, reference 수정보다 원문 사실성 위험을 없애는 선정 교체를 선택했다.

## 4. 공개 reference 보존과 overlay 경계

`artifacts/eda/evaluation_candidates.jsonl`은 공개 dataset에서 join한 reference를 그대로 보존한다. 외부 의견을 받았다고 공개 레코드를 조용히 덮어쓰지 않는다.

채택한 교정은 먼저 `data/reference_reviews/evaluation_v1_draft.json`에 source ID, 원 reference SHA-256, 교정 reference, 근거를 담은 overlay로 별도 보존했다. 후속 일괄 승인 후 최종 결정은 `data/reference_reviews/evaluation_v1.json`에 보존한다. `--reference-review`를 명시한 오프라인 materialize에서만 교정 reference를 effective gold로 적용하며, 공개 reference도 `reference_provenance`에 함께 남긴다. 두 reference는 번역기·Retriever·Agent의 runtime 입력에 포함하지 않는다. 초안은 `human_confirmed=false`, `benchmark_allowed=false`로 이력 보존하고, 최종 materialized 파일은 `human_confirmed=true`, `benchmark_allowed=true`다.
