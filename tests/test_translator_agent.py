from __future__ import annotations

import json
from contextlib import nullcontext

import pytest

from translation_qa.agent import OllamaTranslationAgent, RuleBasedTermAgent
from translation_qa.errors import ComponentExecutionError
from translation_qa.schemas import (
    JudgmentConsistencyIssue,
    MustPreserveConstraint,
    NextAction,
    QualityJudgment,
    RetrievalHit,
    RetrievalMatchType,
    SourceCoverageKind,
    TermConstraint,
    TranslationCandidate,
    TranslationErrorType,
)
from translation_qa.translator import MarianTranslator


class FakeHTTPResponse:
    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(
            {"message": {"content": '{"domain": null, "key_terms": [], "confidence": 0}'}}
        ).encode("utf-8")


class FakeTensor:
    def to(self, device: str) -> FakeTensor:
        return self


class FakeTokenizedTarget:
    def __init__(self, ids: list[int]) -> None:
        self.input_ids = ids


class FakeTokenizer:
    def __call__(self, text: str, **kwargs: object) -> object:
        if kwargs.get("return_tensors") == "pt":
            return {"input_ids": FakeTensor()}
        return FakeTokenizedTarget([len(text), 7])

    def decode(self, output: object, *, skip_special_tokens: bool) -> str:
        return "Constrained deployment completed."


class FakeModel:
    def __init__(self) -> None:
        self.generation_args: dict[str, object] = {}

    def generate(self, **kwargs: object) -> list[list[int]]:
        self.generation_args = kwargs
        return [[1, 2, 3]]


class FakeTorch:
    @staticmethod
    def inference_mode() -> nullcontext[None]:
        return nullcontext()


def test_marian_translator_forwards_lexical_constraints_without_loading_model() -> None:
    translator = MarianTranslator(num_beams=2)
    fake_model = FakeModel()
    translator._tokenizer = FakeTokenizer()
    translator._model = fake_model
    translator._torch = FakeTorch()
    translator._device = "cpu"

    result = translator.translate(
        "배포를 완료했다.",
        constraints=[
            TermConstraint(
                source_term="배포",
                target_term="deployment",
                retrieval_hit_id="t1",
                target_variants=["deploy"],
            )
        ],
    )

    assert result.text == "Constrained deployment completed."
    assert fake_model.generation_args["force_words_ids"] == [
        [[10, 7], [6, 7]]
    ]
    assert fake_model.generation_args["num_beams"] == 4


def test_marian_translator_rejects_empty_source() -> None:
    translator = MarianTranslator()
    with pytest.raises(ValueError, match="empty"):
        translator.translate("   ")


def test_constraint_degeneracy_guard_detects_runaway_repetition() -> None:
    repeated = " ".join(["(Applause) (Applause) (Applause)"] * 12)
    assert MarianTranslator._looks_degenerate("짧은 원문", repeated)
    assert MarianTranslator._looks_degenerate(
        "짧은 원문",
        "A result. (Applause.) (Laughter.) (Music.) deployment",
    )
    assert not MarianTranslator._looks_degenerate(
        "새 버전을 배포한다.",
        "Deploy the new version.",
    )


def test_rule_based_agent_detects_missing_retrieved_term() -> None:
    agent = RuleBasedTermAgent()
    hit = RetrievalHit(
        term_id="t1",
        source_term="배포",
        target_term="deployment",
        domain="software",
        match_type=RetrievalMatchType.EXACT,
        score=1.0,
    )

    failed = agent.judge(
        "새 버전을 배포한다.",
        TranslationCandidate(text="Release the new version.", model_id="fake"),
        [hit],
    )
    passed = agent.judge(
        "새 버전을 배포한다.",
        TranslationCandidate(text="Start the deployment.", model_id="fake"),
        [hit],
    )

    assert not failed.passed
    assert failed.error_types == [TranslationErrorType.TERM]
    assert failed.next_action is NextAction.RETRY_WITH_CONSTRAINTS
    assert passed.passed


def test_rule_based_agent_accepts_documented_target_variant() -> None:
    agent = RuleBasedTermAgent()
    hit = RetrievalHit(
        term_id="t1",
        source_term="배포",
        target_term="deployment",
        accepted_target_variants=["deploy", "deployed"],
        domain="software",
        match_type=RetrievalMatchType.EXACT,
        score=1.0,
    )

    judgment = agent.judge(
        "새 버전을 배포한다.",
        TranslationCandidate(text="Deploy the new version.", model_id="fake"),
        [hit],
    )

    assert judgment.passed


def test_rule_based_agent_does_not_match_variant_inside_larger_word() -> None:
    agent = RuleBasedTermAgent()
    hit = RetrievalHit(
        term_id="t1",
        source_term="지속적 통합",
        target_term="continuous integration",
        accepted_target_variants=["CI"],
        domain="software",
        match_type=RetrievalMatchType.EXACT,
        score=1.0,
    )

    judgment = agent.judge(
        "지속적 통합을 사용한다.",
        TranslationCandidate(text="This is a special process.", model_id="fake"),
        [hit],
    )

    assert not judgment.passed


def test_rule_based_agent_does_not_enforce_term_inside_korean_compound() -> None:
    agent = RuleBasedTermAgent()
    hit = RetrievalHit(
        term_id="deployment",
        source_term="배포",
        target_term="deployment",
        domain="software",
        match_type=RetrievalMatchType.VECTOR,
        score=0.6,
    )

    judgment = agent.judge(
        "우분투는 리눅스 배포판이다.",
        TranslationCandidate(text="Ubuntu is a Linux distribution.", model_id="fake"),
        [hit],
    )

    assert judgment.passed


def test_ollama_agent_validates_analysis_and_judgment(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = OllamaTranslationAgent("fake-local-model")
    calls: list[tuple[str, str]] = []
    responses = iter(
        [
            {"domain": "software", "key_terms": ["배포"], "confidence": 0.8},
            {
                "passed": True,
                "quality_score": 0.9,
                "error_types": [],
                "summary": "The meaning and terminology are acceptable.",
                "confidence": 0.85,
                "next_action": "accept",
                "suggested_query_terms": [],
            },
            {"translation": "Deploy the corrected version."},
        ]
    )

    def fake_chat(
        system: str,
        user: str,
        *,
        think: bool = True,
    ) -> dict[str, object]:
        del think
        calls.append((system, user))
        return next(responses)

    monkeypatch.setattr(agent, "_chat_json", fake_chat)

    analysis = agent.analyze("새 버전을 배포한다.")
    judgment = agent.judge(
        "새 버전을 배포한다.",
        TranslationCandidate(text="Deploy the new version.", model_id="fake"),
        [],
    )
    revision = agent.revise(
        "새 버전을 배포한다.",
        TranslationCandidate(text="Release the new version.", model_id="fake"),
        QualityJudgment(
            passed=False,
            quality_score=0.4,
            error_types=[TranslationErrorType.TERM],
            summary="Use the required deployment term.",
            confidence=0.9,
            next_action=NextAction.REVISE,
            must_preserve_constraints=[
                MustPreserveConstraint(
                    source_term="배포",
                    kinds=[SourceCoverageKind.TECHNICAL_TERM],
                    accepted_targets=["deployment"],
                )
            ],
        ),
        [],
    )

    assert analysis.domain == "software"
    assert judgment.passed
    assert judgment.quality_score == 0.9
    assert revision.text == "Deploy the corrected version."
    assert revision.model_id == "fake-local-model"
    revision_payload = calls[2][1]
    assert "previous_candidate_en" in revision_payload
    assert "reference" not in revision_payload.casefold()
    assert json.loads(revision_payload)["must_preserve"] == [
        {
            "source_term": "배포",
            "kinds": ["technical_term"],
            "accepted_targets": ["deployment"],
        }
    ]


def test_ollama_agent_disables_thinking_for_structured_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_payloads: list[dict[str, object]] = []

    def fake_urlopen(request: object, *, timeout: float) -> FakeHTTPResponse:
        del timeout
        request_data = getattr(request, "data")
        assert isinstance(request_data, bytes)
        captured_payloads.append(json.loads(request_data.decode("utf-8")))
        return FakeHTTPResponse()

    monkeypatch.setattr("translation_qa.agent.urlopen", fake_urlopen)

    agent = OllamaTranslationAgent("fake-local-model")
    analysis = agent.analyze("원문")
    agent._chat_json("judge", "payload")

    assert analysis.key_terms == []
    assert captured_payloads[0]["think"] is False
    assert captured_payloads[1]["think"] is True
    assert captured_payloads[0]["stream"] is False
    assert captured_payloads[0]["format"] == "json"


def test_ollama_agent_derives_revision_from_structured_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = OllamaTranslationAgent("fake-local-model")
    monkeypatch.setattr(
        agent,
        "_chat_json",
        lambda system, user, **kwargs: {
            "passed": True,
            "quality_score": 0.9,
            "error_types": ["other"],
            "summary": "The candidate is acceptable.",
            "confidence": 0.8,
            "next_action": "revise",
            "suggested_query_terms": [],
        },
    )

    judgment = agent.judge(
        "오늘 회의가 시작한다.",
        TranslationCandidate(text="The meeting starts today.", model_id="fake"),
        [],
    )

    assert not judgment.passed
    assert judgment.error_types == [TranslationErrorType.OTHER]
    assert judgment.next_action is NextAction.REVISE
    assert judgment.decision_audit is not None
    assert judgment.decision_audit.reported_passed is True
    assert judgment.decision_audit.consistency_issues == [
        JudgmentConsistencyIssue.REPORTED_PASSED_MISMATCH,
        JudgmentConsistencyIssue.SUMMARY_PASS_WITH_STRUCTURED_ERROR,
    ]


def test_ollama_agent_canonicalizes_unknown_error_type_to_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = OllamaTranslationAgent("fake-local-model")
    monkeypatch.setattr(
        agent,
        "_chat_json",
        lambda system, user, **kwargs: {
            "passed": False,
            "quality_score": 0.6,
            "error_types": ["pronoun"],
            "summary": "The pronoun choice is unnatural.",
            "confidence": 0.8,
            "next_action": "revise",
            "suggested_query_terms": [],
        },
    )

    judgment = agent.judge(
        "사용자는 설정을 변경할 수 있다.",
        TranslationCandidate(
            text="The user can change his settings.",
            model_id="fake",
        ),
        [],
    )

    assert judgment.error_types == [TranslationErrorType.OTHER]
    assert "pronoun" in judgment.summary
    assert judgment.decision_audit is not None
    assert (
        JudgmentConsistencyIssue.UNSUPPORTED_ERROR_TYPE_NORMALIZED
        in judgment.decision_audit.consistency_issues
    )


def test_ollama_agent_preserves_valid_error_types_in_mixed_unknown_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = OllamaTranslationAgent("fake-local-model")
    monkeypatch.setattr(
        agent,
        "_chat_json",
        lambda system, user, **kwargs: {
            "passed": False,
            "quality_score": 0.4,
            "error_types": ["meaning", "pronoun", "fluency_grammar"],
            "summary": "The candidate needs revision.",
            "confidence": 0.9,
            "next_action": "accept",
            "suggested_query_terms": [],
        },
    )

    judgment = agent.judge(
        "그녀는 결과를 검토했다.",
        TranslationCandidate(text="He review the result.", model_id="fake"),
        [],
    )

    assert judgment.error_types == [
        TranslationErrorType.MEANING,
        TranslationErrorType.OTHER,
        TranslationErrorType.FLUENCY_GRAMMAR,
    ]
    assert judgment.next_action is NextAction.REVISE
    assert judgment.summary.startswith("The candidate needs revision.")
    assert judgment.decision_audit is not None
    assert judgment.decision_audit.consistency_issues == [
        JudgmentConsistencyIssue.REPORTED_NEXT_ACTION_MISMATCH,
        JudgmentConsistencyIssue.UNSUPPORTED_ERROR_TYPE_NORMALIZED,
    ]


def test_ollama_agent_derives_pass_from_empty_structured_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = OllamaTranslationAgent("fake-local-model")
    monkeypatch.setattr(
        agent,
        "_chat_json",
        lambda system, user, **kwargs: {
            "passed": False,
            "quality_score": 0.8,
            "error_types": [],
            "summary": "The candidate accurately conveys the source.",
            "confidence": 0.8,
            "next_action": "revise",
            "suggested_query_terms": [],
        },
    )

    judgment = agent.judge(
        "오늘 회의가 시작한다.",
        TranslationCandidate(text="The meeting starts today.", model_id="fake"),
        [],
    )

    assert judgment.passed
    assert judgment.error_types == []
    assert judgment.next_action is NextAction.ACCEPT
    assert judgment.decision_audit is not None
    assert judgment.decision_audit.consistency_issues == [
        JudgmentConsistencyIssue.REPORTED_PASSED_MISMATCH,
        JudgmentConsistencyIssue.REPORTED_NEXT_ACTION_MISMATCH,
    ]


@pytest.mark.parametrize("error_types", [None, "meaning", [1]])
def test_ollama_agent_rejects_malformed_structured_error_list(
    monkeypatch: pytest.MonkeyPatch,
    error_types: object,
) -> None:
    agent = OllamaTranslationAgent("fake-local-model")
    monkeypatch.setattr(
        agent,
        "_chat_json",
        lambda system, user, **kwargs: {
            "passed": True,
            "quality_score": 0.8,
            "error_types": error_types,
            "summary": "The candidate is acceptable.",
            "confidence": 0.8,
            "next_action": "accept",
            "suggested_query_terms": [],
        },
    )

    with pytest.raises(ComponentExecutionError, match="invalid Agent judgment schema"):
        agent.judge(
            "오늘 회의가 시작한다.",
            TranslationCandidate(text="The meeting starts today.", model_id="fake"),
            [],
        )


def test_ollama_agent_rule_overrides_llm_when_required_term_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = OllamaTranslationAgent("fake-local-model")
    monkeypatch.setattr(
        agent,
        "_chat_json",
        lambda system, user, **kwargs: {
            "passed": True,
            "quality_score": 0.95,
            "error_types": [],
            "summary": "The candidate is acceptable.",
            "confidence": 0.9,
            "next_action": "accept",
            "suggested_query_terms": [],
        },
    )
    hit = RetrievalHit(
        term_id="t1",
        source_term="장애 조치",
        target_term="failover",
        domain="software",
        match_type=RetrievalMatchType.EXACT,
        score=1.0,
    )

    judgment = agent.judge(
        "자동 장애 조치를 실행한다.",
        TranslationCandidate(text="Run an automatic disability measure.", model_id="fake"),
        [hit],
    )

    assert not judgment.passed
    assert TranslationErrorType.TERM in judgment.error_types
    assert judgment.next_action is NextAction.REVISE
    assert judgment.decision_audit is not None
    assert judgment.decision_audit.reported_error_types == []
    assert judgment.decision_audit.code_added_error_types == [
        TranslationErrorType.TERM
    ]
    assert judgment.decision_audit.consistency_issues == []
