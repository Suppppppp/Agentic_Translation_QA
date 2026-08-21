from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from translation_qa.config import Settings
from translation_qa.errors import (
    ComponentExecutionError,
    ComponentUnavailableError,
    ConstraintApplicationError,
)
from translation_qa.pipeline import TranslationPipeline
from translation_qa.schemas import (
    ExecutionMode,
    CandidateOrigin,
    NextAction,
    QualityJudgment,
    RetrievalHit,
    RetrievalMatchType,
    RetrievalQuery,
    SelectionReason,
    SourceAnalysis,
    SourceCoverageStatus,
    StopReason,
    TermConstraint,
    TranslationCandidate,
    TranslationErrorType,
    TranslationRequest,
)


def make_judgment(
    *,
    passed: bool,
    score: float,
    action: NextAction | None = None,
    suggested_terms: list[str] | None = None,
) -> QualityJudgment:
    return QualityJudgment(
        passed=passed,
        quality_score=score,
        error_types=[] if passed else [TranslationErrorType.TERM],
        summary="acceptable" if passed else "a required term is missing",
        confidence=0.9,
        next_action=action
        or (NextAction.ACCEPT if passed else NextAction.REVISE),
        suggested_query_terms=suggested_terms or [],
    )


class FakeTranslator:
    model_id = "fake-translator"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[tuple[str, list[TermConstraint]]] = []

    def translate(
        self,
        source_text: str,
        constraints: Sequence[TermConstraint] | None = None,
    ) -> TranslationCandidate:
        self.calls.append((source_text, list(constraints or [])))
        index = min(len(self.calls) - 1, len(self.outputs) - 1)
        return TranslationCandidate(text=self.outputs[index], model_id=self.model_id)


class ConstraintFailingTranslator(FakeTranslator):
    def translate(
        self,
        source_text: str,
        constraints: Sequence[TermConstraint] | None = None,
    ) -> TranslationCandidate:
        if constraints:
            self.calls.append((source_text, list(constraints)))
            raise ConstraintApplicationError("degenerate constrained candidate")
        return super().translate(source_text, constraints)


class FakeRetriever:
    def __init__(self, hits_by_call: list[list[RetrievalHit]]) -> None:
        self.hits_by_call = hits_by_call
        self.calls: list[RetrievalQuery] = []

    def retrieve(self, query: RetrievalQuery) -> list[RetrievalHit]:
        self.calls.append(query)
        index = min(len(self.calls) - 1, len(self.hits_by_call) - 1)
        return self.hits_by_call[index]


class SecondRetrievalFails(FakeRetriever):
    def retrieve(self, query: RetrievalQuery) -> list[RetrievalHit]:
        if self.calls:
            self.calls.append(query)
            raise ComponentExecutionError("targeted retrieval failed")
        return super().retrieve(query)


class FocusAwareDockerRetriever(FakeRetriever):
    def __init__(self) -> None:
        super().__init__([[]])

    def retrieve(self, query: RetrievalQuery) -> list[RetrievalHit]:
        self.calls.append(query)
        if query.source_text != "도커":
            return []
        duplicate = docker_hit().model_copy(
            update={"score": 0.7, "match_type": RetrievalMatchType.VECTOR}
        )
        return [duplicate, docker_hit(), docker_hit()]


class FakeReviser:
    model_id = "fake-reviser"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[
            tuple[str, TranslationCandidate, QualityJudgment, list[RetrievalHit]]
        ] = []

    def revise(
        self,
        source_text: str,
        previous_candidate: TranslationCandidate,
        judgment: QualityJudgment,
        retrieved_terms: Sequence[RetrievalHit],
    ) -> TranslationCandidate:
        self.calls.append(
            (source_text, previous_candidate, judgment, list(retrieved_terms))
        )
        index = min(len(self.calls) - 1, len(self.outputs) - 1)
        return TranslationCandidate(text=self.outputs[index], model_id=self.model_id)


class FailingReviser(FakeReviser):
    def revise(
        self,
        source_text: str,
        previous_candidate: TranslationCandidate,
        judgment: QualityJudgment,
        retrieved_terms: Sequence[RetrievalHit],
    ) -> TranslationCandidate:
        self.calls.append(
            (source_text, previous_candidate, judgment, list(retrieved_terms))
        )
        raise ComponentExecutionError("revision failed")


class FakeAgent:
    model_id = "fake-agent"

    def __init__(
        self,
        judgments: list[QualityJudgment],
        *,
        analysis_terms: list[str] | None = None,
    ) -> None:
        self.judgments = judgments
        self.analysis_terms = ["배포"] if analysis_terms is None else analysis_terms
        self.analysis_calls: list[str] = []
        self.judge_calls: list[str] = []

    def analyze(self, source_text: str) -> SourceAnalysis:
        self.analysis_calls.append(source_text)
        return SourceAnalysis(
            domain="software", key_terms=self.analysis_terms, confidence=0.8
        )

    def judge(
        self,
        source_text: str,
        candidate: TranslationCandidate,
        retrieved_terms: Sequence[RetrievalHit],
    ) -> QualityJudgment:
        self.judge_calls.append(candidate.text)
        return self.judgments[len(self.judge_calls) - 1]


class FailingAgent(FakeAgent):
    def judge(
        self,
        source_text: str,
        candidate: TranslationCandidate,
        retrieved_terms: Sequence[RetrievalHit],
    ) -> QualityJudgment:
        raise ComponentUnavailableError("local Agent is offline")


def glossary_hit() -> RetrievalHit:
    return RetrievalHit(
        term_id="term-deploy",
        source_term="배포",
        target_term="deployment",
        domain="software",
        match_type=RetrievalMatchType.EXACT,
        score=1.0,
        definition="Release software to an environment.",
    )


def docker_hit() -> RetrievalHit:
    return RetrievalHit(
        term_id="term-docker",
        source_term="도커",
        target_term="Docker",
        domain="software",
        match_type=RetrievalMatchType.EXACT,
        score=1.0,
        definition="A container platform.",
    )


def server_hit() -> RetrievalHit:
    return RetrievalHit(
        term_id="term-server",
        source_term="서버",
        target_term="server",
        accepted_target_variants=["servers"],
        domain="software",
        match_type=RetrievalMatchType.EXACT,
        score=1.0,
    )


def test_baseline_calls_only_translator_once() -> None:
    translator = FakeTranslator(["Deploy the new version."])
    pipeline = TranslationPipeline(Settings(), translator)

    response = pipeline.translate(
        TranslationRequest(text="새 버전을 배포한다."),
        ExecutionMode.BASELINE,
    )

    assert len(translator.calls) == 1
    assert response.translation == "Deploy the new version."
    assert response.retry_count == 0
    assert response.final_judgment is None
    assert response.trace.stop_reason is StopReason.BASELINE_COMPLETE


def test_rag_only_passes_retrieved_terms_as_constraints() -> None:
    translator = FakeTranslator(["Start the deployment."])
    retriever = FakeRetriever([[glossary_hit()]])
    pipeline = TranslationPipeline(Settings(), translator, retriever=retriever)

    response = pipeline.translate(
        TranslationRequest(text="배포를 시작한다."),
        ExecutionMode.RAG,
    )

    assert len(retriever.calls) == 1
    assert translator.calls[0][1][0].target_term == "deployment"
    assert response.trace.stop_reason is StopReason.RAG_COMPLETE


def test_vector_only_compound_hit_is_evidence_but_not_an_applied_constraint() -> None:
    translator = FakeTranslator(["Ubuntu is a Linux distribution."])
    vector_hit = glossary_hit().model_copy(
        update={"match_type": RetrievalMatchType.VECTOR, "score": 0.6}
    )
    retriever = FakeRetriever([[vector_hit]])
    pipeline = TranslationPipeline(Settings(), translator, retriever=retriever)

    response = pipeline.translate(
        TranslationRequest(text="우분투는 리눅스 배포판이다."),
        ExecutionMode.RAG,
    )

    assert translator.calls[0][1] == []
    assert response.trace.attempts[0].retrieval_hits == [vector_hit]
    assert response.trace.attempts[0].applied_constraints == []


def test_rag_constraint_failure_falls_back_to_unconstrained_translation() -> None:
    translator = ConstraintFailingTranslator(["Safe baseline candidate."])
    retriever = FakeRetriever([[glossary_hit()]])
    pipeline = TranslationPipeline(Settings(), translator, retriever=retriever)

    response = pipeline.translate(
        TranslationRequest(text="배포를 시작한다."),
        ExecutionMode.RAG,
    )

    assert len(translator.calls) == 2
    assert response.translation == "Safe baseline candidate."
    assert response.trace.attempts[0].applied_constraints == []
    assert response.trace.stop_reason is StopReason.CONSTRAINT_FALLBACK
    assert response.trace.attempts[0].candidate_origin is CandidateOrigin.FALLBACK
    assert response.trace.component_call_counts.translation == 2
    assert "degenerate" in response.trace.warnings[0]
    assert any("not preserved" in warning for warning in response.trace.warnings)


def test_agent_passes_initial_candidate_without_retry() -> None:
    translator = FakeTranslator(["Deploy the new version."])
    agent = FakeAgent([make_judgment(passed=True, score=0.95)])
    pipeline = TranslationPipeline(Settings(), translator, agent=agent)

    response = pipeline.translate(
        TranslationRequest(text="새 버전을 배포한다."),
        ExecutionMode.AGENT,
    )

    assert len(translator.calls) == 1
    assert agent.judge_calls == ["Deploy the new version."]
    assert response.retry_count == 0
    assert response.trace.stop_reason is StopReason.PASSED
    assert response.trace.selection_reason is SelectionReason.PASSED
    timings = response.trace.attempts[0].timings
    assert timings.total_ms >= timings.analysis_ms


def test_agent_failure_then_pass_retries_once() -> None:
    translator = FakeTranslator(["Bad candidate."])
    reviser = FakeReviser(["Improved candidate."])
    agent = FakeAgent(
        [
            make_judgment(passed=False, score=0.3),
            make_judgment(passed=True, score=0.9),
        ]
    )
    pipeline = TranslationPipeline(
        Settings(), translator, agent=agent, reviser=reviser
    )

    response = pipeline.translate(TranslationRequest(text="원문"), ExecutionMode.AGENT)

    assert len(translator.calls) == 1
    assert len(reviser.calls) == 1
    assert reviser.calls[0][1].text == "Bad candidate."
    assert reviser.calls[0][3] == []
    assert response.translation == "Improved candidate."
    assert response.retry_count == 1
    assert response.trace.final_attempt_index == 1
    assert response.trace.stop_reason is StopReason.PASSED
    revision = response.trace.attempts[1]
    assert revision.candidate_origin is CandidateOrigin.AGENT_REVISION
    assert revision.parent_attempt_index == 0
    assert revision.requested_action is NextAction.REVISE
    assert revision.applied_action is NextAction.REVISE
    assert response.trace.component_call_counts.translation == 1
    assert response.trace.component_call_counts.revision == 1


def test_max_retries_selects_highest_scoring_candidate() -> None:
    translator = FakeTranslator(["First."])
    reviser = FakeReviser(["Second.", "Third."])
    agent = FakeAgent(
        [
            make_judgment(passed=False, score=0.4),
            make_judgment(passed=False, score=0.8),
            make_judgment(passed=False, score=0.6),
        ]
    )
    pipeline = TranslationPipeline(
        Settings(max_retries=2), translator, agent=agent, reviser=reviser
    )

    response = pipeline.translate(TranslationRequest(text="원문"), ExecutionMode.AGENT)

    assert len(translator.calls) == 1
    assert len(reviser.calls) == 2
    assert response.retry_count == 2
    assert response.translation == "Second."
    assert response.trace.final_attempt_index == 1
    assert response.trace.stop_reason is StopReason.MAX_RETRIES
    assert (
        response.trace.selection_reason
        is SelectionReason.HIGHEST_QUALITY_ROLLBACK
    )
    assert response.trace.warnings


def test_identical_retry_stops_early_and_rolls_back() -> None:
    translator = FakeTranslator(["Same candidate."])
    reviser = FakeReviser(["Same candidate."])
    agent = FakeAgent([make_judgment(passed=False, score=0.4)])
    pipeline = TranslationPipeline(
        Settings(), translator, agent=agent, reviser=reviser
    )

    response = pipeline.translate(TranslationRequest(text="원문"), ExecutionMode.AGENT)

    assert len(translator.calls) == 1
    assert len(reviser.calls) == 1
    assert len(agent.judge_calls) == 1
    assert response.retry_count == 1
    assert response.trace.final_attempt_index == 0
    assert response.trace.stop_reason is StopReason.UNCHANGED


def test_nfkc_equivalent_revision_stops_as_unchanged() -> None:
    translator = FakeTranslator(["Ａ candidate."])
    reviser = FakeReviser(["A candidate."])
    agent = FakeAgent([make_judgment(passed=False, score=0.4)])
    pipeline = TranslationPipeline(
        Settings(), translator, agent=agent, reviser=reviser
    )

    response = pipeline.translate(TranslationRequest(text="원문"), ExecutionMode.AGENT)

    assert response.trace.stop_reason is StopReason.UNCHANGED
    assert len(agent.judge_calls) == 1


def test_missing_reviser_stops_safely_after_initial_judgment() -> None:
    translator = FakeTranslator(["Candidate."])
    agent = FakeAgent([make_judgment(passed=False, score=0.4)])
    pipeline = TranslationPipeline(Settings(), translator, agent=agent)

    response = pipeline.translate(TranslationRequest(text="원문"), ExecutionMode.AGENT)

    assert response.translation == "Candidate."
    assert response.retry_count == 0
    assert response.trace.stop_reason is StopReason.COMPONENT_FAILURE
    assert response.trace.component_call_counts.revision == 0
    assert any("no TranslationReviser" in warning for warning in response.trace.warnings)


def test_reviser_failure_returns_best_existing_candidate() -> None:
    translator = FakeTranslator(["Candidate."])
    reviser = FailingReviser([])
    agent = FakeAgent([make_judgment(passed=False, score=0.6)])
    pipeline = TranslationPipeline(
        Settings(), translator, agent=agent, reviser=reviser
    )

    response = pipeline.translate(TranslationRequest(text="원문"), ExecutionMode.AGENT)

    assert response.translation == "Candidate."
    assert response.retry_count == 0
    assert response.trace.stop_reason is StopReason.COMPONENT_FAILURE
    assert response.trace.component_call_counts.revision == 1
    assert any("revision failed" in warning for warning in response.trace.warnings)


def test_agent_rag_revision_receives_retrieved_evidence() -> None:
    translator = FakeTranslator(["Bad candidate."])
    reviser = FakeReviser(["Improved deployment candidate."])
    retriever = FakeRetriever([[glossary_hit()]])
    agent = FakeAgent(
        [
            make_judgment(
                passed=False,
                score=0.3,
                action=NextAction.RETRY_WITH_CONSTRAINTS,
            ),
            make_judgment(passed=True, score=0.9),
        ]
    )
    pipeline = TranslationPipeline(
        Settings(),
        translator,
        retriever=retriever,
        agent=agent,
        reviser=reviser,
    )

    response = pipeline.translate(
        TranslationRequest(text="배포한다."), ExecutionMode.AGENT_RAG
    )

    assert len(translator.calls) == 1
    assert [hit.term_id for hit in reviser.calls[0][3]] == ["term-deploy"]
    assert len(retriever.calls) == 1
    assert response.trace.attempts[1].requested_action is NextAction.RETRY_WITH_CONSTRAINTS


def test_retry_with_rag_refreshes_evidence_before_revision() -> None:
    translator = FakeTranslator(["Bad candidate."])
    reviser = FakeReviser(["Improved deployment candidate."])
    retriever = FakeRetriever([[], [glossary_hit()]])
    agent = FakeAgent(
        [
            make_judgment(
                passed=False,
                score=0.3,
                action=NextAction.RETRY_WITH_RAG,
                suggested_terms=["배포"],
            ),
            make_judgment(passed=True, score=0.9),
        ]
    )
    pipeline = TranslationPipeline(
        Settings(),
        translator,
        retriever=retriever,
        agent=agent,
        reviser=reviser,
    )

    response = pipeline.translate(
        TranslationRequest(text="릴리스한다."), ExecutionMode.AGENT_RAG
    )

    assert len(retriever.calls) == 2
    assert retriever.calls[1].attempt_index == 1
    assert retriever.calls[1].source_text == "릴리스한다."
    assert "배포" in retriever.calls[1].key_terms
    assert [hit.term_id for hit in reviser.calls[0][3]] == ["term-deploy"]
    assert response.trace.component_call_counts.retrieval == 2


def test_agent_only_maps_rag_action_to_plain_revision() -> None:
    translator = FakeTranslator(["Bad candidate."])
    reviser = FakeReviser(["Improved candidate."])
    agent = FakeAgent(
        [
            make_judgment(
                passed=False,
                score=0.3,
                action=NextAction.RETRY_WITH_RAG,
            ),
            make_judgment(passed=True, score=0.9),
        ]
    )
    pipeline = TranslationPipeline(
        Settings(), translator, agent=agent, reviser=reviser
    )

    response = pipeline.translate(TranslationRequest(text="원문"), ExecutionMode.AGENT)

    revision = response.trace.attempts[1]
    assert revision.requested_action is NextAction.RETRY_WITH_RAG
    assert revision.applied_action is NextAction.REVISE


def test_source_coverage_overrides_pass_and_rechecks_agent_revision() -> None:
    translator = FakeTranslator(["It's a container platform."])
    reviser = FakeReviser(["Docker is a container platform."])
    agent = FakeAgent(
        [
            make_judgment(
                passed=True,
                score=0.95,
                suggested_terms=["원문에 없는 용어"],
            ),
            make_judgment(passed=True, score=0.95),
        ],
        analysis_terms=["도커", "컨테이너 플랫폼"],
    )
    pipeline = TranslationPipeline(
        Settings(), translator, agent=agent, reviser=reviser
    )

    response = pipeline.translate(
        TranslationRequest(text="도커는 컨테이너 플랫폼입니다."),
        ExecutionMode.AGENT,
    )

    initial, revised = response.trace.attempts
    assert initial.judgment is not None
    assert not initial.judgment.passed
    assert initial.judgment.next_action is NextAction.RETRY_WITH_CONSTRAINTS
    assert any(
        finding.status is SourceCoverageStatus.MISSING
        for finding in initial.coverage_findings
    )
    assert revised.judgment is not None and revised.judgment.passed
    assert any(
        finding.status is SourceCoverageStatus.UNRESOLVED
        for finding in revised.coverage_findings
    )
    assert reviser.calls[0][2].must_preserve_constraints[0].source_term == "도커"
    assert revised.applied_must_preserve_constraints[0].source_term == "도커"
    assert translator.calls[0][1] == []
    assert response.trace.component_call_counts.retrieval == 0


def test_source_coverage_targets_rag_then_enriches_and_deduplicates_hits() -> None:
    translator = FakeTranslator(["It's a container platform."])
    reviser = FakeReviser(["Docker is a container platform."])
    retriever = FocusAwareDockerRetriever()
    agent = FakeAgent(
        [
            make_judgment(
                passed=True,
                score=0.95,
                suggested_terms=["원문에 없는 용어"],
            ),
            make_judgment(passed=True, score=0.95),
        ],
        analysis_terms=["도커", "컨테이너 플랫폼"],
    )
    pipeline = TranslationPipeline(
        Settings(),
        translator,
        retriever=retriever,
        agent=agent,
        reviser=reviser,
    )

    response = pipeline.translate(
        TranslationRequest(text="도커는 컨테이너 플랫폼입니다."),
        ExecutionMode.AGENT_RAG,
    )

    initial, revised = response.trace.attempts
    assert initial.judgment is not None
    assert initial.judgment.next_action is NextAction.RETRY_WITH_RAG
    assert retriever.calls[1].source_text == "도커"
    assert retriever.calls[1].key_terms == ["도커"]
    assert [hit.term_id for hit in revised.retrieval_hits] == ["term-docker"]
    assert revised.retrieval_hits[0].score == 1.0
    applied = revised.applied_must_preserve_constraints
    assert len(applied) == 1
    assert applied[0].accepted_targets == ["Docker"]
    assert reviser.calls[0][2].must_preserve_constraints == applied
    assert not any(
        finding.status is SourceCoverageStatus.MISSING
        for finding in revised.coverage_findings
    )
    assert any(
        finding.requirement.source_term == "도커"
        and finding.status is SourceCoverageStatus.COVERED
        for finding in revised.coverage_findings
    )
    # Coverage constraints are reviser-only; the one NMT call stayed unchanged.
    assert translator.calls[0][1] == []
    assert revised.applied_constraints == []


def test_unchanged_revision_still_records_second_coverage_check() -> None:
    translator = FakeTranslator(["It's a container platform."])
    reviser = FakeReviser(["It's a container platform."])
    agent = FakeAgent(
        [make_judgment(passed=True, score=0.95)],
        analysis_terms=["도커", "컨테이너 플랫폼"],
    )
    pipeline = TranslationPipeline(
        Settings(), translator, agent=agent, reviser=reviser
    )

    response = pipeline.translate(
        TranslationRequest(text="도커는 컨테이너 플랫폼입니다."),
        ExecutionMode.AGENT,
    )

    assert response.trace.stop_reason is StopReason.UNCHANGED
    assert len(response.trace.attempts) == 2
    assert all(attempt.coverage_findings for attempt in response.trace.attempts)
    assert any(
        finding.status is SourceCoverageStatus.MISSING
        for finding in response.trace.attempts[1].coverage_findings
    )


def test_targeted_retrieval_failure_retains_existing_hits_for_revision() -> None:
    translator = FakeTranslator(["It's a container platform on a server."])
    reviser = FakeReviser(["Docker is a container platform on a server."])
    retriever = SecondRetrievalFails([[server_hit()]])
    agent = FakeAgent(
        [
            make_judgment(passed=True, score=0.95),
            make_judgment(passed=True, score=0.95),
        ],
        analysis_terms=["도커", "서버"],
    )
    pipeline = TranslationPipeline(
        Settings(),
        translator,
        retriever=retriever,
        agent=agent,
        reviser=reviser,
    )

    response = pipeline.translate(
        TranslationRequest(text="도커는 서버에서 실행됩니다."),
        ExecutionMode.AGENT_RAG,
    )

    assert len(retriever.calls) == 2
    assert [hit.term_id for hit in reviser.calls[0][3]] == ["term-server"]
    assert [hit.term_id for hit in response.trace.attempts[1].retrieval_hits] == [
        "term-server"
    ]
    assert any("targeted retrieval failed" in warning for warning in response.trace.warnings)


def test_must_preserve_constraint_survives_an_unrelated_second_revision() -> None:
    translator = FakeTranslator(["It's a container platform."])
    reviser = FakeReviser(
        ["Docker needs a wording fix.", "Docker is a container platform."]
    )
    agent = FakeAgent(
        [
            make_judgment(passed=True, score=0.95),
            make_judgment(passed=False, score=0.3),
            make_judgment(passed=True, score=0.95),
        ],
        analysis_terms=["도커", "컨테이너 플랫폼"],
    )
    pipeline = TranslationPipeline(
        Settings(max_retries=2), translator, agent=agent, reviser=reviser
    )

    response = pipeline.translate(
        TranslationRequest(text="도커는 컨테이너 플랫폼입니다."),
        ExecutionMode.AGENT,
    )

    assert response.trace.stop_reason is StopReason.PASSED
    assert len(reviser.calls) == 2
    assert [
        call[2].must_preserve_constraints[0].source_term
        for call in reviser.calls
    ] == ["도커", "도커"]
    assert response.trace.attempts[2].applied_must_preserve_constraints


def test_final_selection_does_not_roll_back_to_missing_source_coverage() -> None:
    translator = FakeTranslator(["It's a container platform."])
    reviser = FakeReviser(["Docker needs a wording fix."])
    agent = FakeAgent(
        [
            make_judgment(passed=True, score=0.95),
            make_judgment(passed=False, score=0.3),
        ],
        analysis_terms=["도커", "컨테이너 플랫폼"],
    )
    pipeline = TranslationPipeline(
        Settings(max_retries=1), translator, agent=agent, reviser=reviser
    )

    response = pipeline.translate(
        TranslationRequest(text="도커는 컨테이너 플랫폼입니다."),
        ExecutionMode.AGENT,
    )

    assert response.translation == "Docker needs a wording fix."
    assert response.trace.final_attempt_index == 1
    assert (
        response.trace.selection_reason
        is SelectionReason.SOURCE_COVERAGE_PRESERVED
    )


def test_translation_request_rejects_more_than_5000_characters() -> None:
    with pytest.raises(ValidationError, match="at most 5000 characters"):
        TranslationRequest(text="가" * 5_001)


def test_agent_runtime_failure_returns_current_translation_with_warning() -> None:
    translator = FakeTranslator(["Candidate."])
    agent = FailingAgent([])
    pipeline = TranslationPipeline(Settings(), translator, agent=agent)

    response = pipeline.translate(TranslationRequest(text="원문"), ExecutionMode.AGENT)

    assert response.translation == "Candidate."
    assert response.final_judgment is None
    assert response.trace.stop_reason is StopReason.COMPONENT_FAILURE
    assert "offline" in response.trace.warnings[0]


@pytest.mark.parametrize("mode", [ExecutionMode.RAG, ExecutionMode.AGENT_RAG])
def test_rag_modes_require_retriever(mode: ExecutionMode) -> None:
    pipeline = TranslationPipeline(
        Settings(),
        FakeTranslator(["Candidate."]),
        agent=FakeAgent([make_judgment(passed=True, score=1.0)]),
    )

    with pytest.raises(ComponentUnavailableError, match="retriever"):
        pipeline.translate(TranslationRequest(text="원문"), mode)


@pytest.mark.parametrize("mode", [ExecutionMode.AGENT, ExecutionMode.AGENT_RAG])
def test_agent_modes_require_agent(mode: ExecutionMode) -> None:
    pipeline = TranslationPipeline(
        Settings(),
        FakeTranslator(["Candidate."]),
        retriever=FakeRetriever([[]]),
    )

    with pytest.raises(ComponentUnavailableError, match="Agent"):
        pipeline.translate(TranslationRequest(text="원문"), mode)
