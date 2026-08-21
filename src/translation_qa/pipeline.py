"""Deterministic orchestration for Baseline, RAG, Agent, and combined modes."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from time import perf_counter

from translation_qa.config import Settings
from translation_qa.contracts import (
    Retriever,
    TranslationAgent,
    TranslationReviser,
    Translator,
)
from translation_qa.coverage import (
    apply_coverage_to_judgment,
    check_source_coverage,
    constraints_for_retry,
    extract_source_coverage_requirements,
    merge_coverage_requirements,
    merge_must_preserve_constraints,
    merge_retrieval_hits,
    missing_coverage_findings,
)
from translation_qa.errors import (
    ComponentExecutionError,
    ComponentUnavailableError,
    ConstraintApplicationError,
)
from translation_qa.retrieval import contains_source_term
from translation_qa.schemas import (
    AttemptTrace,
    CandidateOrigin,
    ComponentCallCounts,
    CoverageRecoveryAction,
    ExecutionMode,
    NextAction,
    QualityJudgment,
    RetrievalHit,
    RetrievalQuery,
    SelectionReason,
    SourceAnalysis,
    SourceCoverageRequirement,
    StageTimings,
    StopReason,
    TermConstraint,
    TranslationCandidate,
    TranslationRequest,
    TranslationResponse,
    TranslationTrace,
)


def _elapsed_ms(started: float) -> float:
    return (perf_counter() - started) * 1000.0


def _same_translation(left: str, right: str) -> bool:
    def normalize(value: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", value).split())

    return normalize(left) == normalize(right)


class TranslationPipeline:
    """Owns retries and stop rules so the Agent cannot create an infinite loop."""

    def __init__(
        self,
        settings: Settings,
        translator: Translator,
        *,
        retriever: Retriever | None = None,
        agent: TranslationAgent | None = None,
        reviser: TranslationReviser | None = None,
    ) -> None:
        self.settings = settings
        self.translator = translator
        self.retriever = retriever
        self.agent = agent
        self.reviser = reviser

    @staticmethod
    def _uses_rag(mode: ExecutionMode) -> bool:
        return mode in {ExecutionMode.RAG, ExecutionMode.AGENT_RAG}

    @staticmethod
    def _uses_agent(mode: ExecutionMode) -> bool:
        return mode in {ExecutionMode.AGENT, ExecutionMode.AGENT_RAG}

    def _constraints(
        self,
        hits: Sequence[RetrievalHit],
        source_text: str,
    ) -> list[TermConstraint]:
        return [
            TermConstraint(
                source_term=hit.source_term,
                target_term=hit.target_term,
                retrieval_hit_id=hit.term_id,
                target_variants=hit.accepted_target_variants,
                replacement_rules=hit.replacement_rules,
            )
            for hit in hits
            if contains_source_term(source_text, hit.source_term)
        ]

    def _response(
        self,
        *,
        request: TranslationRequest,
        mode: ExecutionMode,
        analysis: SourceAnalysis | None,
        coverage_requirements: list[SourceCoverageRequirement],
        attempts: list[AttemptTrace],
        final_index: int,
        stop_reason: StopReason,
        selection_reason: SelectionReason,
        component_call_counts: ComponentCallCounts,
        warnings: list[str],
        started: float,
    ) -> TranslationResponse:
        total_ms = _elapsed_ms(started)
        trace = TranslationTrace(
            source_analysis=analysis,
            coverage_requirements=coverage_requirements,
            attempts=attempts,
            final_attempt_index=final_index,
            stop_reason=stop_reason,
            selection_reason=selection_reason,
            total_latency_ms=total_ms,
            component_call_counts=component_call_counts,
            warnings=warnings,
            model_versions={
                "translator": self.translator.model_id,
                "max_retries": str(self.settings.max_retries),
                "retrieval_top_k": str(self.settings.retrieval_top_k),
                **(
                    {"retriever": type(self.retriever).__name__}
                    if self._uses_rag(mode) and self.retriever is not None
                    else {}
                ),
                **(
                    {"agent": self.agent.model_id}
                    if self._uses_agent(mode) and self.agent is not None
                    else {}
                ),
                **(
                    {"reviser": self.reviser.model_id}
                    if self._uses_agent(mode) and self.reviser is not None
                    else {}
                ),
            },
        )
        final_attempt = attempts[final_index]
        return TranslationResponse(
            mode=mode,
            source_text=request.text,
            translation=final_attempt.candidate.text,
            retry_count=len(attempts) - 1,
            final_judgment=final_attempt.judgment,
            trace=trace,
        )

    def translate(
        self,
        request: TranslationRequest,
        mode: ExecutionMode,
    ) -> TranslationResponse:
        started = perf_counter()
        warnings: list[str] = []
        component_calls = {
            "analysis": 0,
            "retrieval": 0,
            "translation": 0,
            "revision": 0,
            "judgment": 0,
        }
        uses_rag = self._uses_rag(mode)
        uses_agent = self._uses_agent(mode)

        if uses_rag and self.retriever is None:
            raise ComponentUnavailableError("RAG mode requires a configured retriever")
        if uses_agent and self.agent is None:
            raise ComponentUnavailableError("Agent mode requires a configured Agent")

        analysis: SourceAnalysis | None = None
        analysis_ms = 0.0
        if uses_agent:
            analysis_started = perf_counter()
            try:
                assert self.agent is not None
                component_calls["analysis"] += 1
                analysis = self.agent.analyze(request.text)
            except (ComponentExecutionError, ComponentUnavailableError) as exc:
                warnings.append(str(exc))
                analysis = SourceAnalysis(domain=None, key_terms=[], confidence=0.0)
            analysis_ms = _elapsed_ms(analysis_started)

        attempts: list[AttemptTrace] = []
        suggested_terms: list[str] = []
        pending_action: NextAction | None = None
        current_hits: list[RetrievalHit] = []
        stop_reason = StopReason.BASELINE_COMPLETE
        final_index = 0
        constraint_fallback_used = False
        coverage_requirements = extract_source_coverage_requirements(
            request.text, analysis
        )

        max_attempts = self.settings.max_attempts if uses_agent else 1
        for attempt_index in range(max_attempts):
            # Attempt zero owns the one-time source analysis in its timing total.
            attempt_started = started if attempt_index == 0 else perf_counter()
            retrieval_ms = 0.0
            translation_ms = 0.0
            revision_ms = 0.0
            judgment_ms = 0.0
            query: RetrievalQuery | None = None
            requested_action = pending_action if attempt_index > 0 else None
            applied_action: NextAction | None = None
            applied_must_preserve_constraints = []

            should_retrieve = uses_rag and (
                attempt_index == 0 or requested_action is NextAction.RETRY_WITH_RAG
            )
            if should_retrieve:
                coverage_focus_terms = (
                    list(
                        dict.fromkeys(
                            finding.requirement.source_term
                            for finding in missing_coverage_findings(
                                attempts[-1].coverage_findings
                            )
                            if finding.recovery_action
                            is CoverageRecoveryAction.TARGETED_RAG
                        )
                    )
                    if attempt_index > 0 and attempts
                    else []
                )
                is_targeted_refresh = (
                    attempt_index > 0
                    and requested_action is NextAction.RETRY_WITH_RAG
                    and bool(coverage_focus_terms)
                )
                query = RetrievalQuery(
                    source_text=(
                        " ".join(coverage_focus_terms)
                        if is_targeted_refresh
                        else request.text
                    ),
                    domain=analysis.domain if analysis else None,
                    key_terms=list(
                        dict.fromkeys(
                            coverage_focus_terms
                            if is_targeted_refresh
                            else [
                                *(analysis.key_terms if analysis else []),
                                *suggested_terms,
                            ]
                        )
                    ),
                    top_k=self.settings.retrieval_top_k,
                    attempt_index=attempt_index,
                )
                retrieval_started = perf_counter()
                try:
                    assert self.retriever is not None
                    component_calls["retrieval"] += 1
                    discovered_hits = self.retriever.retrieve(query)
                    current_hits = merge_retrieval_hits(
                        current_hits, discovered_hits
                    )
                except (
                    ComponentExecutionError,
                    ComponentUnavailableError,
                    ValueError,
                ) as exc:
                    warnings.append(str(exc))
                retrieval_ms = _elapsed_ms(retrieval_started)

            hits = list(current_hits) if uses_rag else []
            coverage_requirements = merge_coverage_requirements(
                coverage_requirements,
                extract_source_coverage_requirements(request.text, analysis, hits),
            )
            # Only attempt zero calls the NMT translator.  Revision attempts use
            # cumulative hits and the separate must-preserve channel instead.
            constraints = (
                self._constraints(hits, request.text)
                if uses_rag and attempt_index == 0
                else []
            )
            parent_attempt_index: int | None = None

            if attempt_index == 0:
                candidate_origin = CandidateOrigin.NMT
                translation_started = perf_counter()
                try:
                    component_calls["translation"] += 1
                    candidate = self.translator.translate(
                        request.text,
                        constraints=constraints or None,
                    )
                except ConstraintApplicationError as exc:
                    if not constraints:
                        raise
                    warnings.append(str(exc))
                    warnings.append(
                        "The rejected constrained candidate is not preserved because the "
                        "translator exception does not expose it (TODO: capture it)."
                    )
                    constraint_fallback_used = True
                    constraints = []
                    candidate_origin = CandidateOrigin.FALLBACK
                    component_calls["translation"] += 1
                    candidate = self.translator.translate(request.text, constraints=None)
                translation_ms = _elapsed_ms(translation_started)
            else:
                assert attempts
                previous_attempt = attempts[-1]
                previous_judgment = previous_attempt.judgment
                if self.reviser is None or previous_judgment is None:
                    warnings.append(
                        "A revision was requested but no usable TranslationReviser is "
                        "configured; returning the best existing candidate."
                    )
                    stop_reason = StopReason.COMPONENT_FAILURE
                    break
                revision_started = perf_counter()
                try:
                    prior_constraints = merge_must_preserve_constraints(
                        previous_attempt.applied_must_preserve_constraints,
                        previous_judgment.must_preserve_constraints,
                    )
                    applied_must_preserve_constraints = constraints_for_retry(
                        previous_attempt.coverage_findings,
                        coverage_requirements,
                        prior_constraints,
                    )
                    revision_judgment = previous_judgment.model_copy(
                        update={
                            "must_preserve_constraints": (
                                applied_must_preserve_constraints
                            )
                        }
                    )
                    component_calls["revision"] += 1
                    candidate = self.reviser.revise(
                        request.text,
                        previous_attempt.candidate,
                        revision_judgment,
                        hits,
                    )
                except (ComponentExecutionError, ComponentUnavailableError) as exc:
                    warnings.append(str(exc))
                    stop_reason = StopReason.COMPONENT_FAILURE
                    break
                revision_ms = _elapsed_ms(revision_started)
                candidate_origin = CandidateOrigin.AGENT_REVISION
                parent_attempt_index = previous_attempt.attempt_index
                applied_action = (
                    requested_action
                    if uses_rag or requested_action is NextAction.REVISE
                    else NextAction.REVISE
                )

            coverage_findings = check_source_coverage(
                candidate.text,
                coverage_requirements,
                targeted_rag_available=uses_rag,
            )

            if attempts and _same_translation(candidate.text, attempts[-1].candidate.text):
                attempts.append(
                    AttemptTrace(
                        attempt_index=attempt_index,
                        candidate_origin=candidate_origin,
                        parent_attempt_index=parent_attempt_index,
                        requested_action=requested_action,
                        applied_action=applied_action,
                        retrieval_query=query,
                        retrieval_hits=hits,
                        applied_constraints=constraints,
                        applied_must_preserve_constraints=(
                            applied_must_preserve_constraints
                        ),
                        coverage_findings=coverage_findings,
                        candidate=candidate,
                        judgment=None,
                        timings=StageTimings(
                            retrieval_ms=retrieval_ms,
                            translation_ms=translation_ms,
                            revision_ms=revision_ms,
                            total_ms=_elapsed_ms(attempt_started),
                        ),
                    )
                )
                stop_reason = StopReason.UNCHANGED
                break

            judgment: QualityJudgment | None = None
            if uses_agent:
                judgment_started = perf_counter()
                try:
                    assert self.agent is not None
                    component_calls["judgment"] += 1
                    judgment = self.agent.judge(request.text, candidate, hits)
                    judgment = apply_coverage_to_judgment(
                        judgment, coverage_findings
                    )
                except (ComponentExecutionError, ComponentUnavailableError) as exc:
                    warnings.append(str(exc))
                judgment_ms = _elapsed_ms(judgment_started)

            attempts.append(
                AttemptTrace(
                    attempt_index=attempt_index,
                    candidate_origin=candidate_origin,
                    parent_attempt_index=parent_attempt_index,
                    requested_action=requested_action,
                    applied_action=applied_action,
                    retrieval_query=query,
                    retrieval_hits=hits,
                    applied_constraints=constraints,
                    applied_must_preserve_constraints=(
                        applied_must_preserve_constraints
                    ),
                    coverage_findings=coverage_findings,
                    candidate=candidate,
                    judgment=judgment,
                    timings=StageTimings(
                        analysis_ms=analysis_ms if attempt_index == 0 else 0.0,
                        retrieval_ms=retrieval_ms,
                        translation_ms=translation_ms,
                        revision_ms=revision_ms,
                        judgment_ms=judgment_ms,
                        total_ms=_elapsed_ms(attempt_started),
                    ),
                )
            )

            if not uses_agent:
                stop_reason = (
                    StopReason.CONSTRAINT_FALLBACK
                    if constraint_fallback_used
                    else StopReason.NO_RETRIEVAL_MATCH
                    if uses_rag and not hits
                    else StopReason.RAG_COMPLETE
                    if uses_rag
                    else StopReason.BASELINE_COMPLETE
                )
                break

            if judgment is None:
                stop_reason = StopReason.COMPONENT_FAILURE
                break
            if judgment.passed:
                stop_reason = StopReason.PASSED
                final_index = attempt_index
                break
            if judgment.next_action is NextAction.STOP:
                stop_reason = StopReason.AGENT_STOP
                break
            if attempt_index == max_attempts - 1:
                stop_reason = StopReason.MAX_RETRIES
                break
            if self.reviser is None:
                warnings.append(
                    "The Agent requested a revision but no TranslationReviser is "
                    "configured; returning the best existing candidate."
                )
                stop_reason = StopReason.COMPONENT_FAILURE
                break
            pending_action = judgment.next_action
            suggested_terms = list(judgment.suggested_query_terms)

        judged = [
            (index, attempt.judgment)
            for index, attempt in enumerate(attempts)
            if attempt.judgment is not None
        ]
        passed = [item for item in judged if item[1].passed]
        coverage_clear = [
            item
            for item in judged
            if not missing_coverage_findings(attempts[item[0]].coverage_findings)
        ]
        coverage_priority_used = False
        if passed:
            final_index = max(passed, key=lambda item: item[1].quality_score)[0]
        elif coverage_clear:
            final_index = max(
                coverage_clear, key=lambda item: item[1].quality_score
            )[0]
            unconstrained_index = max(
                judged, key=lambda item: item[1].quality_score
            )[0]
            coverage_priority_used = final_index != unconstrained_index
        elif judged:
            final_index = max(judged, key=lambda item: item[1].quality_score)[0]
        else:
            final_index = 0
        if coverage_priority_used:
            warnings.append(
                "A lower-scoring candidate was selected to preserve required "
                "source coverage."
            )
            selection_reason = SelectionReason.SOURCE_COVERAGE_PRESERVED
        elif final_index != len(attempts) - 1:
            warnings.append("A previous higher-scoring candidate was selected as final output.")
            selection_reason = SelectionReason.HIGHEST_QUALITY_ROLLBACK
        elif judged and judged[-1][1].passed:
            selection_reason = SelectionReason.PASSED
        elif judged:
            selection_reason = SelectionReason.HIGHEST_QUALITY
        else:
            selection_reason = SelectionReason.ONLY_CANDIDATE

        return self._response(
            request=request,
            mode=mode,
            analysis=analysis,
            coverage_requirements=coverage_requirements,
            attempts=attempts,
            final_index=final_index,
            stop_reason=stop_reason,
            selection_reason=selection_reason,
            component_call_counts=ComponentCallCounts(**component_calls),
            warnings=warnings,
            started=started,
        )
