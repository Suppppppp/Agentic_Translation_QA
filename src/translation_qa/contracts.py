"""Synchronous component contracts used for dependency injection and tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from translation_qa.schemas import (
    BenchmarkRequest,
    BenchmarkResponse,
    ExecutionMode,
    QualityJudgment,
    RetrievalHit,
    RetrievalQuery,
    SourceAnalysis,
    TermConstraint,
    TranslationCandidate,
    TranslationRequest,
    TranslationResponse,
)


@runtime_checkable
class Translator(Protocol):
    """Translate with the same backend, optionally using glossary constraints."""

    @property
    def model_id(self) -> str: ...

    def translate(
        self,
        source_text: str,
        constraints: Sequence[TermConstraint] | None = None,
    ) -> TranslationCandidate: ...


@runtime_checkable
class Retriever(Protocol):
    """Retrieve glossary/domain knowledge for a structured query."""

    def retrieve(self, query: RetrievalQuery) -> list[RetrievalHit]: ...


@runtime_checkable
class TranslationAgent(Protocol):
    """Analyze source text and judge a candidate without controlling retries."""

    @property
    def model_id(self) -> str: ...

    def analyze(self, source_text: str) -> SourceAnalysis: ...

    def judge(
        self,
        source_text: str,
        candidate: TranslationCandidate,
        retrieved_terms: Sequence[RetrievalHit],
    ) -> QualityJudgment: ...


@runtime_checkable
class TranslationReviser(Protocol):
    """Post-edit a candidate from structured feedback without using references."""

    @property
    def model_id(self) -> str: ...

    def revise(
        self,
        source_text: str,
        previous_candidate: TranslationCandidate,
        judgment: QualityJudgment,
        retrieved_terms: Sequence[RetrievalHit],
    ) -> TranslationCandidate: ...


@runtime_checkable
class TranslationService(Protocol):
    """Application service consumed by FastAPI routes."""

    def translate(
        self,
        request: TranslationRequest,
        mode: ExecutionMode,
    ) -> TranslationResponse: ...


@runtime_checkable
class BenchmarkService(Protocol):
    """Synchronous benchmark service for the small assignment dataset."""

    def run(self, request: BenchmarkRequest) -> BenchmarkResponse: ...
