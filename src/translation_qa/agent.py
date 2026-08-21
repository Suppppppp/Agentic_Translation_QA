"""Translation analysis and quality-judgment agents."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import ValidationError

from translation_qa.errors import ComponentExecutionError, ComponentUnavailableError
from translation_qa.judgment import canonicalize_llm_judgment
from translation_qa.metrics import contains_accepted_target
from translation_qa.retrieval import contains_source_term
from translation_qa.schemas import (
    NextAction,
    QualityJudgment,
    RetrievalHit,
    RevisionResult,
    SourceAnalysis,
    TranslationCandidate,
    TranslationErrorType,
)


class RuleBasedTermAgent:
    """Deterministic fallback used for contract tests and term-only checks.

    It is not a replacement for the required local LLM Agent. Its explicit model
    identifier keeps benchmark traces from mistaking it for one.
    """

    @property
    def model_id(self) -> str:
        return "rule-based-term-checker"

    def analyze(self, source_text: str) -> SourceAnalysis:
        return SourceAnalysis(domain=None, key_terms=[], confidence=0.0)

    def judge(
        self,
        source_text: str,
        candidate: TranslationCandidate,
        retrieved_terms: Sequence[RetrievalHit],
    ) -> QualityJudgment:
        missing = [
            hit
            for hit in retrieved_terms
            if contains_source_term(source_text, hit.source_term)
            and not contains_accepted_target(
                candidate.text,
                [hit.target_term, *hit.accepted_target_variants],
            )
        ]
        if not missing:
            return QualityJudgment(
                passed=True,
                quality_score=1.0,
                error_types=[],
                summary="Required retrieved terms are present.",
                confidence=1.0,
                next_action=NextAction.ACCEPT,
            )
        missing_targets = ", ".join(hit.target_term for hit in missing)
        return QualityJudgment(
            passed=False,
            quality_score=0.4,
            error_types=[TranslationErrorType.TERM],
            summary=f"Missing required target terms: {missing_targets}",
            confidence=1.0,
            next_action=NextAction.RETRY_WITH_CONSTRAINTS,
            suggested_query_terms=[hit.source_term for hit in missing],
        )


class OllamaTranslationAgent:
    """Local Ollama chat adapter that returns validated structured decisions."""

    def __init__(
        self,
        model_id: str,
        *,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 120.0,
    ) -> None:
        self._model_id = model_id
        self._chat_url = f"{base_url.rstrip('/')}/api/chat"
        self._timeout_seconds = timeout_seconds

    @property
    def model_id(self) -> str:
        return self._model_id

    def _chat_json(
        self,
        system: str,
        user: str,
        *,
        think: bool = True,
    ) -> dict[str, Any]:
        payload = {
            "model": self._model_id,
            "stream": False,
            # Source routing is simple enough to run without a reasoning phase,
            # while quality judgment and revision retain it.  A pilot with
            # thinking disabled globally was faster but missed clear grammar
            # errors, so the caller selects the quality/latency trade-off.
            "think": think,
            "format": "json",
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        request = Request(
            self._chat_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise ComponentUnavailableError(f"Ollama request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ComponentExecutionError("Ollama returned invalid response JSON") from exc

        content = response_payload.get("message", {}).get("content")
        if not isinstance(content, str):
            raise ComponentExecutionError("Ollama response did not include message content")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ComponentExecutionError("Agent content was not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise ComponentExecutionError("Agent JSON must be an object")
        return parsed

    def analyze(self, source_text: str) -> SourceAnalysis:
        system = (
            "Analyze a Korean source sentence for translation retrieval. Return only JSON "
            "with domain (string or null), key_terms (array of Korean strings), and "
            "confidence (0 to 1). Do not translate the sentence."
        )
        try:
            return SourceAnalysis.model_validate(
                self._chat_json(system, source_text, think=False)
            )
        except ValidationError as exc:
            raise ComponentExecutionError(f"invalid Agent analysis schema: {exc}") from exc

    def judge(
        self,
        source_text: str,
        candidate: TranslationCandidate,
        retrieved_terms: Sequence[RetrievalHit],
    ) -> QualityJudgment:
        evidence = [
            {
                "source_term": hit.source_term,
                "target_term": hit.target_term,
                "accepted_target_variants": hit.accepted_target_variants,
                "definition": hit.definition,
            }
            for hit in retrieved_terms
        ]
        system = (
            "Judge a Korean-to-English translation. Use only the source, candidate, and "
            "retrieved glossary evidence. error_types is the authoritative structured "
            "error list: include every blocking defect mentioned in summary, and leave "
            "it empty only when the candidate should pass. Return only JSON with: "
            "passed (boolean), "
            "quality_score (0 to 1), error_types (array chosen from term, meaning, "
            "omission_addition, entity_value, fluency_grammar, other), summary (one short "
            "English sentence, not hidden reasoning), confidence (0 to 1), next_action "
            "(accept, revise, retry_with_rag, retry_with_constraints, or stop), and "
            "suggested_query_terms (array of Korean terms). passed and next_action must "
            "agree with error_types, but application code derives the final binary "
            "decision from error_types."
        )
        user = json.dumps(
            {
                "source_ko": source_text,
                "candidate_en": candidate.text,
                "retrieved_glossary": evidence,
            },
            ensure_ascii=False,
        )
        payload = self._chat_json(system, user)
        missing = [
            hit
            for hit in retrieved_terms
            if contains_source_term(source_text, hit.source_term)
            and not contains_accepted_target(
                candidate.text,
                [hit.target_term, *hit.accepted_target_variants],
            )
        ]
        try:
            judgment = canonicalize_llm_judgment(
                payload,
                additional_error_types=(
                    [TranslationErrorType.TERM] if missing else []
                ),
                forced_error_action=NextAction.REVISE if missing else None,
            )
        except ValidationError as exc:
            raise ComponentExecutionError(f"invalid Agent judgment schema: {exc}") from exc
        if not missing:
            return judgment

        missing_targets = ", ".join(hit.target_term for hit in missing)
        return QualityJudgment(
            passed=False,
            quality_score=min(judgment.quality_score, 0.4),
            error_types=judgment.error_types,
            summary=f"Required glossary targets are missing: {missing_targets}.",
            confidence=1.0,
            next_action=NextAction.REVISE,
            suggested_query_terms=list(
                dict.fromkeys(
                    [
                        *(hit.source_term for hit in missing),
                        *judgment.suggested_query_terms,
                    ]
                )
            ),
            decision_audit=judgment.decision_audit,
        )

    def revise(
        self,
        source_text: str,
        previous_candidate: TranslationCandidate,
        judgment: QualityJudgment,
        retrieved_terms: Sequence[RetrievalHit],
    ) -> TranslationCandidate:
        """Post-edit one candidate using structured feedback and glossary evidence."""

        evidence = [
            {
                "source_term": hit.source_term,
                "target_term": hit.target_term,
                "accepted_target_variants": hit.accepted_target_variants,
                "definition": hit.definition,
            }
            for hit in retrieved_terms
        ]
        system = (
            "Post-edit a Korean-to-English translation. Preserve all source meaning and "
            "change only what is needed to address the structured quality judgment. Use "
            "retrieved glossary evidence only when its Korean source term is applicable. "
            "For each must_preserve item, explicitly realize the source item rather than "
            "replacing a standalone subject or entity with an unanchored pronoun. When "
            "accepted_targets are supplied, use one of those target forms. "
            "Never invent facts and never request or infer a reference translation. Return "
            "only JSON with one non-empty field: translation."
        )
        user = json.dumps(
            {
                "source_ko": source_text,
                "previous_candidate_en": previous_candidate.text,
                "judgment": {
                    "error_types": [error.value for error in judgment.error_types],
                    "summary": judgment.summary,
                    "requested_action": judgment.next_action.value,
                    "suggested_query_terms": judgment.suggested_query_terms,
                },
                "retrieved_glossary": evidence,
                "must_preserve": [
                    {
                        "source_term": constraint.source_term,
                        "kinds": [kind.value for kind in constraint.kinds],
                        "accepted_targets": constraint.accepted_targets,
                    }
                    for constraint in judgment.must_preserve_constraints
                ],
            },
            ensure_ascii=False,
        )
        try:
            result = RevisionResult.model_validate(self._chat_json(system, user))
        except ValidationError as exc:
            raise ComponentExecutionError(f"invalid Agent revision schema: {exc}") from exc
        return TranslationCandidate(text=result.translation, model_id=self._model_id)
