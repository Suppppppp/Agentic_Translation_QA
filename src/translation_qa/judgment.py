"""Pure normalization for model-produced translation quality judgments."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from pydantic import Field, StrictBool

from translation_qa.schemas import (
    Confidence,
    JudgmentConsistencyIssue,
    JudgmentDecisionAudit,
    NextAction,
    NonEmptyText,
    QualityJudgment,
    SchemaModel,
    TranslationErrorType,
)


class RawLLMQualityJudgment(SchemaModel):
    """Strict redundant claims returned by the LLM before code decides."""

    passed: StrictBool
    quality_score: Confidence
    error_types: list[str]
    summary: NonEmptyText
    confidence: Confidence
    next_action: NextAction
    suggested_query_terms: list[NonEmptyText] = Field(default_factory=list)


_NEGATED_ERROR_PATTERN = re.compile(
    r"\b(?:without|no)\s+(?:any\s+)?(?:blocking\s+|major\s+|significant\s+)?"
    r"(?:errors?|issues?|problems?)\b",
    re.IGNORECASE,
)
_EXPLICIT_ERROR_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\berrors?\b",
        r"\bincorrect(?:ly)?\b",
        r"\bmistranslat(?:e|ed|es|ion)\b",
        r"\bomits?(?:ted)?\b",
        r"\bomission\b",
        r"\bmissing\b",
        r"\binstead of\b",
        r"\bambiguous\b",
        r"\bawkward(?:ly)?\b",
        r"\bwrong\b",
        r"\badds?\b.{0,100}\b(?:isn't|is not|not)\b.{0,50}"
        r"\b(?:original|source)\b",
    )
)
_EXPLICIT_PASS_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bacceptable\b",
        r"\baccurately\s+(?:captures?|conveys?|reflects?|renders?)\b",
        r"\bwithout\s+(?:any\s+)?(?:errors?|issues?|problems?)\b",
        r"\bno\s+(?:blocking\s+|major\s+|significant\s+)?"
        r"(?:errors?|issues?|problems?)\b",
        r"\bhigh quality\b",
    )
)


def _has_unnegated_match(patterns: Sequence[re.Pattern[str]], text: str) -> bool:
    """Ignore a narrow English negation window around diagnostic keywords."""

    for pattern in patterns:
        for match in pattern.finditer(text):
            prefix = text[max(0, match.start() - 40) : match.start()]
            if re.search(
                r"\b(?:no|not|never|without|neither)\b(?:\W+\w+){0,5}\W*$",
                prefix,
                re.IGNORECASE,
            ):
                continue
            if re.search(
                r"\b(?:does|do|did|is|are|was|were)\s+not\b"
                r"(?:\W+\w+){0,4}\W*$",
                prefix,
                re.IGNORECASE,
            ):
                continue
            return True
    return False


def summary_explicitly_reports_error(summary: str) -> bool:
    """Conservatively identify an explicit defect claim for audit purposes only."""

    without_negated_errors = _NEGATED_ERROR_PATTERN.sub("", summary)
    return _has_unnegated_match(_EXPLICIT_ERROR_PATTERNS, without_negated_errors)


def summary_explicitly_reports_pass(summary: str) -> bool:
    """Identify an explicit acceptance claim without interpreting translation quality."""

    return _has_unnegated_match(_EXPLICIT_PASS_PATTERNS, summary)


def canonicalize_llm_judgment(
    payload: dict[str, Any],
    *,
    additional_error_types: Sequence[TranslationErrorType] = (),
    forced_error_action: NextAction | None = None,
) -> QualityJudgment:
    """Validate model claims and derive the binary decision from structured errors.

    The free-text summary and the model's ``passed`` field are retained only as
    auditable claims.  They never delete or override a structured error type.
    ``next_action`` remains a retry-routing hint when it is compatible with a
    failed decision; accept/revise consistency is enforced by code.
    """

    raw = RawLLMQualityJudgment.model_validate(payload)

    reported_error_types = list(raw.error_types)
    reported_canonical_error_types: list[TranslationErrorType] = []
    unsupported = False
    for label in raw.error_types:
        try:
            error_type = TranslationErrorType(label)
        except ValueError:
            error_type = TranslationErrorType.OTHER
            unsupported = True
        if error_type not in reported_canonical_error_types:
            reported_canonical_error_types.append(error_type)

    reported_derived_passed = not reported_canonical_error_types
    if reported_derived_passed:
        reported_derived_action = NextAction.ACCEPT
    elif raw.next_action is NextAction.ACCEPT:
        reported_derived_action = NextAction.REVISE
    else:
        reported_derived_action = raw.next_action

    issues: list[JudgmentConsistencyIssue] = []
    if raw.passed is not reported_derived_passed:
        issues.append(JudgmentConsistencyIssue.REPORTED_PASSED_MISMATCH)
    if raw.next_action is not reported_derived_action:
        issues.append(JudgmentConsistencyIssue.REPORTED_NEXT_ACTION_MISMATCH)

    summary_reports_error = summary_explicitly_reports_error(raw.summary)
    if summary_reports_error and not reported_canonical_error_types:
        issues.append(
            JudgmentConsistencyIssue.SUMMARY_ERROR_WITHOUT_STRUCTURED_ERROR
        )
    elif (
        reported_canonical_error_types
        and not summary_reports_error
        and summary_explicitly_reports_pass(raw.summary)
    ):
        issues.append(JudgmentConsistencyIssue.SUMMARY_PASS_WITH_STRUCTURED_ERROR)

    if unsupported:
        issues.append(
            JudgmentConsistencyIssue.UNSUPPORTED_ERROR_TYPE_NORMALIZED
        )

    canonical_error_types = list(reported_canonical_error_types)
    code_added_error_types: list[TranslationErrorType] = []
    for error_type in additional_error_types:
        if error_type not in code_added_error_types:
            code_added_error_types.append(error_type)
        if error_type not in canonical_error_types:
            canonical_error_types.append(error_type)

    derived_passed = not canonical_error_types
    if derived_passed:
        derived_action = NextAction.ACCEPT
    elif forced_error_action is not None:
        if forced_error_action is NextAction.ACCEPT:
            raise ValueError("forced_error_action cannot accept a structured error")
        derived_action = forced_error_action
    elif raw.next_action is NextAction.ACCEPT:
        derived_action = NextAction.REVISE
    else:
        derived_action = raw.next_action

    return QualityJudgment(
        passed=derived_passed,
        quality_score=raw.quality_score,
        error_types=canonical_error_types,
        summary=raw.summary,
        confidence=raw.confidence,
        next_action=derived_action,
        suggested_query_terms=raw.suggested_query_terms,
        decision_audit=JudgmentDecisionAudit(
            reported_passed=raw.passed,
            reported_next_action=raw.next_action,
            reported_error_types=reported_error_types,
            code_added_error_types=code_added_error_types,
            consistency_issues=issues,
        ),
    )
