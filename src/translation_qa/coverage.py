"""Deterministic source-coverage extraction, checking, and recovery routing.

The rules in this module are intentionally conservative.  They use the Korean
source, source analysis, and retrieved glossary evidence only; a reference
translation and human labels are never inputs.  An item that cannot be checked
reliably is reported as ``unresolved`` rather than being treated as missing.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import TypeVar

from translation_qa.metrics import contains_accepted_target
from translation_qa.retrieval import contains_source_term
from translation_qa.schemas import (
    CoverageRecoveryAction,
    JudgmentDecisionAudit,
    MustPreserveConstraint,
    NextAction,
    QualityJudgment,
    RetrievalHit,
    RetrievalMatchType,
    SourceAnalysis,
    SourceCoverageExtractionRule,
    SourceCoverageFinding,
    SourceCoverageKind,
    SourceCoverageRequirement,
    SourceCoverageStatus,
    TranslationErrorType,
)


_LEADING_TOPIC_SUBJECT = re.compile(
    r"^\s*[\"'\u2018\u2019\u201c\u201d(\[]*\s*"
    # A period is allowed inside identifiers such as ``Node.js``.  The source
    # sentence itself is still bounded by the first topic/subject particle.
    r"(?P<subject>[^,;:!?\n]{1,60}?)(?:은|는|이|가)"
    r"(?=\s|[,.;:!?]|$)"
)
_LATIN_IDENTIFIER = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"[A-Za-z][A-Za-z0-9]*(?:\.[A-Za-z0-9]+)*(?:\+\+|#)?"
    r"(?:[-_/][A-Za-z0-9]+)*"
    r"(?![A-Za-z0-9_])"
)
_LEADING_UNANCHORED_PRONOUN = re.compile(
    r"^\s*[\"'\u2018\u2019\u201c\u201d(\[]*\s*"
    r"(?P<pronoun>it(?:['\u2019]s|\s+(?:is|was|has|does|can|will))?"
    r"|they(?:['\u2019]re|\s+(?:are|were|have|do|can|will))?"
    r"|this\s+(?:is|was|has|does|can|will)"
    r"|that\s+(?:is|was|has|does|can|will))\b",
    re.IGNORECASE,
)
_SOURCE_DEMONSTRATIVE_PRONOUNS = {
    "이",
    "그",
    "저",
    "이것",
    "그것",
    "저것",
}
_NON_BLOCKING_GENERIC_SUBJECTS = {
    "경우",
    "결과",
    "과정",
    "내용",
    "문제",
    "방법",
    "사실",
    "상황",
    "이유",
    "점",
}
_MATCH_PRIORITY = {
    RetrievalMatchType.EXACT: 3,
    RetrievalMatchType.HYBRID: 2,
    RetrievalMatchType.VECTOR: 1,
}
_T = TypeVar("_T")


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _stable_union(left: Sequence[_T], right: Sequence[_T]) -> list[_T]:
    result = list(left)
    for item in right:
        if item not in result:
            result.append(item)
    return result


def _stable_text_union(left: Sequence[str], right: Sequence[str]) -> list[str]:
    result = list(left)
    seen = {_normalized(item) for item in result}
    for item in right:
        key = _normalized(item)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _subject(source_text: str) -> str | None:
    match = _LEADING_TOPIC_SUBJECT.search(source_text)
    if match is None:
        return None
    subject = " ".join(match.group("subject").split()).strip("\"'\u2018\u2019\u201c\u201d()[] ")
    if not subject or _normalized(subject) in _SOURCE_DEMONSTRATIVE_PRONOUNS:
        return None
    return subject


def _latin_identifiers(source_text: str) -> list[str]:
    identifiers: list[str] = []
    for match in _LATIN_IDENTIFIER.finditer(source_text):
        value = match.group(0)
        # Lowercase prose fragments are not strong proper-name evidence.  Keep
        # identifiers with capitalization, digits, or product punctuation.
        if not any(character.isupper() or character.isdigit() for character in value):
            if not any(character in ".+#" for character in value):
                continue
        if value not in identifiers:
            identifiers.append(value)
    return identifiers


def _add_requirement(
    requirements: list[SourceCoverageRequirement],
    *,
    source_term: str,
    kinds: Sequence[SourceCoverageKind],
    extraction_rules: Sequence[SourceCoverageExtractionRule],
    accepted_targets: Sequence[str] = (),
    evidence_hit_ids: Sequence[str] = (),
) -> None:
    key = _normalized(source_term)
    for index, requirement in enumerate(requirements):
        if _normalized(requirement.source_term) != key:
            continue
        requirements[index] = SourceCoverageRequirement(
            source_term=requirement.source_term,
            kinds=_stable_union(requirement.kinds, kinds),
            accepted_targets=_stable_text_union(
                requirement.accepted_targets, accepted_targets
            ),
            extraction_rules=_stable_union(
                requirement.extraction_rules, extraction_rules
            ),
            evidence_hit_ids=_stable_union(
                requirement.evidence_hit_ids, evidence_hit_ids
            ),
        )
        return
    requirements.append(
        SourceCoverageRequirement(
            source_term=source_term,
            kinds=_stable_union([], kinds),
            accepted_targets=_stable_text_union([], accepted_targets),
            extraction_rules=_stable_union([], extraction_rules),
            evidence_hit_ids=_stable_union([], evidence_hit_ids),
        )
    )


def extract_source_coverage_requirements(
    source_text: str,
    analysis: SourceAnalysis | None = None,
    retrieved_hits: Sequence[RetrievalHit] = (),
) -> list[SourceCoverageRequirement]:
    """Extract source-anchored requirements without reference-side leakage."""

    requirements: list[SourceCoverageRequirement] = []
    subject = _subject(source_text)
    identifiers = _latin_identifiers(source_text)
    anchored_key_terms = [
        key_term
        for key_term in (analysis.key_terms if analysis is not None else ())
        if contains_source_term(source_text, key_term)
    ]
    applicable_hits = [
        hit
        for hit in retrieved_hits
        if contains_source_term(source_text, hit.source_term)
    ]
    subject_anchors = [
        *identifiers,
        *anchored_key_terms,
        *(hit.source_term for hit in applicable_hits),
    ]
    # Both values below are literal spans captured from ``source_text``.  Do
    # not re-run the glossary boundary matcher here: Latin identifiers can
    # legitimately take an attached Korean topic particle (for example
    # ``Node.js는``), which that Korean-glossary matcher intentionally rejects.
    subject_key = _normalized(subject) if subject is not None else None
    subject_is_required = subject_key is not None and any(
        _normalized(anchor) in subject_key or subject_key in _normalized(anchor)
        for anchor in subject_anchors
    )
    if (
        subject is not None
        and subject_is_required
        and subject_key not in _NON_BLOCKING_GENERIC_SUBJECTS
    ):
        _add_requirement(
            requirements,
            source_term=subject,
            kinds=[SourceCoverageKind.STANDALONE_SUBJECT],
            extraction_rules=[
                SourceCoverageExtractionRule.LEADING_TOPIC_SUBJECT
            ],
        )

    for identifier in identifiers:
        _add_requirement(
            requirements,
            source_term=identifier,
            kinds=[SourceCoverageKind.PROPER_NAME],
            accepted_targets=[identifier],
            extraction_rules=[SourceCoverageExtractionRule.LATIN_IDENTIFIER],
        )

    for key_term in anchored_key_terms:
        _add_requirement(
            requirements,
            source_term=key_term,
            kinds=[SourceCoverageKind.TECHNICAL_TERM],
            extraction_rules=[SourceCoverageExtractionRule.ANALYSIS_KEY_TERM],
        )

    for hit in applicable_hits:
        _add_requirement(
            requirements,
            source_term=hit.source_term,
            kinds=[SourceCoverageKind.TECHNICAL_TERM],
            accepted_targets=[hit.target_term, *hit.accepted_target_variants],
            extraction_rules=[SourceCoverageExtractionRule.RETRIEVED_GLOSSARY],
            evidence_hit_ids=[hit.term_id],
        )
    return requirements


def merge_coverage_requirements(
    existing: Sequence[SourceCoverageRequirement],
    discovered: Sequence[SourceCoverageRequirement],
) -> list[SourceCoverageRequirement]:
    """Stable-union requirements while retaining newly discovered target evidence."""

    merged: list[SourceCoverageRequirement] = []
    for requirement in [*existing, *discovered]:
        _add_requirement(
            merged,
            source_term=requirement.source_term,
            kinds=requirement.kinds,
            accepted_targets=requirement.accepted_targets,
            extraction_rules=requirement.extraction_rules,
            evidence_hit_ids=requirement.evidence_hit_ids,
        )
    return merged


def _matched_target(candidate_text: str, targets: Sequence[str]) -> str | None:
    for target in targets:
        normalized_target = unicodedata.normalize("NFKC", target)
        is_upper_acronym = (
            any(character.isalpha() for character in normalized_target)
            and normalized_target.isupper()
            and len(normalized_target) <= 5
        )
        if is_upper_acronym:
            present = re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(normalized_target)}"
                rf"(?![A-Za-z0-9_])",
                unicodedata.normalize("NFKC", candidate_text),
            ) is not None
        else:
            present = contains_accepted_target(candidate_text, [target])
        if present:
            return target
    return None


def check_source_coverage(
    candidate_text: str,
    requirements: Sequence[SourceCoverageRequirement],
    *,
    targeted_rag_available: bool,
) -> list[SourceCoverageFinding]:
    """Check one candidate, blocking only on positive evidence of omission."""

    findings: list[SourceCoverageFinding] = []
    pronoun_match = _LEADING_UNANCHORED_PRONOUN.search(candidate_text)
    starts_with_pronoun = (
        pronoun_match is not None
        and not pronoun_match.group("pronoun").split(maxsplit=1)[0].isupper()
    )
    for requirement in requirements:
        # A target term appearing later as an object or modifier must not hide
        # loss of the standalone source subject (for example, ``It uses Docker
        # images`` for a sentence whose subject is Docker).
        if (
            SourceCoverageKind.STANDALONE_SUBJECT in requirement.kinds
            and starts_with_pronoun
        ):
            findings.append(
                SourceCoverageFinding(
                    requirement=requirement,
                    status=SourceCoverageStatus.MISSING,
                    recovery_action=(
                        CoverageRecoveryAction.MUST_PRESERVE
                        if requirement.accepted_targets
                        else CoverageRecoveryAction.TARGETED_RAG
                        if targeted_rag_available
                        else CoverageRecoveryAction.MUST_PRESERVE
                    ),
                    reason=(
                        "The standalone source subject was replaced by an "
                        "unanchored target pronoun."
                    ),
                )
            )
            continue

        matched_target = _matched_target(
            candidate_text, requirement.accepted_targets
        )
        if matched_target is not None:
            findings.append(
                SourceCoverageFinding(
                    requirement=requirement,
                    status=SourceCoverageStatus.COVERED,
                    matched_target=matched_target,
                    reason="An accepted target form is present as whole tokens.",
                )
            )
            continue

        if requirement.accepted_targets:
            findings.append(
                SourceCoverageFinding(
                    requirement=requirement,
                    status=SourceCoverageStatus.MISSING,
                    recovery_action=CoverageRecoveryAction.MUST_PRESERVE,
                    reason="No accepted target form is present in the candidate.",
                )
            )
            continue

        findings.append(
            SourceCoverageFinding(
                requirement=requirement,
                status=SourceCoverageStatus.UNRESOLVED,
                reason=(
                    "No accepted target mapping is available and there is no "
                    "strong omission signal."
                ),
            )
        )
    return findings


def missing_coverage_findings(
    findings: Sequence[SourceCoverageFinding],
) -> list[SourceCoverageFinding]:
    return [
        finding
        for finding in findings
        if finding.status is SourceCoverageStatus.MISSING
    ]


def _constraint_from_requirement(
    requirement: SourceCoverageRequirement,
) -> MustPreserveConstraint:
    return MustPreserveConstraint(
        source_term=requirement.source_term,
        kinds=requirement.kinds,
        accepted_targets=requirement.accepted_targets,
    )


def merge_must_preserve_constraints(
    existing: Sequence[MustPreserveConstraint],
    discovered: Sequence[MustPreserveConstraint],
) -> list[MustPreserveConstraint]:
    merged: list[MustPreserveConstraint] = []
    for constraint in [*existing, *discovered]:
        key = _normalized(constraint.source_term)
        for index, current in enumerate(merged):
            if _normalized(current.source_term) != key:
                continue
            merged[index] = MustPreserveConstraint(
                source_term=current.source_term,
                kinds=_stable_union(current.kinds, constraint.kinds),
                accepted_targets=_stable_text_union(
                    current.accepted_targets, constraint.accepted_targets
                ),
            )
            break
        else:
            merged.append(constraint)
    return merged


def constraints_for_retry(
    previous_findings: Sequence[SourceCoverageFinding],
    current_requirements: Sequence[SourceCoverageRequirement],
    existing: Sequence[MustPreserveConstraint] = (),
) -> list[MustPreserveConstraint]:
    """Build reviser-only constraints after any targeted retrieval refresh."""

    current_by_term = {
        _normalized(requirement.source_term): requirement
        for requirement in current_requirements
    }
    recovered: list[MustPreserveConstraint] = []
    for finding in missing_coverage_findings(previous_findings):
        current = current_by_term.get(
            _normalized(finding.requirement.source_term), finding.requirement
        )
        recovered.append(_constraint_from_requirement(current))
    return merge_must_preserve_constraints(existing, recovered)


def apply_coverage_to_judgment(
    judgment: QualityJudgment,
    findings: Sequence[SourceCoverageFinding],
) -> QualityJudgment:
    """Make coverage findings authoritative over an LLM PASS decision."""

    missing = missing_coverage_findings(findings)
    if not missing:
        return judgment

    added_errors: list[TranslationErrorType] = []
    for finding in missing:
        kinds = finding.requirement.kinds
        if SourceCoverageKind.STANDALONE_SUBJECT in kinds:
            added_errors = _stable_union(
                added_errors, [TranslationErrorType.OMISSION_ADDITION]
            )
        if SourceCoverageKind.PROPER_NAME in kinds:
            added_errors = _stable_union(
                added_errors, [TranslationErrorType.ENTITY_VALUE]
            )
        if (
            SourceCoverageKind.TECHNICAL_TERM in kinds
            and SourceCoverageKind.STANDALONE_SUBJECT not in kinds
        ):
            added_errors = _stable_union(added_errors, [TranslationErrorType.TERM])

    errors = _stable_union(judgment.error_types, added_errors)
    needs_targeted_rag = any(
        finding.recovery_action is CoverageRecoveryAction.TARGETED_RAG
        for finding in missing
    )
    next_action = (
        NextAction.RETRY_WITH_RAG
        if needs_targeted_rag
        else NextAction.RETRY_WITH_CONSTRAINTS
    )
    direct_constraints = [
        _constraint_from_requirement(finding.requirement)
        for finding in missing
        if finding.recovery_action is CoverageRecoveryAction.MUST_PRESERVE
    ]
    constraints = merge_must_preserve_constraints(
        judgment.must_preserve_constraints, direct_constraints
    )

    audit: JudgmentDecisionAudit | None = judgment.decision_audit
    if audit is not None:
        audit = audit.model_copy(
            update={
                "code_added_error_types": _stable_union(
                    audit.code_added_error_types, added_errors
                )
            }
        )

    missing_terms = [finding.requirement.source_term for finding in missing]
    return judgment.model_copy(
        update={
            "passed": False,
            "quality_score": min(judgment.quality_score, 0.4),
            "error_types": errors,
            "summary": (
                judgment.summary.rstrip()
                + " Required source coverage is missing: "
                + ", ".join(missing_terms)
                + "."
            ),
            "confidence": max(judgment.confidence, 0.9),
            "next_action": next_action,
            "suggested_query_terms": _stable_union(
                missing_terms, judgment.suggested_query_terms
            ),
            "must_preserve_constraints": constraints,
            "decision_audit": audit,
        }
    )


def merge_retrieval_hits(
    existing: Sequence[RetrievalHit],
    discovered: Sequence[RetrievalHit],
) -> list[RetrievalHit]:
    """Stable existing-first merge, deduplicated by immutable glossary term ID."""

    merged = list(existing)
    index_by_id = {hit.term_id: index for index, hit in enumerate(merged)}
    if len(index_by_id) != len(merged):
        raise ValueError("existing retrieval hits contain duplicate term IDs")

    for hit in discovered:
        index = index_by_id.get(hit.term_id)
        if index is None:
            index_by_id[hit.term_id] = len(merged)
            merged.append(hit)
            continue
        current = merged[index]
        current_variants = {
            _normalized(value) for value in current.accepted_target_variants
        }
        discovered_variants = {
            _normalized(value) for value in hit.accepted_target_variants
        }
        if (
            _normalized(current.source_term) != _normalized(hit.source_term)
            or _normalized(current.target_term) != _normalized(hit.target_term)
            or _normalized(current.domain or "") != _normalized(hit.domain or "")
            or _normalized(current.definition or "")
            != _normalized(hit.definition or "")
            or current_variants != discovered_variants
            or current.replacement_rules != hit.replacement_rules
        ):
            raise ValueError(
                f"retrieval term ID {hit.term_id!r} changed immutable identity metadata"
            )
        current_rank = (current.score, _MATCH_PRIORITY[current.match_type])
        discovered_rank = (hit.score, _MATCH_PRIORITY[hit.match_type])
        if discovered_rank > current_rank:
            merged[index] = hit
    return merged


__all__ = [
    "apply_coverage_to_judgment",
    "check_source_coverage",
    "constraints_for_retry",
    "extract_source_coverage_requirements",
    "merge_coverage_requirements",
    "merge_must_preserve_constraints",
    "merge_retrieval_hits",
    "missing_coverage_findings",
]
