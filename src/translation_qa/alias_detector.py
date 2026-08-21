"""Evidence-bound alias checks for frozen translation candidates.

This module is intentionally detached from the translation pipeline.  It
reuses the glossary record and source-coverage contracts, but it neither
changes a judgment nor requests a retry.  Only an explicit disallowed form or
a narrowly defined confusable form is an error; absent evidence remains
``unverifiable``.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from translation_qa.retrieval import GlossaryEntry, contains_source_term
from translation_qa.schemas import SourceCoverageRequirement


_DASHES = str.maketrans({character: " " for character in "-‐‑‒–—―﹘﹣－"})
_WHITESPACE = re.compile(r"\s+")
_ALIAS_TOKEN = re.compile(r"[a-z]+\+\+|[a-z0-9]+(?:[.+#][a-z0-9]+)*")
CONFUSABLE_POLICY = "single_edit_with_inflection"


class AliasDetectionStatus(str, Enum):
    """A lexical alias result, not an overall translation-quality verdict."""

    VERIFIED = "verified"
    ERROR = "error"
    UNVERIFIABLE = "unverifiable"


class AliasEvidenceKind(str, Enum):
    ACCEPTED_VARIANT = "accepted_variant"
    EXPLICIT_DISALLOWED_VARIANT = "explicit_disallowed_variant"
    CONFUSABLE_VARIANT = "confusable_variant"
    CONFLICTING_VARIANTS = "conflicting_variants"
    MISSING_COVERAGE_EVIDENCE = "missing_coverage_evidence"
    MISSING_PROVENANCE = "missing_provenance"
    NO_TARGET_EVIDENCE = "no_target_evidence"


@dataclass(frozen=True, slots=True)
class AliasEntry:
    """Alias-facing projection of the existing :class:`GlossaryEntry`.

    The field names make the detector contract explicit while the loader and
    storage format remain the project's existing glossary CSV schema.
    """

    alias_id: str
    source: str
    preferred: str
    accepted_variants: tuple[str, ...]
    disallowed_variants: tuple[str, ...]
    domain: str | None
    provenance: str | None
    glossary_version: str | None = None
    confusable_policy: str | None = None

    @classmethod
    def from_glossary(cls, entry: GlossaryEntry) -> AliasEntry:
        return cls(
            alias_id=entry.term_id,
            source=entry.source_term,
            preferred=entry.target_term,
            accepted_variants=entry.accepted_variants,
            disallowed_variants=entry.disallowed_variants,
            domain=entry.domain,
            provenance=entry.source,
            glossary_version=entry.glossary_version,
            confusable_policy=(
                CONFUSABLE_POLICY
                if entry.notes is not None
                and f"confusable_policy={CONFUSABLE_POLICY}"
                in {part.strip() for part in entry.notes.split(";")}
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class AliasFinding:
    alias: AliasEntry
    status: AliasDetectionStatus
    evidence_kind: AliasEvidenceKind
    matched_variants: tuple[str, ...]
    reason: str

    @property
    def is_error(self) -> bool:
        return self.status is AliasDetectionStatus.ERROR

    def to_dict(self) -> dict[str, object]:
        return {
            "alias_id": self.alias.alias_id,
            "source": self.alias.source,
            "preferred": self.alias.preferred,
            "accepted_variants": list(self.alias.accepted_variants),
            "disallowed_variants": list(self.alias.disallowed_variants),
            "domain": self.alias.domain,
            "provenance": self.alias.provenance,
            "glossary_version": self.alias.glossary_version,
            "confusable_policy": self.alias.confusable_policy,
            "status": self.status.value,
            "evidence_kind": self.evidence_kind.value,
            "matched_variants": list(self.matched_variants),
            "reason": self.reason,
        }


def _normalized_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().translate(_DASHES)
    return _WHITESPACE.sub(" ", normalized).strip()


def _unique_normalized(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalized_text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(value)
    return tuple(result)


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = _normalized_text(text)
    normalized_phrase = _normalized_text(phrase)
    if not normalized_phrase:
        return False
    return (
        re.search(
            rf"(?<![a-z0-9_]){re.escape(normalized_phrase)}(?![a-z0-9_])",
            normalized_text,
        )
        is not None
    )


def _edit_distance_at_most_one(left: str, right: str) -> bool:
    if left == right:
        return False
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right, strict=True)) == 1
    short, long = (left, right) if len(left) < len(right) else (right, left)
    short_index = 0
    long_index = 0
    skipped = 0
    while short_index < len(short) and long_index < len(long):
        if short[short_index] == long[long_index]:
            short_index += 1
            long_index += 1
            continue
        skipped += 1
        if skipped > 1:
            return False
        long_index += 1
    return True


def _inflection_bases(token: str) -> set[str]:
    bases = {token}
    if token.endswith("ing") and len(token) > 5:
        stem = token[:-3]
        bases.add(stem)
        bases.add(stem + "e")
        if len(stem) > 2 and stem[-1] == stem[-2]:
            bases.add(stem[:-1])
    if token.endswith("ed") and len(token) > 4:
        stem = token[:-2]
        bases.update({stem, stem + "e"})
    if token.endswith("es") and len(token) > 4:
        bases.update({token[:-2], token[:-1]})
    elif token.endswith("s") and len(token) > 3:
        bases.add(token[:-1])
    return bases


def _confusable_windows(candidate_text: str, preferred: str) -> tuple[str, ...]:
    """Return only close lexical attempts, never arbitrary paraphrases.

    A window must have the preferred token count, every token must be exact,
    inflectionally related, or at edit distance one, and at least one token must
    have the edit-distance-one signal.  This catches a likely spelling
    corruption such as ``road balancing`` for ``load balancing`` without
    declaring every missing preferred term to be an error.
    """

    candidate_tokens = _ALIAS_TOKEN.findall(_normalized_text(candidate_text))
    preferred_tokens = _ALIAS_TOKEN.findall(_normalized_text(preferred))
    if not preferred_tokens or len(candidate_tokens) < len(preferred_tokens):
        return ()

    matches: list[str] = []
    width = len(preferred_tokens)
    for start in range(len(candidate_tokens) - width + 1):
        window = candidate_tokens[start : start + width]
        has_edit_signal = False
        compatible = True
        for candidate_token, preferred_token in zip(
            window, preferred_tokens, strict=True
        ):
            if candidate_token == preferred_token:
                continue
            if _inflection_bases(candidate_token) & _inflection_bases(preferred_token):
                continue
            if (
                min(len(candidate_token), len(preferred_token)) >= 4
                and _edit_distance_at_most_one(candidate_token, preferred_token)
            ):
                has_edit_signal = True
                continue
            compatible = False
            break
        if compatible and has_edit_signal:
            value = " ".join(window)
            if value not in matches:
                matches.append(value)
    return tuple(matches)


class AliasDetector:
    """Detect evidence-backed lexical alias errors without mutating runtime state."""

    def __init__(self, entries: Sequence[GlossaryEntry | AliasEntry]) -> None:
        aliases = tuple(
            entry
            if isinstance(entry, AliasEntry)
            else AliasEntry.from_glossary(entry)
            for entry in entries
        )
        identifiers = [entry.alias_id for entry in aliases]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("alias IDs must be unique")
        for entry in aliases:
            accepted = {
                _normalized_text(value)
                for value in (entry.preferred, *entry.accepted_variants)
            }
            disallowed = {
                _normalized_text(value) for value in entry.disallowed_variants
            }
            if accepted & disallowed:
                raise ValueError(
                    f"alias {entry.alias_id} has overlapping accepted/disallowed forms"
                )
        self.entries = aliases

    def detect(
        self,
        *,
        source_text: str,
        candidate_text: str,
        domain: str | None,
        coverage_requirements: Sequence[SourceCoverageRequirement],
    ) -> list[AliasFinding]:
        requirements = {
            _normalized_text(requirement.source_term): requirement
            for requirement in coverage_requirements
        }
        findings: list[AliasFinding] = []
        for entry in self.entries:
            if not contains_source_term(source_text, entry.source):
                continue
            if (
                domain is None
                or entry.domain is None
                or _normalized_text(domain) != _normalized_text(entry.domain)
            ):
                continue

            if _normalized_text(entry.source) not in requirements:
                findings.append(
                    AliasFinding(
                        alias=entry,
                        status=AliasDetectionStatus.UNVERIFIABLE,
                        evidence_kind=AliasEvidenceKind.MISSING_COVERAGE_EVIDENCE,
                        matched_variants=(),
                        reason=(
                            "The source/domain match, but source coverage did not "
                            "independently extract this term."
                        ),
                    )
                )
                continue
            if entry.provenance is None or not entry.provenance.strip():
                findings.append(
                    AliasFinding(
                        alias=entry,
                        status=AliasDetectionStatus.UNVERIFIABLE,
                        evidence_kind=AliasEvidenceKind.MISSING_PROVENANCE,
                        matched_variants=(),
                        reason="The alias has no independent provenance.",
                    )
                )
                continue

            accepted_forms = _unique_normalized(
                (entry.preferred, *entry.accepted_variants)
            )
            accepted_matches = tuple(
                form for form in accepted_forms if _contains_phrase(candidate_text, form)
            )
            disallowed_matches = tuple(
                form
                for form in _unique_normalized(entry.disallowed_variants)
                if _contains_phrase(candidate_text, form)
            )
            confusable_matches = (
                _confusable_windows(candidate_text, entry.preferred)
                if entry.confusable_policy == CONFUSABLE_POLICY
                else ()
            )

            if accepted_matches and (disallowed_matches or confusable_matches):
                findings.append(
                    AliasFinding(
                        alias=entry,
                        status=AliasDetectionStatus.UNVERIFIABLE,
                        evidence_kind=AliasEvidenceKind.CONFLICTING_VARIANTS,
                        matched_variants=_unique_normalized(
                            (*accepted_matches, *disallowed_matches, *confusable_matches)
                        ),
                        reason=(
                            "Accepted and non-accepted alias evidence both occur; "
                            "the detector will not infer which mention is translated."
                        ),
                    )
                )
            elif accepted_matches:
                findings.append(
                    AliasFinding(
                        alias=entry,
                        status=AliasDetectionStatus.VERIFIED,
                        evidence_kind=AliasEvidenceKind.ACCEPTED_VARIANT,
                        matched_variants=accepted_matches,
                        reason="An accepted alias form is present as a whole phrase.",
                    )
                )
            elif disallowed_matches:
                findings.append(
                    AliasFinding(
                        alias=entry,
                        status=AliasDetectionStatus.ERROR,
                        evidence_kind=AliasEvidenceKind.EXPLICIT_DISALLOWED_VARIANT,
                        matched_variants=disallowed_matches,
                        reason=(
                            "An independently versioned glossary explicitly marks "
                            "the matched form as disallowed."
                        ),
                    )
                )
            elif confusable_matches:
                findings.append(
                    AliasFinding(
                        alias=entry,
                        status=AliasDetectionStatus.ERROR,
                        evidence_kind=AliasEvidenceKind.CONFUSABLE_VARIANT,
                        matched_variants=confusable_matches,
                        reason=(
                            "This provenance-backed alias explicitly opts into the "
                            "single-edit confusable policy, and the matched close "
                            "lexical attempt is not an accepted variant."
                        ),
                    )
                )
            else:
                findings.append(
                    AliasFinding(
                        alias=entry,
                        status=AliasDetectionStatus.UNVERIFIABLE,
                        evidence_kind=AliasEvidenceKind.NO_TARGET_EVIDENCE,
                        matched_variants=(),
                        reason=(
                            "Neither an accepted alias nor positive evidence of a "
                            "specific wrong alias is present."
                        ),
                    )
                )
        return findings


__all__ = [
    "AliasDetectionStatus",
    "AliasDetector",
    "AliasEntry",
    "AliasEvidenceKind",
    "AliasFinding",
    "CONFUSABLE_POLICY",
]
