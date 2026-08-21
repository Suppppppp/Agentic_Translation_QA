"""Deterministic glossary post-editing for the RAG-only condition."""

from __future__ import annotations

import re
from collections.abc import Sequence

from translation_qa.contracts import Translator
from translation_qa.metrics import contains_accepted_target
from translation_qa.schemas import TermConstraint, TranslationCandidate


def _whole_phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Match a phrase case-insensitively without matching inside a larger token."""

    return re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", flags=re.IGNORECASE)


def _case_aware_replacement(match: re.Match[str], replacement: str) -> str:
    if match.group(0)[:1].isupper() and replacement[:1].isalpha():
        return replacement[:1].upper() + replacement[1:]
    return replacement


class GlossaryPostEditingTranslator:
    """Wrap one NMT backend and apply only retrieved, explicit replacement rules.

    The base model is called without lexical constraints because the selected
    Marian model produced degenerate constrained-beam candidates in the pilot.
    Rules never insert a term blindly: a documented source phrase must already be
    present in the candidate, and a candidate that already contains an accepted
    target is left untouched.
    """

    def __init__(self, base_translator: Translator) -> None:
        self.base_translator = base_translator

    @property
    def model_id(self) -> str:
        return f"{self.base_translator.model_id}+glossary-postedit"

    def translate(
        self,
        source_text: str,
        constraints: Sequence[TermConstraint] | None = None,
    ) -> TranslationCandidate:
        candidate = self.base_translator.translate(source_text, constraints=None)
        if not constraints:
            return TranslationCandidate(text=candidate.text, model_id=self.model_id)

        edited = candidate.text
        for constraint in constraints:
            accepted = [constraint.target_term, *constraint.target_variants]
            if contains_accepted_target(edited, accepted):
                continue

            rules = sorted(
                constraint.replacement_rules.items(),
                key=lambda pair: len(pair[0]),
                reverse=True,
            )
            for unwanted, replacement in rules:
                pattern = _whole_phrase_pattern(unwanted)
                edited, count = pattern.subn(
                    lambda match, value=replacement: _case_aware_replacement(
                        match, value
                    ),
                    edited,
                )
                if count and contains_accepted_target(edited, accepted):
                    break

        return TranslationCandidate(text=edited, model_id=self.model_id)


__all__ = ["GlossaryPostEditingTranslator"]
