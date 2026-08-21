from __future__ import annotations

from collections.abc import Sequence

from translation_qa.postedit import GlossaryPostEditingTranslator
from translation_qa.schemas import TermConstraint, TranslationCandidate


class StaticTranslator:
    model_id = "base-nmt"

    def __init__(self, text: str) -> None:
        self.text = text
        self.constraints_seen: list[Sequence[TermConstraint] | None] = []

    def translate(
        self,
        source_text: str,
        constraints: Sequence[TermConstraint] | None = None,
    ) -> TranslationCandidate:
        self.constraints_seen.append(constraints)
        return TranslationCandidate(text=self.text, model_id=self.model_id)


def deployment_constraint() -> TermConstraint:
    return TermConstraint(
        source_term="배포",
        target_term="deployment",
        target_variants=["deploy", "deploys"],
        retrieval_hit_id="sw-deployment",
        replacement_rules={"distributes": "deploys"},
    )


def test_posteditor_replaces_documented_whole_phrase_and_preserves_case() -> None:
    base = StaticTranslator("Development team distributes a new version.")
    translator = GlossaryPostEditingTranslator(base)

    result = translator.translate("개발팀이 새 버전을 배포한다.", [deployment_constraint()])

    assert result.text == "Development team deploys a new version."
    assert result.model_id == "base-nmt+glossary-postedit"
    assert base.constraints_seen == [None]


def test_posteditor_does_not_change_an_already_accepted_target() -> None:
    base = StaticTranslator("The team deploys a new version.")
    translator = GlossaryPostEditingTranslator(base)

    result = translator.translate("새 버전을 배포한다.", [deployment_constraint()])

    assert result.text == "The team deploys a new version."


def test_posteditor_does_not_replace_inside_a_larger_word() -> None:
    base = StaticTranslator("The proxy redistributes traffic.")
    translator = GlossaryPostEditingTranslator(base)

    result = translator.translate("새 버전을 배포한다.", [deployment_constraint()])

    assert result.text == "The proxy redistributes traffic."


def test_posteditor_applies_multiple_retrieved_term_rules() -> None:
    base = StaticTranslator("Cachet and load dispersal reduce latency.")
    translator = GlossaryPostEditingTranslator(base)
    constraints = [
        TermConstraint(
            source_term="캐시",
            target_term="cache",
            target_variants=["caching"],
            retrieval_hit_id="sw-cache",
            replacement_rules={"cachet": "cache"},
        ),
        TermConstraint(
            source_term="부하 분산",
            target_term="load balancing",
            retrieval_hit_id="sw-load-balancing",
            replacement_rules={"load dispersal": "load balancing"},
        ),
    ]

    result = translator.translate("캐시와 부하 분산을 사용한다.", constraints)

    assert result.text == "Cache and load balancing reduce latency."
