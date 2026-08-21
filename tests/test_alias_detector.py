from __future__ import annotations

from translation_qa.alias_detector import (
    AliasDetectionStatus,
    AliasDetector,
    AliasEntry,
    AliasEvidenceKind,
    CONFUSABLE_POLICY,
)
from translation_qa.coverage import extract_source_coverage_requirements
from translation_qa.retrieval import GlossaryEntry
from translation_qa.schemas import SourceAnalysis


def _entry(
    *,
    source: str = "로드 밸런싱",
    preferred: str = "load balancing",
    accepted: tuple[str, ...] = ("load-balancing",),
    disallowed: tuple[str, ...] = (),
    domain: str = "software",
    provenance: str | None = "official bilingual documentation",
    confusable: bool = True,
) -> GlossaryEntry:
    return GlossaryEntry(
        term_id="load-balancing-alias",
        source_term=source,
        target_term=preferred,
        accepted_variants=accepted,
        disallowed_variants=disallowed,
        domain=domain,
        source=provenance,
        notes=(
            f"confusable_policy={CONFUSABLE_POLICY}" if confusable else None
        ),
    )


def _requirements(source: str = "로드 밸런싱은 트래픽을 분산합니다."):
    return extract_source_coverage_requirements(
        source,
        SourceAnalysis(
            domain="기술", key_terms=["로드 밸런싱", "트래픽"], confidence=0.9
        ),
    )


def test_nfkc_case_and_hyphen_variants_are_verified() -> None:
    detector = AliasDetector([_entry()])
    source = "로드 밸런싱은 트래픽을 분산합니다."

    spaced = detector.detect(
        source_text=source,
        candidate_text="LOAD BALANCING distributes traffic.",
        domain="software",
        coverage_requirements=_requirements(source),
    )
    hyphenated = detector.detect(
        source_text=source,
        candidate_text="Load‑Balancing distributes traffic.",
        domain="software",
        coverage_requirements=_requirements(source),
    )
    nfkc = detector.detect(
        source_text=source,
        candidate_text="ＬＯＡＤ－ＢＡＬＡＮＣＩＮＧ distributes traffic.",
        domain="Software",
        coverage_requirements=_requirements(source),
    )

    assert spaced[0].status is AliasDetectionStatus.VERIFIED
    assert hyphenated[0].status is AliasDetectionStatus.VERIFIED
    assert nfkc[0].status is AliasDetectionStatus.VERIFIED


def test_road_balance_and_road_balancing_are_confusable_errors() -> None:
    detector = AliasDetector([_entry()])
    source = "로드 밸런싱은 트래픽을 분산합니다."

    for candidate in (
        "Road balance distributes traffic.",
        "Road balancing distributes traffic.",
    ):
        finding = detector.detect(
            source_text=source,
            candidate_text=candidate,
            domain="software",
            coverage_requirements=_requirements(source),
        )[0]
        assert finding.status is AliasDetectionStatus.ERROR
        assert finding.evidence_kind is AliasEvidenceKind.CONFUSABLE_VARIANT


def test_confusable_detection_requires_explicit_alias_opt_in() -> None:
    detector = AliasDetector([_entry(confusable=False)])
    source = "로드 밸런싱은 트래픽을 분산합니다."

    finding = detector.detect(
        source_text=source,
        candidate_text="Road balancing distributes traffic.",
        domain="software",
        coverage_requirements=_requirements(source),
    )[0]

    assert finding.status is AliasDetectionStatus.UNVERIFIABLE
    assert finding.evidence_kind is AliasEvidenceKind.NO_TARGET_EVIDENCE


def test_explicit_disallowed_variant_uses_existing_glossary_evidence() -> None:
    entry = _entry(
        source="배포",
        preferred="deployment",
        accepted=("deploy",),
        disallowed=("distribution",),
    )
    source = "애플리케이션 배포를 자동화합니다."
    requirements = extract_source_coverage_requirements(
        source,
        SourceAnalysis(domain="기술", key_terms=["배포"], confidence=0.9),
    )

    finding = AliasDetector([entry]).detect(
        source_text=source,
        candidate_text="It automates application distribution.",
        domain="software",
        coverage_requirements=requirements,
    )[0]

    assert finding.status is AliasDetectionStatus.ERROR
    assert finding.evidence_kind is AliasEvidenceKind.EXPLICIT_DISALLOWED_VARIANT


def test_missing_target_or_provenance_is_unverifiable() -> None:
    source = "로드 밸런싱은 트래픽을 분산합니다."
    requirements = _requirements(source)
    missing_target = AliasDetector([_entry()]).detect(
        source_text=source,
        candidate_text="It distributes traffic.",
        domain="software",
        coverage_requirements=requirements,
    )[0]
    missing_provenance = AliasDetector([_entry(provenance=None)]).detect(
        source_text=source,
        candidate_text="Road balance distributes traffic.",
        domain="software",
        coverage_requirements=requirements,
    )[0]

    assert missing_target.status is AliasDetectionStatus.UNVERIFIABLE
    assert missing_target.evidence_kind is AliasEvidenceKind.NO_TARGET_EVIDENCE
    assert missing_provenance.status is AliasDetectionStatus.UNVERIFIABLE
    assert missing_provenance.evidence_kind is AliasEvidenceKind.MISSING_PROVENANCE

    blank_provenance = AliasDetector(
        [
            AliasEntry(
                alias_id="blank-provenance",
                source="로드 밸런싱",
                preferred="load balancing",
                accepted_variants=(),
                disallowed_variants=(),
                domain="software",
                provenance="   ",
                confusable_policy=CONFUSABLE_POLICY,
            )
        ]
    ).detect(
        source_text=source,
        candidate_text="Road balance distributes traffic.",
        domain="software",
        coverage_requirements=requirements,
    )[0]
    assert blank_provenance.status is AliasDetectionStatus.UNVERIFIABLE
    assert blank_provenance.evidence_kind is AliasEvidenceKind.MISSING_PROVENANCE


def test_source_domain_and_coverage_must_all_support_application() -> None:
    detector = AliasDetector([_entry()])
    requirements = _requirements()

    assert detector.detect(
        source_text="도로의 균형은 차량 안전에 중요합니다.",
        candidate_text="Road balance matters for vehicle safety.",
        domain="transportation",
        coverage_requirements=[],
    ) == []
    assert detector.detect(
        source_text="로드 밸런싱은 트래픽을 분산합니다.",
        candidate_text="Road balance distributes traffic.",
        domain="transportation",
        coverage_requirements=requirements,
    ) == []

    no_coverage = detector.detect(
        source_text="로드 밸런싱은 트래픽을 분산합니다.",
        candidate_text="Road balance distributes traffic.",
        domain="software",
        coverage_requirements=[],
    )[0]
    assert no_coverage.status is AliasDetectionStatus.UNVERIFIABLE
    assert no_coverage.evidence_kind is AliasEvidenceKind.MISSING_COVERAGE_EVIDENCE


def test_accepted_and_wrong_mentions_together_are_not_overclaimed() -> None:
    detector = AliasDetector([_entry()])
    finding = detector.detect(
        source_text="로드 밸런싱은 트래픽을 분산합니다.",
        candidate_text="Load balancing, not road balancing, distributes traffic.",
        domain="software",
        coverage_requirements=_requirements(),
    )[0]

    assert finding.status is AliasDetectionStatus.UNVERIFIABLE
    assert finding.evidence_kind is AliasEvidenceKind.CONFLICTING_VARIANTS
