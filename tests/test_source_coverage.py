from __future__ import annotations

import pytest

from translation_qa.coverage import (
    apply_coverage_to_judgment,
    check_source_coverage,
    constraints_for_retry,
    extract_source_coverage_requirements,
    merge_coverage_requirements,
    merge_retrieval_hits,
)
from translation_qa.schemas import (
    CoverageRecoveryAction,
    NextAction,
    QualityJudgment,
    RetrievalHit,
    RetrievalMatchType,
    SourceAnalysis,
    SourceCoverageExtractionRule,
    SourceCoverageKind,
    SourceCoverageStatus,
    TranslationErrorType,
)


def _hit(
    *,
    term_id: str = "docker",
    source_term: str = "도커",
    target_term: str = "Docker",
    score: float = 1.0,
    match_type: RetrievalMatchType = RetrievalMatchType.EXACT,
) -> RetrievalHit:
    return RetrievalHit(
        term_id=term_id,
        source_term=source_term,
        target_term=target_term,
        match_type=match_type,
        score=score,
    )


def _pass_judgment() -> QualityJudgment:
    return QualityJudgment(
        passed=True,
        quality_score=0.9,
        error_types=[],
        summary="The candidate is acceptable.",
        confidence=0.8,
        next_action=NextAction.ACCEPT,
    )


def test_extraction_is_source_anchored_and_merges_evidence() -> None:
    requirements = extract_source_coverage_requirements(
        "Node.js는 C++ 서버 배포를 자동화합니다.",
        SourceAnalysis(
            domain="software",
            key_terms=["Node.js", "서버", "원문에 없는 용어"],
            confidence=0.9,
        ),
        [
            _hit(
                term_id="server",
                source_term="서버",
                target_term="server",
            )
        ],
    )

    by_term = {item.source_term: item for item in requirements}
    assert "원문에 없는 용어" not in by_term
    assert by_term["Node.js"].kinds == [
        SourceCoverageKind.STANDALONE_SUBJECT,
        SourceCoverageKind.PROPER_NAME,
    ]
    assert by_term["Node.js"].accepted_targets == ["Node.js"]
    assert by_term["Node.js"].extraction_rules == [
        SourceCoverageExtractionRule.LEADING_TOPIC_SUBJECT,
        SourceCoverageExtractionRule.LATIN_IDENTIFIER,
    ]
    assert by_term["C++"].kinds == [SourceCoverageKind.PROPER_NAME]
    assert by_term["C++"].accepted_targets == ["C++"]
    assert by_term["서버"].accepted_targets == ["server"]
    assert by_term["서버"].evidence_hit_ids == ["server"]


def test_unknown_standalone_subject_blocks_only_on_strong_pronoun_signal() -> None:
    requirements = extract_source_coverage_requirements(
        "도커는 컨테이너 플랫폼입니다.",
        SourceAnalysis(domain="software", key_terms=["도커"], confidence=0.9),
    )

    rag_missing = check_source_coverage(
        "It's a container platform.", requirements, targeted_rag_available=True
    )
    agent_missing = check_source_coverage(
        "It's a container platform.", requirements, targeted_rag_available=False
    )
    lexical_candidate = check_source_coverage(
        "Docker is a container platform.",
        requirements,
        targeted_rag_available=True,
    )

    subject_rag = next(
        item
        for item in rag_missing
        if SourceCoverageKind.STANDALONE_SUBJECT in item.requirement.kinds
    )
    subject_agent = next(
        item
        for item in agent_missing
        if SourceCoverageKind.STANDALONE_SUBJECT in item.requirement.kinds
    )
    subject_lexical = next(
        item
        for item in lexical_candidate
        if SourceCoverageKind.STANDALONE_SUBJECT in item.requirement.kinds
    )
    assert subject_rag.status is SourceCoverageStatus.MISSING
    assert subject_rag.recovery_action is CoverageRecoveryAction.TARGETED_RAG
    assert subject_agent.recovery_action is CoverageRecoveryAction.MUST_PRESERVE
    assert subject_lexical.status is SourceCoverageStatus.UNRESOLVED


def test_pronoun_guard_avoids_it_acronym_demonstrative_determiner_and_generic_topic() -> None:
    it_requirements = extract_source_coverage_requirements(
        "IT 시스템은 서버를 관리합니다.",
        SourceAnalysis(
            domain="software", key_terms=["IT 시스템", "서버"], confidence=0.9
        ),
    )
    docker_requirements = extract_source_coverage_requirements(
        "도커는 애플리케이션을 실행합니다.",
        SourceAnalysis(domain="software", key_terms=["도커", "애플리케이션"], confidence=0.9),
    )
    generic_requirements = extract_source_coverage_requirements(
        "문제는 복잡합니다.",
        SourceAnalysis(domain="general", key_terms=["문제"], confidence=0.9),
    )

    it_findings = check_source_coverage(
        "IT systems manage servers.",
        it_requirements,
        targeted_rag_available=True,
    )
    lost_acronym_findings = check_source_coverage(
        "It is responsible for managing servers.",
        it_requirements,
        targeted_rag_available=True,
    )
    demonstrative_findings = check_source_coverage(
        "This application runs in containers.",
        docker_requirements,
        targeted_rag_available=True,
    )
    generic_findings = check_source_coverage(
        "It is a complex problem.",
        generic_requirements,
        targeted_rag_available=True,
    )

    assert not any(item.status is SourceCoverageStatus.MISSING for item in it_findings)
    assert any(
        item.status is SourceCoverageStatus.MISSING
        for item in lost_acronym_findings
    )
    assert not any(
        item.status is SourceCoverageStatus.MISSING
        for item in demonstrative_findings
    )
    assert not any(
        item.status is SourceCoverageStatus.MISSING for item in generic_findings
    )


def test_known_target_uses_whole_token_coverage_and_must_preserve() -> None:
    requirements = extract_source_coverage_requirements(
        "도커는 컨테이너 플랫폼입니다.",
        SourceAnalysis(domain="software", key_terms=["도커"], confidence=0.9),
        [_hit()],
    )

    covered = check_source_coverage(
        "Docker is a platform.", requirements, targeted_rag_available=True
    )
    missing = check_source_coverage(
        "Dockers is a platform.", requirements, targeted_rag_available=True
    )
    misplaced_subject = check_source_coverage(
        "It uses Docker images.", requirements, targeted_rag_available=True
    )

    assert all(item.status is SourceCoverageStatus.COVERED for item in covered)
    assert all(item.status is SourceCoverageStatus.MISSING for item in missing)
    assert all(
        item.recovery_action is CoverageRecoveryAction.MUST_PRESERVE
        for item in missing
    )
    assert all(
        item.status is SourceCoverageStatus.MISSING
        for item in misplaced_subject
    )


def test_coverage_errors_override_llm_pass_and_drive_retry_constraint() -> None:
    requirements = extract_source_coverage_requirements(
        "도커는 컨테이너 플랫폼입니다.",
        SourceAnalysis(domain="software", key_terms=["도커"], confidence=0.9),
    )
    findings = check_source_coverage(
        "It's a container platform.",
        requirements,
        targeted_rag_available=False,
    )

    judgment = apply_coverage_to_judgment(_pass_judgment(), findings)
    retry_constraints = constraints_for_retry(
        findings, requirements, judgment.must_preserve_constraints
    )

    assert not judgment.passed
    assert judgment.error_types == [TranslationErrorType.OMISSION_ADDITION]
    assert judgment.next_action is NextAction.RETRY_WITH_CONSTRAINTS
    assert judgment.suggested_query_terms[0] == "도커"
    assert judgment.summary.startswith("The candidate is acceptable.")
    assert "Required source coverage is missing" in judgment.summary
    assert [item.source_term for item in retry_constraints] == ["도커"]


def test_proper_name_and_technical_omissions_map_to_structured_errors() -> None:
    proper_requirements = extract_source_coverage_requirements(
        "Node.js는 서버 런타임입니다."
    )
    proper_findings = check_source_coverage(
        "It's a server runtime.",
        proper_requirements,
        targeted_rag_available=False,
    )
    proper_judgment = apply_coverage_to_judgment(
        _pass_judgment(), proper_findings
    )

    technical_requirements = extract_source_coverage_requirements(
        "서비스는 서버를 사용합니다.",
        SourceAnalysis(domain="software", key_terms=["서비스", "서버"], confidence=0.9),
        [_hit(term_id="server", source_term="서버", target_term="server")],
    )
    technical_findings = check_source_coverage(
        "The service uses a host.",
        technical_requirements,
        targeted_rag_available=True,
    )
    technical_judgment = apply_coverage_to_judgment(
        _pass_judgment(), technical_findings
    )

    assert proper_judgment.error_types == [
        TranslationErrorType.OMISSION_ADDITION,
        TranslationErrorType.ENTITY_VALUE,
    ]
    assert technical_judgment.error_types == [TranslationErrorType.TERM]
    assert technical_judgment.next_action is NextAction.RETRY_WITH_CONSTRAINTS


def test_targeted_rag_hit_enriches_retry_must_preserve_constraint() -> None:
    initial = extract_source_coverage_requirements(
        "도커는 컨테이너 플랫폼입니다.",
        SourceAnalysis(domain="software", key_terms=["도커"], confidence=0.9),
    )
    findings = check_source_coverage(
        "It's a container platform.", initial, targeted_rag_available=True
    )
    enriched = merge_coverage_requirements(
        initial,
        extract_source_coverage_requirements(
            "도커는 컨테이너 플랫폼입니다.",
            SourceAnalysis(domain="software", key_terms=["도커"], confidence=0.9),
            [_hit()],
        ),
    )

    constraints = constraints_for_retry(findings, enriched)

    assert len(constraints) == 1
    assert constraints[0].source_term == "도커"
    assert constraints[0].accepted_targets == ["Docker"]


def test_retrieval_merge_is_stable_deduplicated_and_rejects_identity_change() -> None:
    old = _hit(
        term_id="server",
        source_term="서버",
        target_term="server",
        score=0.8,
        match_type=RetrievalMatchType.VECTOR,
    )
    refreshed = old.model_copy(
        update={"score": 1.0, "match_type": RetrievalMatchType.EXACT}
    )
    docker = _hit()

    merged = merge_retrieval_hits([old], [refreshed, docker, docker])

    assert [item.term_id for item in merged] == ["server", "docker"]
    assert merged[0] == refreshed
    with pytest.raises(ValueError, match="changed immutable identity metadata"):
        merge_retrieval_hits(
            [old],
            [old.model_copy(update={"target_term": "host"})],
        )


def test_requirement_target_merge_is_normalized_and_revalidated() -> None:
    source = "서비스는 서버를 사용합니다."
    analysis = SourceAnalysis(
        domain="software", key_terms=["서비스", "서버"], confidence=0.9
    )
    requirements = extract_source_coverage_requirements(
        source,
        analysis,
        [
            _hit(term_id="server-a", source_term="서버", target_term="server"),
            _hit(term_id="server-b", source_term="서버", target_term="Server"),
        ],
    )

    server = next(item for item in requirements if item.source_term == "서버")
    assert server.accepted_targets == ["server"]
    assert server.evidence_hit_ids == ["server-a", "server-b"]
