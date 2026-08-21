from __future__ import annotations

from collections.abc import Sequence

import pytest

from translation_qa.retrieval import (
    ExactFirstHybridGlossaryRetriever,
    ExactGlossaryRetriever,
    GlossaryEntry,
    HybridGlossaryRetriever,
    VectorGlossaryRetriever,
    contains_source_term,
    load_glossary_csv,
)
from translation_qa.schemas import RetrievalMatchType, RetrievalQuery


class KeywordEmbedder:
    """Deterministic semantic stand-in that needs no ML package."""

    keywords = ("면역", "보험", "세포")

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [
            [float(keyword in text) for keyword in self.keywords]
            for text in texts
        ]


def test_load_glossary_csv_supports_documented_columns_and_json(tmp_path) -> None:
    path = tmp_path / "glossary.csv"
    path.write_text(
        "glossary_version,term_id,domain,source_term_ko,preferred_target_en,"
        "accepted_variants_json,disallowed_variants_json,replacement_rules_json,"
        "definition,source,notes\n"
        'v1,t-1,oncology,면역항암제,immunotherapy drug,'
        '"[""immuno-oncology drug"", ""immunotherapy agent""]",'
        '"[""immune medicine""]","{""immune drug"": ""immunotherapy drug""}",'
        '면역 반응을 이용하는 치료제,WHO,독립 출처\n',
        encoding="utf-8",
    )

    entries = load_glossary_csv(path)

    assert entries == [
        GlossaryEntry(
            term_id="t-1",
            source_term="면역항암제",
            target_term="immunotherapy drug",
            domain="oncology",
            definition="면역 반응을 이용하는 치료제",
            accepted_variants=("immuno-oncology drug", "immunotherapy agent"),
            disallowed_variants=("immune medicine",),
            replacement_rules=(("immune drug", "immunotherapy drug"),),
            source="WHO",
            glossary_version="v1",
            notes="독립 출처",
        )
    ]


def test_exact_retriever_uses_source_key_terms_domain_and_top_k() -> None:
    retriever = ExactGlossaryRetriever(
        [
            GlossaryEntry("long", "면역항암제", "immunotherapy drug", "oncology"),
            GlossaryEntry("short", "항암제", "anticancer drug", "oncology"),
            GlossaryEntry("wrong-domain", "면역항암제", "insurance product", "finance"),
            GlossaryEntry("global", "바이오마커", "biomarker"),
        ]
    )

    hits = retriever.retrieve(
        RetrievalQuery(
            source_text="환자에게 면역항암제를 투여했다.",
            key_terms=["바이오마커"],
            domain="oncology",
            top_k=2,
        )
    )

    assert [hit.term_id for hit in hits] == ["long", "global"]
    assert all(hit.match_type is RetrievalMatchType.EXACT for hit in hits)
    assert "short" not in {hit.term_id for hit in hits}
    assert "wrong-domain" not in {hit.term_id for hit in hits}


def test_source_term_boundary_allows_particles_but_rejects_compounds() -> None:
    assert contains_source_term("새 버전을 배포한다.", "배포")
    assert contains_source_term("배포를 시작했다.", "배포")
    assert not contains_source_term("우분투는 리눅스 배포판이다.", "배포")
    assert not contains_source_term("면역항암제를 투여했다.", "항암제")


def test_vector_retriever_ranks_semantic_match_and_honours_domain() -> None:
    retriever = VectorGlossaryRetriever(
        [
            GlossaryEntry("oncology", "면역항암제", "immunotherapy drug", "oncology"),
            GlossaryEntry("finance", "면역 보험", "immune insurance", "finance"),
            GlossaryEntry("cell", "세포독성", "cytotoxicity", "oncology"),
        ],
        KeywordEmbedder(),
        min_score=0.5,
    )

    hits = retriever.retrieve(
        RetrievalQuery(
            source_text="면역 반응을 활성화하는 치료를 시작했다.",
            domain="oncology",
            top_k=1,
        )
    )

    assert [hit.term_id for hit in hits] == ["oncology"]
    assert hits[0].match_type is RetrievalMatchType.VECTOR
    assert hits[0].score == pytest.approx(1.0)


def test_vector_retriever_returns_no_match_below_threshold() -> None:
    retriever = VectorGlossaryRetriever(
        [GlossaryEntry("cell", "세포독성", "cytotoxicity", "oncology")],
        KeywordEmbedder(),
        min_score=0.5,
    )

    hits = retriever.retrieve(
        RetrievalQuery(source_text="오늘 날씨가 맑다.", domain="oncology")
    )

    assert hits == []


def test_hybrid_retriever_deduplicates_and_marks_shared_hit() -> None:
    entries = [
        GlossaryEntry("exact", "면역항암제", "immunotherapy drug", "oncology"),
        GlossaryEntry(
            "semantic",
            "면역치료",
            "immunotherapy",
            "oncology",
            definition="면역 반응을 활용하는 치료",
        ),
    ]
    exact = ExactGlossaryRetriever(entries)
    vector = VectorGlossaryRetriever(entries, KeywordEmbedder(), min_score=0.5)
    hybrid = HybridGlossaryRetriever(exact, vector)

    hits = hybrid.retrieve(
        RetrievalQuery(
            source_text="면역항암제를 처방했다.",
            domain="oncology",
            top_k=2,
        )
    )

    assert [hit.term_id for hit in hits].count("exact") == 1
    shared = next(hit for hit in hits if hit.term_id == "exact")
    assert shared.match_type is RetrievalMatchType.HYBRID
    assert hits[0].term_id == "exact"


def test_exact_first_hybrid_suppresses_vector_only_distractors() -> None:
    entries = [
        GlossaryEntry("exact", "면역항암제", "immunotherapy drug", "oncology"),
        GlossaryEntry(
            "distractor",
            "면역치료",
            "immunotherapy",
            "oncology",
            definition="면역 반응을 활용하는 치료",
        ),
    ]
    retriever = ExactFirstHybridGlossaryRetriever(
        ExactGlossaryRetriever(entries),
        VectorGlossaryRetriever(entries, KeywordEmbedder(), min_score=0.5),
    )

    hits = retriever.retrieve(
        RetrievalQuery(
            source_text="면역항암제를 처방했다.",
            domain="oncology",
            top_k=3,
        )
    )

    assert [hit.term_id for hit in hits] == ["exact"]
    assert hits[0].match_type is RetrievalMatchType.HYBRID


def test_exact_first_hybrid_limits_semantic_fallbacks() -> None:
    entries = [
        GlossaryEntry("a", "면역항암제", "immunotherapy drug", "oncology"),
        GlossaryEntry("b", "면역치료", "immunotherapy", "oncology"),
    ]
    retriever = ExactFirstHybridGlossaryRetriever(
        ExactGlossaryRetriever(entries),
        VectorGlossaryRetriever(entries, KeywordEmbedder(), min_score=0.5),
        max_vector_fallbacks=1,
    )

    hits = retriever.retrieve(
        RetrievalQuery(
            source_text="면역 반응을 활성화한다.",
            domain="oncology",
            top_k=3,
        )
    )

    assert len(hits) == 1
    assert hits[0].match_type is RetrievalMatchType.VECTOR


def test_exact_retriever_returns_no_match() -> None:
    retriever = ExactGlossaryRetriever(
        [GlossaryEntry("term", "면역항암제", "immunotherapy drug", "oncology")]
    )

    assert retriever.retrieve(RetrievalQuery(source_text="일반 문장입니다.")) == []


def test_exact_literal_match_is_not_blocked_by_agent_domain_label() -> None:
    retriever = ExactGlossaryRetriever(
        [GlossaryEntry("term", "장애 조치", "failover", "software")]
    )

    literal_hits = retriever.retrieve(
        RetrievalQuery(source_text="자동 장애 조치를 실행한다.", domain="IT")
    )
    suggested_only_hits = retriever.retrieve(
        RetrievalQuery(
            source_text="자동 전환을 실행한다.",
            key_terms=["장애 조치"],
            domain="finance",
        )
    )

    assert [hit.term_id for hit in literal_hits] == ["term"]
    assert suggested_only_hits == []
