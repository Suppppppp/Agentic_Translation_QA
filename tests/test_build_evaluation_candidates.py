from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.build_evaluation_candidates import (
    ENGLISH_REFERENCE_POLICY,
    SelectionConfig,
    attach_manual_review_references,
    load_fixed_source_lexicon,
    main,
    select_source_candidates,
)
from translation_qa.retrieval import contains_source_term


def _config(**overrides: object) -> SelectionConfig:
    values: dict[str, object] = {
        "seed": 17,
        "per_term_limit": 2,
        "min_korean_chars": 10,
        "max_korean_chars": 100,
        "max_sentence_boundaries": 1,
        "dataset_id": "fixture/korean-parallel",
        "dataset_revision": "fixture-v1",
        "shard_id": "train-00000-of-00001",
    }
    values.update(overrides)
    return SelectionConfig(**values)  # type: ignore[arg-type]


def test_korean_term_boundary_allows_inflection_but_rejects_compound() -> None:
    assert contains_source_term("새 버전을 운영 환경에 배포한다.", "배포")
    assert contains_source_term("승인된 패키지의 배포를 시작한다.", "배포")
    assert not contains_source_term("새 리눅스 배포판을 설치한다.", "배포")


def test_source_selection_is_literal_deduplicated_and_reference_independent() -> None:
    korean = pd.Series(
        [
            "새 버전을 운영 환경에 배포한다.",
            "새 버전을 운영 환경에 배포한다.",  # exact source duplicate
            "설정 변경 뒤 오래된 캐시를 즉시 삭제한다.",
            "용어가 없는 평범한 한국어 문장입니다.",
            "벡터 검색을 실행한다. 결과를 확인한다.",  # two sentence boundaries
            "캐시",  # below the configured source length
            "배포 과정에서 캐시를 함께 비운다.",
        ],
        dtype="string",
    )
    terms = ["배포", "캐시", "벡터 검색"]

    selected, metadata = select_source_candidates(korean, terms, _config())
    records_a = attach_manual_review_references(
        selected,
        pd.Series(
            [
                "Deploy a new version.",
                "Different duplicate reference.",
                "Intentionally unverified reference.",
                "This English says vector search, cache, and deployment.",
                "Run vector search. Check the results.",
                "Cache.",
                "Clear the cache during deployment.",
            ],
            dtype="string",
        ),
        metadata,
        _config(),
    )
    records_b = attach_manual_review_references(
        selected,
        pd.Series([None] * len(korean), dtype="string"),
        metadata,
        _config(),
    )

    assert [record["source_record_id"] for record in records_a] == [
        record["source_record_id"] for record in records_b
    ]
    assert {candidate.source_row for candidate in selected} == {0, 2, 6}
    assert metadata["filter_diagnostics_non_exclusive"][
        "normalized_source_duplicates_beyond_first"
    ] == 1
    assert metadata["per_term"]["벡터 검색"]["status"] == "no_eligible_candidates"
    assert all(record["review_status"] == "unreviewed" for record in records_a)
    assert all(
        record["english_reference_usage"] == ENGLISH_REFERENCE_POLICY
        for record in records_a
    )
    assert all(
        "english" not in record["source_only_selection_reason"]
        for record in records_a
    )


def test_fixed_seed_term_sampling_is_deterministic_and_reports_shortage() -> None:
    sources = pd.Series(
        [
            *(f"운영 환경에 새 버전 {index}을 배포한다." for index in range(8)),
            "설정 변경 뒤 캐시 항목을 정리한다.",
        ],
        dtype="string",
    )
    config = _config(seed=42, per_term_limit=3)

    first, first_metadata = select_source_candidates(
        sources, ["배포", "캐시"], config
    )
    second, second_metadata = select_source_candidates(
        sources, ["배포", "캐시"], config
    )
    different_seed, different_metadata = select_source_candidates(
        sources, ["배포", "캐시"], _config(seed=99, per_term_limit=3)
    )

    first_ids = [candidate.source_record_id for candidate in first]
    assert first_ids == [candidate.source_record_id for candidate in second]
    assert first_metadata["selection_hash"] == second_metadata["selection_hash"]
    assert first_metadata["selection_hash"] != different_metadata["selection_hash"]
    assert first_ids != [candidate.source_record_id for candidate in different_seed]
    assert first_metadata["per_term"]["배포"]["sampled_for_term"] == 3
    assert first_metadata["per_term"]["캐시"]["shortfall"] == 2
    assert first_metadata["per_term"]["캐시"]["status"] == "shortage"


def _write_fixture_glossary(path: Path) -> None:
    path.write_text(
        "source_term_ko,preferred_target_en\n"
        "배포,deployment\n"
        "캐시,cache\n"
        "벡터 검색,vector search\n",
        encoding="utf-8",
    )


def test_cli_writes_unreviewed_pool_and_separate_shortage_summary(
    tmp_path: Path,
) -> None:
    glossary_path = tmp_path / "glossary.csv"
    _write_fixture_glossary(glossary_path)
    terms, column = load_fixed_source_lexicon(glossary_path)
    assert terms == ["배포", "캐시", "벡터 검색"]
    assert column == "source_term_ko"

    parquet_path = tmp_path / "train-00000-of-00001.parquet"
    pd.DataFrame(
        {
            "korean": [
                "새 버전을 운영 환경에 배포한다.",
                "설정 변경 뒤 오래된 캐시를 삭제한다.",
                "일반적인 한국어 원문만 있는 문장입니다.",
            ],
            "english": [
                "Deploy a new version to production.",
                "Clear the stale cache after a configuration change.",
                "This reference alone mentions vector search and deployment.",
            ],
        }
    ).to_parquet(parquet_path, index=False)
    output_path = tmp_path / "evaluation_candidates.jsonl"
    summary_path = tmp_path / "evaluation_candidates_summary.json"

    exit_code = main(
        [
            "--parquet",
            str(parquet_path),
            "--glossary",
            str(glossary_path),
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
            "--dataset-revision",
            "fixture-revision",
            "--per-term-limit",
            "2",
            "--min-korean-chars",
            "10",
            "--max-korean-chars",
            "100",
        ]
    )

    assert exit_code == 0
    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(records) == 2
    assert all(record["review_status"] == "unreviewed" for record in records)
    assert not any("벡터 검색" in record["hit_terms"] for record in records)
    assert summary["status"] == "UNREVIEWED_REVIEW_POOL_NOT_FINAL_DATASET"
    assert summary["final_dataset_frozen"] is False
    assert summary["benchmark_run"] is False
    assert summary["selection_contract"]["reference_used_for_selection"] is False
    assert summary["selection_contract"]["reference_used_for_term_extraction"] is False
    assert summary["exact_pair_deduplication"][
        "remaining_possible_exact_pair_duplicates"
    ] == 0
    shortages = {item["term"]: item for item in summary["shortage_terms"]}
    assert shortages["벡터 검색"]["status"] == "no_eligible_candidates"
    assert summary["candidate_output"]["ordered_source_id_sha256"]
