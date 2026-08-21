from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.compare_datasets import (
    analyze_dataset,
    build_provisional_recommendation,
    main,
)


def _analysis_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ko": pd.Series(
                [
                    "서버에 새 버전을 배포한다.",
                    "서버에 새 버전을 배포한다.",
                    "캐시를 삭제한다.",
                    None,
                    "12345",
                ],
                dtype="string",
            ),
            "en": pd.Series(
                [
                    "Deploy the new version to the server.",
                    "Deploy the new version to the server.",
                    "Clear the cache.",
                    "Missing source.",
                    "12345",
                ],
                dtype="string",
            ),
            "split": ["train"] * 5,
            "source_row": range(5),
        }
    )


def test_analyze_dataset_reports_counts_keywords_and_heuristic_warning() -> None:
    summary, enriched = analyze_dataset(
        "fixture",
        _analysis_frame(),
        {
            "source_files": ["fixture.csv"],
            "schema_by_split": {"train": {"ko": "string", "en": "string"}},
            "rows_by_split": {"train": 5},
            "source_columns": {"korean": "ko", "english": "en"},
        },
        example_limit=3,
    )

    assert summary["row_count"] == 5
    assert summary["null_and_blank"]["korean_null"] == 1
    assert summary["duplicates_beyond_first"]["parallel_pair"] == 1
    screen = summary["software_keyword_screen"]
    assert screen["hit_counts"]["서버"] == 2
    assert screen["hit_counts"]["배포"] == 2
    assert screen["hit_counts"]["캐시"] == 1
    assert screen["rows_with_multiple_keywords"] == 2
    assert screen["screened_candidate_count"] == 2
    assert all(
        item["review_status"] == "UNREVIEWED_HEURISTIC_CANDIDATE"
        for item in screen["candidate_examples"]
    )
    assert "do not prove" in summary["heuristic_anomalies"]["warning"]
    assert int(enriched["screened_candidate"].sum()) == 2


def test_recommendation_is_provisional_and_requires_manual_review() -> None:
    strong = {
        "software_keyword_screen": {
            "distinct_keywords_with_hits": 8,
            "screened_candidate_count": 70,
        },
        "duplicates_beyond_first": {"parallel_pair_rate_pct": 1.0},
        "heuristic_anomalies": {"rows_with_any_flag_rate_pct": 2.0},
    }
    weak = {
        "software_keyword_screen": {
            "distinct_keywords_with_hits": 4,
            "screened_candidate_count": 200,
        },
        "duplicates_beyond_first": {"parallel_pair_rate_pct": 0.0},
        "heuristic_anomalies": {"rows_with_any_flag_rate_pct": 0.0},
    }

    result = build_provisional_recommendation({"strong": strong, "weak": weak})

    assert result["provisional_dataset"] == "strong"
    assert result["selection_status"] == "PROVISIONAL_FOR_MANUAL_REVIEW"
    assert result["not_a_quality_verdict"] is True
    assert result["mandatory_hold_points"]


def _write_fixture_datasets(root: Path) -> tuple[Path, Path]:
    moo_dir = root / "moo"
    moo_dir.mkdir()
    rows = {
        "train": [("서버 버전을 배포한다.", "Deploy the server version.")],
        "dev": [("캐시를 비운다.", "Clear the cache.")],
        "test": [("오늘 회의가 열린다.", "The meeting is today.")],
    }
    for split, pairs in rows.items():
        pd.DataFrame(pairs, columns=["ko", "en"]).to_csv(
            moo_dir / f"{split}.csv", index=False
        )

    lemon_path = root / "lemon.parquet"
    pd.DataFrame(
        {
            "korean": ["데이터베이스 서버를 점검한다.", "네트워크 장애를 복구한다."],
            "english": [
                "Inspect the database server.",
                "Recover from the network failure.",
            ],
        }
    ).to_parquet(lemon_path, index=False)
    return moo_dir, lemon_path


def test_cli_reads_explicit_offline_fixtures_and_writes_json_and_png(
    tmp_path: Path,
) -> None:
    moo_dir, lemon_path = _write_fixture_datasets(tmp_path)
    output_dir = tmp_path / "eda"

    exit_code = main(
        [
            "--moo-dir",
            str(moo_dir),
            "--lemon-parquet",
            str(lemon_path),
            "--output-dir",
            str(output_dir),
            "--example-limit",
            "2",
        ]
    )

    assert exit_code == 0
    summary_path = output_dir / "dataset_comparison.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["analysis_scope"] == "offline cached snapshots only"
    assert payload["datasets"]["moo"]["row_count"] == 3
    assert payload["datasets"]["lemon_mint"]["row_count"] == 2
    assert payload["provisional_selection"]["not_a_quality_verdict"] is True

    for filename in ("length_distributions.png", "software_keyword_hits.png"):
        image = output_dir / filename
        assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert image.stat().st_size > 1_000
