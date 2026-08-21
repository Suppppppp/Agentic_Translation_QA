from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.export_manual_review import export_review


def test_export_review_joins_reference_only_in_offline_sheet(tmp_path: Path) -> None:
    dataset = tmp_path / "evaluation.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "source_text": "원문",
                "reference_text": "Reference",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "case_id": "case-1",
                        "mode": "agent_rag",
                        "response": {
                            "translation": "Final",
                            "trace": {
                                "stop_reason": "passed",
                                "attempts": [
                                    {
                                        "candidate": {"text": "Initial"},
                                        "judgment": {
                                            "passed": False,
                                            "error_types": ["meaning"],
                                        },
                                    },
                                    {"candidate": {"text": "Final"}},
                                ],
                            },
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "review.csv"

    assert export_review(dataset, artifact, output) == 1

    with output.open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["source_text"] == "원문"
    assert row["reference_text"] == "Reference"
    assert row["initial_translation"] == "Initial"
    assert row["final_translation"] == "Final"
    assert row["retry_count"] == "1"
    assert row["manual_initial_needs_revision"] == ""
    assert row["manual_severity"] == ""
    assert row["review_status"] == ""


def test_export_review_rejects_unjoinable_artifact_case(tmp_path: Path) -> None:
    dataset = tmp_path / "evaluation.jsonl"
    dataset.write_text(
        '{"case_id":"case-1","source_text":"x","reference_text":"y"}\n',
        encoding="utf-8",
    )
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        '{"results":[{"case_id":"missing","response":{}}]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="absent from dataset"):
        export_review(dataset, artifact, tmp_path / "review.csv")
