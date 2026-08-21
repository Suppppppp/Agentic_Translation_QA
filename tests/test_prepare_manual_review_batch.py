from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.prepare_manual_review_batch import (
    LEGACY_MANUAL_COLUMNS,
    MANUAL_COLUMNS,
    prepare_batch,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(tmp_path: Path) -> Path:
    reference_review_hash = "review-hash"
    source_feedback_hash = "feedback-hash"
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps(
                {
                    "case_id": case_id,
                    "source_text": f"source {case_id}",
                    "reference_text": f"reference {case_id}",
                    "reference_provenance": {
                        "reference_review_sha256": reference_review_hash,
                        "source_feedback_sha256": source_feedback_hash,
                    },
                }
            )
            for case_id in ("case-a", "case-b")
        )
        + "\n",
        encoding="utf-8",
    )
    results = []
    for case_id, mode, passed, retries in (
        ("case-a", "agent", False, 1),
        ("case-a", "agent_rag", True, 0),
        ("case-b", "agent", True, 0),
        ("case-b", "agent_rag", None, 0),
    ):
        attempts = [
            {
                "candidate": {"text": f"initial {case_id} {mode}"},
                "judgment": (
                    None
                    if passed is None
                    else {
                        "passed": passed,
                        "error_types": ["meaning"] if not passed else [],
                        "summary": f"summary {case_id} {mode}",
                    }
                ),
            }
        ]
        if retries:
            attempts.append({"candidate": {"text": "revised"}, "judgment": None})
        results.append(
            {
                "case_id": case_id,
                "mode": mode,
                "response": {
                    "translation": f"final {case_id} {mode}",
                    "trace": {
                        "attempts": attempts,
                        "stop_reason": "component_failure" if passed is None else "passed",
                    },
                },
            }
        )
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "dataset_id": "evaluation_v1",
                "dataset_sha256": _sha256(dataset),
                "config_sha256": "config-hash",
                "run_config": {
                    "dataset_id": "evaluation_v1",
                    "dataset_sha256": _sha256(dataset),
                },
                "results": results,
            }
        ),
        encoding="utf-8",
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "batch_id": "batch-1",
                "partial_representative_sample": True,
                "artifact_file": "artifact.json",
                "artifact_sha256": _sha256(artifact),
                "artifact_config_sha256": "config-hash",
                "run_id": "run-1",
                "dataset_id": "evaluation_v1",
                "dataset_file": "dataset.jsonl",
                "dataset_sha256": _sha256(dataset),
                "reference_review_sha256": reference_review_hash,
                "source_feedback_sha256": source_feedback_hash,
                "selected": [
                    {"case_id": "case-b", "mode": "agent_rag", "selection_reason": "failure trace"},
                    {"case_id": "case-a", "mode": "agent", "selection_reason": "retry trace"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return selection


def test_prepare_batch_preserves_manifest_order_and_leaves_labels_blank(
    tmp_path: Path,
) -> None:
    selection = _write_fixture(tmp_path)
    output = tmp_path / "review.csv"

    assert prepare_batch(selection, output, tmp_path) == 2

    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["review_key"] for row in rows] == [
        "case-b::agent_rag",
        "case-a::agent",
    ]
    assert all(
        all(row[column] == "" for column in LEGACY_MANUAL_COLUMNS) for row in rows
    )
    assert all("manual_severity" not in row for row in rows)
    assert rows[0]["agent_initial_passed"] == ""
    assert rows[0]["agent_initial_error_types"] == "[]"
    assert rows[0]["stop_reason"] == "component_failure"
    assert rows[1]["agent_initial_passed"] == "False"
    assert rows[1]["agent_initial_error_types"] == '["meaning"]'
    assert rows[1]["retry_count"] == "1"


def test_prepare_v2_batch_adds_blank_severity_without_inference(
    tmp_path: Path,
) -> None:
    selection_path = _write_fixture(tmp_path)
    selection = json.loads(selection_path.read_text())
    selection["manual_review_schema_version"] = 2
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    output = tmp_path / "review.csv"

    assert prepare_batch(selection_path, output, tmp_path) == 2

    with output.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        assert reader.fieldnames is not None
        severity_index = reader.fieldnames.index("manual_severity")
        assert reader.fieldnames[severity_index - 1] == (
            "manual_initial_needs_revision"
        )
    assert all(all(row[column] == "" for column in MANUAL_COLUMNS) for row in rows)
    assert rows[0]["agent_initial_passed"] == ""
    assert rows[1]["agent_initial_passed"] == "False"


@pytest.mark.parametrize("version", [0, 3, True, "2"])
def test_prepare_batch_rejects_invalid_manual_review_schema_version(
    tmp_path: Path, version: object
) -> None:
    selection_path = _write_fixture(tmp_path)
    selection = json.loads(selection_path.read_text())
    selection["manual_review_schema_version"] = version
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    output = tmp_path / "review.csv"

    with pytest.raises(ValueError, match="manual_review_schema_version"):
        prepare_batch(selection_path, output, tmp_path)

    assert not output.exists()


def test_prepare_batch_rejects_hash_mismatch(tmp_path: Path) -> None:
    selection_path = _write_fixture(tmp_path)
    selection = json.loads(selection_path.read_text())
    selection["artifact_sha256"] = "0" * 64
    selection_path.write_text(json.dumps(selection), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact SHA-256"):
        prepare_batch(selection_path, tmp_path / "review.csv", tmp_path)


def test_prepare_batch_rejects_duplicate_pair(tmp_path: Path) -> None:
    selection_path = _write_fixture(tmp_path)
    selection = json.loads(selection_path.read_text())
    selection["selected"].append(dict(selection["selected"][0]))
    selection_path.write_text(json.dumps(selection), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate selected pair"):
        prepare_batch(selection_path, tmp_path / "review.csv", tmp_path)


def test_prepare_batch_rejects_run_or_dataset_mismatch(tmp_path: Path) -> None:
    selection_path = _write_fixture(tmp_path)
    selection = json.loads(selection_path.read_text())
    selection["run_id"] = "wrong-run"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")

    with pytest.raises(ValueError, match="run_id"):
        prepare_batch(selection_path, tmp_path / "review.csv", tmp_path)


def test_prepare_batch_refuses_to_overwrite_labels(tmp_path: Path) -> None:
    selection = _write_fixture(tmp_path)
    output = tmp_path / "review.csv"
    output.write_text("human labels", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_batch(selection, output, tmp_path)
    assert output.read_text(encoding="utf-8") == "human labels"
