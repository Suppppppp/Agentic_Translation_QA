#!/usr/bin/env python3
"""Replay frozen Agent judgments without calling models or translation services."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from translation_qa.judgment import canonicalize_llm_judgment
from translation_qa.schemas import JudgmentConsistencyIssue


RAW_JUDGMENT_FIELDS = (
    "passed",
    "quality_score",
    "error_types",
    "summary",
    "confidence",
    "next_action",
    "suggested_query_terms",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"path must be inside project root: {path}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"review CSV is empty: {path}")
    return rows


def _selected_keys(manifest: dict[str, Any]) -> list[str]:
    selected = manifest.get("selected")
    if not isinstance(selected, list) or not selected:
        raise ValueError("selection manifest must contain selected rows")
    keys: list[str] = []
    for item in selected:
        if not isinstance(item, dict):
            raise ValueError("selection item must be an object")
        case_id = item.get("case_id")
        mode = item.get("mode")
        if not isinstance(case_id, str) or not isinstance(mode, str):
            raise ValueError("selection item must contain case_id and mode")
        expected_key = f"{case_id}::{mode}"
        if item.get("review_key", expected_key) != expected_key:
            raise ValueError(f"selection review_key mismatch: {expected_key}")
        keys.append(expected_key)
    if len(keys) != len(set(keys)):
        raise ValueError("selection contains duplicate review keys")
    return keys


def _parse_bool(value: str, *, field: str, review_key: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"{review_key} has invalid {field}: {value!r}")


def _is_component_failure(trace: dict[str, Any]) -> bool:
    """Use the frozen stop reason even if an attempt cached a judgment."""

    return trace.get("stop_reason") == "component_failure"


def _confusion(human: list[bool], predicted: list[bool]) -> dict[str, Any]:
    if len(human) != len(predicted):
        raise ValueError("human and predicted label counts differ")
    tp = sum(gold and guess for gold, guess in zip(human, predicted, strict=True))
    tn = sum(
        not gold and not guess
        for gold, guess in zip(human, predicted, strict=True)
    )
    fp = sum(
        not gold and guess
        for gold, guess in zip(human, predicted, strict=True)
    )
    fn = sum(gold and not guess for gold, guess in zip(human, predicted, strict=True))
    denominator = len(human)
    positive_count = tp + fn
    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "denominator": denominator,
        "accuracy_pct": (tp + tn) / denominator * 100 if denominator else None,
        "revision_recall_pct": tp / positive_count * 100 if positive_count else None,
    }


def replay(
    *,
    artifact_path: Path,
    review_batches: list[tuple[Path, Path, Path]],
    project_root: Path,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    artifact_path = artifact_path.resolve()
    artifact = _load_json(artifact_path)
    artifact_sha256 = _sha256(artifact_path)

    raw_results = artifact.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("benchmark artifact must contain results")
    results: dict[str, dict[str, Any]] = {}
    for item in raw_results:
        if not isinstance(item, dict):
            raise ValueError("benchmark result must be an object")
        key = f"{item.get('case_id')}::{item.get('mode')}"
        if key in results:
            raise ValueError(f"benchmark contains duplicate result: {key}")
        results[key] = item

    rows: list[dict[str, str]] = []
    provenance_batches: list[dict[str, Any]] = []
    seen_review_keys: set[str] = set()
    for selection_path, review_path, ingestion_path in review_batches:
        selection_path = selection_path.resolve()
        review_path = review_path.resolve()
        ingestion_path = ingestion_path.resolve()
        manifest = _load_json(selection_path)
        ingestion = _load_json(ingestion_path)

        if manifest.get("artifact_sha256") != artifact_sha256:
            raise ValueError("selection artifact SHA-256 does not match replay artifact")
        manifest_artifact = project_root / str(manifest.get("artifact_file", ""))
        if manifest_artifact.resolve() != artifact_path:
            raise ValueError("selection artifact path does not match replay artifact")
        for manifest_field, artifact_field in (
            ("run_id", "run_id"),
            ("dataset_id", "dataset_id"),
            ("dataset_sha256", "dataset_sha256"),
            ("artifact_config_sha256", "config_sha256"),
        ):
            if manifest.get(manifest_field) != artifact.get(artifact_field):
                raise ValueError(
                    f"selection {manifest_field} does not match benchmark artifact"
                )
        if ingestion.get("selection_manifest_sha256") != _sha256(selection_path):
            raise ValueError("ingestion selection SHA-256 mismatch")
        if ingestion.get("output_csv_sha256") != _sha256(review_path):
            raise ValueError("ingestion review CSV SHA-256 mismatch")
        if ingestion.get("selection_manifest_file") != _relative(
            selection_path, project_root
        ):
            raise ValueError("ingestion selection path mismatch")
        if ingestion.get("output_csv_file") != _relative(review_path, project_root):
            raise ValueError("ingestion review CSV path mismatch")

        batch_rows = _load_csv(review_path)
        expected_keys = _selected_keys(manifest)
        observed_keys = [row.get("review_key", "") for row in batch_rows]
        if observed_keys != expected_keys:
            raise ValueError("review CSV keys/order do not match selection manifest")
        if any(key in seen_review_keys for key in observed_keys):
            raise ValueError("review batches contain duplicate review keys")
        seen_review_keys.update(observed_keys)
        rows.extend(batch_rows)
        provenance_batches.append(
            {
                "batch_id": manifest.get("batch_id"),
                "selection_file": _relative(selection_path, project_root),
                "selection_sha256": _sha256(selection_path),
                "review_csv_file": _relative(review_path, project_root),
                "review_csv_sha256": _sha256(review_path),
                "ingestion_file": _relative(ingestion_path, project_root),
                "ingestion_sha256": _sha256(ingestion_path),
                "selected_count": len(batch_rows),
            }
        )

    replayed_entries: list[dict[str, Any]] = []
    component_failure_count = 0
    changed_keys: list[str] = []
    action_changed_keys: list[str] = []
    summary_conflicts: list[dict[str, Any]] = []
    consistency_issue_counts = {
        issue.value: 0 for issue in JudgmentConsistencyIssue
    }
    consistency_issue_rows: list[dict[str, Any]] = []

    # First pass: derive every replay decision from immutable benchmark evidence.
    # Human labels are deliberately not read in this pass.
    for row in rows:
        review_key = row["review_key"]
        result = results.get(review_key)
        if result is None:
            raise ValueError(f"benchmark result is missing: {review_key}")
        response = result.get("response")
        if not isinstance(response, dict):
            raise ValueError(f"benchmark response is invalid: {review_key}")
        trace = response.get("trace")
        attempts = trace.get("attempts") if isinstance(trace, dict) else None
        if not isinstance(attempts, list) or not attempts:
            raise ValueError(f"benchmark attempts are missing: {review_key}")
        first_attempt = attempts[0]
        if not isinstance(first_attempt, dict):
            raise ValueError(f"first attempt is invalid: {review_key}")

        candidate = first_attempt.get("candidate")
        if (
            row.get("source_text") != response.get("source_text")
            or not isinstance(candidate, dict)
            or row.get("initial_translation") != candidate.get("text")
            or row.get("agent_initial_summary", "")
            != ((first_attempt.get("judgment") or {}).get("summary", ""))
        ):
            raise ValueError(f"immutable replay evidence mismatch: {review_key}")

        judgment_payload = first_attempt.get("judgment")
        if judgment_payload is not None and not isinstance(judgment_payload, dict):
            raise ValueError(f"judgment is invalid: {review_key}")
        expected_passed = (
            "" if judgment_payload is None else str(judgment_payload.get("passed"))
        )
        expected_error_types = (
            [] if judgment_payload is None else judgment_payload.get("error_types")
        )
        try:
            recorded_error_types = json.loads(row.get("agent_initial_error_types", ""))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"review CSV Agent error evidence is invalid: {review_key}"
            ) from exc
        if (
            row.get("agent_initial_passed", "") != expected_passed
            or recorded_error_types != expected_error_types
            or row.get("retry_count") != str(response.get("retry_count"))
            or row.get("stop_reason") != trace.get("stop_reason")
        ):
            raise ValueError(f"immutable Agent evidence mismatch: {review_key}")

        component_failure = _is_component_failure(trace)
        if component_failure:
            component_failure_count += 1
            replayed_entries.append(
                {
                    "review_key": review_key,
                    "component_failure": True,
                    "before_passed": None,
                    "after_passed": None,
                }
            )
            continue
        if judgment_payload is None:
            raise ValueError(f"non-failure row has no judgment: {review_key}")
        raw_payload = {
            field: judgment_payload[field]
            for field in RAW_JUDGMENT_FIELDS
            if field in judgment_payload
        }
        normalized = canonicalize_llm_judgment(raw_payload)
        before_passed = judgment_payload.get("passed")
        if type(before_passed) is not bool:
            raise ValueError(f"stored passed value is invalid: {review_key}")

        if before_passed is not normalized.passed:
            changed_keys.append(review_key)
        before_action = judgment_payload.get("next_action")
        if before_action != normalized.next_action.value:
            action_changed_keys.append(review_key)

        audit = normalized.decision_audit
        assert audit is not None
        observed_issues = [issue.value for issue in audit.consistency_issues]
        for issue in audit.consistency_issues:
            consistency_issue_counts[issue.value] += 1
        if observed_issues:
            consistency_issue_rows.append(
                {
                    "review_key": review_key,
                    "issues": observed_issues,
                    "reported_passed": audit.reported_passed,
                    "derived_passed": normalized.passed,
                    "reported_next_action": audit.reported_next_action.value,
                    "derived_next_action": normalized.next_action.value,
                    "reported_error_types": audit.reported_error_types,
                    "derived_error_types": [
                        error.value for error in normalized.error_types
                    ],
                }
            )
        summary_issue_values = {
            JudgmentConsistencyIssue.SUMMARY_ERROR_WITHOUT_STRUCTURED_ERROR,
            JudgmentConsistencyIssue.SUMMARY_PASS_WITH_STRUCTURED_ERROR,
        }
        observed_summary_issues = [
            issue.value
            for issue in audit.consistency_issues
            if issue in summary_issue_values
        ]
        if observed_summary_issues:
            summary_conflicts.append(
                {
                    "review_key": review_key,
                    "issues": observed_summary_issues,
                    "stored_error_types": judgment_payload.get("error_types"),
                    "stored_passed": before_passed,
                    "summary": judgment_payload.get("summary"),
                }
            )

        replayed_entries.append(
            {
                "review_key": review_key,
                "component_failure": False,
                "before_passed": before_passed,
                "after_passed": normalized.passed,
            }
        )

    # Second pass: join the already-derived decisions to confirmed human labels.
    before_human: list[bool] = []
    before_predicted: list[bool] = []
    after_predicted: list[bool] = []
    for row, entry in zip(rows, replayed_entries, strict=True):
        review_key = row["review_key"]
        if entry["review_key"] != review_key:
            raise ValueError("internal replay row order changed before label join")
        if row.get("review_status") != "confirmed":
            raise ValueError(f"{review_key} is not a confirmed human label")
        human_needs_revision = _parse_bool(
            row.get("manual_initial_needs_revision", ""),
            field="manual_initial_needs_revision",
            review_key=review_key,
        )
        if entry["component_failure"]:
            continue
        before_human.append(human_needs_revision)
        before_predicted.append(not entry["before_passed"])
        after_predicted.append(not entry["after_passed"])

    return {
        "schema_version": 1,
        "replay_kind": "frozen_judgment_consistency_no_model",
        "decision_basis": "structured_error_types",
        "quality_claims_allowed": False,
        "provenance": {
            "artifact_file": _relative(artifact_path, project_root),
            "artifact_sha256": artifact_sha256,
            "run_id": artifact.get("run_id"),
            "dataset_id": artifact.get("dataset_id"),
            "dataset_sha256": artifact.get("dataset_sha256"),
            "artifact_config_sha256": artifact.get("config_sha256"),
            "batches": provenance_batches,
        },
        "selected_count": len(rows),
        "scorable_judgment_count": len(before_human),
        "component_failure_count": component_failure_count,
        "decision_change_count": len(changed_keys),
        "decision_change_keys": changed_keys,
        "next_action_change_count": len(action_changed_keys),
        "next_action_change_keys": action_changed_keys,
        "consistency_issue_counts": consistency_issue_counts,
        "consistency_issue_rows": consistency_issue_rows,
        "summary_conflict_count": len(summary_conflicts),
        "summary_conflicts": summary_conflicts,
        "before": _confusion(before_human, before_predicted),
        "after": _confusion(before_human, after_predicted),
        "limitations": [
            "The artifact stores judgments after the previous canonicalizer ran.",
            "Original raw LLM error lists that may have been erased cannot be recovered.",
            "Summary conflicts are diagnostic only and do not determine PASS or REVISE.",
            "Summary diagnostics use conservative English-only patterns.",
            "Human labels are joined only after replay decisions are derived.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--review-batch",
        action="append",
        nargs=3,
        required=True,
        metavar=("SELECTION_JSON", "REVIEW_CSV", "INGESTION_JSON"),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = replay(
        artifact_path=args.artifact,
        review_batches=[tuple(Path(value) for value in batch) for batch in args.review_batch],
        project_root=args.project_root,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
        return
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
