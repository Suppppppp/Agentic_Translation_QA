#!/usr/bin/env python3
"""Replay source coverage on frozen first attempts without calling any model."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from scripts.replay_judgment_consistency import (
        _is_component_failure,
        _load_csv,
        _load_json,
        _parse_bool,
        _relative,
        _selected_keys,
        _sha256,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from replay_judgment_consistency import (  # type: ignore[no-redef]
        _is_component_failure,
        _load_csv,
        _load_json,
        _parse_bool,
        _relative,
        _selected_keys,
        _sha256,
    )
from translation_qa.coverage import (
    check_source_coverage,
    extract_source_coverage_requirements,
    missing_coverage_findings,
)
from translation_qa.schemas import RetrievalHit, SourceAnalysis


def _percent(numerator: int, denominator: int) -> float | None:
    return numerator / denominator * 100 if denominator else None


def replay(
    *,
    artifact_path: Path,
    review_batches: list[tuple[Path, Path, Path]],
    project_root: Path,
) -> dict[str, Any]:
    """Replay coverage first, then join frozen Sup labels for the 7 FN/6 TN audit."""

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
        review_key = f"{item.get('case_id')}::{item.get('mode')}"
        if review_key in results:
            raise ValueError(f"benchmark contains duplicate result: {review_key}")
        results[review_key] = item

    review_rows: list[dict[str, str]] = []
    provenance_batches: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for selection_path, review_path, ingestion_path in review_batches:
        selection_path = selection_path.resolve()
        review_path = review_path.resolve()
        ingestion_path = ingestion_path.resolve()
        manifest = _load_json(selection_path)
        ingestion = _load_json(ingestion_path)
        if manifest.get("artifact_sha256") != artifact_sha256:
            raise ValueError("selection artifact SHA-256 does not match replay artifact")
        if (project_root / str(manifest.get("artifact_file", ""))).resolve() != artifact_path:
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

        rows = _load_csv(review_path)
        expected_keys = _selected_keys(manifest)
        observed_keys = [row.get("review_key", "") for row in rows]
        if observed_keys != expected_keys:
            raise ValueError("review CSV keys/order do not match selection manifest")
        if any(key in seen_keys for key in observed_keys):
            raise ValueError("review batches contain duplicate review keys")
        seen_keys.update(observed_keys)
        review_rows.extend(rows)
        provenance_batches.append(
            {
                "batch_id": manifest.get("batch_id"),
                "selection_file": _relative(selection_path, project_root),
                "selection_sha256": _sha256(selection_path),
                "review_csv_file": _relative(review_path, project_root),
                "review_csv_sha256": _sha256(review_path),
                "ingestion_file": _relative(ingestion_path, project_root),
                "ingestion_sha256": _sha256(ingestion_path),
                "selected_count": len(rows),
            }
        )

    derived: list[dict[str, Any]] = []
    source_component_failure_count = 0
    # First pass: source/candidate/trace only.  References and manual fields are
    # not read while any coverage decision is being derived.
    for row in review_rows:
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
        judgment = first_attempt.get("judgment")
        if (
            row.get("source_text") != response.get("source_text")
            or not isinstance(candidate, dict)
            or row.get("initial_translation") != candidate.get("text")
            or row.get("agent_initial_summary", "")
            != ((judgment or {}).get("summary", ""))
        ):
            raise ValueError(f"immutable replay evidence mismatch: {review_key}")
        if judgment is not None and not isinstance(judgment, dict):
            raise ValueError(f"initial judgment is invalid: {review_key}")
        expected_passed = "" if judgment is None else str(judgment.get("passed"))
        expected_error_types = [] if judgment is None else judgment.get("error_types")
        try:
            recorded_error_types = json.loads(
                row.get("agent_initial_error_types", "")
            )
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

        if _is_component_failure(trace):
            source_component_failure_count += 1
            derived.append(
                {
                    "review_key": review_key,
                    "component_failure": True,
                    "stored_passed": None,
                    "findings": [],
                    "missing": [],
                }
            )
            continue
        if judgment is None or type(judgment.get("passed")) is not bool:
            raise ValueError(f"non-failure row has no valid judgment: {review_key}")

        source_text = str(response["source_text"])
        analysis_payload = trace.get("source_analysis")
        analysis = (
            SourceAnalysis.model_validate(analysis_payload)
            if analysis_payload is not None
            else None
        )
        raw_hits = first_attempt.get("retrieval_hits", [])
        if not isinstance(raw_hits, list):
            raise ValueError(f"retrieval hits are invalid: {review_key}")
        hits = [RetrievalHit.model_validate(hit) for hit in raw_hits]
        requirements = extract_source_coverage_requirements(
            source_text, analysis, hits
        )
        findings = check_source_coverage(
            str(candidate["text"]),
            requirements,
            targeted_rag_available=result.get("mode") == "agent_rag",
        )
        missing = missing_coverage_findings(findings)
        derived.append(
            {
                "review_key": review_key,
                "component_failure": False,
                "stored_passed": judgment["passed"],
                "findings": findings,
                "missing": missing,
            }
        )

    replay_rows: list[dict[str, Any]] = []
    # Second pass: join already-derived decisions to confirmed human labels and
    # retain only the seven old false negatives and six old true negatives.
    for row, entry in zip(review_rows, derived, strict=True):
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
        if entry["component_failure"] or entry["stored_passed"] is not True:
            continue
        baseline_group = (
            "false_negative" if human_needs_revision else "true_negative"
        )
        missing = entry["missing"]
        replay_rows.append(
            {
                "review_key": review_key,
                "baseline_group": baseline_group,
                "human_needs_revision": human_needs_revision,
                "coverage_requests_revision": bool(missing),
                "missing_source_terms": [
                    finding.requirement.source_term for finding in missing
                ],
                "recovery_actions": [
                    finding.recovery_action.value for finding in missing
                ],
                "finding_statuses": {
                    finding.requirement.source_term: finding.status.value
                    for finding in entry["findings"]
                },
            }
        )

    tp = sum(
        row["human_needs_revision"] and row["coverage_requests_revision"]
        for row in replay_rows
    )
    tn = sum(
        not row["human_needs_revision"] and not row["coverage_requests_revision"]
        for row in replay_rows
    )
    fp = sum(
        not row["human_needs_revision"] and row["coverage_requests_revision"]
        for row in replay_rows
    )
    fn = sum(
        row["human_needs_revision"] and not row["coverage_requests_revision"]
        for row in replay_rows
    )
    positive_count = tp + fn
    action_counts = Counter(
        action for row in replay_rows for action in row["recovery_actions"]
    )
    return {
        "schema_version": 1,
        "replay_kind": "frozen_source_coverage_first_attempt_no_model",
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
        "source_selected_count": len(review_rows),
        "source_component_failure_count": source_component_failure_count,
        "replayed_count": len(replay_rows),
        "distinct_source_case_count": len(
            {row["review_key"].split("::", 1)[0] for row in replay_rows}
        ),
        "baseline_false_negative_count": sum(
            row["baseline_group"] == "false_negative" for row in replay_rows
        ),
        "baseline_true_negative_count": sum(
            row["baseline_group"] == "true_negative" for row in replay_rows
        ),
        "coverage_confusion": {
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "denominator": len(replay_rows),
            "accuracy_pct": _percent(tp + tn, len(replay_rows)),
            "revision_recall_pct": _percent(tp, positive_count),
            "true_negative_rate_pct": _percent(tn, tn + fp),
        },
        "recovery_action_counts": dict(sorted(action_counts.items())),
        "rows": replay_rows,
        "limitations": [
            "This is a deliberately selected 13-row diagnostic replay, not a benchmark estimate.",
            "Coverage uses only source text, frozen analysis, candidate, and retrieval hits.",
            "Human labels are joined only after all coverage decisions are derived.",
            "Unmapped lexical noun phrases remain unresolved instead of being called missing.",
            "The replay checks frozen candidates; it does not execute translation or revision.",
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
        review_batches=[
            tuple(Path(value) for value in batch) for batch in args.review_batch
        ],
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
