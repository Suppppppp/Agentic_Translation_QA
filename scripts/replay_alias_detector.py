#!/usr/bin/env python3
"""Replay the deterministic alias detector on frozen candidates only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts.replay_judgment_consistency import (
        _confusion,
        _load_csv,
        _load_json,
        _parse_bool,
        _relative,
        _selected_keys,
        _sha256,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from replay_judgment_consistency import (  # type: ignore[no-redef]
        _confusion,
        _load_csv,
        _load_json,
        _parse_bool,
        _relative,
        _selected_keys,
        _sha256,
    )
from translation_qa.alias_detector import AliasDetector
from translation_qa.coverage import (
    check_source_coverage,
    extract_source_coverage_requirements,
    missing_coverage_findings,
)
from translation_qa.retrieval import load_glossary_csv
from translation_qa.schemas import RetrievalHit, SourceAnalysis


FROZEN_DIAGNOSTIC_KEYS = (
    "evaluation-v1-009::agent_rag",
    "evaluation-v1-019::agent_rag",
    "evaluation-v1-027::agent",
    "evaluation-v1-008::agent",
    "evaluation-v1-008::agent_rag",
    "evaluation-v1-012::agent",
    "evaluation-v1-012::agent_rag",
    "evaluation-v1-013::agent",
    "evaluation-v1-013::agent_rag",
    "evaluation-v1-015::agent",
    "evaluation-v1-015::agent_rag",
    "evaluation-v1-038::agent",
    "evaluation-v1-038::agent_rag",
)
RESERVE_SANITY_KEYS = (
    "evaluation-v1-004::agent",
    "evaluation-v1-004::agent_rag",
)


def _source_metadata(payload: dict[str, Any]) -> dict[str, str | None]:
    """Project a dataset row to source-only fields.

    Reference and manual-gold fields are intentionally neither accessed nor
    retained by this function.
    """

    case_id = payload.get("case_id")
    source_text = payload.get("source_text")
    domain = payload.get("domain")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("dataset case_id must be a non-empty string")
    if not isinstance(source_text, str) or not source_text.strip():
        raise ValueError(f"{case_id} source_text must be a non-empty string")
    if domain is not None and (not isinstance(domain, str) or not domain.strip()):
        raise ValueError(f"{case_id} domain must be null or a non-empty string")
    return {"case_id": case_id, "source_text": source_text, "domain": domain}


def _load_source_metadata(path: Path) -> dict[str, dict[str, str | None]]:
    result: dict[str, dict[str, str | None]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line)
            if not isinstance(payload, dict):
                raise ValueError(f"dataset line {line_number} must be an object")
            projected = _source_metadata(payload)
            case_id = str(projected["case_id"])
            if case_id in result:
                raise ValueError(f"dataset contains duplicate case_id: {case_id}")
            result[case_id] = projected
    if not result:
        raise ValueError("dataset is empty")
    return result


def _artifact_results(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_results = artifact.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("benchmark artifact must contain results")
    results: dict[str, dict[str, Any]] = {}
    for result in raw_results:
        if not isinstance(result, dict):
            raise ValueError("benchmark result must be an object")
        key = f"{result.get('case_id')}::{result.get('mode')}"
        if key in results:
            raise ValueError(f"benchmark contains duplicate result: {key}")
        results[key] = result
    return results


def _validate_batch_selection(
    *,
    selection_path: Path,
    artifact_path: Path,
    artifact: dict[str, Any],
    artifact_sha256: str,
    dataset_path: Path,
    dataset_sha256: str,
    project_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    manifest = _load_json(selection_path)
    if manifest.get("artifact_sha256") != artifact_sha256:
        raise ValueError("selection artifact SHA-256 mismatch")
    if (project_root / str(manifest.get("artifact_file", ""))).resolve() != artifact_path:
        raise ValueError("selection artifact path mismatch")
    if manifest.get("dataset_sha256") != dataset_sha256:
        raise ValueError("selection dataset SHA-256 mismatch")
    if (project_root / str(manifest.get("dataset_file", ""))).resolve() != dataset_path:
        raise ValueError("selection dataset path mismatch")
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
    keys = _selected_keys(manifest)
    return manifest, keys


def _validate_post_decision_review_provenance(
    *,
    selection_path: Path,
    review_path: Path,
    ingestion_path: Path,
    project_root: Path,
    selected_count: int,
) -> dict[str, Any]:
    """Read reviewed artifacts only after all detector decisions exist."""

    ingestion = _load_json(ingestion_path)
    if ingestion.get("selection_manifest_sha256") != _sha256(selection_path):
        raise ValueError("ingestion selection SHA-256 mismatch")
    if ingestion.get("selection_manifest_file") != _relative(
        selection_path, project_root
    ):
        raise ValueError("ingestion selection path mismatch")
    if ingestion.get("output_csv_sha256") != _sha256(review_path):
        raise ValueError("ingestion review CSV SHA-256 mismatch")
    if ingestion.get("output_csv_file") != _relative(review_path, project_root):
        raise ValueError("ingestion review CSV path mismatch")
    return {
        "batch_id": ingestion.get("batch_id"),
        "selection_file": _relative(selection_path, project_root),
        "selection_sha256": _sha256(selection_path),
        "review_csv_file": _relative(review_path, project_root),
        "review_csv_sha256": _sha256(review_path),
        "ingestion_file": _relative(ingestion_path, project_root),
        "ingestion_sha256": _sha256(ingestion_path),
        "selected_count": selected_count,
    }


def _score(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    eligible = [row for row in rows if not row["detector_component_failure"]]
    result = _confusion(
        [bool(row["human_needs_revision"]) for row in eligible],
        [bool(row[field]) for row in eligible],
    )
    result["component_failure_count"] = len(rows) - len(eligible)
    return result


def replay(
    *,
    artifact_path: Path,
    dataset_path: Path,
    glossary_paths: list[Path],
    review_batches: list[tuple[Path, Path, Path]],
    project_root: Path,
) -> dict[str, Any]:
    """Derive all lexical decisions first, then join confirmed human labels."""

    project_root = project_root.resolve()
    artifact_path = artifact_path.resolve()
    dataset_path = dataset_path.resolve()
    glossary_paths = [path.resolve() for path in glossary_paths]
    artifact = _load_json(artifact_path)
    artifact_sha256 = _sha256(artifact_path)
    dataset_sha256 = _sha256(dataset_path)
    results = _artifact_results(artifact)

    entries = []
    glossary_provenance: list[dict[str, str]] = []
    for path in glossary_paths:
        entries.extend(load_glossary_csv(path))
        glossary_provenance.append(
            {
                "file": _relative(path, project_root),
                "sha256": _sha256(path),
            }
        )
    detector = AliasDetector(entries)
    source_metadata = _load_source_metadata(dataset_path)

    selected_keys: list[str] = []
    batch_metadata: list[
        tuple[Path, Path, Path, dict[str, Any], list[str]]
    ] = []
    seen_keys: set[str] = set()
    for selection_raw, review_raw, ingestion_raw in review_batches:
        selection_path = selection_raw.resolve()
        review_path = review_raw.resolve()
        ingestion_path = ingestion_raw.resolve()
        manifest, keys = _validate_batch_selection(
            selection_path=selection_path,
            artifact_path=artifact_path,
            artifact=artifact,
            artifact_sha256=artifact_sha256,
            dataset_path=dataset_path,
            dataset_sha256=dataset_sha256,
            project_root=project_root,
        )
        if any(key in seen_keys for key in keys):
            raise ValueError("review batches contain duplicate review keys")
        seen_keys.update(keys)
        selected_keys.extend(keys)
        batch_metadata.append(
            (selection_path, review_path, ingestion_path, manifest, keys)
        )

    missing_frozen = set(FROZEN_DIAGNOSTIC_KEYS) - set(selected_keys)
    missing_reserve = set(RESERVE_SANITY_KEYS) - set(selected_keys)
    if missing_frozen or missing_reserve:
        raise ValueError(
            "review manifests are missing required replay keys: "
            + ", ".join(sorted(missing_frozen | missing_reserve))
        )

    derived_by_key: dict[str, dict[str, Any]] = {}
    detector_failure_count = 0
    stored_trace_component_failure_count = 0
    # Decision pass.  Review CSVs are not opened until every selected key has
    # a deterministic result.  Only source/candidate/domain/trace evidence is
    # available here; reference text and manual labels are absent.
    for review_key in selected_keys:
        result = results.get(review_key)
        if result is None:
            raise ValueError(f"benchmark result is missing: {review_key}")
        case_id, mode = review_key.split("::", 1)
        metadata = source_metadata.get(case_id)
        if metadata is None:
            raise ValueError(f"source metadata is missing: {case_id}")
        response = result.get("response")
        if not isinstance(response, dict):
            raise ValueError(f"benchmark response is invalid: {review_key}")
        if response.get("source_text") != metadata["source_text"]:
            raise ValueError(f"dataset/artifact source mismatch: {review_key}")
        trace = response.get("trace")
        attempts = trace.get("attempts") if isinstance(trace, dict) else None
        if not isinstance(attempts, list) or not attempts:
            raise ValueError(f"benchmark attempts are missing: {review_key}")
        first_attempt = attempts[0]
        candidate = first_attempt.get("candidate") if isinstance(first_attempt, dict) else None
        if not isinstance(candidate, dict) or not isinstance(candidate.get("text"), str):
            raise ValueError(f"first candidate is invalid: {review_key}")
        stored_trace_failure = trace.get("stop_reason") == "component_failure"
        stored_trace_component_failure_count += int(stored_trace_failure)

        try:
            raw_analysis = trace.get("source_analysis")
            analysis = (
                SourceAnalysis.model_validate(raw_analysis)
                if raw_analysis is not None
                else None
            )
            raw_hits = first_attempt.get("retrieval_hits", [])
            if not isinstance(raw_hits, list):
                raise ValueError("retrieval hits must be a list")
            hits = [RetrievalHit.model_validate(hit) for hit in raw_hits]
            requirements = extract_source_coverage_requirements(
                str(metadata["source_text"]), analysis, hits
            )
            coverage_findings = check_source_coverage(
                str(candidate["text"]),
                requirements,
                targeted_rag_available=mode == "agent_rag",
            )
            coverage_missing = missing_coverage_findings(coverage_findings)
            alias_findings = detector.detect(
                source_text=str(metadata["source_text"]),
                candidate_text=str(candidate["text"]),
                domain=(str(metadata["domain"]) if metadata["domain"] is not None else None),
                coverage_requirements=requirements,
            )
            alias_errors = [finding for finding in alias_findings if finding.is_error]
            derived_by_key[review_key] = {
                "review_key": review_key,
                "case_id": case_id,
                "mode": mode,
                "source_text": metadata["source_text"],
                "candidate_text": candidate["text"],
                "authoritative_domain": metadata["domain"],
                "detector_component_failure": False,
                "detector_failure": None,
                "stored_trace_component_failure": stored_trace_failure,
                "alias_requests_revision": bool(alias_errors),
                "coverage_requests_revision": bool(coverage_missing),
                "combined_requests_revision": bool(alias_errors or coverage_missing),
                "alias_findings": [finding.to_dict() for finding in alias_findings],
                "alias_error_ids": [
                    finding.alias.alias_id for finding in alias_errors
                ],
                "coverage_missing_source_terms": [
                    finding.requirement.source_term for finding in coverage_missing
                ],
            }
        except Exception as exc:  # deterministic failure is part of the replay gate
            detector_failure_count += 1
            derived_by_key[review_key] = {
                "review_key": review_key,
                "case_id": case_id,
                "mode": mode,
                "source_text": metadata["source_text"],
                "candidate_text": candidate["text"],
                "authoritative_domain": metadata["domain"],
                "detector_component_failure": True,
                "detector_failure": f"{type(exc).__name__}: {exc}",
                "stored_trace_component_failure": stored_trace_failure,
                "alias_requests_revision": False,
                "coverage_requests_revision": False,
                "combined_requests_revision": False,
                "alias_findings": [],
                "alias_error_ids": [],
                "coverage_missing_source_terms": [],
            }

    # Label pass.  Every detector decision above is complete before a reviewed
    # CSV is opened.  Reference columns may be parsed by DictReader here, but
    # they are never accessed or exposed to the detector.
    labels_by_key: dict[str, dict[str, Any]] = {}
    provenance_batches: list[dict[str, Any]] = []
    for selection_path, review_path, ingestion_path, manifest, keys in batch_metadata:
        provenance_batches.append(
            _validate_post_decision_review_provenance(
                selection_path=selection_path,
                review_path=review_path,
                ingestion_path=ingestion_path,
                project_root=project_root,
                selected_count=len(keys),
            )
        )
        rows = _load_csv(review_path)
        observed_keys = [row.get("review_key", "") for row in rows]
        if observed_keys != keys or observed_keys != _selected_keys(manifest):
            raise ValueError("review CSV keys/order do not match selection manifest")
        for row in rows:
            review_key = row["review_key"]
            derived = derived_by_key[review_key]
            if (
                row.get("source_text") != derived["source_text"]
                or row.get("initial_translation") != derived["candidate_text"]
            ):
                raise ValueError(f"immutable review evidence mismatch: {review_key}")
            if row.get("review_status") != "confirmed" or row.get("reviewer") != "Sup":
                raise ValueError(f"review label is not confirmed by Sup: {review_key}")
            labels_by_key[review_key] = {
                "human_needs_revision": _parse_bool(
                    row.get("manual_initial_needs_revision", ""),
                    field="manual_initial_needs_revision",
                    review_key=review_key,
                ),
                "manual_severity": row.get("manual_severity", ""),
            }

    def joined(keys: tuple[str, ...]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for key in keys:
            if key not in labels_by_key:
                raise ValueError(f"manual label is missing after decision pass: {key}")
            rows.append({**derived_by_key[key], **labels_by_key[key]})
        return rows

    primary_rows = joined(FROZEN_DIAGNOSTIC_KEYS)
    reserve_rows = joined(RESERVE_SANITY_KEYS)
    alias_confusion = _score(primary_rows, "alias_requests_revision")
    coverage_confusion = _score(primary_rows, "coverage_requests_revision")
    combined_confusion = _score(primary_rows, "combined_requests_revision")
    incremental_keys = [
        row["review_key"]
        for row in primary_rows
        if row["human_needs_revision"]
        and row["alias_requests_revision"]
        and not row["coverage_requests_revision"]
    ]
    tn_alias_false_positives = [
        row["review_key"]
        for row in primary_rows
        if not row["human_needs_revision"] and row["alias_requests_revision"]
    ]
    case_009 = next(
        row
        for row in primary_rows
        if row["review_key"] == "evaluation-v1-009::agent_rag"
    )

    acceptance = {
        "detector_component_failure_zero": detector_failure_count == 0,
        "case_009_detected": bool(case_009["alias_requests_revision"]),
        "tn_false_positive_zero": not tn_alias_false_positives,
        "combined_recovers_all_frozen_fn": (
            combined_confusion["true_positive"] == 7
            and combined_confusion["false_negative"] == 0
        ),
    }
    acceptance["all_passed"] = all(acceptance.values())

    return {
        "schema_version": 1,
        "replay_kind": "frozen_alias_detector_first_attempt_no_model",
        "quality_claims_allowed": False,
        "automatic_gold_labels_generated": False,
        "guardrails": {
            "reference_exposed_to_detector": False,
            "manual_labels_loaded_after_all_detector_decisions": True,
            "detector_input_fields": [
                "source_text",
                "candidate_text",
                "authoritative_domain",
                "source_coverage_requirements",
                "versioned_glossary_aliases",
            ],
            "dataset_fields_projected_before_decisions": [
                "case_id",
                "source_text",
                "domain",
            ],
            "judge_called": False,
            "translator_called": False,
            "retriever_called": False,
            "reviser_called": False,
            "retry_pipeline_called": False,
        },
        "provenance": {
            "artifact_file": _relative(artifact_path, project_root),
            "artifact_sha256": artifact_sha256,
            "run_id": artifact.get("run_id"),
            "dataset_id": artifact.get("dataset_id"),
            "dataset_file": _relative(dataset_path, project_root),
            "dataset_sha256": dataset_sha256,
            "artifact_config_sha256": artifact.get("config_sha256"),
            "glossaries": glossary_provenance,
            "review_batches": provenance_batches,
        },
        "selected_source_pool_count": len(selected_keys),
        "frozen_replay_count": len(primary_rows),
        "frozen_human_revision_count": sum(
            bool(row["human_needs_revision"]) for row in primary_rows
        ),
        "frozen_human_pass_count": sum(
            not bool(row["human_needs_revision"]) for row in primary_rows
        ),
        "detector_component_failure_count": detector_failure_count,
        "stored_source_trace_component_failure_count": (
            stored_trace_component_failure_count
        ),
        "alias_only_confusion": alias_confusion,
        "source_coverage_only_confusion": coverage_confusion,
        "combined_confusion": combined_confusion,
        "incremental_alias_recovery_keys": incremental_keys,
        "tn_alias_false_positive_keys": tn_alias_false_positives,
        "acceptance": acceptance,
        "rows": primary_rows,
        "reserve": {
            "kind": "confirmed_pair_not_used_for_alias_authoring",
            "quality_claims_allowed": False,
            "reviewed_count": len(reserve_rows),
            "rows": reserve_rows,
            "interpretation": (
                "The pair checks an evaluation-preexisting deployment alias: "
                "one explicit disallowed form and one accepted form."
            ),
        },
        "limitations": [
            "The 13 primary rows are a deliberately selected diagnostic set, not a benchmark estimate.",
            "Alias VERIFIED means only that an applicable lexical alias is preserved; it is not a translation PASS.",
            "The new load-balancing mapping comes from paired official AWS documentation, not evaluation references or manual labels.",
            "The 009 candidate is used only as a regression assertion; no regression spelling is stored in the alias glossary.",
            "There is no reviewed human-PASS normal-context reserve beyond the primary TN rows; broader validation needs new manual review.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--glossary", type=Path, action="append", required=True)
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
        dataset_path=args.dataset,
        glossary_paths=args.glossary,
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
