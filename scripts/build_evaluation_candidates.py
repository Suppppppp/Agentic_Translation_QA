#!/usr/bin/env python3
"""Build an unreviewed, source-only evaluation candidate pool.

The deterministic selection stage reads only the Korean source column and a
pre-existing Korean glossary lexicon.  English references are joined after the
source row IDs have been selected and are included solely for bilingual manual
alignment review.  This script does not freeze a final dataset or run a
benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from translation_qa.retrieval import contains_source_term


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GLOSSARY = PROJECT_ROOT / "data" / "glossary_evaluation_v2.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "eda" / "evaluation_candidates.jsonl"
DEFAULT_SUMMARY = (
    PROJECT_ROOT / "artifacts" / "eda" / "evaluation_candidates_summary.json"
)
LEMON_CACHE_NAME = "datasets--lemon-mint--korean_parallel_sentences_v1.1"
DEFAULT_DATASET_ID = "lemon-mint/korean_parallel_sentences_v1.1"
ALGORITHM_VERSION = "source-only-term-stratified-v1"
ENGLISH_REFERENCE_POLICY = (
    "manual_alignment_review_only_not_used_for_selection_or_term_extraction"
)

_HANGUL_RE = re.compile(r"[가-힣]")
_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_BOUNDARY_RE = re.compile(
    r"[.!?。！？]+(?=(?:[\"'”’)}\]〉》」』]+)?(?:\s|$))"
)


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    seed: int = 42
    per_term_limit: int = 30
    min_korean_chars: int = 20
    max_korean_chars: int = 180
    max_sentence_boundaries: int = 1
    dataset_id: str = DEFAULT_DATASET_ID
    dataset_revision: str = "unknown"
    shard_id: str = "train-00000-of-00001"

    def validate(self) -> None:
        if self.per_term_limit < 1:
            raise ValueError("per_term_limit must be at least one")
        if self.min_korean_chars < 1:
            raise ValueError("min_korean_chars must be at least one")
        if self.max_korean_chars < self.min_korean_chars:
            raise ValueError("max_korean_chars must be >= min_korean_chars")
        if self.max_sentence_boundaries < 0:
            raise ValueError("max_sentence_boundaries must be non-negative")
        if not self.dataset_id.strip():
            raise ValueError("dataset_id must not be blank")
        if not self.dataset_revision.strip():
            raise ValueError("dataset_revision must not be blank")
        if not self.shard_id.strip():
            raise ValueError("shard_id must not be blank")


@dataclass(frozen=True, slots=True)
class SourceCandidate:
    source_row: int
    source_record_id: str
    korean: str
    normalized_korean: str
    korean_char_length: int
    sentence_boundary_count: int
    hit_terms: tuple[str, ...]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_artifact_path(path: Path) -> str:
    """Avoid embedding workstation-specific absolute paths in artifacts."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.name


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _display_source(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return unicodedata.normalize("NFKC", str(value)).strip()


def normalize_source(value: Any) -> str:
    """Normalize Korean source text for literal matching and exact deduplication."""

    return _WHITESPACE_RE.sub(" ", _display_source(value))


def _sentence_boundary_count(source: str) -> int:
    return len(_SENTENCE_BOUNDARY_RE.findall(source))


def _source_record_id(config: SelectionConfig, source_row: int) -> str:
    return (
        f"{config.dataset_id}@{config.dataset_revision}:"
        f"{config.shard_id}:{source_row:09d}"
    )


def locate_cached_lemon_parquet(cache_root: Path | None = None) -> Path:
    root = cache_root or Path.home() / ".cache" / "huggingface" / "hub"
    snapshots = root / LEMON_CACHE_NAME / "snapshots"
    for snapshot in sorted(snapshots.glob("*"), reverse=True):
        candidates = sorted((snapshot / "data").glob("*.parquet"))
        if candidates:
            if len(candidates) != 1:
                raise ValueError(
                    "candidate extraction currently expects one local parquet shard"
                )
            return candidates[0]
    raise FileNotFoundError("no local lemon-mint parquet snapshot was found")


def infer_dataset_revision(parquet_path: Path) -> str:
    for parent in parquet_path.parents:
        if parent.parent.name == "snapshots":
            return parent.name
    return f"sha256-{_file_sha256(parquet_path)[:16]}"


def load_fixed_source_lexicon(glossary_path: Path) -> tuple[list[str], str]:
    """Load only the Korean source-term column from the pre-existing glossary."""

    if not glossary_path.is_file():
        raise FileNotFoundError(f"glossary does not exist: {glossary_path}")
    columns = list(pd.read_csv(glossary_path, nrows=0).columns)
    source_column = next(
        (column for column in ("source_term_ko", "source_term") if column in columns),
        None,
    )
    if source_column is None:
        raise ValueError("glossary must contain source_term_ko or source_term")

    values = pd.read_csv(
        glossary_path,
        usecols=[source_column],
        dtype={source_column: "string"},
    )[source_column]
    terms: list[str] = []
    seen: set[str] = set()
    for row_number, value in enumerate(values, start=2):
        term = normalize_source(value)
        if not term:
            raise ValueError(f"blank glossary source term at CSV row {row_number}")
        if not _HANGUL_RE.search(term):
            raise ValueError(
                f"glossary source term has no Hangul at CSV row {row_number}: {term}"
            )
        if term in seen:
            raise ValueError(f"duplicate normalized glossary source term: {term}")
        terms.append(term)
        seen.add(term)
    if not terms:
        raise ValueError("glossary contains no Korean source terms")
    return terms, source_column


def select_source_candidates(
    korean_sources: Sequence[Any] | pd.Series,
    source_terms: Sequence[str],
    config: SelectionConfig,
) -> tuple[list[SourceCandidate], dict[str, Any]]:
    """Select row IDs using Korean source data only.

    English references are intentionally absent from this function's API.
    """

    config.validate()
    terms = tuple(normalize_source(term) for term in source_terms)
    if not terms or any(not term for term in terms):
        raise ValueError("source_terms must contain non-blank terms")
    if len(set(terms)) != len(terms):
        raise ValueError("source_terms must be unique after normalization")

    diagnostics = {
        "total_source_rows": int(len(korean_sources)),
        "blank_source": 0,
        "no_literal_glossary_term": 0,
        "no_hangul": 0,
        "below_min_korean_chars": 0,
        "above_max_korean_chars": 0,
        "contains_line_break": 0,
        "too_many_sentence_boundaries": 0,
        "normalized_source_duplicates_beyond_first": 0,
    }
    raw_literal_counts = {term: 0 for term in terms}
    rows_with_any_literal_term = 0
    seen_sources: set[str] = set()
    eligible: list[SourceCandidate] = []

    for source_row, value in enumerate(korean_sources):
        korean = _display_source(value)
        normalized = normalize_source(value)
        hits = tuple(
            term
            for term in terms
            if term in normalized and contains_source_term(normalized, term)
        )
        for term in hits:
            raw_literal_counts[term] += 1
        if hits:
            rows_with_any_literal_term += 1

        blank = not normalized
        has_hangul = bool(_HANGUL_RE.search(normalized))
        length = len(normalized)
        contains_line_break = "\n" in korean or "\r" in korean
        boundary_count = _sentence_boundary_count(korean)
        duplicate_source = bool(normalized and normalized in seen_sources)
        if normalized:
            seen_sources.add(normalized)

        diagnostics["blank_source"] += int(blank)
        diagnostics["no_literal_glossary_term"] += int(not hits)
        diagnostics["no_hangul"] += int(not has_hangul)
        diagnostics["below_min_korean_chars"] += int(
            length < config.min_korean_chars
        )
        diagnostics["above_max_korean_chars"] += int(
            length > config.max_korean_chars
        )
        diagnostics["contains_line_break"] += int(contains_line_break)
        diagnostics["too_many_sentence_boundaries"] += int(
            boundary_count > config.max_sentence_boundaries
        )
        diagnostics["normalized_source_duplicates_beyond_first"] += int(
            duplicate_source
        )

        if (
            blank
            or not hits
            or not has_hangul
            or length < config.min_korean_chars
            or length > config.max_korean_chars
            or contains_line_break
            or boundary_count > config.max_sentence_boundaries
            or duplicate_source
        ):
            continue

        eligible.append(
            SourceCandidate(
                source_row=source_row,
                source_record_id=_source_record_id(config, source_row),
                korean=korean,
                normalized_korean=normalized,
                korean_char_length=length,
                sentence_boundary_count=boundary_count,
                hit_terms=hits,
            )
        )

    selected_for_terms: dict[str, set[str]] = {}
    sampled_ids_by_term: dict[str, list[str]] = {}
    eligible_counts = {
        term: sum(term in candidate.hit_terms for candidate in eligible)
        for term in terms
    }
    for term in terms:
        ranked = sorted(
            (candidate for candidate in eligible if term in candidate.hit_terms),
            key=lambda candidate: hashlib.sha256(
                (
                    f"{ALGORITHM_VERSION}\x1f{config.seed}\x1f{term}\x1f"
                    f"{candidate.source_record_id}\x1f{candidate.normalized_korean}"
                ).encode("utf-8")
            ).hexdigest(),
        )
        sampled = ranked[: config.per_term_limit]
        sampled_ids_by_term[term] = [item.source_record_id for item in sampled]
        for candidate in sampled:
            selected_for_terms.setdefault(candidate.source_record_id, set()).add(term)

    selected = sorted(
        (
            candidate
            for candidate in eligible
            if candidate.source_record_id in selected_for_terms
        ),
        key=lambda candidate: candidate.source_row,
    )
    selected_ids = [candidate.source_record_id for candidate in selected]
    selected_id_set = set(selected_ids)
    per_term: dict[str, dict[str, Any]] = {}
    for term in terms:
        sampled_count = len(sampled_ids_by_term[term])
        shortfall = max(0, config.per_term_limit - sampled_count)
        output_hit_count = sum(term in candidate.hit_terms for candidate in selected)
        per_term[term] = {
            "raw_literal_source_rows": raw_literal_counts[term],
            "source_only_eligible_unique_rows": eligible_counts[term],
            "requested_review_pool_rows": config.per_term_limit,
            "sampled_for_term": sampled_count,
            "output_rows_containing_term": output_hit_count,
            "shortfall": shortfall,
            "status": (
                "target_met"
                if shortfall == 0
                else "no_eligible_candidates"
                if sampled_count == 0
                else "shortage"
            ),
        }

    metadata = {
        "rows_with_any_literal_term": rows_with_any_literal_term,
        "eligible_unique_source_rows": len(eligible),
        "selected_unique_rows": len(selected),
        "selected_multi_term_rows": sum(len(item.hit_terms) > 1 for item in selected),
        "filter_diagnostics_non_exclusive": diagnostics,
        "per_term": per_term,
        "selected_for_terms": {
            source_id: [term for term in terms if term in selected_terms]
            for source_id, selected_terms in selected_for_terms.items()
            if source_id in selected_id_set
        },
        "selection_hash": hashlib.sha256(
            "\n".join(selected_ids).encode("utf-8")
        ).hexdigest(),
    }
    return selected, metadata


def attach_manual_review_references(
    candidates: Sequence[SourceCandidate],
    english_references: Sequence[Any] | pd.Series,
    selection_metadata: dict[str, Any],
    config: SelectionConfig,
) -> list[dict[str, Any]]:
    """Join opaque English references after source-only selection has finished."""

    selected_for_terms = selection_metadata["selected_for_terms"]
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.source_row >= len(english_references):
            raise ValueError("English reference row count does not match Korean sources")
        raw_reference = english_references.iloc[candidate.source_row] if isinstance(
            english_references, pd.Series
        ) else english_references[candidate.source_row]
        reference = None if raw_reference is None or pd.isna(raw_reference) else str(
            raw_reference
        )
        records.append(
            {
                "source_record_id": candidate.source_record_id,
                "source_row": candidate.source_row,
                "korean": candidate.korean,
                "english_reference": reference,
                "english_reference_usage": ENGLISH_REFERENCE_POLICY,
                "hit_terms": list(candidate.hit_terms),
                "selected_for_terms": selected_for_terms[candidate.source_record_id],
                "source_only_selection_reason": {
                    "literal_fixed_lexicon_match": True,
                    "korean_char_length": candidate.korean_char_length,
                    "accepted_korean_char_range": [
                        config.min_korean_chars,
                        config.max_korean_chars,
                    ],
                    "hangul_present": True,
                    "line_break_present": False,
                    "sentence_boundary_count": candidate.sentence_boundary_count,
                    "max_sentence_boundaries": config.max_sentence_boundaries,
                    "normalized_source_unique": True,
                    "exact_pair_unique_by_source_uniqueness": True,
                    "sampling_algorithm": ALGORITHM_VERSION,
                    "seed": config.seed,
                },
                "review_status": "unreviewed",
            }
        )
    return records


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    path.write_text(text, encoding="utf-8")


def build_candidate_pool(
    parquet_path: Path,
    glossary_path: Path,
    output_path: Path,
    summary_path: Path,
    config: SelectionConfig,
) -> dict[str, Any]:
    if not parquet_path.is_file():
        raise FileNotFoundError(f"parquet does not exist: {parquet_path}")
    source_terms, source_term_column = load_fixed_source_lexicon(glossary_path)

    # Selection is completed while only the Korean parquet column is loaded.
    source_frame = pd.read_parquet(parquet_path, columns=["korean"])
    candidates, selection = select_source_candidates(
        source_frame["korean"], source_terms, config
    )

    # The reference column is loaded only after candidate IDs are immutable.
    reference_frame = pd.read_parquet(parquet_path, columns=["english"])
    if len(reference_frame) != len(source_frame):
        raise ValueError("Korean and English parquet columns have different row counts")
    records = attach_manual_review_references(
        candidates, reference_frame["english"], selection, config
    )
    _write_jsonl(output_path, records)

    shortage_terms = [
        {
            "term": term,
            "shortfall": stats["shortfall"],
            "sampled_for_term": stats["sampled_for_term"],
            "requested_review_pool_rows": stats["requested_review_pool_rows"],
            "status": stats["status"],
        }
        for term, stats in selection["per_term"].items()
        if stats["shortfall"] > 0
    ]
    selection_contract = {
        "algorithm_version": ALGORITHM_VERSION,
        "seed": config.seed,
        "per_term_review_pool_target": config.per_term_limit,
        "korean_character_range_inclusive": [
            config.min_korean_chars,
            config.max_korean_chars,
        ],
        "max_sentence_boundaries": config.max_sentence_boundaries,
        "normalization": "NFKC, trim, collapse whitespace",
        "literal_match": "fixed pre-review Korean glossary source terms only",
        "term_matcher": (
            "translation_qa.retrieval.contains_source_term; Korean particles and "
            "light-verb inflections allowed, longer lexical compounds rejected"
        ),
        "deduplication": (
            "keep first row for each normalized Korean source; this also guarantees "
            "that no exact source-reference pair can repeat without inspecting English"
        ),
        "fields_used_for_selection": [
            "korean",
            "original parquet row offset",
            "fixed glossary source_term lexicon",
        ],
        "reference_used_for_selection": False,
        "reference_used_for_term_extraction": False,
    }
    config_hash = _json_sha256(
        {
            "selection_contract": selection_contract,
            "source_terms": source_terms,
            "dataset_id": config.dataset_id,
            "dataset_revision": config.dataset_revision,
            "shard_id": config.shard_id,
        }
    )
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "UNREVIEWED_REVIEW_POOL_NOT_FINAL_DATASET",
        "final_dataset_frozen": False,
        "benchmark_run": False,
        "quality_warning": (
            "Source-only eligibility is a reproducible screening heuristic, not a "
            "translation-quality or source-reference alignment verdict."
        ),
        "english_reference_policy": ENGLISH_REFERENCE_POLICY,
        "dataset": {
            "dataset_id": config.dataset_id,
            "dataset_revision": config.dataset_revision,
            "shard_id": config.shard_id,
            "local_parquet": _portable_artifact_path(parquet_path),
            "local_path_redacted": True,
            "parquet_sha256": _file_sha256(parquet_path),
            "row_count": len(source_frame),
            "source_record_id_pattern": (
                f"{config.dataset_id}@{config.dataset_revision}:"
                f"{config.shard_id}:<zero-padded-row-offset>"
            ),
        },
        "glossary_lexicon": {
            "path": _portable_artifact_path(glossary_path),
            "sha256": _file_sha256(glossary_path),
            "source_term_column": source_term_column,
            "source_terms": source_terms,
            "only_source_term_column_loaded_for_lexicon": True,
        },
        "selection_contract": selection_contract,
        "selection_config_hash": config_hash,
        "counts": {
            "rows_with_any_literal_term": selection["rows_with_any_literal_term"],
            "source_only_eligible_unique_rows": selection[
                "eligible_unique_source_rows"
            ],
            "output_unique_rows": selection["selected_unique_rows"],
            "output_multi_term_rows": selection["selected_multi_term_rows"],
        },
        "filter_diagnostics_non_exclusive": selection[
            "filter_diagnostics_non_exclusive"
        ],
        "exact_pair_deduplication": {
            "english_inspected": False,
            "remaining_possible_exact_pair_duplicates": 0,
            "reason": (
                "Every emitted normalized Korean source is unique, so two emitted "
                "rows cannot have the same (Korean source, English reference) pair."
            ),
        },
        "per_term": selection["per_term"],
        "shortage_terms": shortage_terms,
        "candidate_output": {
            "path": _portable_artifact_path(output_path),
            "row_count": len(records),
            "sha256": _file_sha256(output_path),
            "ordered_source_id_sha256": selection["selection_hash"],
            "review_status": "unreviewed",
        },
        "next_required_gate": (
            "Bilingual manual alignment and domain-sense review; do not benchmark or "
            "freeze the final 30-50 set from this file automatically."
        ),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an offline source-only lemon-mint manual-review pool."
    )
    parser.add_argument("--parquet", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--glossary", type=Path, default=DEFAULT_GLOSSARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--dataset-revision")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-term-limit", type=int, default=30)
    parser.add_argument("--min-korean-chars", type=int, default=20)
    parser.add_argument("--max-korean-chars", type=int, default=180)
    parser.add_argument("--max-sentence-boundaries", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        parquet_path = args.parquet or locate_cached_lemon_parquet(args.cache_root)
        revision = args.dataset_revision or infer_dataset_revision(parquet_path)
        config = SelectionConfig(
            seed=args.seed,
            per_term_limit=args.per_term_limit,
            min_korean_chars=args.min_korean_chars,
            max_korean_chars=args.max_korean_chars,
            max_sentence_boundaries=args.max_sentence_boundaries,
            dataset_id=args.dataset_id,
            dataset_revision=revision,
            shard_id=parquet_path.stem,
        )
        summary = build_candidate_pool(
            parquet_path,
            args.glossary,
            args.output,
            args.summary_output,
            config,
        )
    except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")

    print(
        json.dumps(
            {
                "status": summary["status"],
                "candidate_output": summary["candidate_output"]["path"],
                "summary_output": str(args.summary_output.resolve()),
                "candidate_count": summary["counts"]["output_unique_rows"],
                "shortage_terms": summary["shortage_terms"],
                "benchmark_run": False,
                "final_dataset_frozen": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
