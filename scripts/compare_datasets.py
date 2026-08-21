#!/usr/bin/env python3
"""Offline EDA for the two candidate Korean-English parallel datasets.

This script reads only explicitly supplied files or already-downloaded Hugging
Face snapshots. Its anomaly rules and keyword matches are screening heuristics;
they do not establish that an English reference is a correct translation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


# Sandboxed/offline runs may not be able to create ``~/.matplotlib``. Keep the
# font cache in the system temporary directory while respecting an explicit
# caller configuration.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "translation-qa-matplotlib")
)


SOFTWARE_KEYWORDS: tuple[str, ...] = (
    "서버",
    "소프트웨어",
    "배포",
    "캐시",
    "장애",
    "부하 분산",
    "지속적 통합",
    "회귀 테스트",
    "롤백",
    "메모리 누수",
    "접근 제어",
    "벡터 검색",
    "데이터베이스",
    "네트워크",
    "버전",
)

KEYWORD_PLOT_LABELS: dict[str, str] = {
    "서버": "server",
    "소프트웨어": "software",
    "배포": "deployment",
    "캐시": "cache",
    "장애": "failure",
    "부하 분산": "load balancing",
    "지속적 통합": "continuous integration",
    "회귀 테스트": "regression testing",
    "롤백": "rollback",
    "메모리 누수": "memory leak",
    "접근 제어": "access control",
    "벡터 검색": "vector search",
    "데이터베이스": "database",
    "네트워크": "network",
    "버전": "version",
}

MOO_CACHE_NAME = "datasets--Moo--korean-parallel-corpora"
LEMON_CACHE_NAME = "datasets--lemon-mint--korean_parallel_sentences_v1.1"


def _normalised(series: pd.Series) -> pd.Series:
    return series.fillna("").astype("string").str.normalize("NFKC").str.strip()


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator * 100.0


def _number(value: Any) -> int | float | None:
    if value is None or pd.isna(value):
        return None
    numeric = float(value)
    return int(numeric) if numeric.is_integer() else numeric


def _distribution(series: pd.Series) -> dict[str, int | float | None]:
    values = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if values.empty:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "p05": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(values.size),
        "min": _number(values.min()),
        "mean": _number(values.mean()),
        "median": _number(values.median()),
        "p05": _number(values.quantile(0.05)),
        "p95": _number(values.quantile(0.95)),
        "p99": _number(values.quantile(0.99)),
        "max": _number(values.max()),
    }


def _schema(frame: pd.DataFrame) -> dict[str, str]:
    return {str(column): str(dtype) for column, dtype in frame.dtypes.items()}


def locate_cached_moo_dir(cache_root: Path | None = None) -> Path:
    root = cache_root or Path.home() / ".cache" / "huggingface" / "hub"
    snapshots = root / MOO_CACHE_NAME / "snapshots"
    for candidate in sorted(snapshots.glob("*"), reverse=True):
        if all((candidate / f"{split}.csv").is_file() for split in ("train", "dev", "test")):
            return candidate
    raise FileNotFoundError(
        "no complete local Moo/korean-parallel-corpora snapshot was found"
    )


def locate_cached_lemon_parquet(cache_root: Path | None = None) -> Path:
    root = cache_root or Path.home() / ".cache" / "huggingface" / "hub"
    snapshots = root / LEMON_CACHE_NAME / "snapshots"
    for snapshot in sorted(snapshots.glob("*"), reverse=True):
        candidates = sorted((snapshot / "data").glob("*.parquet"))
        if candidates:
            if len(candidates) != 1:
                raise ValueError(
                    "the offline EDA currently expects one lemon-mint parquet shard"
                )
            return candidates[0]
    raise FileNotFoundError(
        "no local lemon-mint/korean_parallel_sentences_v1.1 parquet was found"
    )


def load_moo(moo_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    schemas: dict[str, dict[str, str]] = {}
    rows_by_split: dict[str, int] = {}
    source_files: list[str] = []
    for split in ("train", "dev", "test"):
        path = moo_dir / f"{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing Moo split: {path}")
        frame = pd.read_csv(path, dtype={"ko": "string", "en": "string"})
        required = {"ko", "en"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{path.name} must contain ko and en columns")
        schemas[split] = _schema(frame)
        rows_by_split[split] = int(len(frame))
        source_files.append(path.name)
        selected = frame.loc[:, ["ko", "en"]].copy()
        selected["split"] = split
        selected["source_row"] = range(len(selected))
        frames.append(selected)
    combined = pd.concat(frames, ignore_index=True)
    return combined, {
        "source_files": source_files,
        "schema_by_split": schemas,
        "rows_by_split": rows_by_split,
        "source_columns": {"korean": "ko", "english": "en"},
    }


def load_lemon(parquet_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not parquet_path.is_file():
        raise FileNotFoundError(f"missing lemon-mint parquet: {parquet_path}")
    raw = pd.read_parquet(parquet_path, columns=["korean", "english"])
    required = {"korean", "english"}
    if not required.issubset(raw.columns):
        raise ValueError("lemon-mint parquet must contain korean and english columns")
    metadata = {
        "source_files": [parquet_path.name],
        "schema_by_split": {"train": _schema(raw)},
        "rows_by_split": {"train": int(len(raw))},
        "source_columns": {"korean": "korean", "english": "english"},
    }
    frame = raw.rename(columns={"korean": "ko", "english": "en"}).copy()
    frame["ko"] = frame["ko"].astype("string")
    frame["en"] = frame["en"].astype("string")
    frame["split"] = "train"
    frame["source_row"] = range(len(frame))
    return frame, metadata


def analyze_dataset(
    dataset_id: str,
    frame: pd.DataFrame,
    source_metadata: dict[str, Any],
    *,
    example_limit: int = 12,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return JSON-safe EDA statistics and an enriched plotting frame."""

    required = {"ko", "en", "split", "source_row"}
    if not required.issubset(frame.columns):
        missing = sorted(required - set(frame.columns))
        raise ValueError(f"analysis frame is missing columns: {', '.join(missing)}")
    if example_limit < 1:
        raise ValueError("example_limit must be at least one")

    working = frame.loc[:, ["ko", "en", "split", "source_row"]].copy()
    row_count = int(len(working))
    ko_null = working["ko"].isna()
    en_null = working["en"].isna()
    ko = _normalised(working["ko"])
    en = _normalised(working["en"])
    ko_blank = ko.eq("")
    en_blank = en.eq("")

    ko_length = ko.str.len().astype("int64")
    en_length = en.str.len().astype("int64")
    valid_ratio = ko_length.gt(0) & en_length.gt(0)
    ratio = pd.Series(float("nan"), index=working.index, dtype="float64")
    ratio.loc[valid_ratio] = (
        en_length.loc[valid_ratio].astype(float)
        / ko_length.loc[valid_ratio].astype(float)
    )

    duplicate_pair = working.assign(ko_norm=ko, en_norm=en).duplicated(
        ["ko_norm", "en_norm"]
    )
    duplicate_ko = ko.duplicated()
    duplicate_en = en.duplicated()

    anomaly_masks: dict[str, pd.Series] = {
        "null_or_blank_side": ko_null | en_null | ko_blank | en_blank,
        "no_hangul_on_korean_side": ~ko.str.contains(r"[가-힣]", regex=True),
        "no_latin_on_english_side": ~en.str.contains(r"[A-Za-z]", regex=True),
        "very_short_side": ko_length.lt(2) | en_length.lt(2),
        "very_long_side_over_500_chars": ko_length.gt(500) | en_length.gt(500),
        "extreme_en_to_ko_char_ratio": valid_ratio & (ratio.lt(0.2) | ratio.gt(5.0)),
        "identical_cross_language_text": ko.str.casefold().eq(en.str.casefold()),
        "contains_url_or_markup": ko.str.contains(
            r"https?://|<[^>]+>", regex=True, case=False
        )
        | en.str.contains(r"https?://|<[^>]+>", regex=True, case=False),
    }
    anomaly_count = sum(mask.astype("int16") for mask in anomaly_masks.values())

    keyword_masks: dict[str, pd.Series] = {}
    keyword_counts: dict[str, int] = {}
    keyword_counts_by_split: dict[str, dict[str, int]] = {}
    for keyword in SOFTWARE_KEYWORDS:
        mask = ko.str.contains(re.escape(keyword), regex=True, na=False)
        keyword_masks[keyword] = mask
        keyword_counts[keyword] = int(mask.sum())
        keyword_counts_by_split[keyword] = {
            str(split): int(mask[working["split"].eq(split)].sum())
            for split in working["split"].drop_duplicates().tolist()
        }

    hit_count = sum(mask.astype("int16") for mask in keyword_masks.values())
    any_keyword = hit_count.gt(0)
    screened = (
        any_keyword
        & ~anomaly_masks["null_or_blank_side"]
        & ~anomaly_masks["no_hangul_on_korean_side"]
        & ~anomaly_masks["no_latin_on_english_side"]
        & ko_length.between(8, 400)
        & en_length.between(8, 500)
        & ratio.between(0.25, 4.0)
        & ~duplicate_pair
    )

    enriched = working.assign(
        ko_norm=ko,
        en_norm=en,
        ko_length=ko_length,
        en_length=en_length,
        en_to_ko_char_ratio=ratio,
        keyword_hit_count=hit_count,
        anomaly_count=anomaly_count,
        screened_candidate=screened,
    )

    ranked = enriched.loc[screened].copy()
    ranked["ratio_distance"] = (ranked["en_to_ko_char_ratio"] - 1.25).abs()
    ranked = ranked.sort_values(
        ["keyword_hit_count", "anomaly_count", "ratio_distance", "source_row"],
        ascending=[False, True, True, True],
        kind="stable",
    ).head(example_limit)

    examples: list[dict[str, Any]] = []
    for index, row in ranked.iterrows():
        source = str(row["ko_norm"])
        examples.append(
            {
                "split": str(row["split"]),
                "source_row": int(row["source_row"]),
                "source_ko": source,
                "reference_en": str(row["en_norm"]),
                "keyword_hits": [
                    keyword for keyword in SOFTWARE_KEYWORDS if keyword in source
                ],
                "ko_chars": int(row["ko_length"]),
                "en_chars": int(row["en_length"]),
                "en_to_ko_char_ratio": round(
                    float(row["en_to_ko_char_ratio"]), 4
                ),
                "review_status": "UNREVIEWED_HEURISTIC_CANDIDATE",
            }
        )

    anomaly_counts = {
        name: int(mask.sum()) for name, mask in anomaly_masks.items()
    }
    any_anomaly = anomaly_count.gt(0)
    distinct_keyword_coverage = sum(count > 0 for count in keyword_counts.values())

    summary = {
        "dataset_id": dataset_id,
        **source_metadata,
        "row_count": row_count,
        "null_and_blank": {
            "korean_null": int(ko_null.sum()),
            "english_null": int(en_null.sum()),
            "either_null": int((ko_null | en_null).sum()),
            "korean_blank_after_normalization": int(ko_blank.sum()),
            "english_blank_after_normalization": int(en_blank.sum()),
        },
        "duplicates_beyond_first": {
            "parallel_pair": int(duplicate_pair.sum()),
            "korean_source": int(duplicate_ko.sum()),
            "english_reference": int(duplicate_en.sum()),
            "parallel_pair_rate_pct": _rate(int(duplicate_pair.sum()), row_count),
        },
        "lengths": {
            "korean_chars": _distribution(ko_length.mask(ko_blank)),
            "english_chars": _distribution(en_length.mask(en_blank)),
            "english_to_korean_char_ratio": _distribution(ratio),
        },
        "heuristic_anomalies": {
            "counts": anomaly_counts,
            "rows_with_any_flag": int(any_anomaly.sum()),
            "rows_with_any_flag_rate_pct": _rate(int(any_anomaly.sum()), row_count),
            "warning": (
                "These flags identify records for review; they do not prove that a "
                "parallel pair is mistranslated or unusable."
            ),
        },
        "software_keyword_screen": {
            "keywords": list(SOFTWARE_KEYWORDS),
            "hit_counts": keyword_counts,
            "hit_counts_by_split": keyword_counts_by_split,
            "rows_with_any_keyword": int(any_keyword.sum()),
            "rows_with_multiple_keywords": int(hit_count.gt(1).sum()),
            "distinct_keywords_with_hits": distinct_keyword_coverage,
            "screened_candidate_count": int(screened.sum()),
            "screened_candidate_rate_pct": _rate(int(screened.sum()), row_count),
            "candidate_examples": examples,
            "warning": (
                "Literal Korean keyword matching is recall-oriented and cannot confirm "
                "software-domain relevance or English reference quality."
            ),
        },
    }
    return summary, enriched


def build_provisional_recommendation(
    dataset_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    def rank(item: tuple[str, dict[str, Any]]) -> tuple[int, int, float, float]:
        _, summary = item
        screen = summary["software_keyword_screen"]
        duplicates = summary["duplicates_beyond_first"]
        anomalies = summary["heuristic_anomalies"]
        return (
            int(screen["distinct_keywords_with_hits"]),
            int(screen["screened_candidate_count"]),
            -float(duplicates["parallel_pair_rate_pct"] or 0.0),
            -float(anomalies["rows_with_any_flag_rate_pct"] or 0.0),
        )

    ranked = sorted(dataset_summaries.items(), key=rank, reverse=True)
    selected_id, selected = ranked[0]
    screen = selected["software_keyword_screen"]
    ready_by_volume = (
        int(screen["screened_candidate_count"]) >= 50
        and int(screen["distinct_keywords_with_hits"]) >= 5
    )
    return {
        "provisional_dataset": selected_id if ready_by_volume else None,
        "selection_status": (
            "PROVISIONAL_FOR_MANUAL_REVIEW"
            if ready_by_volume
            else "INSUFFICIENT_HEURISTIC_CANDIDATES"
        ),
        "ranking_basis": [
            "number of distinct requested software keywords with at least one hit",
            "number of non-duplicate candidates passing basic length/script heuristics",
            "lower duplicate-pair rate",
            "lower basic anomaly-flag rate",
        ],
        "selected_evidence": {
            "screened_candidate_count": int(screen["screened_candidate_count"]),
            "distinct_keywords_with_hits": int(screen["distinct_keywords_with_hits"]),
        },
        "mandatory_hold_points": [
            "A bilingual reviewer must verify source-reference alignment and translation quality.",
            "Keyword hits must be checked for actual software-domain meaning and ambiguity.",
            "At least five glossary terms need enough manually approved examples for a balanced 30-50 sentence set.",
            "Near-duplicates and cross-dataset overlap require a second pass before freezing the final set.",
            "Licensing, attribution, provenance, and dataset-card limitations remain part of the final selection decision.",
        ],
        "not_a_quality_verdict": True,
    }


def _plot_lengths(
    frames: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    colors = {"moo": "#3569b8", "lemon_mint": "#dc7b2f"}
    for dataset_id, frame in frames.items():
        for axis, column, title in (
            (axes[0], "ko_length", "Korean character length"),
            (axes[1], "en_length", "English character length"),
        ):
            values = frame[column].dropna().astype(float)
            upper = max(1.0, float(values.quantile(0.99)))
            clipped = values.clip(upper=upper)
            axis.hist(
                clipped,
                bins=50,
                density=True,
                histtype="step",
                linewidth=1.8,
                label=f"{dataset_id} (p99 clipped)",
                color=colors.get(dataset_id),
            )
            axis.set_title(title)
            axis.set_xlabel("characters")
            axis.set_ylabel("density")
            axis.grid(alpha=0.2)
    for axis in axes:
        axis.legend(fontsize=8)
    figure.suptitle("Parallel sentence length distributions (heuristic EDA)")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_keyword_hits(
    summaries: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = list(SOFTWARE_KEYWORDS)
    y = np.arange(len(labels))
    height = 0.36
    figure, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
    for offset, (dataset_id, summary) in zip(
        (-height / 2, height / 2), summaries.items(), strict=True
    ):
        row_count = int(summary["row_count"])
        counts = summary["software_keyword_screen"]["hit_counts"]
        rates = [float(counts[label]) / row_count * 100_000 for label in labels]
        axis.barh(y + offset, rates, height=height, label=dataset_id)
    axis.set_yticks(y, [KEYWORD_PLOT_LABELS[label] for label in labels])
    axis.invert_yaxis()
    axis.set_xlabel("literal keyword-hit rows per 100,000 pairs")
    axis.set_title("Software keyword screening density (not domain precision)")
    axis.grid(axis="x", alpha=0.2)
    axis.legend()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def run_comparison(
    moo_dir: Path,
    lemon_parquet: Path,
    output_dir: Path,
    *,
    example_limit: int = 12,
    create_plots: bool = True,
) -> dict[str, Any]:
    moo_frame, moo_metadata = load_moo(moo_dir)
    lemon_frame, lemon_metadata = load_lemon(lemon_parquet)
    moo_summary, moo_enriched = analyze_dataset(
        "moo", moo_frame, moo_metadata, example_limit=example_limit
    )
    lemon_summary, lemon_enriched = analyze_dataset(
        "lemon_mint", lemon_frame, lemon_metadata, example_limit=example_limit
    )
    summaries = {"moo": moo_summary, "lemon_mint": lemon_summary}
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_scope": "offline cached snapshots only",
        "methodology_warning": (
            "All anomaly flags, keyword hits, screening counts, examples, and the "
            "provisional ranking are heuristics. They do not automatically validate "
            "English reference translations."
        ),
        "datasets": summaries,
        "provisional_selection": build_provisional_recommendation(summaries),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "dataset_comparison.json"
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if create_plots:
        _plot_lengths(
            {"moo": moo_enriched, "lemon_mint": lemon_enriched},
            output_dir / "length_distributions.png",
        )
        _plot_keyword_hits(summaries, output_dir / "software_keyword_hits.png")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--moo-dir",
        type=Path,
        help="Offline Moo snapshot directory containing train/dev/test CSV files.",
    )
    parser.add_argument(
        "--lemon-parquet",
        type=Path,
        help="Offline lemon-mint train parquet path.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        help="Optional Hugging Face hub cache root used only for local discovery.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/eda"),
    )
    parser.add_argument("--example-limit", type=int, default=12)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.example_limit < 1 or args.example_limit > 50:
        parser.error("--example-limit must be between 1 and 50")
    try:
        moo_dir = args.moo_dir or locate_cached_moo_dir(args.cache_root)
        lemon_parquet = args.lemon_parquet or locate_cached_lemon_parquet(
            args.cache_root
        )
        payload = run_comparison(
            moo_dir,
            lemon_parquet,
            args.output_dir,
            example_limit=args.example_limit,
            create_plots=not args.no_plots,
        )
    except (ImportError, OSError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")

    recommendation = payload["provisional_selection"]
    print(
        json.dumps(
            {
                "summary_path": str(
                    (args.output_dir / "dataset_comparison.json").resolve()
                ),
                "provisional_selection": recommendation,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
