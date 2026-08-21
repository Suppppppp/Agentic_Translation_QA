#!/usr/bin/env python3
"""Compare exact, vector, and hybrid glossary retrieval on pilot inputs.

The default deterministic embedder is deliberately small and offline.  It is
useful for exercising ranking, filtering, and result logging, but its scores
are not evidence of semantic retrieval quality.  Pass ``--embedder
sentence-transformer`` for a real multilingual embedding experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from translation_qa.retrieval import (
    Embedder,
    ExactGlossaryRetriever,
    ExactFirstHybridGlossaryRetriever,
    GlossaryEntry,
    HybridGlossaryRetriever,
    SentenceTransformerEmbedder,
    VectorGlossaryRetriever,
    load_glossary_csv,
)
from translation_qa.schemas import RetrievalHit, RetrievalQuery


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GLOSSARY = PROJECT_ROOT / "data" / "glossary_pilot.csv"
DEFAULT_PILOT = PROJECT_ROOT / "data" / "pilot_v1.jsonl"
DEFAULT_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_TOKEN_PATTERN = re.compile(r"[가-힣]+|[a-z0-9]+")


@dataclass(frozen=True, slots=True)
class QueryCase:
    case_id: str
    source_text: str
    domain: str | None = None
    key_terms: tuple[str, ...] = ()


class DeterministicHashEmbedder:
    """Offline feature hasher for deterministic retrieval smoke tests.

    Hangul character n-grams let a glossary term share features with the same
    term plus a Korean particle in a sentence.  Stable BLAKE2 hashing avoids
    Python's process-randomized ``hash()`` and makes repeated CLI runs identical.
    """

    def __init__(self, *, dimensions: int = 1024, min_n: int = 2, max_n: int = 4) -> None:
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        if min_n < 1 or max_n < min_n:
            raise ValueError("n-gram bounds are invalid")
        self.dimensions = dimensions
        self.min_n = min_n
        self.max_n = max_n

    @staticmethod
    def _normalise(text: str) -> str:
        return unicodedata.normalize("NFKC", text).casefold()

    def _features(self, text: str) -> list[str]:
        tokens = _TOKEN_PATTERN.findall(self._normalise(text))
        # When source-side Korean exists, exclude English definition text so the
        # deterministic smoke score measures source-surface overlap only.
        hangul_tokens = [token for token in tokens if any("가" <= c <= "힣" for c in token)]
        selected = hangul_tokens or tokens
        features: list[str] = []
        for token in selected:
            features.append(f"token:{token}")
            for size in range(self.min_n, self.max_n + 1):
                features.extend(
                    f"char{size}:{token[index:index + size]}"
                    for index in range(max(0, len(token) - size + 1))
                )
        return features

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for feature in self._features(text):
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "big") % self.dimensions
                sign = 1.0 if digest[4] & 1 else -1.0
                vector[bucket] += sign
            vectors.append(vector)
        return vectors


def load_pilot_cases(path: str | Path) -> list[QueryCase]:
    """Load only runtime-safe fields; references and expected terms are ignored."""

    cases: list[QueryCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
                case_id = str(record["case_id"]).strip()
                source_text = str(record["source_text"]).strip()
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid pilot JSONL record on line {line_number}") from exc
            if not case_id or not source_text:
                raise ValueError(f"empty case_id or source_text on line {line_number}")
            raw_domain = record.get("domain")
            domain = str(raw_domain).strip() if raw_domain is not None else None
            cases.append(QueryCase(case_id=case_id, source_text=source_text, domain=domain or None))
    return cases


def compare_cases(
    entries: Sequence[GlossaryEntry],
    cases: Sequence[QueryCase],
    embedder: Embedder,
    *,
    top_k: int = 3,
    min_score: float = 0.12,
) -> list[dict[str, Any]]:
    """Run raw and safety-first retrieval methods with shared query settings."""

    exact = ExactGlossaryRetriever(entries)
    vector = VectorGlossaryRetriever(entries, embedder, min_score=min_score)
    hybrid = HybridGlossaryRetriever(exact, vector)
    safe_hybrid = ExactFirstHybridGlossaryRetriever(exact, vector)
    results: list[dict[str, Any]] = []

    for case in cases:
        query = RetrievalQuery(
            source_text=case.source_text,
            domain=case.domain,
            key_terms=list(case.key_terms),
            top_k=top_k,
        )
        results.append(
            {
                "case_id": case.case_id,
                "source_text": case.source_text,
                "domain": case.domain,
                "exact": exact.retrieve(query),
                "vector": vector.retrieve(query),
                "hybrid": hybrid.retrieve(query),
                "safe_hybrid": safe_hybrid.retrieve(query),
            }
        )
    return results


def _serialise_hit(hit: RetrievalHit) -> dict[str, Any]:
    return hit.model_dump(mode="json")


def serialise_results(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    serialised: list[dict[str, Any]] = []
    for result in results:
        serialised.append(
            {
                **{key: result[key] for key in ("case_id", "source_text", "domain")},
                **{
                    method: [_serialise_hit(hit) for hit in result[method]]
                    for method in ("exact", "vector", "hybrid", "safe_hybrid")
                },
            }
        )
    return serialised


def _print_text(results: Sequence[dict[str, Any]]) -> None:
    for result in results:
        print(f"[{result['case_id']}] {result['source_text']}")
        print(f"  domain: {result['domain'] or '-'}")
        for method in ("exact", "vector", "hybrid", "safe_hybrid"):
            hits: Sequence[RetrievalHit] = result[method]
            if not hits:
                print(f"  {method:11}: no match")
                continue
            rendered = ", ".join(
                f"{hit.term_id}({hit.score:.4f}, {hit.match_type.value})"
                for hit in hits
            )
            print(f"  {method:11}: {rendered}")
        print()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glossary", type=Path, default=DEFAULT_GLOSSARY)
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument(
        "--text",
        action="append",
        help="Compare an ad-hoc source sentence; repeat for multiple inputs.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        help="Limit pilot input to one or more case IDs.",
    )
    parser.add_argument(
        "--domain",
        help="Override the domain for every selected or ad-hoc input.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.12)
    parser.add_argument(
        "--embedder",
        choices=("deterministic", "sentence-transformer"),
        default="deterministic",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def _select_cases(args: argparse.Namespace) -> list[QueryCase]:
    if args.text:
        if args.case_id:
            raise ValueError("--case-id cannot be combined with --text")
        cases = [
            QueryCase(
                case_id=f"cli-{index:03d}",
                source_text=text,
                domain=args.domain,
            )
            for index, text in enumerate(args.text, start=1)
        ]
    else:
        cases = load_pilot_cases(args.pilot)
        if args.case_id:
            selected = set(args.case_id)
            cases = [case for case in cases if case.case_id in selected]
            missing = selected - {case.case_id for case in cases}
            if missing:
                raise ValueError(f"unknown pilot case ID(s): {', '.join(sorted(missing))}")
        if args.domain:
            cases = [
                QueryCase(
                    case_id=case.case_id,
                    source_text=case.source_text,
                    domain=args.domain,
                    key_terms=case.key_terms,
                )
                for case in cases
            ]
    if not cases:
        raise ValueError("no query cases selected")
    return cases


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.top_k <= 20:
        parser.error("--top-k must be between 1 and 20")
    if not -1.0 <= args.min_score <= 1.0:
        parser.error("--min-score must be between -1 and 1")

    try:
        entries = load_glossary_csv(args.glossary)
        cases = _select_cases(args)
        embedder: Embedder
        if args.embedder == "deterministic":
            embedder = DeterministicHashEmbedder()
        else:
            embedder = SentenceTransformerEmbedder(args.model_id)
        results = compare_cases(
            entries,
            cases,
            embedder,
            top_k=args.top_k,
            min_score=args.min_score,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.format == "json":
        payload = {
            "config": {
                "glossary": str(args.glossary),
                "pilot": None if args.text else str(args.pilot),
                "embedder": args.embedder,
                "model_id": args.model_id if args.embedder == "sentence-transformer" else None,
                "top_k": args.top_k,
                "min_score": args.min_score,
            },
            "cases": serialise_results(results),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _print_text(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
