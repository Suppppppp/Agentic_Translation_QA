#!/usr/bin/env python3
"""Compare five RAG term-injection paths with one translation backend.

The default fake backend validates experiment wiring without ML packages or
model downloads.  ``--real-model`` constructs one lazy Marian backend and
reuses it for baseline, constrained decoding, and all source augmentations.
Pilot references and expected answers are intentionally never passed to the
translator.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from translation_qa.retrieval import ExactGlossaryRetriever, load_glossary_csv
from translation_qa.schemas import (
    RetrievalHit,
    RetrievalQuery,
    TermConstraint,
    TranslationCandidate,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GLOSSARY = PROJECT_ROOT / "data" / "glossary_pilot.csv"
DEFAULT_PILOT = PROJECT_ROOT / "data" / "pilot_v1.jsonl"
DEFAULT_MODEL_ID = "Helsinki-NLP/opus-mt-ko-en"

METHODS = (
    "baseline",
    "lexical_constraints",
    "mixed_english",
    "parenthetical",
    "quoted_marker",
)


@dataclass(frozen=True, slots=True)
class PilotCase:
    """Only fields that are safe to expose to the runtime pipeline."""

    case_id: str
    source_text: str
    domain: str | None = None


class TranslatorLike(Protocol):
    @property
    def model_id(self) -> str: ...

    def translate(
        self,
        source_text: str,
        constraints: Sequence[TermConstraint] | None = None,
    ) -> TranslationCandidate: ...


class FakeMarianTranslator:
    """Offline call-shape simulator; its strings are not translation results."""

    @property
    def model_id(self) -> str:
        return "fake-marian-injection-spike"

    def translate(
        self,
        source_text: str,
        constraints: Sequence[TermConstraint] | None = None,
    ) -> TranslationCandidate:
        if constraints:
            exposed = " | ".join(constraint.target_term for constraint in constraints)
            text = f"FAKE OUTPUT: {exposed}"
        else:
            # Echoing makes source-augmentation propagation observable.  It is
            # deliberately labelled fake and must not be scored as translation.
            text = f"FAKE OUTPUT: {source_text}"
        return TranslationCandidate(text=text, model_id=self.model_id)


def load_pilot_cases(path: str | Path) -> list[PilotCase]:
    """Load runtime fields only, discarding reference and gold annotations."""

    cases: list[PilotCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                raw = json.loads(raw_line)
                case_id = str(raw["case_id"]).strip()
                source_text = str(raw["source_text"]).strip()
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid pilot JSONL record on line {line_number}") from exc
            if not case_id or not source_text:
                raise ValueError(f"empty case_id or source_text on line {line_number}")
            raw_domain = raw.get("domain")
            domain = str(raw_domain).strip() if raw_domain is not None else None
            cases.append(
                PilotCase(
                    case_id=case_id,
                    source_text=source_text,
                    domain=domain or None,
                )
            )
    return cases


def _constraints(hits: Sequence[RetrievalHit]) -> list[TermConstraint]:
    return [
        TermConstraint(
            source_term=hit.source_term,
            target_term=hit.target_term,
            retrieval_hit_id=hit.term_id,
            target_variants=hit.accepted_target_variants,
        )
        for hit in hits
    ]


def _replacement_map(hits: Sequence[RetrievalHit]) -> dict[str, RetrievalHit]:
    mapping: dict[str, RetrievalHit] = {}
    for hit in hits:
        mapping.setdefault(hit.source_term, hit)
    return mapping


def _replace_terms(
    source_text: str,
    hits: Sequence[RetrievalHit],
    formatter: Callable[[str, RetrievalHit], str],
) -> str:
    mapping = _replacement_map(hits)
    if not mapping:
        return source_text
    alternatives = sorted(mapping, key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(term) for term in alternatives))
    return pattern.sub(lambda match: formatter(match.group(0), mapping[match.group(0)]), source_text)


def mixed_english_source(source_text: str, hits: Sequence[RetrievalHit]) -> str:
    """Replace each Korean surface term with its preferred English target."""

    return _replace_terms(source_text, hits, lambda _surface, hit: hit.target_term)


def parenthetical_source(source_text: str, hits: Sequence[RetrievalHit]) -> str:
    """Keep the Korean term and place its English target in parentheses."""

    return _replace_terms(
        source_text,
        hits,
        lambda surface, hit: f"{surface} ({hit.target_term})",
    )


def quoted_marker_source(source_text: str, hits: Sequence[RetrievalHit]) -> str:
    """Prefix explicit quoted term mappings while preserving the original."""

    if not hits:
        return source_text
    markers = " ".join(
        f'[TERM "{hit.source_term}" = "{hit.target_term}"]' for hit in hits
    )
    return f"{markers}\n{source_text}"


def _normalise_candidate(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def _contains_variant(candidate: str, variant: str) -> bool:
    candidate = _normalise_candidate(candidate)
    variant = _normalise_candidate(variant).strip()
    if not variant:
        return False
    if variant[0].isalnum() and variant[-1].isalnum():
        return re.search(rf"(?<!\w){re.escape(variant)}(?!\w)", candidate) is not None
    return variant in candidate


def _term_hit_report(
    candidate: str | None,
    hits: Sequence[RetrievalHit],
) -> tuple[list[dict[str, Any]], bool | None]:
    report: list[dict[str, Any]] = []
    for hit in hits:
        variants = list(
            dict.fromkeys([hit.target_term, *hit.accepted_target_variants])
        )
        matched = (
            next(
                (variant for variant in variants if _contains_variant(candidate, variant)),
                None,
            )
            if candidate is not None
            else None
        )
        report.append(
            {
                "term_id": hit.term_id,
                "source_term": hit.source_term,
                "preferred_target": hit.target_term,
                "accepted_targets": variants,
                "hit": matched is not None,
                "matched_variant": matched,
            }
        )
    return report, (all(item["hit"] for item in report) if report else None)


def detect_degeneracy(source_text: str, candidate: str) -> list[str]:
    """Detect structural failures without making a semantic quality judgment."""

    reasons: list[str] = []
    if len(candidate) > max(120, len(source_text) * 4):
        reasons.append("excessive_length")
    if len(re.findall(r"\([^)]{0,40}\)", candidate)) >= 3:
        reasons.append("repeated_parenthetical")

    tokens = candidate.casefold().split()
    if len(tokens) >= 12:
        trigrams = [tuple(tokens[index : index + 3]) for index in range(len(tokens) - 2)]
        if trigrams:
            highest_count = max(trigrams.count(value) for value in set(trigrams))
            if highest_count >= 4 and highest_count / len(trigrams) >= 0.2:
                reasons.append("repeated_trigram")
    return reasons


def _run_method(
    *,
    method: str,
    translator: TranslatorLike,
    original_source: str,
    input_text: str,
    hits: Sequence[RetrievalHit],
    constraints: Sequence[TermConstraint] | None = None,
) -> dict[str, Any]:
    candidate_text: str | None = None
    failure: dict[str, str] | None = None
    degeneracy_reasons: list[str] = []
    try:
        candidate = translator.translate(input_text, constraints=constraints)
        candidate_text = candidate.text
        degeneracy_reasons = detect_degeneracy(original_source, candidate_text)
    except Exception as exc:  # each experimental arm must remain inspectable
        failure = {"type": type(exc).__name__, "message": str(exc)}
        if "degenerate" in str(exc).casefold():
            degeneracy_reasons.append("backend_rejected_degenerate_candidate")

    term_hits, all_terms_hit = _term_hit_report(candidate_text, hits)
    if failure is not None:
        status = "failure"
    elif degeneracy_reasons:
        status = "degenerate"
    else:
        status = "ok"
    return {
        "method": method,
        "input_text": input_text,
        "constraints": [constraint.model_dump(mode="json") for constraint in constraints or ()],
        "candidate": candidate_text,
        "term_hits": term_hits,
        "all_terms_hit": all_terms_hit,
        "degenerate": bool(degeneracy_reasons),
        "degeneracy_reasons": degeneracy_reasons,
        "failure": failure,
        "status": status,
    }


def compare_injections(
    cases: Sequence[PilotCase],
    glossary_path: str | Path,
    translator: TranslatorLike,
    *,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Compare all injection arms while sharing one translator instance."""

    entries = load_glossary_csv(glossary_path)
    retriever = ExactGlossaryRetriever(entries)
    results: list[dict[str, Any]] = []

    for case in cases:
        hits = retriever.retrieve(
            RetrievalQuery(
                source_text=case.source_text,
                domain=case.domain,
                top_k=top_k,
            )
        )
        constraints = _constraints(hits)
        inputs = {
            "baseline": case.source_text,
            "lexical_constraints": case.source_text,
            "mixed_english": mixed_english_source(case.source_text, hits),
            "parenthetical": parenthetical_source(case.source_text, hits),
            "quoted_marker": quoted_marker_source(case.source_text, hits),
        }
        arms = [
            _run_method(
                method=method,
                translator=translator,
                original_source=case.source_text,
                input_text=inputs[method],
                hits=hits,
                constraints=constraints if method == "lexical_constraints" else None,
            )
            for method in METHODS
        ]
        results.append(
            {
                "case_id": case.case_id,
                "source_text": case.source_text,
                "domain": case.domain,
                "retrieval_hits": [hit.model_dump(mode="json") for hit in hits],
                "arms": arms,
            }
        )
    return results


def _print_text(results: Sequence[dict[str, Any]]) -> None:
    for result in results:
        print(f"[{result['case_id']}] {result['source_text']}")
        hit_ids = [hit["term_id"] for hit in result["retrieval_hits"]]
        print(f"  terms: {', '.join(hit_ids) if hit_ids else 'no match'}")
        for arm in result["arms"]:
            if arm["failure"]:
                detail = f"{arm['failure']['type']}: {arm['failure']['message']}"
            else:
                detail = arm["candidate"]
            print(
                f"  {arm['method']:19} status={arm['status']:<10} "
                f"terms={arm['all_terms_hit']} candidate={detail}"
            )
        print()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--glossary", type=Path, default=DEFAULT_GLOSSARY)
    parser.add_argument("--case-id", action="append", help="Select one or more pilot cases.")
    parser.add_argument("--limit", type=int, help="Limit cases after case-id filtering.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument(
        "--real-model",
        action="store_true",
        help="Use one lazy Marian model instead of the offline fake backend.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--num-beams", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser


def _select_cases(args: argparse.Namespace) -> list[PilotCase]:
    cases = load_pilot_cases(args.pilot)
    if args.case_id:
        requested = set(args.case_id)
        cases = [case for case in cases if case.case_id in requested]
        missing = requested - {case.case_id for case in cases}
        if missing:
            raise ValueError(f"unknown pilot case ID(s): {', '.join(sorted(missing))}")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        cases = cases[: args.limit]
    if not cases:
        raise ValueError("no pilot cases selected")
    return cases


def _real_translator(args: argparse.Namespace) -> TranslatorLike:
    # Importing and constructing Marian here keeps the default CLI path fully
    # independent from optional ML packages.  Marian itself loads on first call.
    from translation_qa.translator import MarianTranslator

    return MarianTranslator(
        args.model_id,
        device=args.device,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.top_k <= 20:
        parser.error("--top-k must be between 1 and 20")
    if args.num_beams < 1:
        parser.error("--num-beams must be at least 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")

    try:
        cases = _select_cases(args)
        translator = _real_translator(args) if args.real_model else FakeMarianTranslator()
        results = compare_injections(
            cases,
            args.glossary,
            translator,
            top_k=args.top_k,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.format == "json":
        payload = {
            "config": {
                "pilot": str(args.pilot),
                "glossary": str(args.glossary),
                "backend": "marian" if args.real_model else "fake",
                "model_id": translator.model_id,
                "top_k": args.top_k,
                "num_beams": args.num_beams if args.real_model else None,
                "max_new_tokens": args.max_new_tokens if args.real_model else None,
            },
            "warning": (
                None
                if args.real_model
                else "Fake outputs verify wiring only and are not translation-quality evidence."
            ),
            "cases": results,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        if not args.real_model:
            print("WARNING: fake outputs validate wiring only; they are not translations.\n")
        _print_text(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
