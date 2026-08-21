"""Small, inspectable glossary retrievers with optional embedding support.

The exact and hybrid paths have no ML dependency.  Vector retrieval only
depends on the injected :class:`Embedder` protocol, which keeps tests and
non-ML API startup independent from sentence-transformers.
"""

from __future__ import annotations

import csv
import json
import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Protocol, runtime_checkable

from translation_qa.contracts import Retriever
from translation_qa.errors import ComponentExecutionError, ComponentUnavailableError
from translation_qa.schemas import RetrievalHit, RetrievalMatchType, RetrievalQuery


_WHITESPACE = re.compile(r"\s+")
_HANGUL_SYLLABLE = re.compile(r"[\uac00-\ud7a3]")
_KOREAN_BOUND_SUFFIXES = tuple(
    sorted(
        {
            # Common noun particles.  The suffix must end at a token boundary,
            # so a lexical compound such as ``배포판`` is not a hit for ``배포``.
            "은",
            "는",
            "이",
            "가",
            "을",
            "를",
            "과",
            "와",
            "의",
            "도",
            "만",
            "에",
            "에서",
            "에게",
            "에게서",
            "께",
            "께서",
            "로",
            "으로",
            "부터",
            "까지",
            "보다",
            "처럼",
            "조차",
            "마저",
            "밖에",
            "라도",
            "이나",
            "나",
            "이며",
            "이고",
            "입니다",
            "이다",
            "인",
            # Productive light-verb forms commonly attached to software nouns.
            "합니다",
            "했다",
            "했으며",
            "하였다",
            "한다",
            "하는",
            "하고",
            "하면",
            "하며",
            "하여",
            "하면서",
            "하도록",
            "하기",
            "하다",
            "해",
            "한",
            "할",
            "함",
            "됩니다",
            "되었다",
            "됐다",
            "된다",
            "되는",
            "되고",
            "되면",
            "되어",
            "되다",
            "된",
            "될",
            "됨",
            "돼",
            "시킨다",
            "시키는",
            "시키고",
            "시키다",
            "시킨",
            "시킬",
            "시켜",
        },
        key=len,
        reverse=True,
    )
)


def _normalise_text(value: str) -> str:
    """Normalise text for matching without changing the displayed glossary value."""

    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", value).casefold()).strip()


def contains_source_term(text: str, term: str) -> bool:
    """Match a source term without treating a longer Korean compound as a hit.

    Korean case particles and common light-verb endings are allowed directly
    after a glossary noun (for example ``배포를`` and ``배포한다``), but an
    unrelated compound such as ``배포판`` is rejected.  This is deliberately a
    conservative lexical matcher; uncertain paraphrases belong to vector
    retrieval rather than the exact arm.
    """

    normalized_text = _normalise_text(text)
    normalized_term = _normalise_text(term)
    if not normalized_term:
        return False

    search_from = 0
    while True:
        start = normalized_text.find(normalized_term, search_from)
        if start < 0:
            return False
        end = start + len(normalized_term)
        before_ok = start == 0 or not (
            normalized_text[start - 1].isalnum()
            or normalized_text[start - 1] == "_"
        )
        if not before_ok:
            search_from = start + 1
            continue

        if end == len(normalized_text):
            return True
        next_character = normalized_text[end]
        if not (next_character.isalnum() or next_character == "_"):
            return True

        # Only Korean glossary terms receive attached-particle handling.
        if _HANGUL_SYLLABLE.search(normalized_term) is not None:
            suffix_end = end
            while (
                suffix_end < len(normalized_text)
                and _HANGUL_SYLLABLE.fullmatch(normalized_text[suffix_end]) is not None
            ):
                suffix_end += 1
            attached = normalized_text[end:suffix_end]
            if attached in _KOREAN_BOUND_SUFFIXES:
                return True
        search_from = start + 1


def _required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    cleaned = _WHITESPACE.sub(" ", value).strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    return cleaned


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = _WHITESPACE.sub(" ", value).strip()
    return cleaned or None


def _text_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence of strings")
    cleaned = tuple(_required_text(value, field_name) for value in values)
    # Keep CSV order while removing repeated spelling variants.
    return tuple(dict.fromkeys(cleaned))


def _replacement_tuple(
    values: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    cleaned: list[tuple[str, str]] = []
    seen_sources: set[str] = set()
    for source, target in values:
        source_text = _required_text(source, "replacement rule source")
        target_text = _required_text(target, "replacement rule target")
        normalized_source = _normalise_text(source_text)
        if normalized_source in seen_sources:
            raise ValueError("replacement rule sources must be unique")
        seen_sources.add(normalized_source)
        cleaned.append((source_text, target_text))
    return tuple(cleaned)


@dataclass(frozen=True, slots=True)
class GlossaryEntry:
    """One independently sourced glossary record.

    ``accepted_variants`` and ``disallowed_variants`` describe target-language
    spellings for later terminology evaluation.  Retrieval itself matches the
    Korean ``source_term`` and optionally uses ``definition`` for embeddings.
    """

    term_id: str
    source_term: str
    target_term: str
    domain: str | None = None
    definition: str | None = None
    accepted_variants: tuple[str, ...] = ()
    disallowed_variants: tuple[str, ...] = ()
    replacement_rules: tuple[tuple[str, str], ...] = ()
    source: str | None = None
    glossary_version: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "term_id", _required_text(self.term_id, "term_id"))
        object.__setattr__(
            self, "source_term", _required_text(self.source_term, "source_term")
        )
        object.__setattr__(
            self, "target_term", _required_text(self.target_term, "target_term")
        )
        for field_name in (
            "domain",
            "definition",
            "source",
            "glossary_version",
            "notes",
        ):
            object.__setattr__(self, field_name, _optional_text(getattr(self, field_name)))
        object.__setattr__(
            self,
            "accepted_variants",
            _text_tuple(self.accepted_variants, "accepted_variants"),
        )
        object.__setattr__(
            self,
            "disallowed_variants",
            _text_tuple(self.disallowed_variants, "disallowed_variants"),
        )
        object.__setattr__(
            self,
            "replacement_rules",
            _replacement_tuple(self.replacement_rules),
        )

    @property
    def embedding_text(self) -> str:
        """Text used to build the source-side semantic index."""

        if self.definition is None:
            return self.source_term
        return f"{self.source_term}: {self.definition}"


@runtime_checkable
class Embedder(Protocol):
    """Minimal embedding contract; implementations may return lists or arrays."""

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class SentenceTransformerEmbedder:
    """Lazy sentence-transformers adapter.

    Importing this module or constructing the adapter does not import the ML
    package, allocate a model, or download weights.  That work starts on the
    first non-empty :meth:`embed` call.
    """

    def __init__(self, model_id: str, *, device: str | None = None) -> None:
        self.model_id = _required_text(model_id, "model_id")
        self.device = _optional_text(device)
        self._model: object | None = None
        self._model_lock = Lock()

    def _load_model(self) -> object:
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:  # pragma: no cover - depends on optional extra
                    raise ComponentUnavailableError(
                        "sentence-transformers is required for this embedder; "
                        "install the project with the 'ml' extra"
                    ) from exc
                kwargs = {"device": self.device} if self.device is not None else {}
                try:
                    self._model = SentenceTransformer(self.model_id, **kwargs)
                except Exception as exc:  # model hub/backend exceptions vary by version
                    raise ComponentUnavailableError(
                        f"could not load embedding model {self.model_id}: {exc}"
                    ) from exc
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load_model()
        try:
            encoded = model.encode(  # type: ignore[attr-defined]
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise ComponentExecutionError(f"embedding failed: {exc}") from exc
        return [[float(component) for component in vector] for vector in encoded]


def _validate_entries(entries: Sequence[GlossaryEntry]) -> tuple[GlossaryEntry, ...]:
    frozen = tuple(entries)
    identifiers = [entry.term_id for entry in frozen]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("glossary term_id values must be unique")
    return frozen


def _domain_matches(entry: GlossaryEntry, query: RetrievalQuery) -> bool:
    if query.domain is None or entry.domain is None:
        return True
    return _normalise_text(entry.domain) == _normalise_text(query.domain)


def _hit(
    entry: GlossaryEntry,
    *,
    match_type: RetrievalMatchType,
    score: float,
) -> RetrievalHit:
    return RetrievalHit(
        term_id=entry.term_id,
        source_term=entry.source_term,
        target_term=entry.target_term,
        domain=entry.domain,
        match_type=match_type,
        score=score,
        definition=entry.definition,
        accepted_target_variants=list(entry.accepted_variants),
        replacement_rules=dict(entry.replacement_rules),
    )


class ExactGlossaryRetriever(Retriever):
    """Return glossary terms literally present in source text or key terms."""

    def __init__(self, entries: Sequence[GlossaryEntry]) -> None:
        self.entries = _validate_entries(entries)

    def retrieve(self, query: RetrievalQuery) -> list[RetrievalHit]:
        source_text = _normalise_text(query.source_text)
        key_terms = tuple(_normalise_text(term) for term in query.key_terms)
        matches: list[tuple[GlossaryEntry, RetrievalHit]] = []

        for entry in self.entries:
            source_term = _normalise_text(entry.source_term)
            appears_in_source = contains_source_term(source_text, source_term)
            appears_in_key_terms = any(
                source_term == key_term or contains_source_term(key_term, source_term)
                for key_term in key_terms
            )
            # A literal occurrence in the authoritative source must not be lost
            # because a small Agent labeled the domain differently (for example
            # ``IT`` instead of the glossary's ``software``). Domain filtering is
            # retained for Agent-suggested key terms, which are weaker evidence.
            domain_match = _domain_matches(entry, query)
            if appears_in_source:
                has_matching_domain_entry = any(
                    _normalise_text(peer.source_term) == source_term
                    and _domain_matches(peer, query)
                    for peer in self.entries
                )
                if has_matching_domain_entry and not domain_match:
                    continue
            elif not domain_match:
                continue
            if appears_in_source or appears_in_key_terms:
                # A literal occurrence in the authoritative source sentence is
                # ranked just ahead of an analysis-supplied key term.
                score = 1.0 if appears_in_source else 0.99
                matches.append(
                    (
                        entry,
                        _hit(
                            entry,
                            match_type=RetrievalMatchType.EXACT,
                            score=score,
                        ),
                    )
                )

        matches.sort(
            key=lambda pair: (
                -pair[1].score,
                -len(_normalise_text(pair[0].source_term)),
                pair[0].term_id,
            )
        )
        return [hit for _, hit in matches[: query.top_k]]


def _coerce_vectors(
    vectors: Sequence[Sequence[float]],
    *,
    expected_count: int,
) -> tuple[tuple[float, ...], ...]:
    converted = tuple(tuple(float(value) for value in vector) for vector in vectors)
    if len(converted) != expected_count:
        raise ValueError(
            f"embedder returned {len(converted)} vectors for {expected_count} texts"
        )
    if not converted:
        return converted
    dimensions = len(converted[0])
    if dimensions == 0:
        raise ValueError("embedding vectors must not be empty")
    if any(len(vector) != dimensions for vector in converted):
        raise ValueError("embedding vectors must have a consistent dimension")
    if any(not math.isfinite(value) for vector in converted for value in vector):
        raise ValueError("embedding vectors must contain only finite numbers")
    return converted


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("query and glossary embeddings have different dimensions")
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


class VectorGlossaryRetriever(Retriever):
    """In-memory vector search using injected embeddings and pure Python cosine."""

    def __init__(
        self,
        entries: Sequence[GlossaryEntry],
        embedder: Embedder,
        *,
        min_score: float = 0.35,
    ) -> None:
        if not math.isfinite(min_score) or not -1.0 <= min_score <= 1.0:
            raise ValueError("min_score must be finite and between -1 and 1")
        self.entries = _validate_entries(entries)
        self.embedder = embedder
        self.min_score = min_score
        self._entry_vectors: tuple[tuple[float, ...], ...] | None = None
        self._index_lock = Lock()

    def _index(self) -> tuple[tuple[float, ...], ...]:
        if self._entry_vectors is not None:
            return self._entry_vectors
        with self._index_lock:
            if self._entry_vectors is None:
                raw_vectors = self.embedder.embed(
                    [entry.embedding_text for entry in self.entries]
                )
                self._entry_vectors = _coerce_vectors(
                    raw_vectors,
                    expected_count=len(self.entries),
                )
        return self._entry_vectors

    def retrieve(self, query: RetrievalQuery) -> list[RetrievalHit]:
        if not self.entries:
            return []

        # Key terms enrich the sentence without replacing its context.
        query_parts = [query.source_text, *query.key_terms]
        query_text = "\n".join(dict.fromkeys(query_parts))
        raw_query_vectors = self.embedder.embed([query_text])
        query_vector = _coerce_vectors(raw_query_vectors, expected_count=1)[0]
        entry_vectors = self._index()

        hits: list[RetrievalHit] = []
        for entry, vector in zip(self.entries, entry_vectors, strict=True):
            if not _domain_matches(entry, query):
                continue
            score = _cosine(query_vector, vector)
            if score >= self.min_score:
                hits.append(
                    _hit(
                        entry,
                        match_type=RetrievalMatchType.VECTOR,
                        score=score,
                    )
                )

        hits.sort(key=lambda hit: (-hit.score, hit.term_id))
        return hits[: query.top_k]


class HybridGlossaryRetriever(Retriever):
    """Merge exact and vector results, deduplicating by glossary term ID."""

    def __init__(
        self,
        exact_retriever: Retriever,
        vector_retriever: Retriever,
        *,
        exact_weight: float = 0.7,
        vector_weight: float = 0.3,
    ) -> None:
        if not math.isfinite(exact_weight) or not math.isfinite(vector_weight):
            raise ValueError("hybrid weights must be finite")
        if exact_weight <= 0.0 or vector_weight <= 0.0:
            raise ValueError("hybrid weights must be positive")
        total = exact_weight + vector_weight
        self.exact_retriever = exact_retriever
        self.vector_retriever = vector_retriever
        self.exact_weight = exact_weight / total
        self.vector_weight = vector_weight / total

    @staticmethod
    def _unit_score(score: float) -> float:
        return max(0.0, min(1.0, score))

    def retrieve(self, query: RetrievalQuery) -> list[RetrievalHit]:
        exact_hits = {hit.term_id: hit for hit in self.exact_retriever.retrieve(query)}
        vector_hits = {hit.term_id: hit for hit in self.vector_retriever.retrieve(query)}
        merged: list[RetrievalHit] = []

        for term_id in exact_hits.keys() | vector_hits.keys():
            exact = exact_hits.get(term_id)
            vector = vector_hits.get(term_id)
            representative = exact or vector
            assert representative is not None

            exact_score = self._unit_score(exact.score) if exact is not None else 0.0
            vector_score = (
                self._unit_score(vector.score) if vector is not None else 0.0
            )
            score = (
                self.exact_weight * exact_score
                + self.vector_weight * vector_score
            )
            if exact is not None and vector is not None:
                match_type = RetrievalMatchType.HYBRID
            elif exact is not None:
                match_type = RetrievalMatchType.EXACT
            else:
                match_type = RetrievalMatchType.VECTOR

            merged.append(
                RetrievalHit(
                    term_id=representative.term_id,
                    source_term=representative.source_term,
                    target_term=representative.target_term,
                    domain=representative.domain,
                    match_type=match_type,
                    score=score,
                    definition=representative.definition,
                    accepted_target_variants=representative.accepted_target_variants,
                    replacement_rules=representative.replacement_rules,
                )
            )

        match_priority = {
            RetrievalMatchType.HYBRID: 0,
            RetrievalMatchType.EXACT: 1,
            RetrievalMatchType.VECTOR: 2,
        }
        merged.sort(
            key=lambda hit: (
                -hit.score,
                match_priority[hit.match_type],
                hit.term_id,
            )
        )
        return merged[: query.top_k]


class ExactFirstHybridGlossaryRetriever(Retriever):
    """Safety-first hybrid retrieval for terminology injection.

    A naive union can attach semantically related but absent glossary terms to a
    sentence.  If literal terms are available, this policy keeps only those
    applicable terms and uses vector scores solely as corroborating evidence. If
    there is no literal match, it allows a small number of high-confidence vector
    fallbacks for paraphrases.  The raw union retriever remains available for
    comparison experiments.
    """

    def __init__(
        self,
        exact_retriever: Retriever,
        vector_retriever: Retriever,
        *,
        max_vector_fallbacks: int = 1,
        exact_weight: float = 0.7,
        vector_weight: float = 0.3,
    ) -> None:
        if max_vector_fallbacks < 1:
            raise ValueError("max_vector_fallbacks must be at least one")
        if not math.isfinite(exact_weight) or not math.isfinite(vector_weight):
            raise ValueError("hybrid weights must be finite")
        if exact_weight <= 0.0 or vector_weight <= 0.0:
            raise ValueError("hybrid weights must be positive")
        total = exact_weight + vector_weight
        self.exact_retriever = exact_retriever
        self.vector_retriever = vector_retriever
        self.max_vector_fallbacks = max_vector_fallbacks
        self.exact_weight = exact_weight / total
        self.vector_weight = vector_weight / total

    def retrieve(self, query: RetrievalQuery) -> list[RetrievalHit]:
        exact_hits = self.exact_retriever.retrieve(query)
        vector_hits = self.vector_retriever.retrieve(query)
        if not exact_hits:
            return vector_hits[: min(query.top_k, self.max_vector_fallbacks)]

        vector_by_id = {hit.term_id: hit for hit in vector_hits}
        corroborated: list[RetrievalHit] = []
        for exact in exact_hits:
            vector = vector_by_id.get(exact.term_id)
            if vector is None:
                corroborated.append(exact)
                continue
            score = (
                self.exact_weight * max(0.0, min(1.0, exact.score))
                + self.vector_weight * max(0.0, min(1.0, vector.score))
            )
            corroborated.append(
                RetrievalHit(
                    term_id=exact.term_id,
                    source_term=exact.source_term,
                    target_term=exact.target_term,
                    domain=exact.domain,
                    match_type=RetrievalMatchType.HYBRID,
                    score=score,
                    definition=exact.definition,
                    accepted_target_variants=exact.accepted_target_variants,
                    replacement_rules=exact.replacement_rules,
                )
            )
        corroborated.sort(key=lambda hit: (-hit.score, hit.term_id))
        return corroborated[: query.top_k]


def _resolve_column(
    fieldnames: Sequence[str] | None,
    candidates: Sequence[str],
) -> str:
    available = set(fieldnames or ())
    for candidate in candidates:
        if candidate in available:
            return candidate
    joined = " or ".join(repr(candidate) for candidate in candidates)
    raise ValueError(f"glossary CSV is missing required column {joined}")


def _parse_json_text_list(raw: str | None, *, field_name: str, row_number: int) -> tuple[str, ...]:
    if raw is None or not raw.strip():
        return ()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"row {row_number}: {field_name} must contain a JSON array"
        ) from exc
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise ValueError(f"row {row_number}: {field_name} must be a JSON string array")
    return _text_tuple(parsed, field_name)


def _parse_json_replacements(
    raw: str | None,
    *,
    row_number: int,
) -> tuple[tuple[str, str], ...]:
    if raw is None or not raw.strip():
        return ()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"row {row_number}: replacement_rules_json must contain a JSON object"
        ) from exc
    if not isinstance(parsed, dict) or any(
        not isinstance(source, str) or not isinstance(target, str)
        for source, target in parsed.items()
    ):
        raise ValueError(
            f"row {row_number}: replacement_rules_json must be a string-to-string object"
        )
    return _replacement_tuple(list(parsed.items()))


def load_glossary_csv(path: str | Path) -> list[GlossaryEntry]:
    """Load the documented glossary CSV schema.

    The documented Korean/English column names are preferred, while the shorter
    ``source_term`` and ``target_term`` aliases make small prototypes convenient.
    UTF-8 BOM files are accepted.
    """

    csv_path = Path(path)
    entries: list[GlossaryEntry] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        term_id_column = _resolve_column(reader.fieldnames, ("term_id",))
        source_term_column = _resolve_column(
            reader.fieldnames,
            ("source_term_ko", "source_term"),
        )
        target_term_column = _resolve_column(
            reader.fieldnames,
            ("preferred_target_en", "target_term"),
        )

        for row_number, row in enumerate(reader, start=2):
            if not any(value and value.strip() for value in row.values()):
                continue
            accepted_raw = row.get("accepted_variants_json")
            if accepted_raw is None:
                accepted_raw = row.get("accepted_variants")
            disallowed_raw = row.get("disallowed_variants_json")
            if disallowed_raw is None:
                disallowed_raw = row.get("disallowed_variants")
            try:
                entry = GlossaryEntry(
                    term_id=row.get(term_id_column, ""),
                    source_term=row.get(source_term_column, ""),
                    target_term=row.get(target_term_column, ""),
                    domain=row.get("domain"),
                    definition=row.get("definition"),
                    accepted_variants=_parse_json_text_list(
                        accepted_raw,
                        field_name="accepted_variants_json",
                        row_number=row_number,
                    ),
                    disallowed_variants=_parse_json_text_list(
                        disallowed_raw,
                        field_name="disallowed_variants_json",
                        row_number=row_number,
                    ),
                    replacement_rules=_parse_json_replacements(
                        row.get("replacement_rules_json"),
                        row_number=row_number,
                    ),
                    source=row.get("source"),
                    glossary_version=row.get("glossary_version"),
                    notes=row.get("notes"),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"row {row_number}: invalid glossary entry: {exc}") from exc
            entries.append(entry)

    _validate_entries(entries)
    return entries


# Short aliases keep wiring code readable while retaining descriptive class names.
ExactRetriever = ExactGlossaryRetriever
VectorRetriever = VectorGlossaryRetriever
HybridRetriever = HybridGlossaryRetriever


__all__ = [
    "Embedder",
    "ExactGlossaryRetriever",
    "ExactFirstHybridGlossaryRetriever",
    "ExactRetriever",
    "GlossaryEntry",
    "HybridGlossaryRetriever",
    "HybridRetriever",
    "SentenceTransformerEmbedder",
    "VectorGlossaryRetriever",
    "VectorRetriever",
    "load_glossary_csv",
]
