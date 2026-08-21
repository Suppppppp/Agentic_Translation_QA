"""Pure, dependency-free metrics for the translation QA benchmark."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math
import re
import statistics
import unicodedata


# Hyphenated and apostrophized words remain one token. Other punctuation acts as
# a separator, so an accepted multi-word target must occur as consecutive tokens.
_TOKEN_PATTERN = re.compile(
    r"\d+(?:[.,]\d+)*|[^\W\d_]+(?:[-'’][^\W\d_]+)*",
    flags=re.UNICODE,
)


@dataclass(frozen=True, slots=True)
class RatioMetric:
    """A count-based metric whose value is undefined for an empty denominator."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if self.denominator < 0:
            raise ValueError("denominator must be non-negative")
        if not 0 <= self.numerator <= self.denominator:
            raise ValueError("numerator must be between zero and denominator")

    @property
    def rate(self) -> float | None:
        """Return the metric on a zero-to-one scale."""

        if self.denominator == 0:
            return None
        return self.numerator / self.denominator

    @property
    def percentage(self) -> float | None:
        """Return the metric on a zero-to-100 scale."""

        rate = self.rate
        return None if rate is None else rate * 100.0


@dataclass(frozen=True, slots=True)
class ConfusionMatrix:
    """Binary counts where ``needs revision`` is the positive class."""

    true_positive: int = 0
    true_negative: int = 0
    false_positive: int = 0
    false_negative: int = 0

    @property
    def total(self) -> int:
        return (
            self.true_positive
            + self.true_negative
            + self.false_positive
            + self.false_negative
        )


@dataclass(frozen=True, slots=True)
class AgentJudgmentMetrics:
    """Confusion counts and the three Agent gate metrics in the contract."""

    confusion: ConfusionMatrix
    accuracy: RatioMetric
    revision_recall: RatioMetric
    unnecessary_revision_rate: RatioMetric


@dataclass(frozen=True, slots=True)
class LatencyStatistics:
    """Summary of non-negative latency observations measured in milliseconds."""

    count: int
    mean_ms: float | None
    median_ms: float | None
    p95_ms: float | None


def normalize_text(text: str) -> str:
    """Apply NFKC normalization and collapse all runs of Unicode whitespace."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return " ".join(unicodedata.normalize("NFKC", text).split())


def _tokens(text: str) -> tuple[str, ...]:
    normalized = normalize_text(text).casefold()
    return tuple(match.group(0) for match in _TOKEN_PATTERN.finditer(normalized))


def _contains_consecutive_tokens(
    candidate_tokens: Sequence[str],
    target_tokens: Sequence[str],
) -> bool:
    if not target_tokens or len(target_tokens) > len(candidate_tokens):
        return False

    window_size = len(target_tokens)
    return any(
        tuple(candidate_tokens[start : start + window_size]) == tuple(target_tokens)
        for start in range(len(candidate_tokens) - window_size + 1)
    )


def contains_accepted_target(
    translation: str,
    accepted_targets: Iterable[str],
) -> bool:
    """Return whether any accepted target occurs as consecutive whole tokens.

    Matching is Unicode-normalized and case-insensitive. It never treats a
    substring within a larger token as a match. Each accepted target must contain
    at least one token.
    """

    candidate_tokens = _tokens(translation)
    targets = tuple(accepted_targets)
    if not targets:
        raise ValueError("accepted_targets must contain at least one target")

    for target in targets:
        if not isinstance(target, str):
            raise TypeError("each accepted target must be a string")
        target_tokens = _tokens(target)
        if not target_tokens:
            raise ValueError("accepted targets must contain at least one token")
        if _contains_consecutive_tokens(candidate_tokens, target_tokens):
            return True
    return False


def terminology_accuracy(
    occurrences: Iterable[tuple[str, Iterable[str]]],
) -> RatioMetric:
    """Calculate occurrence-level terminology accuracy.

    Every item represents one required source-term occurrence and consists of
    ``(translation, accepted_targets)``. Repeated source terms therefore count as
    separate denominator entries, as required by the evaluation contract.
    """

    correct = 0
    total = 0
    for translation, accepted_targets in occurrences:
        total += 1
        if contains_accepted_target(translation, accepted_targets):
            correct += 1
    return RatioMetric(numerator=correct, denominator=total)


def sentence_modification_rate(
    sentence_pairs: Iterable[tuple[str, str]],
) -> RatioMetric:
    """Return the share of ``(initial, final)`` pairs that differ after normalization."""

    modified = 0
    total = 0
    for initial, final in sentence_pairs:
        total += 1
        if normalize_text(initial) != normalize_text(final):
            modified += 1
    return RatioMetric(numerator=modified, denominator=total)


def confusion_matrix(
    manual_needs_revision: Iterable[bool],
    agent_needs_revision: Iterable[bool],
) -> ConfusionMatrix:
    """Build a binary confusion matrix using manual labels as ground truth."""

    manual = tuple(manual_needs_revision)
    agent = tuple(agent_needs_revision)
    if len(manual) != len(agent):
        raise ValueError("manual and Agent label sequences must have equal length")
    if any(type(label) is not bool for label in (*manual, *agent)):
        raise TypeError("manual and Agent labels must be booleans")

    tp = tn = fp = fn = 0
    for expected, predicted in zip(manual, agent, strict=True):
        if expected and predicted:
            tp += 1
        elif not expected and not predicted:
            tn += 1
        elif not expected and predicted:
            fp += 1
        else:
            fn += 1

    return ConfusionMatrix(
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
    )


def agent_judgment_metrics(
    manual_needs_revision: Iterable[bool],
    agent_needs_revision: Iterable[bool],
) -> AgentJudgmentMetrics:
    """Calculate Agent accuracy, revision recall and unnecessary revision rate."""

    matrix = confusion_matrix(manual_needs_revision, agent_needs_revision)
    return AgentJudgmentMetrics(
        confusion=matrix,
        accuracy=RatioMetric(
            numerator=matrix.true_positive + matrix.true_negative,
            denominator=matrix.total,
        ),
        revision_recall=RatioMetric(
            numerator=matrix.true_positive,
            denominator=matrix.true_positive + matrix.false_negative,
        ),
        unnecessary_revision_rate=RatioMetric(
            numerator=matrix.false_positive,
            denominator=matrix.false_positive + matrix.true_negative,
        ),
    )


def latency_statistics(latencies_ms: Iterable[float]) -> LatencyStatistics:
    """Return mean, median and nearest-rank p95 for latency observations.

    Empty input produces ``None`` for every aggregate. Values must be finite and
    non-negative. Nearest-rank p95 is used so the result is always an observed
    latency rather than an interpolated value.
    """

    observations: list[float] = []
    for value in latencies_ms:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("latency values must be real numbers")
        numeric_value = float(value)
        if not math.isfinite(numeric_value) or numeric_value < 0:
            raise ValueError("latency values must be finite and non-negative")
        observations.append(numeric_value)

    if not observations:
        return LatencyStatistics(
            count=0,
            mean_ms=None,
            median_ms=None,
            p95_ms=None,
        )

    ordered = sorted(observations)
    p95_rank = math.ceil(0.95 * len(ordered))
    return LatencyStatistics(
        count=len(ordered),
        mean_ms=statistics.fmean(ordered),
        median_ms=statistics.median(ordered),
        p95_ms=ordered[p95_rank - 1],
    )


__all__ = [
    "AgentJudgmentMetrics",
    "ConfusionMatrix",
    "LatencyStatistics",
    "RatioMetric",
    "agent_judgment_metrics",
    "confusion_matrix",
    "contains_accepted_target",
    "latency_statistics",
    "normalize_text",
    "sentence_modification_rate",
    "terminology_accuracy",
]
