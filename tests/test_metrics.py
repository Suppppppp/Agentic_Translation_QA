import math

import pytest

from translation_qa.metrics import (
    RatioMetric,
    agent_judgment_metrics,
    confusion_matrix,
    contains_accepted_target,
    latency_statistics,
    normalize_text,
    sentence_modification_rate,
    terminology_accuracy,
)


def test_normalize_text_applies_nfkc_and_collapses_unicode_whitespace() -> None:
    assert normalize_text("  Ａ\t cafe\u0301\n\u00a0test  ") == "A café test"


def test_normalize_text_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="string"):
        normalize_text(123)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("translation", "accepted_targets"),
    [
        ("The Neural Network is stable.", ["neural network"]),
        ("We used the approved term.", ["preferred term", "approved term"]),
        ("A neural, network model was used.", ["neural network"]),
    ],
)
def test_contains_accepted_target_matches_consecutive_whole_tokens(
    translation: str,
    accepted_targets: list[str],
) -> None:
    assert contains_accepted_target(translation, accepted_targets)


def test_contains_accepted_target_does_not_match_substring_or_nonconsecutive_words() -> None:
    assert not contains_accepted_target("This is only partial.", ["art"])
    assert not contains_accepted_target(
        "The quality was followed by robust assurance.",
        ["quality assurance"],
    )


def test_contains_accepted_target_preserves_hyphenated_token_boundaries() -> None:
    assert not contains_accepted_target("quality-assurance process", ["quality assurance"])
    assert contains_accepted_target(
        "quality-assurance process",
        ["quality-assurance"],
    )


@pytest.mark.parametrize("accepted_targets", [[], ["   "], ["..."]])
def test_contains_accepted_target_rejects_empty_target_definitions(
    accepted_targets: list[str],
) -> None:
    with pytest.raises(ValueError):
        contains_accepted_target("translation", accepted_targets)


def test_terminology_accuracy_counts_required_occurrences() -> None:
    result = terminology_accuracy(
        [
            ("The neural network is ready.", ["neural network"]),
            ("The approved term appears.", ["preferred term", "approved term"]),
            ("The expression is missing.", ["target term"]),
        ]
    )

    assert result == RatioMetric(numerator=2, denominator=3)
    assert result.rate == pytest.approx(2 / 3)
    assert result.percentage == pytest.approx(200 / 3)


def test_terminology_accuracy_counts_repeated_occurrences_separately() -> None:
    result = terminology_accuracy(
        [
            ("Only one occurrence is translated.", ["translated"]),
            ("Only one occurrence is translated.", ["missing target"]),
        ]
    )

    assert result.numerator == 1
    assert result.denominator == 2


def test_empty_terminology_accuracy_has_undefined_rate() -> None:
    result = terminology_accuracy([])

    assert result == RatioMetric(numerator=0, denominator=0)
    assert result.rate is None
    assert result.percentage is None


def test_sentence_modification_rate_uses_normalized_text() -> None:
    result = sentence_modification_rate(
        [
            ("A  sentence", "Ａ\tsentence"),
            ("Same sentence.", "Same sentence!"),
            ("Unchanged", " Unchanged\n"),
        ]
    )

    assert result == RatioMetric(numerator=1, denominator=3)
    assert result.rate == pytest.approx(1 / 3)


def test_empty_sentence_modification_rate_is_undefined() -> None:
    result = sentence_modification_rate([])

    assert result.denominator == 0
    assert result.rate is None


def test_confusion_matrix_uses_needs_revision_as_positive_class() -> None:
    matrix = confusion_matrix(
        [True, True, False, False],
        [True, False, True, False],
    )

    assert matrix.true_positive == 1
    assert matrix.true_negative == 1
    assert matrix.false_positive == 1
    assert matrix.false_negative == 1
    assert matrix.total == 4


def test_agent_judgment_metrics_include_counts_and_rates() -> None:
    result = agent_judgment_metrics(
        [True, True, False, False],
        [True, False, True, False],
    )

    assert result.accuracy == RatioMetric(2, 4)
    assert result.revision_recall == RatioMetric(1, 2)
    assert result.unnecessary_revision_rate == RatioMetric(1, 2)
    assert result.accuracy.percentage == 50.0


def test_agent_judgment_metrics_handle_each_zero_denominator() -> None:
    no_manual_revisions = agent_judgment_metrics([False, False], [False, True])
    assert no_manual_revisions.revision_recall.rate is None
    assert no_manual_revisions.unnecessary_revision_rate.rate == 0.5

    no_manual_passes = agent_judgment_metrics([True, True], [True, False])
    assert no_manual_passes.revision_recall.rate == 0.5
    assert no_manual_passes.unnecessary_revision_rate.rate is None

    empty = agent_judgment_metrics([], [])
    assert empty.accuracy.rate is None
    assert empty.revision_recall.rate is None
    assert empty.unnecessary_revision_rate.rate is None


def test_confusion_matrix_rejects_mismatched_or_non_boolean_labels() -> None:
    with pytest.raises(ValueError, match="equal length"):
        confusion_matrix([True], [])

    with pytest.raises(TypeError, match="booleans"):
        confusion_matrix([True], [1])  # type: ignore[list-item]


def test_latency_statistics_calculate_mean_median_and_nearest_rank_p95() -> None:
    result = latency_statistics(range(1, 21))

    assert result.count == 20
    assert result.mean_ms == 10.5
    assert result.median_ms == 10.5
    assert result.p95_ms == 19.0


def test_latency_statistics_handle_singleton_and_empty_input() -> None:
    singleton = latency_statistics([12.5])
    assert singleton.count == 1
    assert singleton.mean_ms == 12.5
    assert singleton.median_ms == 12.5
    assert singleton.p95_ms == 12.5

    empty = latency_statistics([])
    assert empty.count == 0
    assert empty.mean_ms is None
    assert empty.median_ms is None
    assert empty.p95_ms is None


@pytest.mark.parametrize("value", [-1.0, math.inf, -math.inf, math.nan])
def test_latency_statistics_reject_invalid_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        latency_statistics([value])


def test_latency_statistics_rejects_boolean_values() -> None:
    with pytest.raises(TypeError, match="real numbers"):
        latency_statistics([True])
