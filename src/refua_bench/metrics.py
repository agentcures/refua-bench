from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal

MetricDirection = Literal["higher", "lower"]

_METRIC_DIRECTIONS: dict[str, MetricDirection] = {
    "mae": "lower",
    "rmse": "lower",
    "accuracy": "higher",
    "exact_match": "higher",
    "f1": "higher",
    "enrichment_factor": "higher",
    "ef": "higher",
}


def metric_direction(metric: str) -> MetricDirection:
    try:
        return _METRIC_DIRECTIONS[metric]
    except KeyError as exc:
        raise ValueError(f"Unsupported metric: {metric}") from exc


def compute_metric(
    metric: str,
    expected_values: Sequence[Any],
    predicted_values: Sequence[Any],
    *,
    positive_label: Any = 1,
    enrichment_fraction: float = 0.01,
) -> float:
    if len(expected_values) != len(predicted_values):
        raise ValueError("expected_values and predicted_values must have equal length")
    if not expected_values:
        raise ValueError("Cannot compute a metric on empty values")

    if metric == "mae":
        expected = _to_float_list(expected_values)
        predicted = _to_float_list(predicted_values)
        absolute_errors = [
            abs(exp - pred)
            for exp, pred in zip(expected, predicted, strict=True)
        ]
        return sum(absolute_errors) / len(expected)

    if metric == "rmse":
        expected = _to_float_list(expected_values)
        predicted = _to_float_list(predicted_values)
        squared_errors = [
            (exp - pred) ** 2
            for exp, pred in zip(expected, predicted, strict=True)
        ]
        mse = sum(squared_errors) / len(expected)
        return math.sqrt(mse)

    if metric in {"accuracy", "exact_match"}:
        matches = sum(
            1
            for exp, pred in zip(expected_values, predicted_values, strict=True)
            if exp == pred
        )
        return matches / len(expected_values)

    if metric == "f1":
        return _f1_binary(expected_values, predicted_values, positive_label=positive_label)
    if metric in {"enrichment_factor", "ef"}:
        return _enrichment_factor(
            expected_values,
            predicted_values,
            positive_label=positive_label,
            enrichment_fraction=enrichment_fraction,
        )

    raise ValueError(f"Unsupported metric: {metric}")


def _to_float_list(values: Sequence[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        try:
            result.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Expected numeric value, got {value!r}") from exc
    return result


def _f1_binary(expected: Sequence[Any], predicted: Sequence[Any], *, positive_label: Any) -> float:
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for exp, pred in zip(expected, predicted, strict=True):
        exp_positive = exp == positive_label
        pred_positive = pred == positive_label
        if exp_positive and pred_positive:
            true_positive += 1
        elif (not exp_positive) and pred_positive:
            false_positive += 1
        elif exp_positive and (not pred_positive):
            false_negative += 1

    if true_positive == 0 and false_positive == 0 and false_negative == 0:
        return 1.0

    precision_denom = true_positive + false_positive
    recall_denom = true_positive + false_negative

    precision = 0.0 if precision_denom == 0 else true_positive / precision_denom
    recall = 0.0 if recall_denom == 0 else true_positive / recall_denom

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _enrichment_factor(
    expected: Sequence[Any],
    predicted: Sequence[Any],
    *,
    positive_label: Any,
    enrichment_fraction: float,
) -> float:
    try:
        fraction = float(enrichment_fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError("enrichment_fraction must be numeric") from exc
    if not math.isfinite(fraction) or fraction <= 0 or fraction > 1:
        raise ValueError("enrichment_fraction must be > 0 and <= 1")

    n_cases = len(expected)
    top_k = max(1, math.ceil(n_cases * fraction))
    scores = _to_float_list(predicted)

    ranked = sorted(
        zip(scores, expected, strict=True),
        key=lambda item: item[0],
        reverse=True,
    )
    total_positives = sum(1 for value in expected if value == positive_label)
    if total_positives == 0:
        raise ValueError("enrichment_factor requires at least one positive expected label")

    top_hits = sum(1 for _, label in ranked[:top_k] if label == positive_label)
    baseline_positive_rate = total_positives / n_cases
    top_positive_rate = top_hits / top_k
    return top_positive_rate / baseline_positive_rate
