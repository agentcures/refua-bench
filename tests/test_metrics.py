from __future__ import annotations

import pytest

from refua_bench.metrics import compute_metric, metric_direction


def test_metric_direction() -> None:
    assert metric_direction("mae") == "lower"
    assert metric_direction("accuracy") == "higher"


def test_mae() -> None:
    score = compute_metric("mae", [1.0, 3.0], [2.0, 1.0])
    assert score == pytest.approx(1.5)


def test_rmse() -> None:
    score = compute_metric("rmse", [2.0, 4.0], [1.0, 5.0])
    assert score == pytest.approx(1.0)


def test_accuracy() -> None:
    score = compute_metric("accuracy", [1, 0, 1, 1], [1, 1, 1, 0])
    assert score == pytest.approx(0.5)


def test_f1_binary() -> None:
    score = compute_metric("f1", [1, 1, 0, 0], [1, 0, 1, 0], positive_label=1)
    assert score == pytest.approx(0.5)
