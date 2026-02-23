from __future__ import annotations

import pytest

from refua_bench.metrics import compute_metric, metric_direction


def test_metric_direction() -> None:
    assert metric_direction("mae") == "lower"
    assert metric_direction("accuracy") == "higher"
    assert metric_direction("enrichment_factor") == "higher"
    assert metric_direction("bedroc") == "higher"


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


def test_enrichment_factor() -> None:
    score = compute_metric(
        "enrichment_factor",
        [1, 1, 0, 0],
        [0.9, 0.8, 0.7, 0.1],
        positive_label=1,
        enrichment_fraction=0.5,
    )
    assert score == pytest.approx(2.0)


def test_enrichment_factor_alias() -> None:
    score = compute_metric(
        "ef",
        [1, 0, 1, 0, 0],
        [0.95, 0.1, 0.7, 0.2, 0.3],
        positive_label=1,
        enrichment_fraction=0.2,
    )
    assert score == pytest.approx(2.5)


def test_enrichment_factor_requires_positive_labels() -> None:
    with pytest.raises(ValueError, match="at least one positive"):
        compute_metric(
            "enrichment_factor",
            [0, 0, 0],
            [0.9, 0.8, 0.7],
            positive_label=1,
            enrichment_fraction=0.5,
        )


def test_bedroc_reference_value() -> None:
    score = compute_metric(
        "bedroc",
        [1, 0, 1, 0, 0],
        [0.95, 0.9, 0.8, 0.2, 0.1],
        positive_label=1,
        bedroc_alpha=20.0,
    )
    assert score == pytest.approx(0.9823431104768724)


def test_bedroc_alpha_controls_early_recognition_weight() -> None:
    low_alpha = compute_metric(
        "bedroc",
        [1, 0, 0, 0, 1],
        [0.95, 0.7, 0.5, 0.3, 0.1],
        positive_label=1,
        bedroc_alpha=5.0,
    )
    high_alpha = compute_metric(
        "bedroc",
        [1, 0, 0, 0, 1],
        [0.95, 0.7, 0.5, 0.3, 0.1],
        positive_label=1,
        bedroc_alpha=20.0,
    )
    assert high_alpha > low_alpha


def test_bedroc_requires_positive_labels() -> None:
    with pytest.raises(ValueError, match="at least one positive"):
        compute_metric(
            "bedroc",
            [0, 0, 0],
            [0.9, 0.8, 0.7],
            positive_label=1,
            bedroc_alpha=20.0,
        )
