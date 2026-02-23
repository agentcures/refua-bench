from __future__ import annotations

import json

import pytest

from refua_bench.adapters import FileAdapter, GoldenAdapter
from refua_bench.runner import run_benchmark
from refua_bench.schema import BenchmarkSuite, suite_from_mapping


def test_runner_golden_has_no_failures(suite) -> None:  # type: ignore[no-untyped-def]
    run = run_benchmark(suite, GoldenAdapter())
    payload = run.to_dict()

    assert payload["summary"]["case_failures"] == 0
    assert payload["provenance"] == {}
    first_task = payload["task_results"][0]
    assert first_task["score"] == pytest.approx(0.0)


def test_runner_accepts_provenance_payload(suite) -> None:  # type: ignore[no-untyped-def]
    run = run_benchmark(
        suite,
        GoldenAdapter(),
        provenance={"model": {"name": "demo", "version": "v1"}},
    )
    payload = run.to_dict()
    assert payload["provenance"]["model"]["name"] == "demo"


def test_runner_with_file_adapter(suite, tmp_path) -> None:  # type: ignore[no-untyped-def]
    predictions = {
        "affinity_mae": {
            "a": {"affinity": -8.9},
            "b": {"affinity": -7.8},
        },
        "tox_acc": {
            "c1": {"toxic": 0},
            "c2": {"toxic": 1},
        },
    }
    pred_path = tmp_path / "predictions.json"
    pred_path.write_text(json.dumps(predictions), encoding="utf-8")

    adapter = FileAdapter({"predictions_path": str(pred_path)})
    run = run_benchmark(suite, adapter).to_dict()

    assert run["summary"]["case_failures"] == 0
    assert run["task_results"][0]["score"] == pytest.approx(0.15)


def test_runner_enrichment_factor_uses_task_fraction(tmp_path) -> None:  # type: ignore[no-untyped-def]
    suite = suite_from_mapping(
        {
            "name": "ef-suite",
            "version": "1.0.0",
            "tasks": [
                {
                    "id": "ef_task",
                    "metric": "enrichment_factor",
                    "prediction_key": "score",
                    "expected_key": "active",
                    "positive_label": 1,
                    "enrichment_fraction": 0.5,
                    "cases": [
                        {"id": "a", "input": {}, "expected": {"active": 1}},
                        {"id": "b", "input": {}, "expected": {"active": 1}},
                        {"id": "c", "input": {}, "expected": {"active": 0}},
                        {"id": "d", "input": {}, "expected": {"active": 0}},
                    ],
                }
            ],
        }
    )

    predictions = {
        "ef_task": {
            "a": {"score": 0.9},
            "b": {"score": 0.1},
            "c": {"score": 0.8},
            "d": {"score": 0.2},
        }
    }
    pred_path = tmp_path / "predictions.json"
    pred_path.write_text(json.dumps(predictions), encoding="utf-8")

    adapter = FileAdapter({"predictions_path": str(pred_path)})
    run = run_benchmark(suite, adapter).to_dict()
    assert run["task_results"][0]["score"] == pytest.approx(1.0)


def test_runner_bedroc_uses_task_alpha(tmp_path) -> None:  # type: ignore[no-untyped-def]
    def build_suite(alpha: float) -> BenchmarkSuite:
        return suite_from_mapping(
            {
                "name": "bedroc-suite",
                "version": "1.0.0",
                "tasks": [
                    {
                        "id": "bedroc_task",
                        "metric": "bedroc",
                        "prediction_key": "score",
                        "expected_key": "active",
                        "positive_label": 1,
                        "bedroc_alpha": alpha,
                        "cases": [
                            {"id": "a", "input": {}, "expected": {"active": 1}},
                            {"id": "b", "input": {}, "expected": {"active": 0}},
                            {"id": "c", "input": {}, "expected": {"active": 0}},
                            {"id": "d", "input": {}, "expected": {"active": 0}},
                            {"id": "e", "input": {}, "expected": {"active": 1}},
                        ],
                    }
                ],
            }
        )

    predictions = {
        "bedroc_task": {
            "a": {"score": 0.95},
            "b": {"score": 0.7},
            "c": {"score": 0.5},
            "d": {"score": 0.3},
            "e": {"score": 0.1},
        }
    }
    pred_path = tmp_path / "bedroc_predictions.json"
    pred_path.write_text(json.dumps(predictions), encoding="utf-8")
    adapter = FileAdapter({"predictions_path": str(pred_path)})

    low_alpha_run = run_benchmark(build_suite(5.0), adapter).to_dict()
    high_alpha_run = run_benchmark(build_suite(20.0), adapter).to_dict()

    low_score = float(low_alpha_run["task_results"][0]["score"])
    high_score = float(high_alpha_run["task_results"][0]["score"])
    assert high_score > low_score
