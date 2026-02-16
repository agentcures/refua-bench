from __future__ import annotations

import json

from refua_bench.adapters import FileAdapter, GoldenAdapter
from refua_bench.compare import StatisticalPolicy, compare_runs
from refua_bench.runner import run_benchmark
from refua_bench.schema import suite_from_mapping


def test_compare_passes_identical_runs(suite) -> None:  # type: ignore[no-untyped-def]
    baseline = run_benchmark(suite, GoldenAdapter()).to_dict()
    candidate = run_benchmark(suite, GoldenAdapter()).to_dict()

    report = compare_runs(suite, baseline, candidate).to_dict()
    assert report["summary"]["passed"] is True
    assert report["summary"]["regressions"] == 0


def test_compare_detects_regression(suite, tmp_path) -> None:  # type: ignore[no-untyped-def]
    predictions = {
        "affinity_mae": {
            "a": {"affinity": -8.2},
            "b": {"affinity": -7.1},
        },
        "tox_acc": {
            "c1": {"toxic": 0},
            "c2": {"toxic": 0},
        },
    }
    pred_path = tmp_path / "candidate.json"
    pred_path.write_text(json.dumps(predictions), encoding="utf-8")

    baseline = run_benchmark(suite, GoldenAdapter()).to_dict()
    candidate = run_benchmark(suite, FileAdapter({"predictions_path": str(pred_path)})).to_dict()

    report = compare_runs(suite, baseline, candidate).to_dict()
    assert report["summary"]["passed"] is False
    assert report["summary"]["regressions"] >= 1


def test_compare_min_effect_size_avoids_noise(tmp_path) -> None:  # type: ignore[no-untyped-def]
    suite = suite_from_mapping(
        {
            "name": "stats-suite",
            "version": "1.0.0",
            "tasks": [
                {
                    "id": "mae_task",
                    "metric": "mae",
                    "prediction_key": "value",
                    "regression_tolerance": 0.0,
                    "cases": [
                        {"id": "a", "input": {}, "expected": {"value": 0.0}},
                        {"id": "b", "input": {}, "expected": {"value": 0.0}},
                        {"id": "c", "input": {}, "expected": {"value": 0.0}},
                        {"id": "d", "input": {}, "expected": {"value": 0.0}},
                    ],
                }
            ],
        }
    )
    predictions = {
        "mae_task": {
            "a": {"value": 0.02},
            "b": {"value": 0.02},
            "c": {"value": 0.02},
            "d": {"value": 0.02},
        }
    }
    pred_path = tmp_path / "candidate.json"
    pred_path.write_text(json.dumps(predictions), encoding="utf-8")

    baseline = run_benchmark(suite, GoldenAdapter()).to_dict()
    candidate = run_benchmark(suite, FileAdapter({"predictions_path": str(pred_path)})).to_dict()

    report = compare_runs(
        suite,
        baseline,
        candidate,
        policy=StatisticalPolicy(min_effect_size=0.05),
    ).to_dict()
    assert report["summary"]["regressions"] == 0
    assert report["task_comparisons"][0]["threshold"] == 0.05


def test_compare_bootstrap_confirms_regression(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cases = []
    predictions: dict[str, dict[str, dict[str, float]]] = {"mae_task": {}}
    for idx in range(20):
        case_id = f"c{idx}"
        cases.append({"id": case_id, "input": {}, "expected": {"value": 0.0}})
        predictions["mae_task"][case_id] = {"value": 1.0}

    suite = suite_from_mapping(
        {
            "name": "bootstrap-suite",
            "version": "1.0.0",
            "tasks": [
                {
                    "id": "mae_task",
                    "metric": "mae",
                    "prediction_key": "value",
                    "regression_tolerance": 0.0,
                    "cases": cases,
                }
            ],
        }
    )

    pred_path = tmp_path / "candidate.json"
    pred_path.write_text(json.dumps(predictions), encoding="utf-8")

    baseline = run_benchmark(suite, GoldenAdapter()).to_dict()
    candidate = run_benchmark(suite, FileAdapter({"predictions_path": str(pred_path)})).to_dict()

    report = compare_runs(
        suite,
        baseline,
        candidate,
        policy=StatisticalPolicy(
            bootstrap_resamples=300,
            confidence_level=0.95,
            bootstrap_seed=7,
        ),
    ).to_dict()

    task = report["task_comparisons"][0]
    assert report["summary"]["regressions"] == 1
    assert task["status"] == "regression"
    assert task["ci_low"] is not None
    assert task["ci_low"] > 0


def test_compare_bootstrap_can_be_uncertain(tmp_path) -> None:  # type: ignore[no-untyped-def]
    suite = suite_from_mapping(
        {
            "name": "uncertain-suite",
            "version": "1.0.0",
            "tasks": [
                {
                    "id": "mae_task",
                    "metric": "mae",
                    "prediction_key": "value",
                    "regression_tolerance": 0.1,
                    "cases": [
                        {"id": "a", "input": {}, "expected": {"value": 0.0}},
                        {"id": "b", "input": {}, "expected": {"value": 0.0}},
                    ],
                }
            ],
        }
    )
    predictions = {
        "mae_task": {
            "a": {"value": 0.0},
            "b": {"value": 1.0},
        }
    }
    pred_path = tmp_path / "candidate.json"
    pred_path.write_text(json.dumps(predictions), encoding="utf-8")

    baseline = run_benchmark(suite, GoldenAdapter()).to_dict()
    candidate = run_benchmark(suite, FileAdapter({"predictions_path": str(pred_path)})).to_dict()

    report = compare_runs(
        suite,
        baseline,
        candidate,
        policy=StatisticalPolicy(
            bootstrap_resamples=500,
            confidence_level=0.95,
            bootstrap_seed=17,
        ),
    ).to_dict()

    assert report["summary"]["uncertain"] == 1
    assert report["task_comparisons"][0]["status"] == "uncertain"
