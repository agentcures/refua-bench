from __future__ import annotations

import json
from pathlib import Path

import yaml

from refua_bench.adapters import GoldenAdapter
from refua_bench.gating import gate_suite
from refua_bench.reporting import write_json
from refua_bench.runner import run_benchmark
from refua_bench.schema import load_suite


def _write_suite(path: Path) -> None:
    payload = {
        "name": "gating-suite",
        "version": "1.0.0",
        "tasks": [
            {
                "id": "affinity_mae",
                "metric": "mae",
                "prediction_key": "affinity",
                "regression_tolerance": 0.05,
                "cases": [
                    {"id": "a", "input": {"x": 1}, "expected": {"affinity": -9.0}},
                    {"id": "b", "input": {"x": 2}, "expected": {"affinity": -8.0}},
                ],
            }
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_gate_suite_runs_candidate_and_compare(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yaml"
    baseline_path = tmp_path / "baseline.json"
    predictions_path = tmp_path / "predictions.json"
    candidate_path = tmp_path / "candidate.json"
    comparison_path = tmp_path / "comparison.json"

    _write_suite(suite_path)
    suite = load_suite(suite_path)
    baseline = run_benchmark(suite, GoldenAdapter()).to_dict()
    write_json(baseline_path, baseline)

    predictions_path.write_text(
        json.dumps(
            {
                "affinity_mae": {
                    "a": {"affinity": -8.0},
                    "b": {"affinity": -7.0},
                }
            }
        ),
        encoding="utf-8",
    )

    payload = gate_suite(
        suite_path=suite_path,
        baseline_run_path=baseline_path,
        adapter_spec="file",
        adapter_config={"predictions_path": str(predictions_path)},
        candidate_output_path=candidate_path,
        comparison_output_path=comparison_path,
    )

    assert payload["passed"] is False
    assert candidate_path.exists()
    assert comparison_path.exists()
    assert payload["comparison"]["summary"]["regressions"] >= 1
