from __future__ import annotations

import json
from pathlib import Path

import yaml

from refua_bench.cli import main


def _write_suite(path: Path) -> None:
    payload = {
        "name": "cli-suite",
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


def test_cli_run_and_compare_regression(tmp_path) -> None:  # type: ignore[no-untyped-def]
    suite_path = tmp_path / "suite.yaml"
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    compare_path = tmp_path / "compare.json"
    config_path = tmp_path / "config.yaml"
    predictions_path = tmp_path / "predictions.json"

    _write_suite(suite_path)

    baseline_rc = main(
        [
            "run",
            "--suite",
            str(suite_path),
            "--adapter",
            "golden",
            "--output",
            str(baseline_path),
            "--fail-on-errors",
        ]
    )
    assert baseline_rc == 0
    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert "provenance" in baseline_payload
    assert "runtime" in baseline_payload["provenance"]

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
    config_path.write_text(
        yaml.safe_dump({"predictions_path": str(predictions_path)}, sort_keys=False),
        encoding="utf-8",
    )

    candidate_rc = main(
        [
            "run",
            "--suite",
            str(suite_path),
            "--adapter",
            "file",
            "--adapter-config",
            str(config_path),
            "--output",
            str(candidate_path),
            "--fail-on-errors",
        ]
    )
    assert candidate_rc == 0

    compare_rc = main(
        [
            "compare",
            "--suite",
            str(suite_path),
            "--baseline",
            str(baseline_path),
            "--candidate",
            str(candidate_path),
            "--output",
            str(compare_path),
        ]
    )
    assert compare_rc == 1

    compare_allow_rc = main(
        [
            "compare",
            "--suite",
            str(suite_path),
            "--baseline",
            str(baseline_path),
            "--candidate",
            str(candidate_path),
            "--output",
            str(compare_path),
            "--no-fail-on-regression",
        ]
    )
    assert compare_allow_rc == 0


def test_cli_init_creates_project(tmp_path) -> None:  # type: ignore[no-untyped-def]
    out_dir = tmp_path / "starter"
    rc = main(["init", "--directory", str(out_dir), "--name", "starter-suite"])
    assert rc == 0
    assert (out_dir / "suite.yaml").exists()
    assert (out_dir / "baseline.json").exists()
    assert (out_dir / "command_adapter_config.yaml").exists()


def test_cli_baseline_registry_flow(tmp_path) -> None:  # type: ignore[no-untyped-def]
    suite_path = tmp_path / "suite.yaml"
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    compare_path = tmp_path / "compare.json"
    registry_path = tmp_path / "registry.json"
    list_path = tmp_path / "list.json"
    config_path = tmp_path / "config.yaml"
    predictions_path = tmp_path / "predictions.json"

    _write_suite(suite_path)

    baseline_rc = main(
        [
            "run",
            "--suite",
            str(suite_path),
            "--adapter",
            "golden",
            "--output",
            str(baseline_path),
        ]
    )
    assert baseline_rc == 0

    promote_baseline_rc = main(
        [
            "baseline",
            "promote",
            "--registry",
            str(registry_path),
            "--suite",
            str(suite_path),
            "--baseline-name",
            "stable",
            "--candidate",
            str(baseline_path),
        ]
    )
    assert promote_baseline_rc == 0

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
    config_path.write_text(
        yaml.safe_dump({"predictions_path": str(predictions_path)}, sort_keys=False),
        encoding="utf-8",
    )

    candidate_rc = main(
        [
            "run",
            "--suite",
            str(suite_path),
            "--adapter",
            "file",
            "--adapter-config",
            str(config_path),
            "--output",
            str(candidate_path),
        ]
    )
    assert candidate_rc == 0

    compare_rc = main(
        [
            "compare",
            "--suite",
            str(suite_path),
            "--registry",
            str(registry_path),
            "--baseline-name",
            "stable",
            "--candidate",
            str(candidate_path),
            "--output",
            str(compare_path),
        ]
    )
    assert compare_rc == 1

    promote_regressed_rc = main(
        [
            "baseline",
            "promote",
            "--registry",
            str(registry_path),
            "--suite",
            str(suite_path),
            "--baseline-name",
            "stable",
            "--candidate",
            str(candidate_path),
        ]
    )
    assert promote_regressed_rc == 2

    promote_override_rc = main(
        [
            "baseline",
            "promote",
            "--registry",
            str(registry_path),
            "--suite",
            str(suite_path),
            "--baseline-name",
            "stable",
            "--candidate",
            str(candidate_path),
            "--allow-regression",
        ]
    )
    assert promote_override_rc == 0

    compare_after_promote_rc = main(
        [
            "compare",
            "--suite",
            str(suite_path),
            "--registry",
            str(registry_path),
            "--baseline-name",
            "stable",
            "--candidate",
            str(candidate_path),
            "--output",
            str(compare_path),
        ]
    )
    assert compare_after_promote_rc == 0

    list_rc = main(
        [
            "baseline",
            "list",
            "--registry",
            str(registry_path),
            "--output",
            str(list_path),
        ]
    )
    assert list_rc == 0
    listed = json.loads(list_path.read_text(encoding="utf-8"))
    assert listed["rows"][0]["baseline_name"] == "stable"


def test_cli_compare_fail_on_uncertain(tmp_path) -> None:  # type: ignore[no-untyped-def]
    suite_path = tmp_path / "suite.yaml"
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    compare_path = tmp_path / "compare.json"
    config_path = tmp_path / "config.yaml"
    predictions_path = tmp_path / "predictions.json"

    payload = {
        "name": "uncertain-cli-suite",
        "version": "1.0.0",
        "tasks": [
            {
                "id": "mae_task",
                "metric": "mae",
                "prediction_key": "value",
                "regression_tolerance": 0.1,
                "cases": [
                    {"id": "a", "input": {"x": 1}, "expected": {"value": 0.0}},
                    {"id": "b", "input": {"x": 2}, "expected": {"value": 0.0}},
                ],
            }
        ],
    }
    suite_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    baseline_rc = main(
        [
            "run",
            "--suite",
            str(suite_path),
            "--adapter",
            "golden",
            "--output",
            str(baseline_path),
        ]
    )
    assert baseline_rc == 0

    predictions_path.write_text(
        json.dumps(
            {
                "mae_task": {
                    "a": {"value": 0.0},
                    "b": {"value": 1.0},
                }
            }
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        yaml.safe_dump({"predictions_path": str(predictions_path)}, sort_keys=False),
        encoding="utf-8",
    )
    candidate_rc = main(
        [
            "run",
            "--suite",
            str(suite_path),
            "--adapter",
            "file",
            "--adapter-config",
            str(config_path),
            "--output",
            str(candidate_path),
        ]
    )
    assert candidate_rc == 0

    compare_default_rc = main(
        [
            "compare",
            "--suite",
            str(suite_path),
            "--baseline",
            str(baseline_path),
            "--candidate",
            str(candidate_path),
            "--output",
            str(compare_path),
            "--bootstrap-resamples",
            "500",
            "--bootstrap-seed",
            "17",
        ]
    )
    assert compare_default_rc == 0

    compare_fail_uncertain_rc = main(
        [
            "compare",
            "--suite",
            str(suite_path),
            "--baseline",
            str(baseline_path),
            "--candidate",
            str(candidate_path),
            "--output",
            str(compare_path),
            "--bootstrap-resamples",
            "500",
            "--bootstrap-seed",
            "17",
            "--fail-on-uncertain",
        ]
    )
    assert compare_fail_uncertain_rc == 1


def test_cli_compare_rejects_invalid_candidate_artifact(tmp_path) -> None:  # type: ignore[no-untyped-def]
    suite_path = tmp_path / "suite.yaml"
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate_invalid.json"
    compare_path = tmp_path / "compare.json"

    _write_suite(suite_path)

    baseline_rc = main(
        [
            "run",
            "--suite",
            str(suite_path),
            "--adapter",
            "golden",
            "--output",
            str(baseline_path),
        ]
    )
    assert baseline_rc == 0

    candidate_path.write_text(json.dumps({"run_id": "bad"}), encoding="utf-8")

    compare_rc = main(
        [
            "compare",
            "--suite",
            str(suite_path),
            "--baseline",
            str(baseline_path),
            "--candidate",
            str(candidate_path),
            "--output",
            str(compare_path),
        ]
    )
    assert compare_rc == 2
