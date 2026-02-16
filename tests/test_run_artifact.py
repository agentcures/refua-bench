from __future__ import annotations

import pytest

from refua_bench.adapters import GoldenAdapter
from refua_bench.run_artifact import validate_run_artifact
from refua_bench.runner import run_benchmark
from refua_bench.schema import BenchmarkSuite


def test_validate_run_artifact_accepts_runner_output(suite: BenchmarkSuite) -> None:
    run_payload = run_benchmark(suite, GoldenAdapter()).to_dict()
    validated = validate_run_artifact(
        run_payload,
        source="candidate",
        suite=suite,
    )
    assert validated["suite_name"] == suite.name


def test_validate_run_artifact_rejects_missing_top_level_key(suite: BenchmarkSuite) -> None:
    run_payload = run_benchmark(suite, GoldenAdapter()).to_dict()
    run_payload.pop("summary")

    with pytest.raises(ValueError, match="invalid keys"):
        validate_run_artifact(run_payload, source="candidate", suite=suite)


def test_validate_run_artifact_rejects_summary_mismatch(suite: BenchmarkSuite) -> None:
    run_payload = run_benchmark(suite, GoldenAdapter()).to_dict()
    run_payload["summary"]["cases_total"] = 999

    with pytest.raises(ValueError, match="cases_total"):
        validate_run_artifact(run_payload, source="candidate", suite=suite)


def test_validate_run_artifact_rejects_suite_mismatch(suite: BenchmarkSuite) -> None:
    run_payload = run_benchmark(suite, GoldenAdapter()).to_dict()
    run_payload["suite_name"] = "other-suite"

    with pytest.raises(ValueError, match="suite_name"):
        validate_run_artifact(run_payload, source="candidate", suite=suite)
