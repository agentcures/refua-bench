from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from refua_bench.adapters import load_adapter
from refua_bench.compare import StatisticalPolicy, compare_runs
from refua_bench.reporting import read_json, write_json
from refua_bench.run_artifact import validate_run_artifact
from refua_bench.runner import run_benchmark
from refua_bench.schema import load_suite


def gate_suite(
    *,
    suite_path: Path,
    baseline_run_path: Path,
    adapter_spec: str,
    adapter_config: Mapping[str, Any] | None = None,
    policy: StatisticalPolicy | None = None,
    provenance: Mapping[str, Any] | None = None,
    candidate_output_path: Path | None = None,
    comparison_output_path: Path | None = None,
) -> dict[str, Any]:
    """Run a candidate benchmark and compare it against a baseline artifact."""
    suite = load_suite(suite_path)
    adapter = load_adapter(adapter_spec, adapter_config)

    candidate_run = run_benchmark(
        suite,
        adapter,
        provenance=None if provenance is None else dict(provenance),
    ).to_dict()

    baseline_run = validate_run_artifact(
        read_json(baseline_run_path),
        source=f"baseline run artifact '{baseline_run_path}'",
        suite=suite,
    )
    candidate_validated = validate_run_artifact(
        candidate_run,
        source="candidate run artifact",
        suite=suite,
    )
    comparison = compare_runs(
        suite,
        baseline_run,
        candidate_validated,
        policy=policy,
    ).to_dict()

    if candidate_output_path is not None:
        write_json(candidate_output_path, candidate_validated)
    if comparison_output_path is not None:
        write_json(comparison_output_path, comparison)

    summary = comparison.get("summary")
    passed = bool(summary.get("passed")) if isinstance(summary, dict) else False
    return {
        "passed": passed,
        "suite": {"name": suite.name, "version": suite.version},
        "candidate": candidate_validated,
        "comparison": comparison,
    }
