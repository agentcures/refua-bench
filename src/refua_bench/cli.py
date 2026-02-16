from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from refua_bench.adapters import GoldenAdapter, load_adapter
from refua_bench.baseline_registry import (
    get_baseline_entry,
    list_baselines,
    load_registry,
    promote_baseline,
    resolve_baseline_path,
    save_registry,
)
from refua_bench.compare import StatisticalPolicy, compare_runs
from refua_bench.provenance import collect_provenance
from refua_bench.reporting import (
    read_json,
    render_compare_markdown,
    render_run_markdown,
    write_json,
    write_markdown,
)
from refua_bench.run_artifact import validate_run_artifact
from refua_bench.runner import run_benchmark
from refua_bench.schema import BenchmarkSuite, load_data_file, load_suite, suite_from_mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="refua-bench",
        description="Benchmark and regression gate tooling for Refua model workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a benchmark suite and emit run artifacts")
    _add_common_run_arguments(run_parser)
    _add_provenance_arguments(run_parser)
    run_parser.add_argument("--output", type=Path, required=True, help="JSON run artifact path")
    run_parser.add_argument("--markdown", type=Path, help="Optional markdown run report path")
    run_parser.add_argument(
        "--fail-on-errors",
        action="store_true",
        help="Exit non-zero if any case fails",
    )
    run_parser.set_defaults(handler=_cmd_run)

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare candidate run artifact against baseline",
    )
    compare_parser.add_argument("--suite", type=Path, required=True, help="Benchmark suite config")
    compare_parser.add_argument(
        "--baseline",
        type=Path,
        help="Baseline run artifact. Optional when --registry and --baseline-name are used.",
    )
    compare_parser.add_argument(
        "--registry",
        type=Path,
        help="Baseline registry path for named baseline resolution.",
    )
    compare_parser.add_argument(
        "--baseline-name",
        help="Named baseline in registry. Requires --registry.",
    )
    compare_parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="Candidate run artifact",
    )
    compare_parser.add_argument("--output", type=Path, required=True, help="Comparison JSON report")
    compare_parser.add_argument("--markdown", type=Path, help="Optional markdown comparison report")
    _add_statistical_arguments(compare_parser)
    compare_parser.add_argument(
        "--no-fail-on-regression",
        action="store_true",
        help="Return 0 even when regressions are detected",
    )
    compare_parser.add_argument(
        "--fail-on-uncertain",
        action="store_true",
        help="Fail when bootstrap result is inconclusive",
    )
    compare_parser.set_defaults(handler=_cmd_compare)

    gate_parser = subparsers.add_parser(
        "gate",
        help="Run suite for a model and compare result against baseline in one command",
    )
    _add_common_run_arguments(gate_parser)
    _add_provenance_arguments(gate_parser)
    gate_parser.add_argument(
        "--baseline",
        type=Path,
        help="Baseline run artifact. Optional when --registry and --baseline-name are used.",
    )
    gate_parser.add_argument(
        "--registry",
        type=Path,
        help="Baseline registry path for named baseline resolution.",
    )
    gate_parser.add_argument(
        "--baseline-name",
        help="Named baseline in registry. Requires --registry.",
    )
    gate_parser.add_argument(
        "--candidate-output",
        type=Path,
        required=True,
        help="Path to write candidate run artifact",
    )
    gate_parser.add_argument("--output", type=Path, required=True, help="Comparison JSON report")
    gate_parser.add_argument("--markdown", type=Path, help="Optional markdown comparison report")
    _add_statistical_arguments(gate_parser)
    gate_parser.add_argument(
        "--no-fail-on-regression",
        action="store_true",
        help="Return 0 even when regressions are detected",
    )
    gate_parser.add_argument(
        "--fail-on-uncertain",
        action="store_true",
        help="Fail when bootstrap result is inconclusive",
    )
    gate_parser.set_defaults(handler=_cmd_gate)

    init_parser = subparsers.add_parser(
        "init",
        help="Generate a starter benchmark suite and baseline artifact",
    )
    init_parser.add_argument("--directory", type=Path, required=True, help="Output directory")
    init_parser.add_argument("--name", default="my-refua-suite", help="Suite name")
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files in the target directory",
    )
    init_parser.set_defaults(handler=_cmd_init)

    baseline_parser = subparsers.add_parser(
        "baseline",
        help="Manage named baselines in a registry",
    )
    baseline_subparsers = baseline_parser.add_subparsers(
        dest="baseline_command",
        required=True,
    )

    baseline_list_parser = baseline_subparsers.add_parser(
        "list",
        help="List baselines in a registry",
    )
    baseline_list_parser.add_argument("--registry", type=Path, required=True, help="Registry JSON")
    baseline_list_parser.add_argument("--suite-name", help="Optional suite name filter")
    baseline_list_parser.add_argument("--output", type=Path, help="Optional JSON output path")
    baseline_list_parser.set_defaults(handler=_cmd_baseline_list)

    baseline_resolve_parser = baseline_subparsers.add_parser(
        "resolve",
        help="Resolve a named baseline to its run artifact path",
    )
    baseline_resolve_parser.add_argument(
        "--registry",
        type=Path,
        required=True,
        help="Registry JSON",
    )
    baseline_resolve_parser.add_argument(
        "--suite",
        type=Path,
        required=True,
        help="Benchmark suite config",
    )
    baseline_resolve_parser.add_argument(
        "--baseline-name",
        required=True,
        help="Named baseline to resolve",
    )
    baseline_resolve_parser.set_defaults(handler=_cmd_baseline_resolve)

    baseline_promote_parser = baseline_subparsers.add_parser(
        "promote",
        help="Promote a candidate run to the current named baseline",
    )
    baseline_promote_parser.add_argument(
        "--registry",
        type=Path,
        required=True,
        help="Registry JSON",
    )
    baseline_promote_parser.add_argument(
        "--suite",
        type=Path,
        required=True,
        help="Benchmark suite config",
    )
    baseline_promote_parser.add_argument(
        "--baseline-name",
        required=True,
        help="Name of baseline to update",
    )
    baseline_promote_parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="Candidate run artifact",
    )
    baseline_promote_parser.add_argument(
        "--store-dir",
        type=Path,
        help="Optional root directory where promoted artifacts are stored",
    )
    baseline_promote_parser.add_argument("--notes", help="Optional promotion notes")
    _add_statistical_arguments(baseline_promote_parser)
    baseline_promote_parser.add_argument(
        "--allow-regression",
        action="store_true",
        help="Allow promotion even when candidate regresses against current baseline",
    )
    baseline_promote_parser.add_argument(
        "--fail-on-uncertain",
        action="store_true",
        help="Treat inconclusive bootstrap result as failure",
    )
    baseline_promote_parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output with promotion details",
    )
    baseline_promote_parser.set_defaults(handler=_cmd_baseline_promote)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return int(args.handler(args))
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _add_common_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--suite", type=Path, required=True, help="Benchmark suite config")
    parser.add_argument(
        "--adapter",
        required=True,
        help="Adapter spec. Built-ins: golden, file, command or module.path:AdapterClass",
    )
    parser.add_argument(
        "--adapter-config",
        help="Adapter config JSON string or path to YAML/JSON file",
    )


def _add_provenance_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-name", help="Model name for provenance")
    parser.add_argument("--model-version", help="Model version for provenance")
    parser.add_argument(
        "--model-params",
        help="Model parameter mapping as JSON string or path to YAML/JSON file",
    )
    parser.add_argument(
        "--provenance-extra",
        help="Extra provenance mapping as JSON string or path to YAML/JSON file",
    )
    parser.add_argument(
        "--no-provenance",
        action="store_true",
        help="Disable automatic provenance capture",
    )


def _add_statistical_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--min-effect-size",
        type=float,
        default=0.0,
        help="Minimum practical regression effect required before failing",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=0,
        help="Enable statistical gating with this many bootstrap resamples",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level used for bootstrap CI",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        help="Optional random seed for bootstrap",
    )


def _cmd_run(args: argparse.Namespace) -> int:
    suite = load_suite(args.suite)
    adapter_config = _load_optional_mapping(args.adapter_config)
    adapter = load_adapter(args.adapter, adapter_config)

    provenance = _build_provenance(args, adapter.name, adapter_config)
    run_payload = run_benchmark(suite, adapter, provenance=provenance).to_dict()
    write_json(args.output, run_payload)

    if args.markdown is not None:
        write_markdown(args.markdown, render_run_markdown(run_payload))

    if args.fail_on_errors and int(run_payload["summary"]["case_failures"]) > 0:
        return 1
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    suite = load_suite(args.suite)
    baseline_path = _resolve_baseline_arg(args, suite)

    baseline = validate_run_artifact(
        read_json(baseline_path),
        source=f"baseline run artifact '{baseline_path}'",
        suite=suite,
    )
    candidate = validate_run_artifact(
        read_json(args.candidate),
        source=f"candidate run artifact '{args.candidate}'",
        suite=suite,
    )

    policy = _policy_from_args(args)
    comparison = compare_runs(suite, baseline, candidate, policy=policy).to_dict()
    write_json(args.output, comparison)

    if args.markdown is not None:
        write_markdown(args.markdown, render_compare_markdown(comparison))

    fail_on_regression = not bool(args.no_fail_on_regression)
    if fail_on_regression and _comparison_failed(comparison, args.fail_on_uncertain):
        return 1
    return 0


def _cmd_gate(args: argparse.Namespace) -> int:
    suite = load_suite(args.suite)
    adapter_config = _load_optional_mapping(args.adapter_config)
    adapter = load_adapter(args.adapter, adapter_config)

    provenance = _build_provenance(args, adapter.name, adapter_config)
    candidate_run = run_benchmark(suite, adapter, provenance=provenance).to_dict()
    write_json(args.candidate_output, candidate_run)
    validated_candidate = validate_run_artifact(
        candidate_run,
        source="candidate run artifact (current gate run)",
        suite=suite,
    )

    baseline_path = _resolve_baseline_arg(args, suite)
    baseline = validate_run_artifact(
        read_json(baseline_path),
        source=f"baseline run artifact '{baseline_path}'",
        suite=suite,
    )

    policy = _policy_from_args(args)
    comparison = compare_runs(suite, baseline, validated_candidate, policy=policy).to_dict()
    write_json(args.output, comparison)

    if args.markdown is not None:
        write_markdown(args.markdown, render_compare_markdown(comparison))

    fail_on_regression = not bool(args.no_fail_on_regression)
    if fail_on_regression and _comparison_failed(comparison, args.fail_on_uncertain):
        return 1
    return 0


def _cmd_init(args: argparse.Namespace) -> int:
    directory = args.directory
    directory.mkdir(parents=True, exist_ok=True)

    suite_path = directory / "suite.yaml"
    baseline_path = directory / "baseline.json"
    file_predictions_path = directory / "candidate_predictions.json"
    command_config_path = directory / "command_adapter_config.yaml"
    command_script_path = directory / "adapter_command.py"

    file_paths = [
        suite_path,
        baseline_path,
        file_predictions_path,
        command_config_path,
        command_script_path,
    ]

    if (not args.force) and any(path.exists() for path in file_paths):
        existing = [str(path.name) for path in file_paths if path.exists()]
        raise FileExistsError(
            "Target directory already contains files: "
            + ", ".join(sorted(existing))
            + ". Use --force to overwrite."
        )

    suite_payload = _starter_suite_payload(args.name)
    suite_path.write_text(yaml.safe_dump(suite_payload, sort_keys=False), encoding="utf-8")

    suite = suite_from_mapping(suite_payload)
    baseline = run_benchmark(suite, GoldenAdapter()).to_dict()
    write_json(baseline_path, baseline)

    write_json(file_predictions_path, _starter_candidate_predictions())
    command_config_path.write_text(
        yaml.safe_dump(
            {
                "command": [sys.executable, str(command_script_path)],
                "timeout_seconds": 30,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    command_script_path.write_text(_starter_command_adapter_script(), encoding="utf-8")

    return 0


def _cmd_baseline_list(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry)
    rows = list_baselines(registry, suite_name=args.suite_name)

    if args.output is not None:
        write_json(
            args.output,
            {
                "rows": rows,
            },
        )
    else:
        for row in rows:
            print(
                f"{row['suite_name']}:{row['baseline_name']} -> {row['run_path']} "
                f"(updated {row.get('updated_at')})"
            )

    return 0


def _cmd_baseline_resolve(args: argparse.Namespace) -> int:
    suite = load_suite(args.suite)
    registry = load_registry(args.registry)
    resolved = resolve_baseline_path(
        registry,
        suite_name=suite.name,
        baseline_name=args.baseline_name,
    )
    print(str(resolved))
    return 0


def _cmd_baseline_promote(args: argparse.Namespace) -> int:
    suite = load_suite(args.suite)
    registry = load_registry(args.registry)

    candidate_run = validate_run_artifact(
        read_json(args.candidate),
        source=f"candidate run artifact '{args.candidate}'",
        suite=suite,
    )
    existing_entry = get_baseline_entry(
        registry,
        suite_name=suite.name,
        baseline_name=args.baseline_name,
    )

    compare_payload: dict[str, Any] | None = None
    if existing_entry is not None:
        current_baseline_path = resolve_baseline_path(
            registry,
            suite_name=suite.name,
            baseline_name=args.baseline_name,
        )
        baseline_run = validate_run_artifact(
            read_json(current_baseline_path),
            source=f"baseline run artifact '{current_baseline_path}'",
            suite=suite,
        )

        policy = _policy_from_args(args)
        compare_payload = compare_runs(
            suite,
            baseline_run,
            candidate_run,
            policy=policy,
        ).to_dict()

        failed = _comparison_failed(compare_payload, args.fail_on_uncertain)
        if failed and (not args.allow_regression):
            raise RuntimeError(
                "Candidate did not pass safety checks against current baseline. "
                "Use --allow-regression to override."
            )

    promoted = promote_baseline(
        registry=registry,
        registry_path=args.registry,
        suite_name=suite.name,
        suite_version=suite.version,
        baseline_name=args.baseline_name,
        candidate_path=args.candidate,
        notes=args.notes,
        provenance=_mapping_or_none(candidate_run.get("provenance")),
        compare_summary=None if compare_payload is None else compare_payload.get("summary"),
        store_dir=args.store_dir,
    )
    save_registry(args.registry, registry)

    if args.output is not None:
        write_json(
            args.output,
            {
                "promoted": promoted,
                "compare": compare_payload,
            },
        )

    print(str(promoted["run_path"]))
    return 0


def _comparison_failed(compare_payload: dict[str, Any], fail_on_uncertain: bool) -> bool:
    summary = compare_payload.get("summary", {})
    regressions = int(summary.get("regressions", 0)) if isinstance(summary, dict) else 0
    uncertain = int(summary.get("uncertain", 0)) if isinstance(summary, dict) else 0

    if regressions > 0:
        return True
    if fail_on_uncertain and uncertain > 0:
        return True
    return False


def _resolve_baseline_arg(args: argparse.Namespace, suite: BenchmarkSuite) -> Path:
    if args.baseline is not None:
        return args.baseline

    if args.registry is not None and args.baseline_name:
        registry = load_registry(args.registry)
        return resolve_baseline_path(
            registry,
            suite_name=suite.name,
            baseline_name=args.baseline_name,
        )

    raise ValueError(
        "Provide --baseline, or provide --registry with --baseline-name"
    )


def _policy_from_args(args: argparse.Namespace) -> StatisticalPolicy:
    min_effect_size = float(args.min_effect_size)
    bootstrap_resamples = int(args.bootstrap_resamples)
    confidence_level = float(args.confidence_level)

    if min_effect_size < 0:
        raise ValueError("--min-effect-size must be >= 0")
    if bootstrap_resamples < 0:
        raise ValueError("--bootstrap-resamples must be >= 0")
    if confidence_level <= 0 or confidence_level >= 1:
        raise ValueError("--confidence-level must be between 0 and 1")

    return StatisticalPolicy(
        min_effect_size=min_effect_size,
        bootstrap_resamples=bootstrap_resamples,
        confidence_level=confidence_level,
        bootstrap_seed=args.bootstrap_seed,
        fail_on_uncertain=bool(args.fail_on_uncertain),
    )


def _build_provenance(
    args: argparse.Namespace,
    adapter_name: str,
    adapter_config: Mapping[str, Any],
) -> dict[str, Any]:
    if args.no_provenance:
        return {}

    model_params = _load_optional_mapping(args.model_params)
    provenance_extra = _load_optional_mapping(args.provenance_extra)

    return collect_provenance(
        adapter_name=adapter_name,
        adapter_spec=args.adapter,
        adapter_config=adapter_config,
        model_name=args.model_name,
        model_version=args.model_version,
        model_params=model_params,
        extra=provenance_extra,
    )


def _mapping_or_none(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    return None


def _load_optional_mapping(config_value: str | None) -> dict[str, Any]:
    if config_value is None:
        return {}

    path = Path(config_value)
    if path.exists():
        payload = load_data_file(path)
        return dict(payload)

    try:
        parsed = json.loads(config_value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Expected a valid JSON object or existing YAML/JSON file path"
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError("Decoded JSON value must be an object")

    return parsed


def _starter_suite_payload(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "version": "1.0.0",
        "description": "Starter suite for Refua benchmarking and regression gates.",
        "tasks": [
            {
                "id": "affinity_mae",
                "metric": "mae",
                "prediction_key": "affinity",
                "regression_tolerance": 0.05,
                "weight": 2.0,
                "cases": [
                    {
                        "id": "kras_mrtx1133",
                        "input": {
                            "target": "KRAS",
                            "ligand": "MRTX1133",
                        },
                        "expected": {
                            "affinity": -9.3,
                        },
                    },
                    {
                        "id": "egfr_osimertinib",
                        "input": {
                            "target": "EGFR",
                            "ligand": "Osimertinib",
                        },
                        "expected": {
                            "affinity": -8.7,
                        },
                    },
                ],
            },
            {
                "id": "admet_toxicity_accuracy",
                "metric": "accuracy",
                "prediction_key": "toxic",
                "regression_tolerance": 0.02,
                "cases": [
                    {
                        "id": "case_1",
                        "input": {
                            "smiles": "CCO",
                        },
                        "expected": {
                            "toxic": 0,
                        },
                    },
                    {
                        "id": "case_2",
                        "input": {
                            "smiles": "CC(=O)O",
                        },
                        "expected": {
                            "toxic": 0,
                        },
                    },
                    {
                        "id": "case_3",
                        "input": {
                            "smiles": "N#CCN",
                        },
                        "expected": {
                            "toxic": 1,
                        },
                    },
                ],
            },
        ],
    }


def _starter_candidate_predictions() -> dict[str, Any]:
    return {
        "affinity_mae": {
            "kras_mrtx1133": {"affinity": -9.15},
            "egfr_osimertinib": {"affinity": -8.5},
        },
        "admet_toxicity_accuracy": {
            "case_1": {"toxic": 0},
            "case_2": {"toxic": 1},
            "case_3": {"toxic": 1},
        },
    }


def _starter_command_adapter_script() -> str:
    return """#!/usr/bin/env python3
import json
import sys


def predict(payload):
    prediction_key = payload["prediction_key"]

    # Replace this block with calls into your model runtime.
    if prediction_key == "affinity":
        return {"affinity": -8.0}
    if prediction_key == "toxic":
        return {"toxic": 0}
    return {prediction_key: None}


def main():
    raw = sys.stdin.read()
    payload = json.loads(raw)
    result = predict(payload)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
"""


if __name__ == "__main__":
    raise SystemExit(main())
