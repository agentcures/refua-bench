from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from refua_bench.metrics import metric_direction
from refua_bench.schema import BenchmarkSuite

_TOP_LEVEL_KEYS = {
    "run_id",
    "suite_name",
    "suite_version",
    "adapter",
    "started_at",
    "finished_at",
    "summary",
    "task_results",
    "provenance",
}

_SUMMARY_KEYS = {
    "tasks_total",
    "tasks_with_errors",
    "cases_total",
    "case_failures",
    "all_cases_succeeded",
}

_TASK_KEYS = {
    "task_id",
    "metric",
    "direction",
    "score",
    "cases_total",
    "case_failures",
    "case_results",
}

_CASE_KEYS = {
    "case_id",
    "expected",
    "predicted",
    "duration_ms",
    "error",
}


def validate_run_artifact(
    payload: Mapping[str, Any],
    *,
    source: str,
    suite: BenchmarkSuite | None = None,
) -> dict[str, Any]:
    run = _require_mapping(payload, f"{source}")
    _validate_key_set(run, required=_TOP_LEVEL_KEYS, path=source)

    _require_non_empty_str(run.get("run_id"), f"{source}.run_id")
    suite_name = _require_non_empty_str(run.get("suite_name"), f"{source}.suite_name")
    suite_version = _require_non_empty_str(run.get("suite_version"), f"{source}.suite_version")
    _require_non_empty_str(run.get("adapter"), f"{source}.adapter")
    _require_non_empty_str(run.get("started_at"), f"{source}.started_at")
    _require_non_empty_str(run.get("finished_at"), f"{source}.finished_at")

    summary = _require_mapping(run.get("summary"), f"{source}.summary")
    _validate_key_set(summary, required=_SUMMARY_KEYS, path=f"{source}.summary")

    summary_tasks_total = _require_non_negative_int(
        summary.get("tasks_total"),
        f"{source}.summary.tasks_total",
    )
    summary_tasks_with_errors = _require_non_negative_int(
        summary.get("tasks_with_errors"),
        f"{source}.summary.tasks_with_errors",
    )
    summary_cases_total = _require_non_negative_int(
        summary.get("cases_total"),
        f"{source}.summary.cases_total",
    )
    summary_case_failures = _require_non_negative_int(
        summary.get("case_failures"),
        f"{source}.summary.case_failures",
    )
    summary_all_cases_succeeded = _require_bool(
        summary.get("all_cases_succeeded"),
        f"{source}.summary.all_cases_succeeded",
    )

    task_results_raw = _require_list(run.get("task_results"), f"{source}.task_results")
    provenance = _require_mapping(run.get("provenance"), f"{source}.provenance")

    task_ids: set[str] = set()
    case_totals_sum = 0
    case_failures_sum = 0
    tasks_with_errors = 0
    task_map: dict[str, dict[str, Any]] = {}

    for index, task_item in enumerate(task_results_raw):
        task_path = f"{source}.task_results[{index}]"
        task = _require_mapping(task_item, task_path)
        _validate_key_set(task, required=_TASK_KEYS, path=task_path)

        task_id = _require_non_empty_str(task.get("task_id"), f"{task_path}.task_id")
        if task_id in task_ids:
            raise ValueError(f"Duplicate task_id in {source}: '{task_id}'")
        task_ids.add(task_id)
        task_map[task_id] = task

        _require_non_empty_str(task.get("metric"), f"{task_path}.metric")
        direction = _require_non_empty_str(task.get("direction"), f"{task_path}.direction")
        if direction not in {"higher", "lower"}:
            raise ValueError(f"{task_path}.direction must be 'higher' or 'lower'")

        _require_optional_finite_number(task.get("score"), f"{task_path}.score")
        cases_total = _require_non_negative_int(task.get("cases_total"), f"{task_path}.cases_total")
        case_failures = _require_non_negative_int(
            task.get("case_failures"),
            f"{task_path}.case_failures",
        )
        if case_failures > cases_total:
            raise ValueError(f"{task_path}.case_failures cannot exceed cases_total")

        case_results_raw = _require_list(task.get("case_results"), f"{task_path}.case_results")
        if len(case_results_raw) != cases_total:
            raise ValueError(
                f"{task_path}.cases_total={cases_total} does not match "
                f"len(case_results)={len(case_results_raw)}"
            )

        case_ids: set[str] = set()
        observed_case_failures = 0
        for case_index, case_item in enumerate(case_results_raw):
            case_path = f"{task_path}.case_results[{case_index}]"
            case = _require_mapping(case_item, case_path)
            _validate_key_set(case, required=_CASE_KEYS, path=case_path)

            case_id = _require_non_empty_str(case.get("case_id"), f"{case_path}.case_id")
            if case_id in case_ids:
                raise ValueError(f"Duplicate case_id in {task_path}: '{case_id}'")
            case_ids.add(case_id)

            _require_non_negative_finite_number(case.get("duration_ms"), f"{case_path}.duration_ms")

            error = case.get("error")
            if error is None:
                pass
            elif isinstance(error, str):
                observed_case_failures += 1
            else:
                raise ValueError(f"{case_path}.error must be a string or null")

        if observed_case_failures != case_failures:
            raise ValueError(
                f"{task_path}.case_failures={case_failures} does not match observed failures "
                f"{observed_case_failures}"
            )

        if case_failures > 0 or task.get("score") is None:
            tasks_with_errors += 1

        case_totals_sum += cases_total
        case_failures_sum += case_failures

    if summary_tasks_total != len(task_results_raw):
        raise ValueError(
            f"{source}.summary.tasks_total={summary_tasks_total} does not match "
            f"len(task_results)={len(task_results_raw)}"
        )

    if summary_cases_total != case_totals_sum:
        raise ValueError(
            f"{source}.summary.cases_total={summary_cases_total} does not match "
            f"sum(task.cases_total)={case_totals_sum}"
        )

    if summary_case_failures != case_failures_sum:
        raise ValueError(
            f"{source}.summary.case_failures={summary_case_failures} does not match "
            f"sum(task.case_failures)={case_failures_sum}"
        )

    if summary_tasks_with_errors != tasks_with_errors:
        raise ValueError(
            f"{source}.summary.tasks_with_errors={summary_tasks_with_errors} does not match "
            f"observed value {tasks_with_errors}"
        )

    expected_all_cases_succeeded = summary_case_failures == 0
    if summary_all_cases_succeeded != expected_all_cases_succeeded:
        raise ValueError(
            f"{source}.summary.all_cases_succeeded={summary_all_cases_succeeded} "
            f"does not match (case_failures == 0)={expected_all_cases_succeeded}"
        )

    if suite is not None:
        _validate_suite_alignment(
            suite=suite,
            source=source,
            suite_name=suite_name,
            suite_version=suite_version,
            task_map=task_map,
        )

    return {
        "run_id": run["run_id"],
        "suite_name": run["suite_name"],
        "suite_version": run["suite_version"],
        "adapter": run["adapter"],
        "started_at": run["started_at"],
        "finished_at": run["finished_at"],
        "summary": summary,
        "task_results": task_results_raw,
        "provenance": provenance,
    }


def _validate_suite_alignment(
    *,
    suite: BenchmarkSuite,
    source: str,
    suite_name: str,
    suite_version: str,
    task_map: Mapping[str, dict[str, Any]],
) -> None:
    if suite_name != suite.name:
        raise ValueError(
            f"{source}.suite_name='{suite_name}' does not match suite.name='{suite.name}'"
        )
    if suite_version != suite.version:
        raise ValueError(
            f"{source}.suite_version='{suite_version}' "
            f"does not match suite.version='{suite.version}'"
        )

    expected_task_ids = {task.task_id for task in suite.tasks}
    run_task_ids = set(task_map.keys())
    if run_task_ids != expected_task_ids:
        missing = sorted(expected_task_ids - run_task_ids)
        extra = sorted(run_task_ids - expected_task_ids)
        raise ValueError(
            f"{source}.task_results do not match suite tasks "
            f"(missing={missing}, extra={extra})"
        )

    for task in suite.tasks:
        run_task = task_map[task.task_id]
        run_metric = _require_non_empty_str(
            run_task.get("metric"),
            f"{source}.task_results[{task.task_id}].metric",
        )
        if run_metric != task.metric:
            raise ValueError(
                f"{source} task '{task.task_id}' metric '{run_metric}' does not match "
                f"suite metric '{task.metric}'"
            )

        run_direction = _require_non_empty_str(
            run_task.get("direction"),
            f"{source}.task_results[{task.task_id}].direction",
        )
        expected_direction = metric_direction(task.metric)
        if run_direction != expected_direction:
            raise ValueError(
                f"{source} task '{task.task_id}' direction '{run_direction}' does not match "
                f"expected '{expected_direction}'"
            )

        run_case_results = _require_list(
            run_task.get("case_results"),
            f"{source}.task_results[{task.task_id}].case_results",
        )
        expected_case_ids = {case.case_id for case in task.cases}
        run_case_ids = set()
        for index, case_item in enumerate(run_case_results):
            case = _require_mapping(
                case_item,
                f"{source}.task_results[{task.task_id}].case_results[{index}]",
            )
            run_case_ids.add(
                _require_non_empty_str(
                    case.get("case_id"),
                    f"{source}.task_results[{task.task_id}].case_results[{index}].case_id",
                )
            )

        if run_case_ids != expected_case_ids:
            missing_cases = sorted(expected_case_ids - run_case_ids)
            extra_cases = sorted(run_case_ids - expected_case_ids)
            raise ValueError(
                f"{source} task '{task.task_id}' case ids do not match suite "
                f"(missing={missing_cases}, extra={extra_cases})"
            )

        run_cases_total = _require_non_negative_int(
            run_task.get("cases_total"),
            f"{source}.task_results[{task.task_id}].cases_total",
        )
        if run_cases_total != len(task.cases):
            raise ValueError(
                f"{source} task '{task.task_id}' cases_total={run_cases_total} does not match "
                f"suite case count {len(task.cases)}"
            )


def _validate_key_set(
    payload: Mapping[str, Any],
    *,
    required: set[str],
    path: str,
) -> None:
    keys = set(payload.keys())
    missing = sorted(required - keys)
    extra = sorted(keys - required)
    if missing or extra:
        raise ValueError(f"{path} has invalid keys (missing={missing}, extra={extra})")


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return dict(value)


def _require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list")
    return list(value)


def _require_non_empty_str(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{path} must be a non-empty string")
    return normalized


def _require_non_negative_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{path} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


def _require_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean")
    return value


def _require_non_negative_finite_number(value: Any, path: str) -> float:
    numeric = _require_finite_number(value, path)
    if numeric < 0:
        raise ValueError(f"{path} must be >= 0")
    return numeric


def _require_optional_finite_number(value: Any, path: str) -> float | None:
    if value is None:
        return None
    return _require_finite_number(value, path)


def _require_finite_number(value: Any, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{path} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{path} must be a finite number")
    return numeric
