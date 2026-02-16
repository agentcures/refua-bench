from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_METRICS = {"mae", "rmse", "accuracy", "exact_match", "f1"}


@dataclass(slots=True)
class BenchmarkCase:
    case_id: str
    input: dict[str, Any]
    expected: dict[str, Any]
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class BenchmarkTask:
    task_id: str
    metric: str
    prediction_key: str
    expected_key: str
    regression_tolerance: float = 0.0
    weight: float = 1.0
    positive_label: Any = 1
    cases: list[BenchmarkCase] = field(default_factory=list)


@dataclass(slots=True)
class BenchmarkSuite:
    name: str
    version: str
    description: str
    tasks: list[BenchmarkTask]
    metadata: dict[str, Any] = field(default_factory=dict)


def load_data_file(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    raw = file_path.read_text(encoding="utf-8")
    if file_path.suffix.lower() == ".json":
        data = json.loads(raw)
    else:
        parsed = yaml.safe_load(raw)
        data = {} if parsed is None else parsed
    if not isinstance(data, dict):
        raise ValueError(f"Top-level data in {file_path} must be an object/mapping")
    return data


def dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_suite(path: str | Path) -> BenchmarkSuite:
    payload = load_data_file(path)
    return suite_from_mapping(payload)


def suite_from_mapping(data: Mapping[str, Any]) -> BenchmarkSuite:
    name = _required_str(data, "name")
    version = str(data.get("version", "0.0.1"))
    description = str(data.get("description", ""))

    tasks_raw = data.get("tasks")
    if not isinstance(tasks_raw, list) or not tasks_raw:
        raise ValueError("suite.tasks must be a non-empty list")

    tasks: list[BenchmarkTask] = []
    seen_tasks: set[str] = set()
    for item in tasks_raw:
        task = _parse_task(item)
        if task.task_id in seen_tasks:
            raise ValueError(f"Duplicate task id: {task.task_id}")
        seen_tasks.add(task.task_id)
        tasks.append(task)

    metadata_raw = data.get("metadata", {})
    metadata = dict(metadata_raw) if isinstance(metadata_raw, Mapping) else {}
    return BenchmarkSuite(
        name=name,
        version=version,
        description=description,
        tasks=tasks,
        metadata=metadata,
    )


def _parse_task(item: Any) -> BenchmarkTask:
    if not isinstance(item, Mapping):
        raise ValueError("Each task must be a mapping")

    task_id = _required_str(item, "id")
    metric = _required_str(item, "metric")
    if metric not in SUPPORTED_METRICS:
        allowed = ", ".join(sorted(SUPPORTED_METRICS))
        raise ValueError(f"Unsupported metric '{metric}' for task '{task_id}'. Allowed: {allowed}")

    prediction_key = _required_str(item, "prediction_key")
    expected_key = str(item.get("expected_key", prediction_key))
    regression_tolerance = _float(item.get("regression_tolerance", 0.0), "regression_tolerance")
    weight = _float(item.get("weight", 1.0), "weight")
    positive_label = item.get("positive_label", 1)

    cases_raw = item.get("cases")
    if not isinstance(cases_raw, list) or not cases_raw:
        raise ValueError(f"task '{task_id}' must include a non-empty cases list")

    cases: list[BenchmarkCase] = []
    seen_cases: set[str] = set()
    for raw_case in cases_raw:
        case = _parse_case(raw_case, task_id)
        if case.case_id in seen_cases:
            raise ValueError(f"Duplicate case id '{case.case_id}' in task '{task_id}'")
        seen_cases.add(case.case_id)
        cases.append(case)

    return BenchmarkTask(
        task_id=task_id,
        metric=metric,
        prediction_key=prediction_key,
        expected_key=expected_key,
        regression_tolerance=regression_tolerance,
        weight=weight,
        positive_label=positive_label,
        cases=cases,
    )


def _parse_case(item: Any, task_id: str) -> BenchmarkCase:
    if not isinstance(item, Mapping):
        raise ValueError(f"Cases in task '{task_id}' must be mappings")

    case_id = _required_str(item, "id")

    input_data = item.get("input", {})
    if not isinstance(input_data, Mapping):
        raise ValueError(f"task '{task_id}' case '{case_id}' has non-mapping input")

    expected = item.get("expected")
    if not isinstance(expected, Mapping):
        raise ValueError(f"task '{task_id}' case '{case_id}' must include mapping expected")

    tags_raw = item.get("tags", [])
    if not isinstance(tags_raw, list):
        raise ValueError(f"task '{task_id}' case '{case_id}' tags must be a list")
    tags = [str(tag) for tag in tags_raw]

    return BenchmarkCase(
        case_id=case_id,
        input=dict(input_data),
        expected=dict(expected),
        tags=tags,
    )


def _required_str(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if value is None:
        raise ValueError(f"Missing required key: {key}")
    value_str = str(value).strip()
    if not value_str:
        raise ValueError(f"Key '{key}' must be a non-empty string")
    return value_str


def _float(value: Any, key: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{key}' must be numeric") from exc
