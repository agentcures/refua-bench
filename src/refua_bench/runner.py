from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from uuid import uuid4

from refua_bench.adapters import ModelAdapter
from refua_bench.metrics import compute_metric, metric_direction
from refua_bench.schema import BenchmarkSuite, BenchmarkTask


@dataclass(slots=True)
class CaseResult:
    case_id: str
    expected: Any
    predicted: Any
    duration_ms: float
    error: str | None = None


@dataclass(slots=True)
class TaskResult:
    task_id: str
    metric: str
    direction: str
    score: float | None
    cases_total: int
    case_failures: int
    case_results: list[CaseResult]


@dataclass(slots=True)
class BenchmarkRun:
    run_id: str
    suite_name: str
    suite_version: str
    adapter: str
    started_at: str
    finished_at: str
    summary: dict[str, Any]
    task_results: list[TaskResult]
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_benchmark(
    suite: BenchmarkSuite,
    adapter: ModelAdapter,
    *,
    provenance: dict[str, Any] | None = None,
) -> BenchmarkRun:
    started_at = datetime.now(UTC)
    task_results: list[TaskResult] = []

    cases_total = 0
    case_failures = 0
    tasks_with_errors = 0

    for task in suite.tasks:
        result = _run_task(task, adapter)
        task_results.append(result)
        cases_total += result.cases_total
        case_failures += result.case_failures
        if result.case_failures > 0 or result.score is None:
            tasks_with_errors += 1

    summary = {
        "tasks_total": len(suite.tasks),
        "tasks_with_errors": tasks_with_errors,
        "cases_total": cases_total,
        "case_failures": case_failures,
        "all_cases_succeeded": case_failures == 0,
    }

    finished_at = datetime.now(UTC)
    return BenchmarkRun(
        run_id=uuid4().hex,
        suite_name=suite.name,
        suite_version=suite.version,
        adapter=adapter.name,
        started_at=started_at.isoformat(),
        finished_at=finished_at.isoformat(),
        summary=summary,
        provenance={} if provenance is None else dict(provenance),
        task_results=task_results,
    )


def _run_task(task: BenchmarkTask, adapter: ModelAdapter) -> TaskResult:
    expected_values: list[Any] = []
    predicted_values: list[Any] = []

    case_results: list[CaseResult] = []
    case_failures = 0

    for case in task.cases:
        expected = case.expected.get(task.expected_key)
        predicted: Any = None
        error: str | None = None
        start = perf_counter()

        try:
            output = adapter.predict(task, case)
            if task.prediction_key not in output:
                raise KeyError(
                    "Missing prediction key "
                    f"'{task.prediction_key}' in adapter output for case '{case.case_id}'"
                )
            predicted = output[task.prediction_key]
            expected_values.append(expected)
            predicted_values.append(predicted)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            case_failures += 1

        duration_ms = (perf_counter() - start) * 1000.0
        case_results.append(
            CaseResult(
                case_id=case.case_id,
                expected=expected,
                predicted=predicted,
                duration_ms=duration_ms,
                error=error,
            )
        )

    score: float | None = None
    if expected_values:
        score = compute_metric(
            task.metric,
            expected_values,
            predicted_values,
            positive_label=task.positive_label,
        )

    return TaskResult(
        task_id=task.task_id,
        metric=task.metric,
        direction=metric_direction(task.metric),
        score=score,
        cases_total=len(task.cases),
        case_failures=case_failures,
        case_results=case_results,
    )
