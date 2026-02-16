from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

from refua_bench.metrics import compute_metric, metric_direction
from refua_bench.schema import BenchmarkSuite


@dataclass(slots=True)
class StatisticalPolicy:
    min_effect_size: float = 0.0
    bootstrap_resamples: int = 0
    confidence_level: float = 0.95
    bootstrap_seed: int | None = None
    fail_on_uncertain: bool = False

    @property
    def bootstrap_enabled(self) -> bool:
        return self.bootstrap_resamples > 0


@dataclass(slots=True)
class TaskComparison:
    task_id: str
    metric: str
    direction: str
    baseline_score: float | None
    candidate_score: float | None
    tolerance: float
    threshold: float
    delta: float | None
    regression_amount: float | None
    ci_low: float | None
    ci_high: float | None
    p_regression: float | None
    paired_cases: int
    status: str
    message: str


@dataclass(slots=True)
class ComparisonReport:
    suite_name: str
    suite_version: str
    summary: dict[str, Any]
    policy: dict[str, Any]
    task_comparisons: list[TaskComparison]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_runs(
    suite: BenchmarkSuite,
    baseline_run: dict[str, Any],
    candidate_run: dict[str, Any],
    *,
    policy: StatisticalPolicy | None = None,
) -> ComparisonReport:
    active_policy = StatisticalPolicy() if policy is None else policy

    baseline_tasks = _task_map(baseline_run)
    candidate_tasks = _task_map(candidate_run)

    comparisons: list[TaskComparison] = []
    regressions = 0
    uncertain = 0

    for task_index, task in enumerate(suite.tasks):
        baseline_task = baseline_tasks.get(task.task_id)
        candidate_task = candidate_tasks.get(task.task_id)

        threshold = max(task.regression_tolerance, active_policy.min_effect_size)

        if baseline_task is None:
            regressions += 1
            comparisons.append(
                TaskComparison(
                    task_id=task.task_id,
                    metric=task.metric,
                    direction=metric_direction(task.metric),
                    baseline_score=None,
                    candidate_score=_extract_score(candidate_task),
                    tolerance=task.regression_tolerance,
                    threshold=threshold,
                    delta=None,
                    regression_amount=None,
                    ci_low=None,
                    ci_high=None,
                    p_regression=None,
                    paired_cases=0,
                    status="regression",
                    message="Task missing in baseline run",
                )
            )
            continue

        if candidate_task is None:
            regressions += 1
            comparisons.append(
                TaskComparison(
                    task_id=task.task_id,
                    metric=task.metric,
                    direction=metric_direction(task.metric),
                    baseline_score=_extract_score(baseline_task),
                    candidate_score=None,
                    tolerance=task.regression_tolerance,
                    threshold=threshold,
                    delta=None,
                    regression_amount=None,
                    ci_low=None,
                    ci_high=None,
                    p_regression=None,
                    paired_cases=0,
                    status="regression",
                    message="Task missing in candidate run",
                )
            )
            continue

        baseline_score = _extract_score(baseline_task)
        candidate_score = _extract_score(candidate_task)

        if baseline_score is None or candidate_score is None:
            regressions += 1
            comparisons.append(
                TaskComparison(
                    task_id=task.task_id,
                    metric=task.metric,
                    direction=metric_direction(task.metric),
                    baseline_score=baseline_score,
                    candidate_score=candidate_score,
                    tolerance=task.regression_tolerance,
                    threshold=threshold,
                    delta=None,
                    regression_amount=None,
                    ci_low=None,
                    ci_high=None,
                    p_regression=None,
                    paired_cases=0,
                    status="regression",
                    message="Missing score in one of the runs",
                )
            )
            continue

        direction = metric_direction(task.metric)
        delta = candidate_score - baseline_score
        regression_amount = delta if direction == "lower" else -delta

        ci_low: float | None = None
        ci_high: float | None = None
        p_regression: float | None = None
        paired_cases = 0

        status = "pass"
        message = "Within threshold"

        if regression_amount > threshold:
            status = "regression"
            message = "Exceeded threshold"

        if active_policy.bootstrap_enabled:
            paired_cases_data = _paired_cases(baseline_task, candidate_task)
            paired_cases = len(paired_cases_data)
            bootstrap_stats = _bootstrap_regression_stats(
                metric=task.metric,
                direction=direction,
                positive_label=task.positive_label,
                pairs=paired_cases_data,
                threshold=threshold,
                resamples=active_policy.bootstrap_resamples,
                confidence_level=active_policy.confidence_level,
                seed=_seed_for_task(active_policy.bootstrap_seed, task_index),
            )

            if bootstrap_stats is not None:
                ci_low, ci_high, p_regression = bootstrap_stats
                if regression_amount <= threshold:
                    status = "pass"
                    message = "Within threshold"
                elif ci_low > threshold:
                    status = "regression"
                    message = "Bootstrap CI confirms regression"
                elif ci_high <= threshold:
                    status = "pass"
                    message = "Bootstrap CI rejects regression"
                else:
                    status = "uncertain"
                    message = "Regression signal is inconclusive under bootstrap CI"
            elif regression_amount > threshold:
                status = "uncertain"
                message = "Insufficient paired cases for bootstrap CI"

        if status == "regression":
            regressions += 1
        elif status == "uncertain":
            uncertain += 1

        comparisons.append(
            TaskComparison(
                task_id=task.task_id,
                metric=task.metric,
                direction=direction,
                baseline_score=baseline_score,
                candidate_score=candidate_score,
                tolerance=task.regression_tolerance,
                threshold=threshold,
                delta=delta,
                regression_amount=regression_amount,
                ci_low=ci_low,
                ci_high=ci_high,
                p_regression=p_regression,
                paired_cases=paired_cases,
                status=status,
                message=message,
            )
        )

    passed = regressions == 0 and (
        uncertain == 0 if active_policy.fail_on_uncertain else True
    )
    summary = {
        "tasks_total": len(comparisons),
        "regressions": regressions,
        "uncertain": uncertain,
        "passed": passed,
    }

    policy_payload = {
        "min_effect_size": active_policy.min_effect_size,
        "bootstrap_resamples": active_policy.bootstrap_resamples,
        "confidence_level": active_policy.confidence_level,
        "bootstrap_seed": active_policy.bootstrap_seed,
        "fail_on_uncertain": active_policy.fail_on_uncertain,
    }

    return ComparisonReport(
        suite_name=suite.name,
        suite_version=suite.version,
        summary=summary,
        policy=policy_payload,
        task_comparisons=comparisons,
    )


def _seed_for_task(seed: int | None, task_index: int) -> int | None:
    if seed is None:
        return None
    return seed + task_index


def _task_map(run_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_tasks = run_payload.get("task_results", [])
    if not isinstance(raw_tasks, list):
        return {}

    task_map: dict[str, dict[str, Any]] = {}
    for task in raw_tasks:
        if not isinstance(task, dict):
            continue
        task_id = task.get("task_id")
        if isinstance(task_id, str):
            task_map[task_id] = task
    return task_map


def _extract_score(task_payload: dict[str, Any] | None) -> float | None:
    if not isinstance(task_payload, dict):
        return None
    score = task_payload.get("score")
    if score is None:
        return None
    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def _paired_cases(
    baseline_task: dict[str, Any],
    candidate_task: dict[str, Any],
) -> list[tuple[Any, Any, Any, Any]]:
    baseline_cases = _case_map(baseline_task)
    candidate_cases = _case_map(candidate_task)

    pairs: list[tuple[Any, Any, Any, Any]] = []
    for case_id, baseline_case in baseline_cases.items():
        candidate_case = candidate_cases.get(case_id)
        if candidate_case is None:
            continue

        if baseline_case.get("error") is not None or candidate_case.get("error") is not None:
            continue

        baseline_expected = baseline_case.get("expected")
        baseline_predicted = baseline_case.get("predicted")
        candidate_expected = candidate_case.get("expected")
        candidate_predicted = candidate_case.get("predicted")

        if baseline_expected is None or baseline_predicted is None:
            continue
        if candidate_expected is None or candidate_predicted is None:
            continue

        pairs.append(
            (
                baseline_expected,
                baseline_predicted,
                candidate_expected,
                candidate_predicted,
            )
        )

    return pairs


def _case_map(task_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_cases = task_payload.get("case_results", [])
    if not isinstance(raw_cases, list):
        return {}

    case_map: dict[str, dict[str, Any]] = {}
    for case in raw_cases:
        if not isinstance(case, dict):
            continue
        case_id = case.get("case_id")
        if isinstance(case_id, str):
            case_map[case_id] = case
    return case_map


def _bootstrap_regression_stats(
    *,
    metric: str,
    direction: str,
    positive_label: Any,
    pairs: list[tuple[Any, Any, Any, Any]],
    threshold: float,
    resamples: int,
    confidence_level: float,
    seed: int | None,
) -> tuple[float, float, float] | None:
    if len(pairs) < 2 or resamples <= 0:
        return None

    rng = random.Random(seed)
    n_cases = len(pairs)

    baseline_expected = [item[0] for item in pairs]
    baseline_predicted = [item[1] for item in pairs]
    candidate_expected = [item[2] for item in pairs]
    candidate_predicted = [item[3] for item in pairs]

    regression_samples: list[float] = []
    regression_hits = 0

    for _ in range(resamples):
        indices = [rng.randrange(n_cases) for _ in range(n_cases)]

        b_expected = [baseline_expected[idx] for idx in indices]
        b_predicted = [baseline_predicted[idx] for idx in indices]
        c_expected = [candidate_expected[idx] for idx in indices]
        c_predicted = [candidate_predicted[idx] for idx in indices]

        try:
            baseline_score = compute_metric(
                metric,
                b_expected,
                b_predicted,
                positive_label=positive_label,
            )
            candidate_score = compute_metric(
                metric,
                c_expected,
                c_predicted,
                positive_label=positive_label,
            )
        except ValueError:
            return None

        delta = candidate_score - baseline_score
        regression_amount = delta if direction == "lower" else -delta
        regression_samples.append(regression_amount)
        if regression_amount > threshold:
            regression_hits += 1

    alpha = 1.0 - confidence_level
    ci_low = _quantile(regression_samples, alpha / 2.0)
    ci_high = _quantile(regression_samples, 1.0 - (alpha / 2.0))
    p_regression = regression_hits / len(regression_samples)

    return ci_low, ci_high, p_regression


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot compute quantile on empty values")

    sorted_values = sorted(values)
    if probability <= 0:
        return sorted_values[0]
    if probability >= 1:
        return sorted_values[-1]

    position = probability * (len(sorted_values) - 1)
    low_index = int(position)
    high_index = min(low_index + 1, len(sorted_values) - 1)
    fraction = position - low_index
    low_value = sorted_values[low_index]
    high_value = sorted_values[high_index]
    return low_value + (high_value - low_value) * fraction
