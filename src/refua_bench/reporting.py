from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload at {path} must be an object")
    return payload


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: str | Path, text: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text.strip() + "\n", encoding="utf-8")


def render_run_markdown(run_payload: dict[str, Any]) -> str:
    lines = [
        f"# Benchmark Run: {run_payload.get('suite_name', '<unknown>')}",
        "",
        f"- Run ID: `{run_payload.get('run_id', '<none>')}`",
        f"- Suite Version: `{run_payload.get('suite_version', '<none>')}`",
        f"- Adapter: `{run_payload.get('adapter', '<none>')}`",
    ]

    summary = run_payload.get("summary", {})
    if isinstance(summary, dict):
        lines.extend(
            [
                f"- Cases Total: `{summary.get('cases_total', 0)}`",
                f"- Case Failures: `{summary.get('case_failures', 0)}`",
                f"- Tasks With Errors: `{summary.get('tasks_with_errors', 0)}`",
            ]
        )

    provenance = run_payload.get("provenance")
    if isinstance(provenance, dict):
        model = provenance.get("model", {})
        git = provenance.get("git", {})
        lines.extend(["", "## Provenance"])
        if isinstance(model, dict):
            lines.append(f"- Model Name: `{model.get('name', '<none>')}`")
            lines.append(f"- Model Version: `{model.get('version', '<none>')}`")
            lines.append(f"- Adapter Spec: `{model.get('adapter_spec', '<none>')}`")
        if isinstance(git, dict):
            lines.append(f"- Git Commit: `{git.get('commit', '<none>')}`")
            lines.append(f"- Git Dirty: `{git.get('dirty', '<none>')}`")

    lines.extend(
        [
            "",
            "## Task Scores",
            "",
            "| Task | Metric | Score | Failures |",
            "|---|---:|---:|---:|",
        ]
    )

    tasks = run_payload.get("task_results", [])
    if isinstance(tasks, list):
        for task in tasks:
            if not isinstance(task, dict):
                continue
            score = task.get("score")
            score_str = "n/a" if score is None else f"{float(score):.6f}"
            task_id = task.get("task_id", "<unknown>")
            metric = task.get("metric", "<none>")
            failures = task.get("case_failures", 0)
            lines.append(f"| {task_id} | {metric} | {score_str} | {failures} |")

    return "\n".join(lines)


def render_compare_markdown(report_payload: dict[str, Any]) -> str:
    summary = report_payload.get("summary", {})
    passed = bool(summary.get("passed", False)) if isinstance(summary, dict) else False

    lines = [
        f"# Benchmark Compare: {report_payload.get('suite_name', '<unknown>')}",
        "",
        f"- Suite Version: `{report_payload.get('suite_version', '<none>')}`",
        f"- Passed: `{passed}`",
    ]

    if isinstance(summary, dict):
        lines.extend(
            [
                f"- Tasks Total: `{summary.get('tasks_total', 0)}`",
                f"- Regressions: `{summary.get('regressions', 0)}`",
                f"- Uncertain: `{summary.get('uncertain', 0)}`",
            ]
        )

    policy = report_payload.get("policy", {})
    if isinstance(policy, dict):
        lines.extend(
            [
                f"- Min Effect Size: `{policy.get('min_effect_size', 0)}`",
                f"- Bootstrap Resamples: `{policy.get('bootstrap_resamples', 0)}`",
                f"- Confidence Level: `{policy.get('confidence_level', 0.95)}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Task Deltas",
            "",
            (
                "| Task | Metric | Baseline | Candidate | Delta | Threshold | "
                "CI Low | CI High | Status |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )

    tasks = report_payload.get("task_comparisons", [])
    if isinstance(tasks, list):
        for task in tasks:
            if not isinstance(task, dict):
                continue
            baseline = task.get("baseline_score")
            candidate = task.get("candidate_score")
            delta = task.get("delta")
            baseline_str = "n/a" if baseline is None else f"{float(baseline):.6f}"
            candidate_str = "n/a" if candidate is None else f"{float(candidate):.6f}"
            delta_str = "n/a" if delta is None else f"{float(delta):+.6f}"
            ci_low = task.get("ci_low")
            ci_high = task.get("ci_high")
            ci_low_str = "n/a" if ci_low is None else f"{float(ci_low):.6f}"
            ci_high_str = "n/a" if ci_high is None else f"{float(ci_high):.6f}"
            task_id = task.get("task_id", "<unknown>")
            metric = task.get("metric", "<none>")
            threshold = task.get("threshold", 0)
            status = task.get("status", "<none>")
            lines.append(
                f"| {task_id} | {metric} | {baseline_str} | {candidate_str} | {delta_str} | "
                f"{threshold} | {ci_low_str} | {ci_high_str} | {status} |"
            )

    return "\n".join(lines)
