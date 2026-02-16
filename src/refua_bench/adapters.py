from __future__ import annotations

import importlib
import json
import os
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from refua_bench.schema import BenchmarkCase, BenchmarkTask, load_data_file


@runtime_checkable
class ModelAdapter(Protocol):
    name: str

    def predict(self, task: BenchmarkTask, case: BenchmarkCase) -> Mapping[str, Any]:
        """Return a mapping that includes the task's prediction_key."""


class GoldenAdapter:
    name = "golden"

    def predict(self, task: BenchmarkTask, case: BenchmarkCase) -> Mapping[str, Any]:
        return dict(case.expected)


class FileAdapter:
    name = "file"

    def __init__(self, config: Mapping[str, Any]) -> None:
        predictions_path = config.get("predictions_path")
        if predictions_path is None:
            raise ValueError("file adapter requires 'predictions_path' in adapter config")

        payload = load_data_file(Path(predictions_path))
        self._predictions = self._normalize_predictions(payload)

    @staticmethod
    def _normalize_predictions(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        normalized: dict[str, dict[str, Any]] = {}
        for task_id, task_payload in payload.items():
            if not isinstance(task_payload, Mapping):
                raise ValueError("Each task in predictions file must be a mapping")
            cases: dict[str, Any] = {}
            for case_id, case_prediction in task_payload.items():
                cases[str(case_id)] = case_prediction
            normalized[str(task_id)] = cases
        return normalized

    def predict(self, task: BenchmarkTask, case: BenchmarkCase) -> Mapping[str, Any]:
        task_data = self._predictions.get(task.task_id)
        if task_data is None:
            raise KeyError(f"No predictions for task '{task.task_id}'")
        if case.case_id not in task_data:
            raise KeyError(f"No prediction for case '{case.case_id}' in task '{task.task_id}'")

        case_prediction = task_data[case.case_id]
        if isinstance(case_prediction, Mapping):
            return dict(case_prediction)
        return {task.prediction_key: case_prediction}


class CommandAdapter:
    name = "command"

    def __init__(self, config: Mapping[str, Any]) -> None:
        command = config.get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(item, str) for item in command
        ):
            raise ValueError("command adapter requires a non-empty string list at config.command")
        self._command = command

        env_raw = config.get("env", {})
        if not isinstance(env_raw, Mapping):
            raise ValueError("command adapter 'env' must be a mapping")
        self._env = {str(key): str(value) for key, value in env_raw.items()}

        timeout_value = config.get("timeout_seconds", 60)
        self._timeout_seconds = float(timeout_value)

        include_expected = config.get("include_expected", False)
        self._include_expected = bool(include_expected)

    def predict(self, task: BenchmarkTask, case: BenchmarkCase) -> Mapping[str, Any]:
        payload: dict[str, Any] = {
            "task_id": task.task_id,
            "prediction_key": task.prediction_key,
            "case_id": case.case_id,
            "input": case.input,
        }
        if self._include_expected:
            payload["expected"] = case.expected

        completed = subprocess.run(
            self._command,
            check=False,
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            env={**os.environ, **self._env},
            timeout=self._timeout_seconds,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            raise RuntimeError(
                f"Command adapter failed with code {completed.returncode}: {stderr}"
            )

        stdout = completed.stdout.strip()
        if not stdout:
            raise RuntimeError("Command adapter returned empty stdout")

        parsed = _parse_last_json_line(stdout)
        if not isinstance(parsed, Mapping):
            raise RuntimeError("Command adapter must return a JSON object")
        return dict(parsed)


def _parse_last_json_line(stdout: str) -> Any:
    lines = [line for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError("Unable to parse JSON output from command adapter")


def load_adapter(spec: str, config: Mapping[str, Any] | None = None) -> ModelAdapter:
    adapter_config: Mapping[str, Any] = {} if config is None else config

    builtins: dict[str, Callable[[Mapping[str, Any]], ModelAdapter]] = {
        "golden": lambda _cfg: GoldenAdapter(),
        "file": lambda cfg: FileAdapter(cfg),
        "command": lambda cfg: CommandAdapter(cfg),
    }
    if spec in builtins:
        return builtins[spec](adapter_config)

    module_name, sep, attr_name = spec.partition(":")
    if not sep:
        raise ValueError("Custom adapter must use format 'module.path:AdapterClass'")

    module = importlib.import_module(module_name)
    target = getattr(module, attr_name)

    if isinstance(target, type):
        try:
            instance = target(adapter_config)
        except TypeError:
            instance = target()
    else:
        try:
            instance = target(adapter_config)
        except TypeError:
            instance = target()

    if not isinstance(instance, ModelAdapter):
        raise TypeError(f"Loaded adapter from '{spec}' does not implement ModelAdapter")

    return instance
