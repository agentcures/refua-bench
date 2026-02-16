from __future__ import annotations

import pytest

from refua_bench.schema import load_data_file, suite_from_mapping


def test_suite_defaults(suite_payload: dict[str, object]) -> None:
    suite = suite_from_mapping(suite_payload)
    assert suite.name == "test-suite"
    assert len(suite.tasks) == 2
    assert suite.tasks[0].expected_key == "affinity"


def test_invalid_metric_raises(suite_payload: dict[str, object]) -> None:
    payload = dict(suite_payload)
    tasks = list(payload["tasks"])
    tasks[0] = dict(tasks[0])
    tasks[0]["metric"] = "pearson"
    payload["tasks"] = tasks

    with pytest.raises(ValueError, match="Unsupported metric"):
        suite_from_mapping(payload)


def test_load_data_file_yaml(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "suite.yaml"
    path.write_text("name: a\ntasks: []\n", encoding="utf-8")
    payload = load_data_file(path)
    assert payload["name"] == "a"
