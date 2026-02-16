from __future__ import annotations

import pytest

from refua_bench.schema import BenchmarkSuite, suite_from_mapping


@pytest.fixture
def suite_payload() -> dict[str, object]:
    return {
        "name": "test-suite",
        "version": "1.0.0",
        "description": "test",
        "tasks": [
            {
                "id": "affinity_mae",
                "metric": "mae",
                "prediction_key": "affinity",
                "regression_tolerance": 0.05,
                "cases": [
                    {
                        "id": "a",
                        "input": {"target": "KRAS"},
                        "expected": {"affinity": -9.0},
                    },
                    {
                        "id": "b",
                        "input": {"target": "EGFR"},
                        "expected": {"affinity": -8.0},
                    },
                ],
            },
            {
                "id": "tox_acc",
                "metric": "accuracy",
                "prediction_key": "toxic",
                "regression_tolerance": 0.01,
                "cases": [
                    {
                        "id": "c1",
                        "input": {"smiles": "CCO"},
                        "expected": {"toxic": 0},
                    },
                    {
                        "id": "c2",
                        "input": {"smiles": "N#CCN"},
                        "expected": {"toxic": 1},
                    },
                ],
            },
        ],
    }


@pytest.fixture
def suite(suite_payload: dict[str, object]) -> BenchmarkSuite:
    return suite_from_mapping(suite_payload)
