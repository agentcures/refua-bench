from importlib.metadata import PackageNotFoundError, version

import pytest

import refua_bench


def test_package_version_matches_distribution_metadata() -> None:
    try:
        dist_version = version("refua-bench")
    except PackageNotFoundError:
        pytest.skip("Distribution metadata unavailable in source-only test runs")

    assert refua_bench.__version__ == dist_version
