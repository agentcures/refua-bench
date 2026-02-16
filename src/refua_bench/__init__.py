"""refua-bench package."""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("refua-bench")
except PackageNotFoundError:
    # Source-only usage (without installed dist metadata) still imports cleanly.
    __version__ = "0.0.0"
