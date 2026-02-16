from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REGISTRY_VERSION = 1


def load_registry(path: str | Path) -> dict[str, Any]:
    registry_path = Path(path)
    if not registry_path.exists():
        return {
            "version": REGISTRY_VERSION,
            "baselines": {},
        }

    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Registry JSON must be an object")

    version = payload.get("version")
    baselines = payload.get("baselines")
    if not isinstance(version, int):
        raise ValueError("Registry must include integer 'version'")
    if not isinstance(baselines, dict):
        raise ValueError("Registry must include object 'baselines'")

    return payload


def save_registry(path: str | Path, registry: Mapping[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(dict(registry), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def list_baselines(
    registry: Mapping[str, Any],
    *,
    suite_name: str | None = None,
) -> list[dict[str, Any]]:
    baselines_root = registry.get("baselines")
    if not isinstance(baselines_root, Mapping):
        return []

    rows: list[dict[str, Any]] = []
    for current_suite_name, suite_entries in baselines_root.items():
        if suite_name is not None and current_suite_name != suite_name:
            continue
        if not isinstance(suite_entries, Mapping):
            continue

        for baseline_name, baseline_payload in suite_entries.items():
            if not isinstance(baseline_payload, Mapping):
                continue
            current = baseline_payload.get("current")
            if not isinstance(current, Mapping):
                continue

            rows.append(
                {
                    "suite_name": str(current_suite_name),
                    "baseline_name": str(baseline_name),
                    "run_path": str(current.get("run_path", "")),
                    "suite_version": current.get("suite_version"),
                    "updated_at": current.get("updated_at"),
                    "run_id": current.get("run_id"),
                }
            )

    rows.sort(key=lambda item: (item["suite_name"], item["baseline_name"]))
    return rows


def resolve_baseline_path(
    registry: Mapping[str, Any],
    *,
    suite_name: str,
    baseline_name: str,
) -> Path:
    entry = get_baseline_entry(registry, suite_name=suite_name, baseline_name=baseline_name)
    if entry is None:
        raise KeyError(f"No baseline named '{baseline_name}' for suite '{suite_name}'")

    run_path = entry.get("run_path")
    if not isinstance(run_path, str) or not run_path:
        raise ValueError(
            f"Baseline '{baseline_name}' for suite '{suite_name}' has invalid run_path"
        )

    return Path(run_path)


def get_baseline_entry(
    registry: Mapping[str, Any],
    *,
    suite_name: str,
    baseline_name: str,
) -> dict[str, Any] | None:
    baselines_root = registry.get("baselines")
    if not isinstance(baselines_root, Mapping):
        return None

    suite_entries = baselines_root.get(suite_name)
    if not isinstance(suite_entries, Mapping):
        return None

    baseline_payload = suite_entries.get(baseline_name)
    if not isinstance(baseline_payload, Mapping):
        return None

    current = baseline_payload.get("current")
    if not isinstance(current, Mapping):
        return None

    return dict(current)


def promote_baseline(
    *,
    registry: dict[str, Any],
    registry_path: str | Path,
    suite_name: str,
    suite_version: str,
    baseline_name: str,
    candidate_path: str | Path,
    notes: str | None,
    provenance: Mapping[str, Any] | None,
    compare_summary: Mapping[str, Any] | None,
    store_dir: str | Path | None = None,
) -> dict[str, Any]:
    candidate_file = Path(candidate_path)
    if not candidate_file.exists():
        raise FileNotFoundError(f"Candidate run not found: {candidate_file}")

    promoted_at = datetime.now(UTC).isoformat()
    registry_file = Path(registry_path)

    if store_dir is None:
        target_dir = registry_file.parent / "baselines" / _slug(suite_name) / _slug(baseline_name)
    else:
        target_dir = Path(store_dir) / _slug(suite_name) / _slug(baseline_name)

    target_dir.mkdir(parents=True, exist_ok=True)

    run_payload = json.loads(candidate_file.read_text(encoding="utf-8"))
    if not isinstance(run_payload, dict):
        raise ValueError("Candidate run artifact must be a JSON object")
    run_id = str(run_payload.get("run_id", "unknown-run"))

    timestamp = promoted_at.replace(":", "").replace("+", "Z")
    target_name = f"{timestamp}__{_slug(run_id)}.json"
    target_path = target_dir / target_name

    shutil.copy2(candidate_file, target_path)

    baselines_root = registry.setdefault("baselines", {})
    if not isinstance(baselines_root, dict):
        raise ValueError("Registry 'baselines' must be an object")

    suite_entries = baselines_root.setdefault(suite_name, {})
    if not isinstance(suite_entries, dict):
        raise ValueError(f"Registry entry for suite '{suite_name}' must be an object")

    baseline_entry = suite_entries.setdefault(baseline_name, {})
    if not isinstance(baseline_entry, dict):
        raise ValueError(
            f"Registry entry for suite '{suite_name}' baseline '{baseline_name}' must be an object"
        )

    history = baseline_entry.setdefault("history", [])
    if not isinstance(history, list):
        raise ValueError(
            f"Registry history for suite '{suite_name}' baseline '{baseline_name}' must be a list"
        )

    current = {
        "suite_name": suite_name,
        "suite_version": suite_version,
        "baseline_name": baseline_name,
        "run_path": str(target_path.resolve()),
        "run_id": run_payload.get("run_id"),
        "updated_at": promoted_at,
        "notes": notes,
        "summary": run_payload.get("summary"),
        "provenance": {} if provenance is None else dict(provenance),
        "compare_summary": None if compare_summary is None else dict(compare_summary),
    }

    history.append(current)
    baseline_entry["current"] = current

    return current


def _slug(value: str) -> str:
    result = []
    for char in value.lower().strip():
        if char.isalnum():
            result.append(char)
        else:
            result.append("-")
    slug = "".join(result).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "default"
