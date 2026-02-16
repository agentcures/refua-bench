# refua-bench

`refua-bench` is a standalone benchmark and regression-gating project for Refua model workflows.
It benchmarks current and future models via adapter interfaces and enforces safe regression gates.

## What It Provides

- Benchmark suite schema (`yaml`/`json`) for tasks, metrics, tolerances, and case sets.
- Adapter system for model execution:
  - `golden`: uses expected outputs (sanity checks)
  - `file`: reads predictions from a JSON artifact
  - `command`: calls any executable that reads JSON stdin and returns JSON stdout
  - custom adapters via `module.path:AdapterClass`
- Run artifacts in JSON + markdown.
- Automatic run provenance capture (git/runtime/model/dependencies).
- Statistical regression gating (minimum practical effect + bootstrap confidence intervals).
- Baseline registry with named baselines and safe promotion flow.

## Install

```bash
cd refua-bench
poetry install
```

## Build

```bash
poetry build
```

## CLI

```bash
poetry run refua-bench --help
```

### 1. Run a benchmark

```bash
poetry run refua-bench run \
  --suite benchmarks/sample_suite.yaml \
  --adapter file \
  --adapter-config benchmarks/sample_file_adapter_config.yaml \
  --model-name boltz2-affinity \
  --model-version 2026-02-12 \
  --output artifacts/candidate_run.json \
  --markdown artifacts/candidate_run.md
```

By default, each run stores provenance in `run.provenance`.

### 2. Compare candidate vs baseline

```bash
poetry run refua-bench compare \
  --suite benchmarks/sample_suite.yaml \
  --baseline benchmarks/sample_baseline_run.json \
  --candidate artifacts/candidate_run.json \
  --output artifacts/compare.json \
  --markdown artifacts/compare.md
```

### 3. Statistical gating

```bash
poetry run refua-bench compare \
  --suite benchmarks/sample_suite.yaml \
  --baseline benchmarks/sample_baseline_run.json \
  --candidate artifacts/candidate_run.json \
  --output artifacts/compare_stats.json \
  --min-effect-size 0.02 \
  --bootstrap-resamples 2000 \
  --confidence-level 0.95 \
  --bootstrap-seed 7 \
  --fail-on-uncertain
```

Interpretation:

- `min-effect-size`: ignores changes too small to matter practically.
- `bootstrap-resamples`: enables CI-based robustness checks.
- `--fail-on-uncertain`: optional strict mode for inconclusive bootstrap tasks.

### 4. Run + compare in one command (`gate`)

```bash
poetry run refua-bench gate \
  --suite benchmarks/sample_suite.yaml \
  --baseline benchmarks/sample_baseline_run.json \
  --adapter file \
  --adapter-config benchmarks/sample_file_adapter_config.yaml \
  --candidate-output artifacts/candidate_run.json \
  --output artifacts/gate_report.json \
  --min-effect-size 0.02 \
  --bootstrap-resamples 1000
```

### 5. Baseline registry and promotion

Promote an initial baseline:

```bash
poetry run refua-bench baseline promote \
  --registry artifacts/baseline_registry.json \
  --suite benchmarks/sample_suite.yaml \
  --baseline-name stable \
  --candidate benchmarks/sample_baseline_run.json
```

Compare against named baseline:

```bash
poetry run refua-bench compare \
  --suite benchmarks/sample_suite.yaml \
  --registry artifacts/baseline_registry.json \
  --baseline-name stable \
  --candidate artifacts/candidate_run.json \
  --output artifacts/compare_named.json
```

Promote a new candidate safely (fails if regression is detected):

```bash
poetry run refua-bench baseline promote \
  --registry artifacts/baseline_registry.json \
  --suite benchmarks/sample_suite.yaml \
  --baseline-name stable \
  --candidate artifacts/candidate_run.json \
  --min-effect-size 0.02 \
  --bootstrap-resamples 2000
```

List/resolve baselines:

```bash
poetry run refua-bench baseline list --registry artifacts/baseline_registry.json
poetry run refua-bench baseline resolve \
  --registry artifacts/baseline_registry.json \
  --suite benchmarks/sample_suite.yaml \
  --baseline-name stable
```

### 6. Scaffold a new suite

```bash
poetry run refua-bench init --directory benchmarks/new_suite --name refua-next
```

## Suite Schema

```yaml
name: refua-core-smoke
version: 1.0.0
description: smoke checks
tasks:
  - id: affinity_mae
    metric: mae
    prediction_key: affinity
    expected_key: affinity  # optional, defaults to prediction_key
    regression_tolerance: 0.05
    weight: 2.0
    positive_label: 1       # used by f1
    cases:
      - id: case_1
        input: {target: KRAS, ligand: MRTX1133}
        expected: {affinity: -9.3}
```

Supported metrics:

- `mae`
- `rmse`
- `accuracy`
- `exact_match`
- `f1` (binary)

## Prediction File Format (`file` adapter)

```json
{
  "affinity_mae": {
    "case_1": {"affinity": -9.1}
  }
}
```

## Command Adapter Contract

Input (stdin):

```json
{
  "task_id": "affinity_mae",
  "prediction_key": "affinity",
  "case_id": "case_1",
  "input": {"target": "KRAS", "ligand": "MRTX1133"}
}
```

Output (stdout):

```json
{"affinity": -9.2}
```

## Tests

```bash
poetry run pytest
```
