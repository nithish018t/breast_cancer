# Lab 9 - Failure and Recovery Tests

## Purpose

Verify that malformed breast-cancer input data is rejected instead of being
silently accepted by the schema validator.

## Workflow

The workflow is `.github/workflows/lab9_failure_tests.yml`. It can be started
from the repository's **Actions** tab and runs automatically when the
failure-injection code or its data-validation dependencies change. It first
builds the normal MLOps outputs, then runs
`src/mlops_pipeline/10_failure_injection.py`.

The injection changes a diagnosis label to an unsupported value, introduces a
non-numeric feature, and removes a required feature. The step passes only when
validation rejects the corrupted dataset.

## Evidence

Each run uploads the generated report, logs, and artifacts as
`lab9-failure-tests-*`. Review the GitHub Actions run and artifact for the
result; this document does not claim that a new remote workflow run has
already happened.
