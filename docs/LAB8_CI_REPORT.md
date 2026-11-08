# Lab 8 - MLOps Pipeline CI

## Purpose

Automate the breast-cancer MLOps training and evaluation pipeline and run the
project tests on GitHub-hosted Linux runners.

## Workflow

The workflow is `.github/workflows/lab8_ci.yml`. It starts on relevant pushes
or pull requests to `main`/`master`, and can also be started manually from the
repository's **Actions** tab. It installs the API requirements, runs
`src/mlops_pipeline/breast_cancer_prediction_pipeline.py`, and runs `pytest`.

## Evidence

Each run uploads a `lab8-evidence-*` artifact containing generated reports,
logs, and outputs when those files exist. The GitHub Actions run status is the
authoritative pass/fail result; no historical churn metrics are carried forward
as breast-cancer results.
