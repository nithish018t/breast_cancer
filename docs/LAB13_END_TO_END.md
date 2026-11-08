# Lab 13 - Self-Hosted End-to-End Integration

## Purpose

Verify the breast-cancer project from training through a live API prediction in
Docker and Kubernetes, using a self-hosted Windows runner.

## Runner requirements

Register a GitHub Actions self-hosted runner with labels `self-hosted`,
`Windows`, and `X64`. Keep the runner available and configure Python 3.11,
Docker Desktop, and `kubectl` with a working Kubernetes context. The Kubernetes
cluster must be able to access the locally built `breast-cancer-api:latest`
image (as Docker Desktop Kubernetes does when sharing the Docker image store).

Use the **Lab 13 - Self-Hosted Runner Prerequisites** workflow to check that
Python, Git, Docker, and Kubernetes are accessible. GitHub-hosted runners
cannot substitute for this runner because they do not have the local Docker
Desktop Kubernetes cluster.

## End-to-end workflow

`.github/workflows/lab13_end_to_end.yml` starts automatically on pushes to
`main` and can be run manually from **Actions**. It checks out the current
commit into `GITHUB_WORKSPACE`; it does not depend on a fixed machine-specific
project path. It then:

1. Trains and evaluates the model and runs the MLOps stages.
2. Runs the test suite and Lab 9 invalid-data checks.
3. Builds the Docker image and checks `/health` plus a real `/predict` request.
4. Applies the Kubernetes manifests, waits for rollout, and checks `/health`
   plus a real `/predict` request through the service.
5. Uploads generated reports, logs, and prediction outputs as a workflow
   artifact.

The workflow updates the `breast-cancer-api` Kubernetes Deployment in the
configured cluster. Run it only when that cluster is intended for this project.

## Evidence

The GitHub Actions job status and its `lab13-end-to-end-*` artifact are the
run-specific evidence. Local Docker and Kubernetes smoke tests may be run
independently using the commands in [DEPLOYMENT.md](DEPLOYMENT.md).
