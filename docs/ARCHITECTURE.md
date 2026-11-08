# Architecture

The project is a small train-and-serve workflow for the Wisconsin Diagnostic Breast Cancer dataset:

1. `src/breast_cancer_pipeline.py` validates the configured input columns and `B`/`M` labels, drops the identifier and empty column, performs a stratified holdout split, imputes missing feature values, trains a random forest, and writes metrics and artifacts.
2. `api/` loads the generated model bundle and validates requests against its saved feature list. Predictions return `B`/`M` and the malignant-class probability. Inference logs intentionally do not record measurement values.
3. `Dockerfile` packages the API and model bundle. `k8s/` defines a Deployment with health probes and a ClusterIP Service.

Generated files remain under the project root: model metadata in `deployment/`, processed splits in `data/processed/`, metrics in `reports/`, and predictions in `outputs/`.
