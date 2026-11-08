# Lab 12 - Breast-Cancer Monitoring and Retraining

## What Lab 12 covers

The numbered scripts in `src/mlops_pipeline/12_*.py` retain the Lab 12 entry
points. Their shared implementation is in `lab12_monitoring.py`. All dataset
fields, labels, model metrics, and API logs use the breast-cancer project.

The monitoring exercise generates **synthetic** covariate and label-shift
examples from the public Wisconsin Diagnostic Breast Cancer dataset. They are
clearly marked in their CSVs and reports. The generated examples are only for
testing monitoring; they are not patient data, should not be treated as real
diagnoses, and are never automatically used to retrain or replace the
production model.

## Run the full Lab 12 workflow

From the project root in PowerShell, with the project environment active:

```powershell
python src\breast_cancer_pipeline.py
python src\mlops_pipeline\12_monitor_trigger.py
python src\mlops_pipeline\12_retrain_model.py
python src\mlops_pipeline\12_model_comparison.py
python src\mlops_pipeline\12_update_mlflow_registry.py
```

Or start **Lab 12 - Breast Cancer Monitoring and Retraining** from the
repository's GitHub Actions tab. Its evidence is uploaded as a
`lab12-monitoring-*` artifact.

## Individual scripts

| Script | Purpose |
|---|---|
| `12_create_evolved_dataset.py` | Generate a deterministic synthetic feature-drift sample. |
| `12_create_concept_drift_dataset.py` | Generate a deterministic synthetic label-shift sample. |
| `12_create_updated_dataset.py` | Combine baseline and simulated rows for demonstration only. |
| `12_data_drift.py` | Compare numeric feature distributions with the two-sample KS statistic. |
| `12_prediction_drift.py` | Compare predicted malignant-probability distributions. |
| `12_concept_drift.py` | Compare classification metrics against simulated shifted labels. |
| `12_drift_analysis.py` | Combine the three drift reports and recommend review when needed. |
| `12_performance_monitoring.py` | Evaluate the production model on the fixed stratified holdout. |
| `12_performance_trend.py` | Report holdout metrics across deterministic windows. |
| `12_prediction_monitoring.py` | Summarize prediction counts and probabilities in API logs. |
| `12_api_performance_monitoring.py` | Summarize API inference latency from logs. |
| `12_request_history.py` | Summarize request IDs and failures without feature values. |
| `12_retrain_model.py` | Train a separate candidate from the source training split and, optionally, a validated labeled incoming batch. |
| `12_model_comparison.py` | Compare the candidate and deployed model on the same holdout. |
| `12_update_mlflow_registry.py` | Record the candidate and decision in the local registry manifest. |
| `12_monitor_trigger.py` | Run the monitoring reports and produce the Lab 12 trigger report. |

`12_update_mlflow_registry.py` keeps its historical filename for compatibility;
the current implementation records a local candidate manifest under
`models/registry/` and does not require or contact an MLflow server. Registering
a candidate does not promote or deploy it. Promotion remains a separate,
reviewed operation.

To retrain with newly collected data, place a **labeled** CSV at
`data/monitoring/incoming/labeled_data.csv` before running
`12_retrain_model.py`. It must contain `id`, `diagnosis` (`B` or `M`), and all
30 configured feature columns. IDs must be unique and must not overlap the
source dataset. This incoming-data folder is Git-ignored. The simulation files
under `outputs/` are not accepted as incoming training data. Candidate metrics
are still evaluated on the fixed source holdout; a candidate is not
automatically promoted.

## Outputs

- `outputs/drift/` — synthetic samples and feature, prediction, concept, and
  combined drift reports.
- `outputs/monitoring/` — model performance, API event, latency, request-history,
  and monitor-trigger reports.
- `outputs/retraining/` — challenger evaluation, model comparison, and registry
  decision reports.
- `models/` — ignored local candidate model and candidate registry artifacts.

The API logger stores the number of submitted features and the prediction
result; it does not persist raw measurement values. No metric from synthetic
monitoring data should be interpreted as clinical evidence.
