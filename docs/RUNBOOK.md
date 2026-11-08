# Runbook

## Train

From the repository root, install the API dependencies and run the pipeline:

```powershell
python -m pip install -r requirements-api.txt
python src\breast_cancer_pipeline.py
```

The command fails explicitly if the configured dataset or required columns are absent. It writes the model to `deployment\model.pkl`, feature metadata to `deployment\model_metadata.json`, evaluation results to `reports\evaluation_report.json`, and held-out predictions to `outputs\inference_predictions.csv`.

## Test and serve

```powershell
python -m pip install pytest
python -m pytest -q
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Use `/health` for readiness and `/docs` for the request schema. Every prediction request must provide all 30 numeric features. Labels are `B` (benign) and `M` (malignant).

## Deploy

Build and run Docker after training:

```powershell
docker build -t breast-cancer-api:latest .
docker run --rm -p 8000:8000 breast-cancer-api:latest
```

For a local Kubernetes cluster, follow [`DEPLOYMENT.md`](DEPLOYMENT.md). Kubernetes commands affect the active cluster context; check it with `kubectl config current-context` before applying manifests.

## Clean outputs

Training artifacts are reproducible and ignored by Git. To refresh them, rerun the training command; do not remove `data/raw/data.csv`.
