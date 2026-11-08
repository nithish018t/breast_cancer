# Breast Cancer Diagnosis

This project trains and serves a classifier for the Wisconsin Diagnostic Breast Cancer dataset. It uses `diagnosis` as the target (`M` = malignant, `B` = benign) and the 30 numeric measurement columns as model features.

## Dataset

The source file is [`data/raw/data.csv`](data/raw/data.csv). The pipeline excludes the record identifier (`id`) and the unused `Unnamed: 32` column. The dataset contains 569 records. Do not send real patient data to the API; this example is for research and demonstration, not clinical use.

## Run locally

Use Python 3.11 or newer. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-api.txt
python src\breast_cancer_pipeline.py
```

Training writes the model bundle to `deployment\`, split metadata and arrays to `data\processed\`, evaluation metrics to `reports\evaluation_report.json`, and held-out predictions to `outputs\inference_predictions.csv`. These are generated artifacts and are excluded from Git.

Run the API and tests:

```powershell
python -m pip install pytest
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
python -m pytest -q
```

The API is documented at <http://127.0.0.1:8000/docs>. `POST /predict` accepts a `features` object containing all 30 feature names; `POST /predict/batch` accepts a non-empty `samples` array. The response includes the predicted class (`B` or `M`) and the malignant-class probability.

## Docker

Train the model first so the Docker build can include the generated deployment bundle:

```powershell
python src\breast_cancer_pipeline.py
docker build -t breast-cancer-api:latest .
docker run --rm -p 8000:8000 breast-cancer-api:latest
```

Check `http://localhost:8000/health`. Docker instructions and the API payload shape are in [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

## Kubernetes

The manifests target a local Kubernetes cluster, such as Docker Desktop Kubernetes. After building the image:

```powershell
kubectl apply -f k8s\deployment.yaml
kubectl apply -f k8s\service.yaml
kubectl rollout status deployment/breast-cancer-api
kubectl port-forward service/breast-cancer-api 8000:8000
```

For a remote cluster, push the image to a registry accessible by that cluster and update the image in `k8s\deployment.yaml` first. Remove the local resources with:

```powershell
kubectl delete -f k8s\service.yaml
kubectl delete -f k8s\deployment.yaml
```

## Important files

- `src\breast_cancer_pipeline.py` — dataset validation, stratified training, evaluation, and artifact generation.
- `api\` — FastAPI service, request validation, model loading, and inference logging.
- `Dockerfile` — container image for the API.
- `k8s\` — Kubernetes deployment and service manifests.
- `tests\` — API and deployment tests.

Predictions are not medical diagnoses. This model is not validated for clinical decision-making.
