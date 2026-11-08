# Docker and Kubernetes

Run commands from the repository root. Create the model bundle before building an image:

```powershell
python src\breast_cancer_pipeline.py
docker build -t breast-cancer-api:latest .
```

Start the API locally and check readiness:

```powershell
docker run --rm -p 8000:8000 breast-cancer-api:latest
```

In another terminal:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

## Kubernetes

The supplied manifests use the local `breast-cancer-api:latest` image and are intended for a local cluster that can use Docker's image store (for example, Docker Desktop Kubernetes).

```powershell
kubectl apply -f k8s\deployment.yaml
kubectl apply -f k8s\service.yaml
kubectl rollout status deployment/breast-cancer-api
kubectl port-forward service/breast-cancer-api 8000:8000
```

The API is available at `http://localhost:8000`; Swagger UI is at `/docs`. To deploy to a remote cluster, push the image to a registry it can access and set the deployment manifest's `image` to that registry path. Tear down the resources with `kubectl delete -f k8s\service.yaml` and `kubectl delete -f k8s\deployment.yaml`.

## Prediction request

`POST /predict` expects a JSON object with all 30 numeric features nested under `features`. Example:

```json
{
  "features": {
    "radius_mean": 17.99,
    "texture_mean": 10.38,
    "perimeter_mean": 122.8,
    "area_mean": 1001.0,
    "smoothness_mean": 0.1184,
    "compactness_mean": 0.2776,
    "concavity_mean": 0.3001,
    "concave points_mean": 0.1471,
    "symmetry_mean": 0.2419,
    "fractal_dimension_mean": 0.07871,
    "radius_se": 1.095,
    "texture_se": 0.9053,
    "perimeter_se": 8.589,
    "area_se": 153.4,
    "smoothness_se": 0.006399,
    "compactness_se": 0.04904,
    "concavity_se": 0.05373,
    "concave points_se": 0.01587,
    "symmetry_se": 0.03003,
    "fractal_dimension_se": 0.006193,
    "radius_worst": 25.38,
    "texture_worst": 17.33,
    "perimeter_worst": 184.6,
    "area_worst": 2019.0,
    "smoothness_worst": 0.1622,
    "compactness_worst": 0.6656,
    "concavity_worst": 0.7119,
    "concave points_worst": 0.2654,
    "symmetry_worst": 0.4601,
    "fractal_dimension_worst": 0.1189
  }
}
```

The response contains `prediction` (`0` benign, `1` malignant), `diagnosis` (`B`/`M`), and `malignant_probability`. Do not submit identifiable patient records; inference logs contain only the feature count and model result.
