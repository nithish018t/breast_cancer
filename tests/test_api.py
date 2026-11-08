import json

import pandas as pd
import pytest
from starlette.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_sample():
    frame = pd.read_csv("data/raw/data.csv")
    features = frame.drop(
        columns=["id", "diagnosis", "Unnamed: 32"], errors="ignore"
    )
    return {"features": features.iloc[0].to_dict()}


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}


def test_metadata(client):
    response = client.get("/metadata")
    assert response.status_code == 200
    assert response.json()["service"] == "Breast Cancer Diagnosis API"
    assert response.json()["feature_count"] == 30


def test_valid_prediction(client, valid_sample):
    response = client.post("/predict", json=valid_sample)
    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] in (0, 1)
    assert result["diagnosis"] in ("B", "M")
    assert 0 <= result["malignant_probability"] <= 1


def test_missing_feature_is_rejected(client, valid_sample):
    valid_sample["features"].pop("radius_mean")
    response = client.post("/predict", json=valid_sample)
    assert response.status_code == 422
    assert "missing features" in response.json()["detail"]


def test_non_finite_feature_is_rejected(client, valid_sample):
    valid_sample["features"]["radius_mean"] = float("nan")
    response = client.post(
        "/predict",
        content=json.dumps(valid_sample, allow_nan=True),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_batch_prediction(client, valid_sample):
    response = client.post(
        "/predict/batch", json={"samples": [valid_sample, valid_sample]}
    )
    assert response.status_code == 200
    assert response.json()["count"] == 2
    assert len(response.json()["predictions"]) == 2
