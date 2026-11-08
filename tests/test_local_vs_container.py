import json
import os
from urllib.request import Request, urlopen

import pandas as pd
import pytest
from starlette.testclient import TestClient

from api.main import app


def sample_payload():
    frame = pd.read_csv("data/raw/data.csv")
    features = frame.drop(
        columns=["id", "diagnosis", "Unnamed: 32"], errors="ignore"
    )
    return {"features": features.iloc[0].to_dict()}


def test_local_api_prediction():
    with TestClient(app) as client:
        response = client.post("/predict", json=sample_payload())
    assert response.status_code == 200
    assert response.json()["diagnosis"] in {"B", "M"}


def test_container_matches_local_api():
    base_url = os.getenv("CONTAINER_BASE_URL")
    if not base_url:
        pytest.skip("Set CONTAINER_BASE_URL to compare against a running container.")

    payload = sample_payload()
    with TestClient(app) as client:
        local_response = client.post("/predict", json=payload)
    request = Request(
        f"{base_url.rstrip('/')}/predict",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=10) as response:
        container_result = json.loads(response.read().decode("utf-8"))
    assert response.json() == container_result
