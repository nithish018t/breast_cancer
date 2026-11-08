import statistics
import time

import pandas as pd
from starlette.testclient import TestClient

from api.main import app


def sample_payload():
    frame = pd.read_csv("data/raw/data.csv")
    features = frame.drop(
        columns=["id", "diagnosis", "Unnamed: 32"], errors="ignore"
    )
    return {"features": features.iloc[0].to_dict()}


def test_repeated_prediction_latency():
    payload = sample_payload()
    latencies = []
    with TestClient(app) as client:
        for _ in range(10):
            started_at = time.perf_counter()
            response = client.post("/predict", json=payload)
            latencies.append((time.perf_counter() - started_at) * 1000)
            assert response.status_code == 200

    assert len(latencies) == 10
    assert statistics.median(latencies) > 0


def test_batch_prediction_performance():
    sample = sample_payload()
    payload = {"samples": [sample] * 20}
    with TestClient(app) as client:
        started_at = time.perf_counter()
        response = client.post("/predict/batch", json=payload)
        latency_ms = (time.perf_counter() - started_at) * 1000

    assert response.status_code == 200
    assert response.json()["count"] == 20
    assert latency_ms > 0
