import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "logs" / "inference" / "inference_log.jsonl"
_LOCK = threading.Lock()


def log_inference(
    input_data: dict[str, Any],
    prediction: int | None = None,
    malignant_probability: float | None = None,
    latency_ms: float | None = None,
    status: str = "success",
    error: str | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
    request_id: str | None = None,
) -> str:
    request_id = request_id or uuid.uuid4().hex[:12]
    event = {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": input_data,
        "prediction": prediction,
        "malignant_probability": malignant_probability,
        "latency_ms": latency_ms,
        "status": status,
        "error": error,
        "model_name": model_name,
        "model_version": model_version,
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, default=str) + "\n")
    return request_id
