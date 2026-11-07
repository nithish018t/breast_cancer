import os
from pathlib import Path

import mlflow
from mlflow import MlflowClient
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent

with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

tracking_uri = (
    os.getenv("MLFLOW_TRACKING_URI")
    or config["mlflow"].get("tracking_uri")
    or f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
)
mlflow.set_tracking_uri(tracking_uri)

client = MlflowClient()

model_name = config["registry"]["model_name"]
version = os.getenv("MLFLOW_MODEL_VERSION", "1")

mv = client.get_model_version(model_name, version)

print("MODEL NAME:", mv.name)
print("VERSION:", mv.version)
print("STAGE:", mv.current_stage)
print("RUN ID:", mv.run_id)
print("SOURCE:", mv.source)
print("STATUS:", mv.status)
print("CREATION TIMESTAMP:", mv.creation_timestamp)