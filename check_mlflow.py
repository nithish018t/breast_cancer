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

model = client.get_registered_model(model_name)

print("MODEL:", model.name)
print("VERSIONS:")

versions = client.search_model_versions(f"name='{model_name}'")

for version in versions:
    print(
        f"version={version.version} "
        f"stage={version.current_stage} "
        f"run_id={version.run_id}"
    )