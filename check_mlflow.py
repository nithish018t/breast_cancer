import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
METADATA_PATH = PROJECT_ROOT / "deployment" / "model_metadata.json"

if not METADATA_PATH.is_file():
    raise FileNotFoundError(f"Model metadata not found: {METADATA_PATH}")

with METADATA_PATH.open("r", encoding="utf-8") as file:
    metadata = json.load(file)

print("MODEL:", metadata["model_name"])
print("VERSION:", metadata["model_version"])
print("DATASET:", metadata["dataset_name"])
print("METRICS:", metadata["metrics"])
