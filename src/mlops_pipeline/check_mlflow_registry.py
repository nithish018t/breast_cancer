import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_PATH = PROJECT_ROOT / "deployment" / "model_metadata.json"


def main():
    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    print(f"Model: {metadata['model_name']}")
    print(f"Version: {metadata['model_version']}")
    print(f"Dataset: {metadata['dataset_name']}")
    print(f"Metrics: {metadata['metrics']}")


if __name__ == "__main__":
    main()
