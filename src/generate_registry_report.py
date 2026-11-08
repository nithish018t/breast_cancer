import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def generate_registry_report():
    metadata_path = PROJECT_ROOT / "deployment" / "model_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Model metadata not found: {metadata_path}. Train the model first."
        )
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    report_path = PROJECT_ROOT / "reports" / "model_metadata_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(f"[SUCCESS] Model report saved to {report_path}.")


if __name__ == "__main__":
    generate_registry_report()
