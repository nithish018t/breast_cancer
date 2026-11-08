import json
from pathlib import Path

import pandas as pd
import yaml

from validation import validate_schema


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    dataset_path = PROJECT_ROOT / config["data"]["raw_path"]
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    frame = pd.read_csv(dataset_path)
    if not validate_schema(frame, output_report_name="data_validation_errors.json"):
        raise ValueError("Dataset failed validation; no pipeline stages were run.")

    report_path = PROJECT_ROOT / "artifacts" / "data_load_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "status": "loaded",
                "dataset": config["data"]["raw_path"],
                "rows": len(frame),
                "columns": len(frame.columns),
            },
            file,
            indent=2,
        )
    print(f"[SUCCESS] Loaded {len(frame)} records from {dataset_path}.")


if __name__ == "__main__":
    main()
