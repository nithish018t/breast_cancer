from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "mlops_pipeline"))

from validation import validate_schema
from breast_cancer_pipeline import load_config


def simulate_corrupted_data(frame):
    corrupted = frame.copy()
    corrupted.loc[corrupted.index[0], "diagnosis"] = "unknown"
    corrupted["radius_mean"] = corrupted["radius_mean"].astype(object)
    corrupted.loc[corrupted.index[1], "radius_mean"] = "not-a-number"
    return corrupted.drop(columns=["texture_mean"])


def main():
    dataset_path = PROJECT_ROOT / load_config()["data"]["raw_path"]
    frame = pd.read_csv(dataset_path)
    corrupted = simulate_corrupted_data(frame)
    if validate_schema(
        corrupted, output_report_name="corrupted_data_validation.json"
    ):
        raise AssertionError("Corrupted breast-cancer data unexpectedly passed.")
    print("[SUCCESS] Invalid data was rejected without changing the source dataset.")


if __name__ == "__main__":
    main()
