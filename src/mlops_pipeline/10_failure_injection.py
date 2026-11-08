import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "mlops_pipeline"))

from validation import validate_schema
from breast_cancer_pipeline import load_config


def run_failure_injection():
    config = load_config()
    dataset_path = PROJECT_ROOT / config["data"]["raw_path"]
    source = pd.read_csv(dataset_path)
    corrupted = source.copy()
    corrupted.loc[corrupted.index[0], "diagnosis"] = "invalid"
    corrupted["radius_mean"] = corrupted["radius_mean"].astype(object)
    corrupted.loc[corrupted.index[1], "radius_mean"] = "invalid"
    corrupted = corrupted.drop(columns=["texture_mean"])
    accepted = validate_schema(
        corrupted, output_report_name="failure_injection_report.json"
    )
    if accepted:
        raise AssertionError("The invalid data was not rejected.")
    print("[SUCCESS] Failure-injection validation rejected invalid input.")


if __name__ == "__main__":
    run_failure_injection()
