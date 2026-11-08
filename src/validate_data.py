from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "mlops_pipeline"))

from validation import validate_schema
from breast_cancer_pipeline import load_config


def main():
    config = load_config()
    dataset_path = PROJECT_ROOT / config["data"]["raw_path"]
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    passed = validate_schema(pd.read_csv(dataset_path))
    if not passed:
        raise ValueError("Dataset failed breast-cancer schema validation.")


if __name__ == "__main__":
    main()
