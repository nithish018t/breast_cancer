import json
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def validate_schema(df, output_report_name="schema_validation_errors.json"):
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    data_config = config["data"]
    required = [
        data_config["id_column"],
        data_config["target_column"],
        *data_config["feature_columns"],
    ]
    errors = []
    missing_columns = sorted(set(required).difference(df.columns))
    if missing_columns:
        errors.append(f"Missing columns: {missing_columns}")

    if not missing_columns:
        target = data_config["target_column"]
        invalid_labels = sorted(set(df[target].dropna().astype(str)) - {"B", "M"})
        if invalid_labels:
            errors.append(f"Invalid diagnosis labels: {invalid_labels}")
        features = data_config["feature_columns"]
        original_features = df[features]
        numeric = original_features.apply(pd.to_numeric, errors="coerce")
        invalid_values = original_features.notna() & numeric.isna()
        if invalid_values.any().any():
            invalid_columns = invalid_values.columns[
                invalid_values.any()
            ].tolist()
            errors.append(f"Non-numeric feature values: {invalid_columns}")
        if numeric.isna().all(axis=1).any():
            errors.append("At least one record has no usable numeric features.")
        if df[data_config["id_column"]].isna().any():
            errors.append("Record identifiers must not be missing.")

    if errors:
        report_path = PROJECT_ROOT / "artifacts" / output_report_name
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as file:
            json.dump({"status": "FAILED", "errors": errors}, file, indent=2)
        print("[ERROR] Breast-cancer dataset validation failed.")
        for error in errors:
            print(f"  - {error}")
        return False

    print(f"[SUCCESS] Validated {len(df)} breast-cancer records.")
    return True
