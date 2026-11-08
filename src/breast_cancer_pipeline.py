import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config():
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_dataset(config):
    data_config = config["data"]
    dataset_path = PROJECT_ROOT / data_config["raw_path"]
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Breast cancer dataset not found: {dataset_path}")

    frame = pd.read_csv(dataset_path)
    target_column = data_config["target_column"]
    id_column = data_config["id_column"]
    feature_columns = data_config["feature_columns"]
    required_columns = [id_column, target_column, *feature_columns]
    missing_columns = sorted(set(required_columns).difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    frame = frame.loc[:, required_columns].copy()
    original_features = frame[feature_columns]
    numeric_features = original_features.apply(pd.to_numeric, errors="coerce")
    invalid_values = original_features.notna() & numeric_features.isna()
    if invalid_values.any().any():
        invalid_columns = invalid_values.columns[invalid_values.any()].tolist()
        raise ValueError(
            f"Feature columns contain non-numeric values: {invalid_columns}"
        )
    frame[feature_columns] = numeric_features
    if frame[feature_columns].isna().all(axis=1).any():
        raise ValueError("Dataset contains a row with no usable feature values.")

    labels = frame[target_column].astype(str).str.strip().str.upper()
    invalid_labels = sorted(set(labels).difference({"B", "M"}))
    if invalid_labels:
        raise ValueError(f"Unexpected diagnosis labels: {invalid_labels}")

    frame[target_column] = labels.map({"B": 0, "M": 1}).astype("int64")
    return frame, feature_columns


def prepare_splits(config, frame, feature_columns):
    target_column = config["data"]["target_column"]
    id_column = config["data"]["id_column"]
    return train_test_split(
        frame[feature_columns],
        frame[target_column],
        frame[id_column],
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
        stratify=frame[target_column],
    )


def train_and_evaluate():
    config = load_config()
    frame, feature_columns = load_dataset(config)
    target_column = config["data"]["target_column"]
    id_column = config["data"]["id_column"]
    X_train, X_test, y_train, y_test, _, ids_test = prepare_splits(
        config, frame, feature_columns
    )

    model_config = config["model"]
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=model_config["n_estimators"],
                    max_depth=model_config["max_depth"],
                    random_state=model_config["random_state"],
                    class_weight=model_config["class_weight"],
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    malignant_probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, malignant_probabilities)),
    }

    deployment_dir = PROJECT_ROOT / "deployment"
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    reports_dir = PROJECT_ROOT / "reports"
    outputs_dir = PROJECT_ROOT / "outputs"
    for directory in (deployment_dir, processed_dir, reports_dir, outputs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, PROJECT_ROOT / config["paths"]["model"])
    metadata = {
        "model_name": config["registry"]["model_name"],
        "model_version": "1",
        "model_stage": "Production",
        "dataset_name": "Wisconsin Diagnostic Breast Cancer",
        "target_column": target_column,
        "positive_label": "M",
        "positive_class": 1,
        "feature_names": feature_columns,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "metrics": metrics,
    }
    with (PROJECT_ROOT / config["paths"]["model_metadata"]).open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    report = {
        "dataset": metadata["dataset_name"],
        "target": target_column,
        "positive_label": "M",
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }
    with (reports_dir / "evaluation_report.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    prediction_output = pd.DataFrame(
        {
            id_column: ids_test.to_numpy(),
            "actual_diagnosis": y_test.map({0: "B", 1: "M"}).to_numpy(),
            "predicted_diagnosis": np.where(predictions == 1, "M", "B"),
            "malignant_probability": malignant_probabilities,
        }
    )
    prediction_output.to_csv(
        PROJECT_ROOT / config["paths"]["inference_output"], index=False
    )
    X_train.to_csv(processed_dir / "X_train.csv", index=False)
    X_test.to_csv(processed_dir / "X_test.csv", index=False)
    np.save(processed_dir / "y_train.npy", y_train.to_numpy(dtype=np.int64))
    np.save(processed_dir / "y_test.npy", y_test.to_numpy(dtype=np.int64))

    split_metadata = {
        "dataset_name": metadata["dataset_name"],
        "source_path": config["data"]["raw_path"],
        "target_column": target_column,
        "feature_columns": feature_columns,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "random_state": config["split"]["random_state"],
    }
    with (processed_dir / "dataset_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(split_metadata, file, indent=2)

    print(f"[SUCCESS] Trained breast-cancer classifier on {len(X_train)} rows.")
    print(f"[INFO] Test metrics: {json.dumps(metrics, sort_keys=True)}")
    print(f"[INFO] Model saved to {PROJECT_ROOT / config['paths']['model']}")
    print(
        f"[INFO] Predictions saved to "
        f"{PROJECT_ROOT / config['paths']['inference_output']}"
    )
    return metrics


if __name__ == "__main__":
    train_and_evaluate()
