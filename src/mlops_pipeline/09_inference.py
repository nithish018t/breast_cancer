import json
import os
import sys

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def configure_mlflow(config):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def clean_dataframe(df, config):
    df = df.copy()
    target_column = config["data"]["target_column"]
    id_column = config["data"]["id_column"]

    if id_column in df.columns:
        df = df.drop(columns=[id_column])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if target_column in df.columns:
        df = df.drop(columns=[target_column])

    return df


def load_production_bundle(model_name):
    client = MlflowClient()
    production_models = [
        version
        for version in client.search_model_versions(f"name='{model_name}'")
        if version.current_stage == "Production"
    ]
    if not production_models:
        raise RuntimeError(f"No Production model is registered for {model_name}.")

    production = production_models[0]
    metadata_path = PROJECT_ROOT / "deployment" / "model_metadata.json"
    model_path = PROJECT_ROOT / "deployment" / "model.pkl"
    preprocessor_path = PROJECT_ROOT / "deployment" / "preprocessor.pkl"
    if not all(path.is_file() for path in (metadata_path, model_path, preprocessor_path)):
        raise RuntimeError("The Production deployment bundle is incomplete.")

    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    expected = {
        "model_name": model_name,
        "model_version": str(production.version),
        "run_id": production.run_id,
        "model_source": production.source,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Deployment bundle does not match the Production registry entry: {mismatches}"
        )

    print("[WARNING] MLflow model artifact is unavailable; using the verified Production deployment bundle.")
    return joblib.load(model_path), joblib.load(preprocessor_path)


def run_inference_demo():
    print("[INFO] Starting ML Pipeline Production Inference...")
    os.chdir(PROJECT_ROOT)

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    configure_mlflow(config)
    model_name = config["registry"]["model_name"]
    model_uri = f"models:/{model_name}/Production"

    try:
        print(f"[INFO] Loading Production model from MLflow: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        preprocessor_path = config["paths"]["preprocessor"]
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
        preprocessor = joblib.load(preprocessor_path)
    except MlflowException as exc:
        print(f"[WARNING] Could not load the Production model from MLflow: {exc}")
        try:
            model, preprocessor = load_production_bundle(model_name)
        except (FileNotFoundError, RuntimeError, ValueError) as fallback_error:
            print(f"[ERROR] Could not load a verified Production deployment bundle: {fallback_error}")
            return False

    raw_data_path = config["data"]["raw_path"]

    if not os.path.exists(raw_data_path):
        print(f"[ERROR] Raw dataset not found at {raw_data_path}")
        return False

    df = pd.read_csv(raw_data_path)

    sample = df.sample(n=5, random_state=7)
    X_sample = clean_dataframe(sample, config)
    X_sample = preprocessor.transform(X_sample)

    predictions = model.predict(X_sample)
    probabilities = model.predict_proba(X_sample)[:, 1]

    result = sample.copy()
    result["Predicted_Churn"] = predictions
    result["Churn_Probability"] = np.round(probabilities, 4)

    print("\n--- ML Pipeline Production Inference Results ---")
    print(result[["Predicted_Churn", "Churn_Probability"]].to_string(index=False))
    print("-----------------------------------------\n")

    os.makedirs("outputs", exist_ok=True)
    output_path = config["paths"]["inference_output"]
    result.to_csv(output_path, index=False)

    print(f"[SUCCESS] Predictions saved to {output_path}")
    return True


if __name__ == "__main__":
    passed = run_inference_demo()
    sys.exit(0 if passed else 1)
