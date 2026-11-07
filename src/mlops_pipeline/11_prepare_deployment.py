import json
import os
import tempfile
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import yaml
from mlflow import MlflowClient


MODEL_NAME = "Telco_Churn_Production_Model"


def load_config():
    project_root = Path(__file__).resolve().parents[2]

    with open(project_root / "config.yaml", "r", encoding="utf-8") as file:
        return project_root, yaml.safe_load(file)


def configure_mlflow(config, project_root):
    tracking_uri = (
        os.getenv("MLFLOW_TRACKING_URI")
        or config["mlflow"].get("tracking_uri")
    )
    if not tracking_uri:
        tracking_uri = f"sqlite:///{(project_root / 'mlflow.db').as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)
    print(f"[INFO] MLflow tracking URI: {mlflow.get_tracking_uri()}")


def get_production_model(client, model_name):
    versions = client.search_model_versions(f"name='{model_name}'")

    production_versions = [
        version for version in versions
        if version.current_stage == "Production"
    ]

    if not production_versions:
        raise RuntimeError(f"No Production version found for '{model_name}'.")

    if len(production_versions) > 1:
        raise RuntimeError(
            f"Multiple Production versions found for '{model_name}'."
        )

    return production_versions[0]


def prepare_deployment():
    project_root, config = load_config()
    deployment_dir = project_root / "deployment"
    deployment_dir.mkdir(parents=True, exist_ok=True)

    configure_mlflow(config, project_root)
    client = MlflowClient()

    model_version = get_production_model(client, MODEL_NAME)

    model_name = model_version.name
    version = model_version.version
    run_id = model_version.run_id
    model_source = model_version.source

    print("[INFO] Production model found")
    print(f"[INFO] Model name : {model_name}")
    print(f"[INFO] Version    : {version}")
    print(f"[INFO] Run ID     : {run_id}")
    print(f"[INFO] Source     : {model_source}")

    print("[INFO] Downloading Production model...")

    model_uri = f"models:/{model_name}/{version}"
    model = mlflow.sklearn.load_model(model_uri)

    with tempfile.TemporaryDirectory(
        prefix=".deployment-staging-",
        dir=deployment_dir,
    ) as staging_dir:
        staging_path = Path(staging_dir)
        staged_model_path = staging_path / "model.pkl"
        joblib.dump(model, staged_model_path)

        print("[INFO] Downloading matching preprocessor...")
        downloaded_preprocessor = client.download_artifacts(
            run_id,
            "preprocessing_pipeline/preprocessor.pkl",
            dst_path=str(staging_path),
        )
        staged_preprocessor_path = Path(downloaded_preprocessor)

        staged_metadata_path = staging_path / "model_metadata.json"
        metadata = {
            "model_name": model_name,
            "model_version": str(version),
            "model_stage": model_version.current_stage,
            "run_id": run_id,
            "model_source": model_source,
            "model_id": model_source.split("/")[-1],
            "preprocessor_artifact": "preprocessing_pipeline/preprocessor.pkl",
            "source": "MLflow",
        }
        with staged_metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4)

        print("[INFO] Verifying staged deployment artifacts...")
        if joblib.load(staged_model_path) is None:
            raise RuntimeError("Model verification failed.")
        if joblib.load(staged_preprocessor_path) is None:
            raise RuntimeError("Preprocessor verification failed.")
        with staged_metadata_path.open("r", encoding="utf-8") as file:
            staged_metadata = json.load(file)
        if staged_metadata != metadata:
            raise RuntimeError("Deployment metadata verification failed.")

        model_path = deployment_dir / "model.pkl"
        preprocessor_path = deployment_dir / "preprocessor.pkl"
        metadata_path = deployment_dir / "model_metadata.json"
        os.replace(staged_model_path, model_path)
        os.replace(staged_preprocessor_path, preprocessor_path)
        os.replace(staged_metadata_path, metadata_path)

    print(f"[SUCCESS] Model saved to {model_path}")
    print(f"[SUCCESS] Preprocessor saved to {preprocessor_path}")
    print(f"[SUCCESS] Metadata saved to {metadata_path}")

    print("\n[INFO] Deployment bundle:")
    print("deployment/")
    print("+-- model.pkl")
    print("+-- preprocessor.pkl")
    print("+-- model_metadata.json")

    print("\n[SUCCESS] Deployment preparation completed.")


if __name__ == "__main__":
    prepare_deployment()