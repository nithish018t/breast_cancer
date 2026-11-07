import argparse
import os
from pathlib import Path

import mlflow
from mlflow import MlflowClient
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent


def configure_mlflow(config):
    tracking_uri = (
        os.getenv("MLFLOW_TRACKING_URI")
        or config["mlflow"].get("tracking_uri")
        or f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    )
    mlflow.set_tracking_uri(tracking_uri)


def get_production_run_id(client, model_name):
    production_versions = [
        version
        for version in client.search_model_versions(f"name='{model_name}'")
        if version.current_stage == "Production"
    ]
    if not production_versions:
        raise RuntimeError(f"No Production version found for '{model_name}'.")
    return max(production_versions, key=lambda version: int(version.version)).run_id


def main():
    parser = argparse.ArgumentParser(description="Inspect an MLflow run and its artifacts.")
    parser.add_argument(
        "run_id",
        nargs="?",
        help="MLflow run ID to inspect (defaults to the configured Production model run).",
    )
    args = parser.parse_args()

    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    configure_mlflow(config)

    client = MlflowClient()
    run_id = args.run_id or get_production_run_id(client, config["registry"]["model_name"])

    run = client.get_run(run_id)

    print("RUN ID:", run.info.run_id)
    print("STATUS:", run.info.status)
    print("ARTIFACT URI:", run.info.artifact_uri)

    print("\nARTIFACTS:")

    artifacts = client.list_artifacts(run_id)

    for artifact in artifacts:
        print(f"{artifact.path} | is_dir={artifact.is_dir}")

        if artifact.is_dir:
            children = client.list_artifacts(run_id, artifact.path)

            for child in children:
                print(f"  - {child.path} | is_dir={child.is_dir}")


if __name__ == "__main__":
    main()