import os
import json
import mlflow
import yaml
from pathlib import Path
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def generate_registry_report():
    print("[INFO] Generating Model Registry Report...")

    os.chdir(PROJECT_ROOT)
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    tracking_uri = (
        os.getenv("MLFLOW_TRACKING_URI")
        or config["mlflow"].get("tracking_uri")
        or f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    )
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    model_name = config["registry"]["model_name"]

    try:
        # Search for the model specifically in Production
        production_models = client.search_model_versions(f"name='{model_name}'")
        champion = next((mv for mv in production_models if mv.current_stage == "Production"), None)

        if not champion:
            raise RuntimeError(
                f"No model found in Production stage for '{model_name}' "
                f"using tracking URI '{mlflow.get_tracking_uri()}'."
            )

        # Get the run details to extract metrics and parameters
        run = client.get_run(champion.run_id)

        # Build the deployment-ready artifact dictionary
        report = {
            "registry_status": "READY_FOR_DEPLOYMENT",
            "model_lineage": {
                "registered_name": model_name,
                "version": int(champion.version),
                "current_stage": champion.current_stage,
                "run_id": champion.run_id,
                "artifact_uri": champion.source
            },
            "performance_metrics": {
                "recall": run.data.metrics.get("recall"),
                "f1_score": run.data.metrics.get("f1_score"),
                "roc_auc": run.data.metrics.get("roc_auc"),
                "accuracy": run.data.metrics.get("accuracy"),
                "precision": run.data.metrics.get("precision")
            },
            "hyperparameters": run.data.params,
            "preprocessing_dependency": "models/preprocessor.pkl"
        }

        # Save the report
        report_path = PROJECT_ROOT / "artifacts" / "production_model_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        print(f"[SUCCESS] Report successfully generated for Version {champion.version}")
        print(f"[INFO] Saved to: {report_path}")

    except Exception as e:
        print(f"[ERROR] Failed to generate report: {e}")

if __name__ == "__main__":
    generate_registry_report()