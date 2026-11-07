import os
from pathlib import Path

import numpy as np
import mlflow
import mlflow.sklearn
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def configure_mlflow(config):
    tracking_uri = (
        os.getenv("MLFLOW_TRACKING_URI")
        or config["mlflow"].get("tracking_uri")
        or f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    )
    mlflow.set_tracking_uri(tracking_uri)


def train_and_register_model():
    print("[INFO] --- Starting MLflow Run with Model Registry ---")
    os.chdir(PROJECT_ROOT)

    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    configure_mlflow(config)

    # 2. Load Processed Data
    processed_dir = Path(config["data"]["processed_dir"])
    try:
        X_train = np.load(processed_dir / "X_train_final.npy")
        X_test = np.load(processed_dir / "X_test_final.npy")
        y_train = np.load(processed_dir / "y_train.npy")
        y_test = np.load(processed_dir / "y_test.npy")
    except FileNotFoundError as exc:
        print(f"[ERROR] Processed data file not found: {exc.filename}. Please run the Lab 5 pipeline first.")
        return

    preprocessor_path = Path(config["paths"]["preprocessor"])
    if not preprocessor_path.is_file():
        print(f"[ERROR] Preprocessor not found at {preprocessor_path}. Run preprocessing first.")
        return

    # 1. Set up MLflow only after input validation to avoid creating empty experiments.
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # 3. Define Model Parameters
    # params = {
    #     "n_estimators": 100,
    #     "max_depth": 10,
    #     "random_state": 42,
    #     "class_weight": "balanced"
    # }

    params = {
        "n_estimators": 200,
        "max_depth": 15,
        "random_state": 42,
        "class_weight": "balanced"
    }
    
    # with mlflow.start_run(run_name="RandomForest_Registry_V1") as run:
    with mlflow.start_run(run_name="RandomForest_Registry_V2") as run:
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_family", "RandomForest")
        
        # 4. Train Model
        print("[INFO] Training RandomForest model...")
        rf_model = RandomForestClassifier(**params)
        rf_model.fit(X_train, y_train)
        
        # 5. Evaluate
        print("[INFO] Evaluating model...")
        y_pred = rf_model.predict(X_test)
        y_proba = rf_model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }
        
        mlflow.log_metrics(metrics)
        
        # 6. LOG PREPROCESSOR PIPELINE (Lineage)
        print("[INFO] Attaching preprocessing pipeline to model artifacts...")
        mlflow.log_artifact(str(preprocessor_path), artifact_path="preprocessing_pipeline")
        
        # 7. LOG AND REGISTER THE MODEL
        print("[INFO] Pushing model to MLflow Registry...")
        mlflow.sklearn.log_model(
            sk_model=rf_model,
            # artifact_path="random_forest_model",
            # artifact_path is deprecated in newer MLflow versions
            name="random_forest_model",
            registered_model_name=config["registry"]["model_name"]
        )
        
        print(f"[SUCCESS] Metrics logged: F1 = {metrics['f1_score']:.4f} | ROC-AUC = {metrics['roc_auc']:.4f}")
        print(f"[SUCCESS] Model successfully registered under name: '{config['registry']['model_name']}'")

if __name__ == "__main__":
    train_and_register_model()