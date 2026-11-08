import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_LOG = PROJECT_ROOT / "logs" / "inference" / "inference_log.jsonl"
DRIFT_DIR = PROJECT_ROOT / "outputs" / "drift"
RETRAINING_DIR = PROJECT_ROOT / "outputs" / "retraining"
MONITORING_DIR = PROJECT_ROOT / "outputs" / "monitoring"


def load_config():
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_dataset():
    config = load_config()
    data_config = config["data"]
    source_path = PROJECT_ROOT / data_config["raw_path"]
    if not source_path.is_file():
        raise FileNotFoundError(f"Breast-cancer dataset not found: {source_path}")
    frame = pd.read_csv(source_path)
    required = [
        data_config["id_column"],
        data_config["target_column"],
        *data_config["feature_columns"],
    ]
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"Breast-cancer dataset is missing columns: {missing}")
    frame = frame.loc[:, required].copy()
    features = data_config["feature_columns"]
    original_features = frame[features]
    numeric_features = original_features.apply(pd.to_numeric, errors="coerce")
    invalid_values = original_features.notna() & numeric_features.isna()
    if invalid_values.any().any():
        invalid_columns = invalid_values.columns[invalid_values.any()].tolist()
        raise ValueError(f"Non-numeric feature values: {invalid_columns}")
    if np.isinf(numeric_features.to_numpy(dtype=float)).any():
        raise ValueError("Feature values must be finite.")
    if numeric_features.isna().all(axis=1).any():
        raise ValueError("Dataset contains a row without usable measurements.")
    if frame[data_config["id_column"]].isna().any():
        raise ValueError("Record identifiers must not be missing.")
    frame[features] = numeric_features
    labels = frame[data_config["target_column"]].astype(str).str.strip().str.upper()
    invalid_labels = sorted(set(labels).difference({"B", "M"}))
    if invalid_labels:
        raise ValueError(f"Unexpected diagnosis labels: {invalid_labels}")
    frame[data_config["target_column"]] = labels.map({"B": 0, "M": 1}).astype(int)
    if frame[features].isna().all(axis=1).any():
        raise ValueError("Dataset contains a row without usable measurements.")
    return config, frame


def split_dataset(frame, config):
    features = config["data"]["feature_columns"]
    target = config["data"]["target_column"]
    return train_test_split(
        frame[features],
        frame[target],
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
        stratify=frame[target],
    )


def load_production_model(config=None):
    config = config or load_config()
    model_path = PROJECT_ROOT / config["paths"]["model"]
    metadata_path = PROJECT_ROOT / config["paths"]["model_metadata"]
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "Train the production model first with "
            "`python src/breast_cancer_pipeline.py`."
        )
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    return joblib.load(model_path), metadata


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)


def _metrics(model, X, y):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1_score": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
    }


def _ensure_drift_inputs():
    evolved_path = DRIFT_DIR / "evolved_data.csv"
    concept_path = DRIFT_DIR / "concept_drift_data.csv"
    config, frame = load_dataset()
    _, X_test, _, _ = split_dataset(frame, config)
    if not evolved_path.is_file():
        create_evolved_dataset()
    if not concept_path.is_file():
        create_concept_drift_dataset()
    evolved = pd.read_csv(evolved_path)
    concept = pd.read_csv(concept_path)
    if "simulation" not in evolved or not evolved["simulation"].eq(
        "synthetic_covariate_drift"
    ).all():
        create_evolved_dataset()
    if (
        len(concept) != len(X_test)
        or "simulation" not in concept
        or not concept["simulation"].eq("synthetic_label_shift").all()
    ):
        create_concept_drift_dataset()
    return pd.read_csv(evolved_path), pd.read_csv(concept_path)


def create_evolved_dataset():
    config, frame = load_dataset()
    features = config["data"]["feature_columns"]
    fraction = float(config["monitoring"]["synthetic_drift_fraction"])
    sample_count = max(20, int(len(frame) * fraction))
    sample = frame.sample(
        n=min(sample_count, len(frame)),
        random_state=int(config["split"]["random_state"]) + 101,
    ).copy()
    rng = np.random.default_rng(int(config["split"]["random_state"]) + 202)
    shifted_features = [name for name in features if name.endswith("_worst")]
    for column in shifted_features:
        sample[column] = sample[column] * rng.normal(1.12, 0.015, len(sample))
    sample[config["data"]["id_column"]] = [
        f"SYNTHETIC-COVARIATE-{index:04d}" for index in range(len(sample))
    ]
    sample["simulation"] = "synthetic_covariate_drift"
    path = DRIFT_DIR / "evolved_data.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(path, index=False)
    print(f"[SUCCESS] Saved simulated covariate-drift data to {path}")
    return path


def create_concept_drift_dataset():
    config, frame = load_dataset()
    fraction = float(config["monitoring"]["synthetic_label_flip_fraction"])
    _, X_test, _, _ = split_dataset(frame, config)
    sample = frame.loc[X_test.index].copy()
    flip_count = max(1, int(len(sample) * fraction))
    flip_indices = sample.sample(
        n=flip_count, random_state=int(config["split"]["random_state"]) + 404
    ).index
    target = config["data"]["target_column"]
    sample.loc[flip_indices, target] = 1 - sample.loc[flip_indices, target]
    sample[config["data"]["id_column"]] = [
        f"SYNTHETIC-CONCEPT-{index:04d}" for index in range(len(sample))
    ]
    sample["simulation"] = "synthetic_label_shift"
    path = DRIFT_DIR / "concept_drift_data.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(path, index=False)
    print(f"[SUCCESS] Saved simulated label-shift data to {path}")
    return path


def create_updated_dataset():
    config, frame = load_dataset()
    _, concept_data = _ensure_drift_inputs()
    features = config["data"]["feature_columns"]
    target = config["data"]["target_column"]
    identifier = config["data"]["id_column"]
    columns = [identifier, *features, target]
    updated = pd.concat(
        [frame[columns], concept_data[columns]],
        ignore_index=True,
    )
    path = RETRAINING_DIR / "simulated_updated_dataset.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(path, index=False)
    _write_json(
        RETRAINING_DIR / "simulated_updated_dataset_summary.json",
        {
            "dataset_type": "simulation_only",
            "baseline_rows": len(frame),
            "synthetic_rows": len(concept_data),
            "updated_rows": len(updated),
            "warning": "Synthetic labels are for monitoring exercises only; "
            "do not use this dataset for production retraining.",
        },
    )
    print(f"[SUCCESS] Saved simulation-only combined data to {path}")
    return path


def analyze_data_drift():
    config, baseline = load_dataset()
    evolved, _ = _ensure_drift_inputs()
    features = config["data"]["feature_columns"]
    threshold = float(config["monitoring"]["drift_ks_threshold"])
    results = []
    for feature in features:
        base = baseline[feature].dropna().astype(float)
        current = evolved[feature].dropna().astype(float)
        statistic = float(ks_2samp(base, current).statistic)
        results.append(
            {
                "feature": feature,
                "baseline_rows": len(base),
                "current_rows": len(current),
                "ks_statistic": statistic,
                "threshold": threshold,
                "status": "DRIFT" if statistic >= threshold else "OK",
            }
        )
    report = pd.DataFrame(results)
    path = DRIFT_DIR / "data_drift_report.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(path, index=False)
    print(f"[SUCCESS] Data drift report saved to {path}")
    return report


def analyze_prediction_drift():
    config, frame = load_dataset()
    model, _ = load_production_model(config)
    evolved, _ = _ensure_drift_inputs()
    features = config["data"]["feature_columns"]
    reference = model.predict_proba(frame[features])[:, 1]
    current = model.predict_proba(evolved[features])[:, 1]
    statistic = float(ks_2samp(reference, current).statistic)
    threshold = float(config["monitoring"]["drift_ks_threshold"])
    report = pd.DataFrame(
        [
            {
                "baseline_rows": len(reference),
                "current_rows": len(current),
                "baseline_malignant_probability_mean": float(np.mean(reference)),
                "current_malignant_probability_mean": float(np.mean(current)),
                "ks_statistic": statistic,
                "threshold": threshold,
                "status": "DRIFT" if statistic >= threshold else "OK",
            }
        ]
    )
    path = DRIFT_DIR / "prediction_drift_report.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(path, index=False)
    print(f"[SUCCESS] Prediction drift report saved to {path}")
    return report


def analyze_concept_drift():
    config, frame = load_dataset()
    model, _ = load_production_model(config)
    _, concept = _ensure_drift_inputs()
    features = config["data"]["feature_columns"]
    target = config["data"]["target_column"]
    _, X_test, _, y_test = split_dataset(frame, config)
    baseline_metrics = _metrics(model, X_test, y_test)
    current_metrics = _metrics(model, concept[features], concept[target])
    results = []
    for metric, baseline_value in baseline_metrics.items():
        current_value = current_metrics[metric]
        results.append(
            {
                "metric": metric,
                "baseline": baseline_value,
                "current": current_value,
                "change": current_value - baseline_value,
                "status": "DEGRADED" if current_value < baseline_value else "OK",
            }
        )
    report = pd.DataFrame(results)
    path = DRIFT_DIR / "concept_drift_report.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(path, index=False)
    print(f"[SUCCESS] Concept drift report saved to {path}")
    return report


def analyze_all_drift():
    data_report = analyze_data_drift()
    prediction_report = analyze_prediction_drift()
    concept_report = analyze_concept_drift()
    drift_features = data_report.loc[data_report["status"] == "DRIFT", "feature"].tolist()
    prediction_drift = bool((prediction_report["status"] == "DRIFT").any())
    degraded_metrics = concept_report.loc[
        concept_report["status"] == "DEGRADED", "metric"
    ].tolist()
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "Wisconsin Diagnostic Breast Cancer",
        "simulation_only": True,
        "data_drift_features": drift_features,
        "prediction_drift": prediction_drift,
        "degraded_metrics": degraded_metrics,
        "recommendation": (
            "REVIEW_LABELED_DATA"
            if drift_features or prediction_drift or degraded_metrics
            else "NO_ACTION"
        ),
        "automatic_retraining": False,
    }
    _write_json(DRIFT_DIR / "drift_analysis_report.json", summary)
    pd.DataFrame(
        [
            {
                "data_drift_features": len(drift_features),
                "prediction_drift": prediction_drift,
                "degraded_metrics": len(degraded_metrics),
                "recommendation": summary["recommendation"],
                "simulation_only": True,
            }
        ]
    ).to_csv(DRIFT_DIR / "drift_analysis_report.csv", index=False)
    return summary


def monitor_performance():
    config, frame = load_dataset()
    model, metadata = load_production_model(config)
    _, X_test, _, y_test = split_dataset(frame, config)
    metrics = _metrics(model, X_test, y_test)
    report = {
        "model_name": metadata.get("model_name"),
        "model_version": metadata.get("model_version"),
        "evaluation_rows": len(X_test),
        "metrics": metrics,
        "evaluation": "fixed_stratified_holdout",
    }
    path = MONITORING_DIR / "performance_report.json"
    _write_json(path, report)
    print(f"[SUCCESS] Performance report saved to {path}")
    return report


def monitor_performance_trend():
    config, frame = load_dataset()
    model, _ = load_production_model(config)
    _, X_test, _, y_test = split_dataset(frame, config)
    window_count = int(config["monitoring"]["performance_windows"])
    indices = np.array_split(np.arange(len(X_test)), window_count)
    rows = []
    for window, index in enumerate(indices, start=1):
        if not len(index):
            continue
        x_window = X_test.iloc[index]
        y_window = y_test.iloc[index]
        probabilities = model.predict_proba(x_window)[:, 1]
        predictions = model.predict(x_window)
        rows.append(
            {
                "window": window,
                "rows": len(index),
                "accuracy": float(accuracy_score(y_window, predictions)),
                "precision": float(
                    precision_score(y_window, predictions, zero_division=0)
                ),
                "recall": float(recall_score(y_window, predictions, zero_division=0)),
                "f1_score": float(f1_score(y_window, predictions, zero_division=0)),
                "roc_auc": (
                    float(roc_auc_score(y_window, probabilities))
                    if y_window.nunique() == 2
                    else None
                ),
            }
        )
    report = pd.DataFrame(rows)
    path = MONITORING_DIR / "performance_trend.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(path, index=False)
    print(f"[SUCCESS] Performance trend saved to {path}")
    return report


def load_inference_events():
    if not INFERENCE_LOG.is_file():
        return []
    events = []
    with INFERENCE_LOG.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed inference event at line {line_number}: {exc}"
                ) from exc
            events.append(event)
    return events


def monitor_predictions():
    events = load_inference_events()
    predictions = [event for event in events if event.get("prediction") in (0, 1)]
    probabilities = [
        float(event["malignant_probability"])
        for event in events
        if event.get("malignant_probability") is not None
    ]
    report = {
        "events": len(events),
        "successful_predictions": len(predictions),
        "failures": sum(event.get("status") != "success" for event in events),
        "benign_predictions": sum(event.get("prediction") == 0 for event in predictions),
        "malignant_predictions": sum(
            event.get("prediction") == 1 for event in predictions
        ),
        "mean_malignant_probability": (
            float(np.mean(probabilities)) if probabilities else None
        ),
        "raw_measurements_logged": False,
    }
    path = MONITORING_DIR / "prediction_monitoring_report.json"
    _write_json(path, report)
    print(f"[SUCCESS] Prediction monitoring report saved to {path}")
    return report


def monitor_api_performance():
    events = load_inference_events()
    latencies = [
        float(event["latency_ms"])
        for event in events
        if event.get("latency_ms") is not None
    ]
    report = {
        "events": len(events),
        "latency_samples": len(latencies),
        "latency_mean_ms": float(np.mean(latencies)) if latencies else None,
        "latency_p50_ms": float(np.percentile(latencies, 50)) if latencies else None,
        "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
        "raw_measurements_logged": False,
    }
    path = MONITORING_DIR / "api_performance_report.json"
    _write_json(path, report)
    print(f"[SUCCESS] API performance report saved to {path}")
    return report


def review_request_history():
    events = load_inference_events()
    failures = [
        {
            "request_id": event.get("request_id"),
            "timestamp": event.get("timestamp"),
            "status": event.get("status"),
            "error": event.get("error"),
        }
        for event in events
        if event.get("status") != "success"
    ]
    report = {
        "events": len(events),
        "failures": len(failures),
        "latest_request_id": events[-1].get("request_id") if events else None,
        "failure_details": failures[-100:],
        "raw_measurements_included": False,
    }
    path = MONITORING_DIR / "request_history_report.json"
    _write_json(path, report)
    print(f"[SUCCESS] Request history saved to {path}")
    return report


def retrain_candidate():
    config, frame = load_dataset()
    features = config["data"]["feature_columns"]
    target = config["data"]["target_column"]
    identifier = config["data"]["id_column"]
    X_train, X_test, y_train, y_test = split_dataset(frame, config)
    incoming_path = PROJECT_ROOT / config["monitoring"]["incoming_data_path"]
    incoming_rows = 0
    if incoming_path.is_file():
        incoming = pd.read_csv(incoming_path)
        required = [identifier, target, *features]
        missing = sorted(set(required).difference(incoming.columns))
        if missing:
            raise ValueError(f"Labeled incoming data is missing columns: {missing}")
        incoming = incoming.loc[:, required].copy()
        if incoming[identifier].isna().any() or incoming[identifier].duplicated().any():
            raise ValueError("Incoming record identifiers must be present and unique.")
        existing_ids = set(frame[identifier].astype(str))
        duplicate_ids = set(incoming[identifier].astype(str)).intersection(existing_ids)
        if duplicate_ids:
            raise ValueError(
                "Incoming rows must have identifiers not present in the source "
                f"dataset; duplicates found: {len(duplicate_ids)}."
            )
        original_features = incoming[features]
        incoming_features = original_features.apply(pd.to_numeric, errors="coerce")
        invalid_values = original_features.notna() & incoming_features.isna()
        if invalid_values.any().any():
            invalid_columns = invalid_values.columns[invalid_values.any()].tolist()
            raise ValueError(f"Incoming feature values are non-numeric: {invalid_columns}")
        if np.isinf(incoming_features.to_numpy(dtype=float)).any():
            raise ValueError("Incoming feature values must be finite.")
        if incoming_features.isna().all(axis=1).any():
            raise ValueError("Incoming row contains no usable measurements.")
        labels = incoming[target].astype(str).str.strip().str.upper()
        invalid_labels = sorted(set(labels).difference({"B", "M"}))
        if invalid_labels:
            raise ValueError(f"Unexpected incoming diagnosis labels: {invalid_labels}")
        incoming_labels = labels.map({"B": 0, "M": 1}).astype(int)
        X_train = pd.concat([X_train, incoming_features], ignore_index=True)
        y_train = pd.concat([y_train.reset_index(drop=True), incoming_labels], ignore_index=True)
        incoming_rows = len(incoming)
    model_config = config["model"]
    model = Pipeline(
        [
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
    metrics = _metrics(model, X_test, y_test)
    candidate_path = PROJECT_ROOT / "models" / "candidate_model.pkl"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, candidate_path)
    report = {
        "candidate_model": str(candidate_path.relative_to(PROJECT_ROOT)),
        "dataset": "Wisconsin Diagnostic Breast Cancer",
        "training_rows": len(X_train),
        "evaluation_rows": len(X_test),
        "incoming_rows": incoming_rows,
        "incoming_data_path": (
            str(incoming_path.relative_to(PROJECT_ROOT))
            if incoming_rows
            else None
        ),
        "metrics": metrics,
        "decision_metric": config["registry"]["decision_metric"],
        "simulation_data_used": False,
        "status": "CANDIDATE_TRAINED",
    }
    _write_json(RETRAINING_DIR / "retraining_report.json", report)
    print(f"[SUCCESS] Trained candidate model and saved report to {RETRAINING_DIR}")
    return report


def compare_models():
    config, frame = load_dataset()
    production, _ = load_production_model(config)
    candidate_path = PROJECT_ROOT / "models" / "candidate_model.pkl"
    if not candidate_path.is_file():
        raise FileNotFoundError("Run 12_retrain_model.py before model comparison.")
    candidate = joblib.load(candidate_path)
    _, X_test, _, y_test = split_dataset(frame, config)
    production_metrics = _metrics(production, X_test, y_test)
    candidate_metrics = _metrics(candidate, X_test, y_test)
    metric = config["registry"]["decision_metric"]
    minimum = float(config["quality_gate"]["minimum_value"])
    production_score = production_metrics[metric]
    candidate_score = candidate_metrics[metric]
    report = {
        "decision_metric": metric,
        "minimum_value": minimum,
        "production_model": {"metrics": production_metrics},
        "candidate_model": {"metrics": candidate_metrics},
        "candidate_passes_quality_gate": candidate_score >= minimum,
        "candidate_improves_production": candidate_score > production_score,
        "candidate_recommended": (
            candidate_score >= minimum and candidate_score > production_score
        ),
    }
    _write_json(RETRAINING_DIR / "model_comparison_report.json", report)
    pd.DataFrame(
        [
            {"model": "production", **production_metrics},
            {"model": "candidate", **candidate_metrics},
        ]
    ).to_csv(RETRAINING_DIR / "model_comparison_report.csv", index=False)
    return report


def update_local_registry():
    comparison_path = RETRAINING_DIR / "model_comparison_report.json"
    if not comparison_path.is_file():
        raise FileNotFoundError(
            "Run 12_model_comparison.py before updating the candidate registry."
        )
    with comparison_path.open("r", encoding="utf-8") as file:
        comparison = json.load(file)
    candidate_path = PROJECT_ROOT / "models" / "candidate_model.pkl"
    if not candidate_path.is_file():
        raise FileNotFoundError("Run 12_retrain_model.py before registry update.")
    metric = comparison["decision_metric"]
    registry_dir = PROJECT_ROOT / "models" / "registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    version_prefix = "candidate-" + datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S"
    )
    version = version_prefix
    suffix = 1
    while (registry_dir / f"{version}.pkl").exists():
        version = f"{version_prefix}-{suffix}"
        suffix += 1
    registered_model_path = registry_dir / f"{version}.pkl"
    shutil.copy2(candidate_path, registered_model_path)
    manifest_path = registry_dir / "candidate_registry.json"
    previous_candidates = []
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as file:
            previous_candidates = json.load(file).get("candidates", [])
    candidate = {
        "version": version,
        "model_path": str(registered_model_path.relative_to(PROJECT_ROOT)),
        "decision_metric": metric,
        "score": comparison["candidate_model"]["metrics"][metric],
        "passes_quality_gate": comparison["candidate_passes_quality_gate"],
        "improves_production": comparison["candidate_improves_production"],
        "promoted": comparison["candidate_recommended"],
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    previous_candidates.append(candidate)
    _write_json(
        manifest_path,
        {
            "model_name": load_config()["registry"]["model_name"],
            "candidates": previous_candidates,
        },
    )
    report = {
        **candidate,
        "production_model_changed": False,
        "note": (
            "Candidate is recorded in the local registry manifest. Production "
            "model promotion is intentionally a separate, reviewed deployment step."
        ),
    }
    _write_json(RETRAINING_DIR / "model_registry_update_report.json", report)
    print(f"[SUCCESS] Registered candidate {version} at {registered_model_path}")
    return report


def run_monitor_trigger():
    for action in (create_evolved_dataset, create_concept_drift_dataset):
        action()
    create_updated_dataset()
    drift = analyze_all_drift()
    performance = monitor_performance()
    prediction = monitor_predictions()
    api_performance = monitor_api_performance()
    history = review_request_history()
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "Wisconsin Diagnostic Breast Cancer",
        "simulation_only": True,
        "drift": drift,
        "production_performance": performance,
        "api_predictions": prediction,
        "api_performance": api_performance,
        "request_history": {
            "events": history["events"],
            "failures": history["failures"],
        },
        "automatic_retraining": False,
        "status": "COMPLETED",
    }
    _write_json(MONITORING_DIR / "monitor_trigger_report.json", report)
    print(f"[SUCCESS] Lab 12 monitoring completed: {drift['recommendation']}")
    return report


def run_performance_trend():
    return monitor_performance_trend()


def main():
    run_monitor_trigger()


if __name__ == "__main__":
    main()
