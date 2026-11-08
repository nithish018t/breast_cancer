import json
from pathlib import Path

from sklearn.metrics import f1_score

from breast_cancer_pipeline import load_config, load_dataset, prepare_splits
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_deterministic_test():
    config = load_config()
    frame, feature_columns = load_dataset(config)
    X_train, X_test, y_train, y_test, _, _ = prepare_splits(
        config, frame, feature_columns
    )
    scores = []
    for _ in range(2):
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=config["model"]["n_estimators"],
                        max_depth=config["model"]["max_depth"],
                        random_state=config["model"]["random_state"],
                        class_weight=config["model"]["class_weight"],
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train)
        scores.append(float(f1_score(y_test, model.predict(X_test))))

    reproducible = scores[0] == scores[1]
    report = {
        "dataset": "Wisconsin Diagnostic Breast Cancer",
        "execution_f1": scores,
        "is_strictly_reproducible": reproducible,
        "status": "PASSED" if reproducible else "FAILED",
    }
    report_path = PROJECT_ROOT / "reports" / "reproducibility_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    if not reproducible:
        raise AssertionError("Repeated training did not produce identical F1 scores.")
    print(f"[SUCCESS] Deterministic F1 score: {scores[0]:.6f}")


if __name__ == "__main__":
    run_deterministic_test()
