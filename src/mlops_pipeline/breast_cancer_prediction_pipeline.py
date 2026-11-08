import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGES = (
    ("Load and validate data", "src/mlops_pipeline/01_load_data.py"),
    ("Validate source schema", "src/mlops_pipeline/02_validate_data.py"),
    ("Preprocess", "src/mlops_pipeline/03_preprocessing.py"),
    ("Validate processed data", "src/mlops_pipeline/04_validate_outputs.py"),
    ("Train and evaluate", "src/mlops_pipeline/05_train_registry.py"),
    ("Report evaluation", "src/mlops_pipeline/06_model_evaluation.py"),
    ("Apply quality gate", "src/mlops_pipeline/07_quality_gate.py"),
    ("Record lifecycle decision", "src/mlops_pipeline/08_automate_lifecycle.py"),
    ("Run sample inference", "src/mlops_pipeline/09_inference.py"),
)


def run_pipeline():
    results = []
    for name, script in STAGES:
        print(f"\n--- {name} ---")
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / script)],
            cwd=PROJECT_ROOT,
            check=False,
        )
        results.append(
            {"stage": name, "status": "PASSED" if result.returncode == 0 else "FAILED"}
        )
        if result.returncode:
            break

    log_path = PROJECT_ROOT / "logs" / "breast_cancer_pipeline_run_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "pipeline": "Breast Cancer Diagnosis",
                "stages": results,
                "status": "PASSED"
                if len(results) == len(STAGES)
                and all(stage["status"] == "PASSED" for stage in results)
                else "FAILED",
            },
            file,
            indent=2,
        )
    return len(results) == len(STAGES) and all(
        stage["status"] == "PASSED" for stage in results
    )


if __name__ == "__main__":
    sys.exit(0 if run_pipeline() else 1)
