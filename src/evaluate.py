import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_evaluation():
    report_path = PROJECT_ROOT / "reports" / "evaluation_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(
            f"Evaluation report not found: {report_path}. Train the model first."
        )
    with report_path.open("r", encoding="utf-8") as file:
        report = json.load(file)
    print(f"Dataset: {report['dataset']}")
    for metric, value in report["metrics"].items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    run_evaluation()
