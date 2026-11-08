import json
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    with (PROJECT_ROOT / config["paths"]["evaluation_report"]).open(
        "r", encoding="utf-8"
    ) as file:
        evaluation = json.load(file)
    metric = config["registry"]["decision_metric"]
    score = evaluation["metrics"][metric]
    gate = config["quality_gate"]
    promoted = score >= float(gate["minimum_value"])
    report = {
        "status": "success",
        "model_name": config["registry"]["model_name"],
        "decision_metric": metric,
        "candidate_metric": score,
        "promoted": promoted,
        "decision": "PROMOTED" if promoted else "QUALITY_GATE_FAILED",
    }
    report_path = PROJECT_ROOT / "reports" / "lifecycle_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(f"[INFO] Lifecycle decision: {report['decision']}")


if __name__ == "__main__":
    main()
