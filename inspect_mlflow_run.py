import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Inspect the latest local breast-cancer evaluation report."
    )
    parser.add_argument(
        "--report",
        default="reports/evaluation_report.json",
        help="Evaluation report path relative to the project root.",
    )
    args = parser.parse_args()
    report_path = PROJECT_ROOT / args.report
    with report_path.open("r", encoding="utf-8") as file:
        report = json.load(file)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
