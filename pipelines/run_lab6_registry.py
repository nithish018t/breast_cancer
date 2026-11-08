from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def execute_pipeline():
    scripts = (
        "src/train_registry.py",
        "src/mlops_pipeline/07_quality_gate.py",
        "src/automate_lifecycle.py",
        "src/generate_registry_report.py",
    )
    for script in scripts:
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / script)],
            cwd=PROJECT_ROOT,
            check=False,
        )
        if result.returncode:
            raise SystemExit(f"Pipeline stage failed: {script}")
    print("[SUCCESS] Breast-cancer training and lifecycle pipeline completed.")


if __name__ == "__main__":
    execute_pipeline()
