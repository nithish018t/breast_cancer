from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def execute_pipeline():
    print("[INFO] =========================================")
    print("[INFO] Starting Lab 4: MLflow Experiment Tracking")
    print("[INFO] =========================================")
    
    scripts = [
        "src/preprocess.py",
        "src/train_mlflow.py",
        "src/validate_reproducibility.py"
    ]
    
    for script in scripts:
        print(f"\n[INFO] ---> Executing {script}...")
        script_path = PROJECT_ROOT / script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
        )
        
        if result.returncode != 0:
            print(f"[ERROR] Pipeline halted. {script} failed.")
            sys.exit(1)
            
    print("\n[SUCCESS] Lab 4 Pipeline fully executed!")

if __name__ == "__main__":
    execute_pipeline()