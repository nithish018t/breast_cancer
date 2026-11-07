from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def execute_pipeline():
    print("[INFO] =========================================")
    print("[INFO] Starting Lab 3: Baseline ML Pipeline")
    print("[INFO] =========================================")
    
    scripts = [
        "src/preprocess.py",
        "src/train.py",
        "src/evaluate.py"
    ]
    
    for script in scripts:
        print(f"\n[INFO] ---> Executing {script}...")
        script_path = PROJECT_ROOT / script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
        )
        
        if result.returncode != 0:
            print(f"[ERROR] Pipeline halted. {script} failed with exit code {result.returncode}.")
            sys.exit(1)
            
    print("\n[SUCCESS] Lab 3 Pipeline fully executed!")

if __name__ == "__main__":
    execute_pipeline()