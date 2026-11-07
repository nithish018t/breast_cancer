from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def execute_pipeline():
    print("[INFO] =========================================")
    print("[INFO] Starting Lab 6: Model Registry & Lifecycle")
    print("[INFO] =========================================")
    
    scripts = [
        "src/train_registry.py",          # Trains and registers a new model version
        "src/automate_lifecycle.py",      # Compares versions and promotes the Champion
        "src/generate_registry_report.py" # Generates the JSON deployment artifact
    ]
    
    for script in scripts:
        print(f"\n[INFO] ---> Executing {script}...")
        script_path = PROJECT_ROOT / script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
        )
        
        if result.returncode != 0:
            print(f"[ERROR] Pipeline halted. {script} caught an error.")
            sys.exit(1)
            
    print("\n[SUCCESS] Lab 6 Model Registry Pipeline fully executed!")

if __name__ == "__main__":
    execute_pipeline()