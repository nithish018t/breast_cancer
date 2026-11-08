import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from breast_cancer_pipeline import train_and_evaluate


if __name__ == "__main__":
    train_and_evaluate()
