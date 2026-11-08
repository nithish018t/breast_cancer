import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from preprocess import run_preprocessing


if __name__ == "__main__":
    run_preprocessing()
