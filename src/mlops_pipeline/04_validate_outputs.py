import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def main():
    required = (
        PROCESSED_DIR / "X_train.csv",
        PROCESSED_DIR / "X_test.csv",
        PROCESSED_DIR / "y_train.npy",
        PROCESSED_DIR / "y_test.npy",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Processed outputs are missing: {missing}")

    X_train = pd.read_csv(required[0])
    X_test = pd.read_csv(required[1])
    y_train = np.load(required[2])
    y_test = np.load(required[3])
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Training and test feature columns do not match.")
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Feature and target row counts do not match.")
    if not set(np.unique(np.concatenate((y_train, y_test)))).issubset({0, 1}):
        raise ValueError("Targets must contain only 0 (benign) and 1 (malignant).")

    print(
        f"[SUCCESS] Processed outputs validated: "
        f"{len(X_train)} train / {len(X_test)} test rows."
    )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
