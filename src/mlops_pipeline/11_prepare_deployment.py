import json
from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "deployment" / "model.pkl"
METADATA_PATH = PROJECT_ROOT / "deployment" / "model_metadata.json"


def main():
    for path in (MODEL_PATH, METADATA_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Deployment artifact not found: {path}")
    model = joblib.load(MODEL_PATH)
    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    if not metadata.get("feature_names"):
        raise ValueError("Model metadata does not contain feature names.")
    if not hasattr(model, "predict_proba"):
        raise TypeError("Deployment model must provide class probabilities.")
    print(
        f"[SUCCESS] Verified {metadata['model_name']} v{metadata['model_version']} "
        f"with {len(metadata['feature_names'])} features."
    )


if __name__ == "__main__":
    main()
