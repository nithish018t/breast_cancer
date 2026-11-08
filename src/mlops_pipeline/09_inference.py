import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from api.model_loader import ModelLoader


def main():
    config_path = PROJECT_ROOT / "config.yaml"
    import yaml

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    dataset_path = PROJECT_ROOT / config["data"]["raw_path"]
    frame = pd.read_csv(dataset_path)
    loader = ModelLoader()
    loader.load()
    features = frame[loader.feature_names].iloc[:5]
    predictions = loader.model.predict(features)
    probabilities = loader.model.predict_proba(features)[:, 1]
    output = pd.DataFrame(
        {
            config["data"]["id_column"]: frame[config["data"]["id_column"]].iloc[:5],
            "predicted_diagnosis": ["M" if value else "B" for value in predictions],
            "malignant_probability": probabilities,
        }
    )
    output_path = PROJECT_ROOT / config["paths"]["inference_output"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"[SUCCESS] Saved inference results to {output_path}.")


if __name__ == "__main__":
    main()
