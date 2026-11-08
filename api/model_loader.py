import json
from pathlib import Path
from typing import Any

import joblib
import yaml


class ModelLoader:
    def __init__(self, config_path: str = "config.yaml"):
        self.project_root = Path(__file__).resolve().parents[1]
        self.config_path = self.project_root / config_path
        self.config: dict[str, Any] = {}
        self.model = None
        self.metadata: dict[str, Any] = {}
        self.feature_names: list[str] = []
        self.model_name: str | None = None
        self.model_version: str | None = None
        self.model_uri: str | None = None

    def load(self) -> None:
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        with self.config_path.open("r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

        model_path = self.project_root / self.config["paths"]["model"]
        metadata_path = self.project_root / self.config["paths"]["model_metadata"]
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Model metadata not found: {metadata_path}")

        with metadata_path.open("r", encoding="utf-8") as file:
            self.metadata = json.load(file)
        self.model = joblib.load(model_path)
        self.feature_names = self.metadata["feature_names"]
        self.model_name = self.metadata.get("model_name")
        self.model_version = str(self.metadata.get("model_version", ""))
        self.model_uri = str(model_path)

    def is_ready(self) -> bool:
        return self.model is not None and bool(self.feature_names)
