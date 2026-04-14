from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "What The Food API")
    app_version: str = os.getenv("APP_VERSION", "0.1.0")
    model_id: str = os.getenv("MODEL_ID", "openai/clip-vit-base-patch32")
    device: str = os.getenv("DEVICE", "cpu")

    text_threshold: float = float(os.getenv("TEXT_THRESHOLD", "0.30"))
    image_weight: float = float(os.getenv("IMAGE_WEIGHT", "0.8"))
    text_weight: float = float(os.getenv("TEXT_WEIGHT", "0.2"))
    text_embedding_batch_size: int = int(os.getenv("TEXT_EMBEDDING_BATCH_SIZE", "128"))
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "file:/app/mlruns")
    mlflow_experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "what-the-food-retrieval")

    @property
    def backend_dir(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def project_dir(self) -> Path:
        return self.backend_dir.parent

    @property
    def data_dir(self) -> Path:
        dir_path = os.getenv("DATA_DIR")
        return Path(dir_path) if dir_path else (self.project_dir / "data")

    @property
    def artifacts_dir(self) -> Path:
        dir_path = os.getenv("ARTIFACTS_DIR")
        return Path(dir_path) if dir_path else (self.project_dir / "artifacts")

    @property
    def mlruns_dir(self) -> Path:
        dir_path = os.getenv("MLRUNS_DIR")
        return Path(dir_path) if dir_path else (self.project_dir / "mlruns")

    @property
    def dataset_path(self) -> Path:
        path = os.getenv("DATASET_PATH")
        return Path(path) if path else (self.data_dir / "dataset.csv")

    @property
    def image_embeddings_path(self) -> Path:
        path = os.getenv("IMAGE_EMBEDDINGS_PATH")
        return Path(path) if path else (self.data_dir / "image_embeddings.pt")
    
    @property
    def text_embeddings_path(self) -> Path: 
        path = os.getenv("TEXT_EMBEDDINGS_PATH")
        return Path(path) if path else (self.data_dir / "text_embeddings.pt")

settings = Settings()
