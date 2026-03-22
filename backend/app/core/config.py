from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str = "What The Food API"
    app_version: str = "0.1.0"
    model_id: str = "openai/clip-vit-base-patch32"
    device: str = "cpu"

    text_threshold: float = 0.30
    image_weight: float = 0.8
    text_weight: float = 0.2
    text_embedding_batch_size: int = 128
    mlflow_tracking_uri: str = "file:/app/mlruns"
    mlflow_experiment_name: str = "what-the-food-retrieval"



    @property
    def backend_dir(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def project_dir(self) -> Path:
        return self.backend_dir.parent

    @property
    def data_dir(self) -> Path:
        return self.project_dir / "data"

    @property
    def artifacts_dir(self) -> Path:
        return self.project_dir / "artifacts"

    @property
    def mlruns_dir(self) -> Path:
        return self.project_dir / "mlruns"

    @property
    def dataset_path(self) -> Path:
        return self.data_dir / "dataset.csv"

    @property
    def image_embeddings_path(self) -> Path:
        return self.data_dir / "image_embeddings.pt"
    
    @property
    def text_embeddings_path(self) -> Path: 
        return self.data_dir / "text_embeddings.pt"
    



settings = Settings()
