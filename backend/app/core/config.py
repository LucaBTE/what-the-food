from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str = "What The Food API"
    app_version: str = "0.1.0"
    model_id: str = "openai/clip-vit-base-patch32"
    device: str = "cpu"

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
    def dataset_path(self) -> Path:
        return self.data_dir / "dataset.csv"

    @property
    def image_embeddings_path(self) -> Path:
        return self.data_dir / "image_embeddings.pt"


settings = Settings()
