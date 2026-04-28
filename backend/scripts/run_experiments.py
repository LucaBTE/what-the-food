from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import mlflow
import pandas as pd
import torch
from transformers import CLIPModel, CLIPProcessor

from app.core.config import settings
from app.services.recipe_service import RecipeService

#calculates hash SHA356 of dataset file
#if dataset changes, hash changes
def compute_file_sha256(path: Path) -> str:
    sha = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha.update(chunk)

    return sha.hexdigest()

#it creates a run_summary.json
def build_run_summary(
    dataset_sha256: str,
    raw_row_count: int,
    cleaned_row_count: int,
    rows_removed: int,
    build_seconds: float,
    text_embedding_shape: tuple[int, ...],
    image_embedding_shape: tuple[int, ...] | None,
    used_cached_text_embeddings: bool,
) -> dict[str, object]:
    return {
        "model_id": settings.model_id,
        "device": settings.device,
        "dataset_path": str(settings.dataset_path),
        "dataset_sha256": dataset_sha256,
        "text_threshold": settings.text_threshold,
        "image_weight": settings.image_weight,
        "text_weight": settings.text_weight,
        "text_embedding_batch_size": settings.text_embedding_batch_size,
        "raw_row_count": raw_row_count,
        "cleaned_row_count": cleaned_row_count,
        "rows_removed": rows_removed,
        "build_seconds": round(build_seconds, 3),
        "used_cached_text_embeddings": used_cached_text_embeddings,
        "text_embedding_shape": list(text_embedding_shape),
        "image_embedding_shape": list(image_embedding_shape) if image_embedding_shape else None,
        "text_embeddings_path": str(settings.text_embeddings_path),
        "image_embeddings_path": str(settings.image_embeddings_path),
    }


def main() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    dataset_path = settings.dataset_path
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    raw_df = pd.read_csv(dataset_path, encoding="utf-8")
    dataset_sha256 = compute_file_sha256(dataset_path)
    raw_row_count = len(raw_df)
    dataset_size_bytes = dataset_path.stat().st_size

    artifact_dir = settings.artifacts_dir / "mlflow"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    run_name = (
        f"clip-thr-{settings.text_threshold}"
        f"-img-{settings.image_weight}"
        f"-txt-{settings.text_weight}"
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model_id": settings.model_id,
                "device": settings.device,
                "dataset_path": str(dataset_path),
                "dataset_sha256": dataset_sha256,
                "dataset_size_bytes": dataset_size_bytes,
                "raw_row_count": raw_row_count,
                "text_threshold": settings.text_threshold,
                "image_weight": settings.image_weight,
                "text_weight": settings.text_weight,
                "text_embedding_batch_size": settings.text_embedding_batch_size,
            }
        )

        service = RecipeService()
        cleaned_df = service._load_dataset()
        cleaned_row_count = len(cleaned_df)
        rows_removed = raw_row_count - cleaned_row_count

        mlflow.log_metrics(
            {
                "cleaned_row_count": cleaned_row_count,
                "rows_removed": rows_removed,
            }
        )

        cleaned_dataset_artifact = artifact_dir / "cleaned_dataset.csv"
        cleaned_df.to_csv(cleaned_dataset_artifact, index=False)
        mlflow.log_artifact(str(cleaned_dataset_artifact))

        service.df = cleaned_df
        service.model = CLIPModel.from_pretrained(settings.model_id).to(settings.device)
        service.processor = CLIPProcessor.from_pretrained(settings.model_id)

        text_embeddings_cached = settings.text_embeddings_path.exists()
        build_start = time.time()
        
        if not text_embeddings_cached:
            raise FileNotFoundError("Text embeddings must be built via build_embeddings.py first.")
        text_embeddings = torch.load(settings.text_embeddings_path, map_location="cpu")
        
        build_seconds = time.time() - build_start

        image_embedding_shape: tuple[int, ...] | None = None
        if settings.image_embeddings_path.exists():
            image_embeddings = torch.load(settings.image_embeddings_path, map_location="cpu")
            image_embedding_shape = tuple(image_embeddings.shape)
            mlflow.log_metric("image_embedding_count", image_embeddings.shape[0])

        mlflow.log_metrics(
            {
                "embedding_build_seconds": build_seconds,
                "text_embedding_count": text_embeddings.shape[0],
                "text_embedding_dimension": text_embeddings.shape[1],
            }
        )

        if settings.text_embeddings_path.exists():
            mlflow.log_artifact(str(settings.text_embeddings_path))

        if settings.image_embeddings_path.exists():
            mlflow.log_artifact(str(settings.image_embeddings_path))

        run_summary = build_run_summary(
            dataset_sha256=dataset_sha256,
            raw_row_count=raw_row_count,
            cleaned_row_count=cleaned_row_count,
            rows_removed=rows_removed,
            build_seconds=build_seconds,
            text_embedding_shape=tuple(text_embeddings.shape),
            image_embedding_shape=image_embedding_shape,
            used_cached_text_embeddings=text_embeddings_cached,
        )

        run_summary_artifact = artifact_dir / "run_summary.json"
        run_summary_artifact.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(run_summary_artifact))


if __name__ == "__main__":
    main()
