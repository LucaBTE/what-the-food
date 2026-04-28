from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, Form
from fastapi.responses import FileResponse

from app.core.config import settings
from app.models.recipe import RecipePrediction
from app.services.recipe_service import recipe_service

router = APIRouter(prefix="/recipes", tags=["recipes"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@router.post("/predict", response_model=RecipePrediction)
async def predict_recipe(file: UploadFile = File(...), dish_name: str | None = Form(None)) -> RecipePrediction:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported image format.")

    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(await file.read())
            temp_path = Path(temp_file.name)

        return recipe_service.identify_recipe_via_image(temp_path, dish_name)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)


@router.get("/reference-image/{image_name:path}")
def get_reference_image(image_name: str) -> FileResponse:
    images_dir = settings.food_images_dir.resolve()
    requested_path = (images_dir / image_name).resolve()

    if images_dir not in requested_path.parents:
        raise HTTPException(status_code=400, detail="Invalid image path.")

    image_path = _resolve_reference_image(images_dir, image_name, requested_path)

    if image_path is None:
        raise HTTPException(status_code=404, detail="Reference image not found.")

    return FileResponse(image_path)


def _resolve_reference_image(
    images_dir: Path,
    image_name: str,
    requested_path: Path,
) -> Path | None:
    if requested_path.exists() and requested_path.is_file():
        return requested_path

    requested_stem = Path(image_name).stem.lower()
    if not requested_stem:
        return None

    exact_stem_matches = [
        candidate
        for candidate in images_dir.iterdir()
        if candidate.is_file()
        and candidate.suffix.lower() in ALLOWED_IMAGE_SUFFIXES
        and candidate.stem.lower() == requested_stem
    ]
    if exact_stem_matches:
        return sorted(exact_stem_matches)[0]

    prefix_matches = [
        candidate
        for candidate in images_dir.iterdir()
        if candidate.is_file()
        and candidate.suffix.lower() in ALLOWED_IMAGE_SUFFIXES
        and candidate.stem.lower().startswith(f"{requested_stem}-")
    ]
    if prefix_matches:
        return sorted(prefix_matches)[0]

    return None
