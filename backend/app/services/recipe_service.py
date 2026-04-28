from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from fastapi import HTTPException
from torch.nn import functional as F
from transformers import CLIPModel, CLIPProcessor

from app.core.config import settings
from app.models.recipe import RecipePrediction


class RecipeService:
    def __init__(self) -> None:
        self._is_loaded = False
        self.device = settings.device
        self.model_id = settings.model_id
        self.df: pd.DataFrame | None = None
        self.model: CLIPModel | None = None
        self.processor: CLIPProcessor | None = None
        self.saved_image_features: torch.Tensor | None = None
        self.saved_text_features: torch.Tensor | None = None  

    def load(self) -> None:
        if self._is_loaded:
            return

        self.df = self._load_dataset()
        self.model = CLIPModel.from_pretrained(self.model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        
        if not settings.image_embeddings_path.exists():
            raise FileNotFoundError(f"Missing image embeddings at {settings.image_embeddings_path}")
        if not settings.text_embeddings_path.exists():
            raise FileNotFoundError(f"Missing text embeddings at {settings.text_embeddings_path}. Run build_embeddings.py first.")

        self.saved_image_features = torch.load(
            settings.image_embeddings_path,
            map_location="cpu",
        )
        
        # loading pre-computed embeddings 
        self.saved_text_features = torch.load(
            settings.text_embeddings_path,
            map_location="cpu",
        )

        if len(self.df) != self.saved_image_features.shape[0]:
            raise RuntimeError(
                "Dataset and image embeddings are misaligned: "
                f"{len(self.df)} rows vs {self.saved_image_features.shape[0]} embeddings."
            )

        self._is_loaded = True

    # dish_name is completely optional. Its purpose is to "help" the model.
    def identify_recipe_via_image(self, image_path: str | Path, dish_name: str | None = None) -> RecipePrediction:
        self.load()

        if self.model is None or self.processor is None or self.saved_image_features is None or self.df is None:
            raise RuntimeError("Recipe service not initialized correctly.")

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        self.model.eval()
        with torch.no_grad():
            vision_outputs = self.model.vision_model(**inputs)
            image_features = self.model.visual_projection(vision_outputs.pooler_output)
            image_features = F.normalize(image_features, p=2, dim=-1).cpu()
            image_similarities = (self.saved_image_features @ image_features.T).squeeze(1)

            # If dish_name is provided by the user
            if(dish_name):
                text_inputs = self.processor(text=[dish_name], return_tensors="pt", padding=True).to(self.device)
                text_outputs = self.model.text_model(**text_inputs)
                text_features = self.model.text_projection(text_outputs.pooler_output)
                text_features = F.normalize(text_features, p=2, dim=-1).cpu()

                text_similarities  = (self.saved_text_features @ text_features.T).squeeze(1)

                if(text_similarities.max().item() >= settings.text_threshold): # If the text is relevant enough
                    # img weights 0.8, text 0.2
                   similarities = settings.image_weight * image_similarities + settings.text_weight * text_similarities   
                else:
                    similarities = image_similarities
            else:
               similarities = image_similarities
        best_idx = similarities.argmax().item()
        row = self.df.iloc[best_idx]

        return RecipePrediction(
            title=str(row["Title"]),
            ingredients=self._parse_ingredients(row["Cleaned_Ingredients"]),
            instructions=str(row["Instructions"]),
            similarity=round(float(similarities[best_idx].item()), 3),
            reference_image_name=str(row["Image_Name"]) if pd.notna(row["Image_Name"]) else None,
        )

    def _load_dataset(self) -> pd.DataFrame:
        if not settings.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {settings.dataset_path}")

        df = pd.read_csv(settings.dataset_path, encoding="utf-8")
        df.drop(columns=["Unnamed: 0", ""], inplace=True, errors="ignore")

        df_missing = df[df.isnull().any(axis=1)]
        df = df.dropna()
        df = df[df["Image_Name"] != "#NAME?"]
        df = df[~df["Ingredients"].astype(str).isin(["[]", "['']"])]
        df = df[~df["Cleaned_Ingredients"].astype(str).isin(["[]", "['']"])]
        df = df.drop_duplicates(subset=["Title", "Ingredients", "Instructions"], keep="first")

        if not df_missing.empty:
            missing_image_names = set(df_missing["Image_Name"].astype(str).tolist())
            df = df[~df["Image_Name"].astype(str).isin(missing_image_names)]

        return df.reset_index(drop=True)

    @staticmethod
    def _parse_ingredients(raw_ingredients: object) -> list[str]:
        if isinstance(raw_ingredients, list):
            return [str(item) for item in raw_ingredients]

        if not isinstance(raw_ingredients, str):
            return []

        try:
            parsed = ast.literal_eval(raw_ingredients)
        except (ValueError, SyntaxError):
            return [raw_ingredients]

        if isinstance(parsed, list):
            return [str(item) for item in parsed]

        return [str(parsed)]

recipe_service = RecipeService()
