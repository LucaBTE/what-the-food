import torch
from torch.nn import functional as F
from transformers import CLIPModel, CLIPProcessor

from app.core.config import settings
from app.services.recipe_service import RecipeService

def main():
    print(f"Loading dataset...")
    service = RecipeService()
    df = service._load_dataset()
    
    print(f"Loading model {settings.model_id} on {settings.device}...")
    model = CLIPModel.from_pretrained(settings.model_id).to(settings.device)
    processor = CLIPProcessor.from_pretrained(settings.model_id)
    
    print("Building text embeddings...")
    titles = df["Title"].astype(str).tolist()
    batch_size = settings.text_embedding_batch_size
    all_text_features = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(titles), batch_size):
            batch = titles[i:i + batch_size]
            inputs = processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(settings.device)

            text_outputs = model.text_model(**inputs)
            text_features = model.text_projection(text_outputs.pooler_output)
            text_features = F.normalize(text_features, p=2, dim=-1)
            all_text_features.append(text_features.cpu())

    saved_text = torch.cat(all_text_features, dim=0)
    
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving text embeddings to {settings.text_embeddings_path}...")
    torch.save(saved_text, settings.text_embeddings_path)
    
    print("Done!")

if __name__ == "__main__":
    main()