from pydantic import BaseModel


class RecipePrediction(BaseModel):
    title: str
    ingredients: list[str]
    instructions: str
    similarity: float
