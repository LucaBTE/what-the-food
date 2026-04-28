#pure unit tests

import pandas as pd

from app.services.recipe_service import RecipeService


def test_parse_ingredients_from_list():
    raw = ["egg", "milk"]
    result = RecipeService._parse_ingredients(raw)
    assert result == ["egg", "milk"]


def test_parse_ingredients_from_stringified_list():
    raw = "['egg', 'milk']"
    result = RecipeService._parse_ingredients(raw)
    assert result == ["egg", "milk"]


def test_parse_ingredients_from_invalid_string():
    raw = "egg, milk"
    result = RecipeService._parse_ingredients(raw)
    assert result == ["egg, milk"]


def test_parse_ingredients_from_none():
    result = RecipeService._parse_ingredients(None)
    assert result == []


def test_load_dataset_drops_rows_with_missing_values(monkeypatch):
    fake_df = pd.DataFrame([
        {
            "Title": "Soup",
            "Ingredients": "['water']",
            "Cleaned_Ingredients": "['water']",
            "Instructions": "Boil",
            "Image_Name": "soup.jpg",
        },
        {
            "Title": "Broken",
            "Ingredients": None,
            "Cleaned_Ingredients": "['x']",
            "Instructions": "Bad",
            "Image_Name": "broken.jpg",
        },
    ])

    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: fake_df)

    service = RecipeService()
    result = service._load_dataset()

    assert len(result) == 1
    assert result.iloc[0]["Title"] == "Soup"


def test_load_dataset_drops_invalid_image_name_rows(monkeypatch):
    fake_df = pd.DataFrame([
        {
            "Title": "Good",
            "Ingredients": "['egg']",
            "Cleaned_Ingredients": "['egg']",
            "Instructions": "Cook",
            "Image_Name": "good.jpg",
        },
        {
            "Title": "Bad",
            "Ingredients": "['egg']",
            "Cleaned_Ingredients": "['egg']",
            "Instructions": "Cook",
            "Image_Name": "#NAME?",
        },
    ])

    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: fake_df)

    service = RecipeService()
    result = service._load_dataset()

    assert len(result) == 1
    assert result.iloc[0]["Title"] == "Good"


def test_load_dataset_drops_empty_ingredients(monkeypatch):
    fake_df = pd.DataFrame([
        {
            "Title": "Good",
            "Ingredients": "['egg']",
            "Cleaned_Ingredients": "['egg']",
            "Instructions": "Cook",
            "Image_Name": "good.jpg",
        },
        {
            "Title": "Bad",
            "Ingredients": "[]",
            "Cleaned_Ingredients": "[]",
            "Instructions": "Cook",
            "Image_Name": "bad.jpg",
        },
    ])

    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: fake_df)

    service = RecipeService()
    result = service._load_dataset()

    assert len(result) == 1
    assert result.iloc[0]["Title"] == "Good"


def test_load_dataset_drops_duplicates(monkeypatch):
    fake_df = pd.DataFrame([
        {
            "Title": "Pasta",
            "Ingredients": "['pasta']",
            "Cleaned_Ingredients": "['pasta']",
            "Instructions": "Boil",
            "Image_Name": "pasta1.jpg",
        },
        {
            "Title": "Pasta",
            "Ingredients": "['pasta']",
            "Cleaned_Ingredients": "['pasta']",
            "Instructions": "Boil",
            "Image_Name": "pasta2.jpg",
        },
    ])

    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: fake_df)

    service = RecipeService()
    result = service._load_dataset()

    assert len(result) == 1


def test_load_dataset_resets_index(monkeypatch):
    fake_df = pd.DataFrame([
        {
            "Title": "Good 1",
            "Ingredients": "['a']",
            "Cleaned_Ingredients": "['a']",
            "Instructions": "Do",
            "Image_Name": "a.jpg",
        },
        {
            "Title": "Bad",
            "Ingredients": None,
            "Cleaned_Ingredients": "['x']",
            "Instructions": "Bad",
            "Image_Name": "bad.jpg",
        },
        {
            "Title": "Good 2",
            "Ingredients": "['b']",
            "Cleaned_Ingredients": "['b']",
            "Instructions": "Do",
            "Image_Name": "b.jpg",
        },
    ])

    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: fake_df)

    service = RecipeService()
    result = service._load_dataset()

    assert list(result.index) == [0, 1]
