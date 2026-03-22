from app.services.recipe_service import recipe_service


#we are mocking recipe_service.load and recipe_service.identify_recipe_via_image
def test_predict_recipe_success(client, monkeypatch):
    def fake_predict(image_path, dish_name=None):
        return {
            "title": "Lasagna",
            "ingredients": ["pasta", "meat"],
            "instructions": "Bake it",
            "similarity": 0.91,
        }

    monkeypatch.setattr(recipe_service, "identify_recipe_via_image", fake_predict)

    response = client.post(
        "/recipes/predict",
        files={"file": ("dish.jpg", b"fake-image-bytes", "image/jpeg")},
        data={"dish_name": "lasagna"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "Lasagna"
    assert body["ingredients"] == ["pasta", "meat"]
    assert body["similarity"] == 0.91


#no need to mock anything because the endpoint will fail before
def test_predict_recipe_rejects_unsupported_format(client):
    response = client.post(
        "/recipes/predict",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported image format."


#testing try/except
def test_predict_recipe_returns_500_on_service_error(client, monkeypatch):
    def fake_predict(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(recipe_service, "identify_recipe_via_image", fake_predict)

    response = client.post(
        "/recipes/predict",
        files={"file": ("dish.jpg", b"fake-image-bytes", "image/jpeg")},
    )

    assert response.status_code == 500
    assert "Prediction failed" in response.json()["detail"]
