from app.services.recipe_service import recipe_service


#we are mocking recipe_service.load and recipe_service.identify_recipe_via_image
def test_predict_recipe_success(client, monkeypatch):
    def fake_predict(image_path, dish_name=None):
        return {
            "title": "Lasagna",
            "ingredients": ["pasta", "meat"],
            "instructions": "Bake it",
            "similarity": 0.91,
            "reference_image_name": "lasagna.jpg",
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
    assert body["reference_image_name"] == "lasagna.jpg"


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


def test_get_reference_image_success(client, tmp_path, monkeypatch):
    image_dir = tmp_path / "Food Images"
    image_dir.mkdir()
    image_path = image_dir / "dish.jpg"
    image_path.write_bytes(b"fake-jpg")

    from app.api.routes import recipes as recipes_module

    monkeypatch.setattr(
        recipes_module.settings.__class__,
        "food_images_dir",
        property(lambda self: image_dir),
    )

    response = client.get("/recipes/reference-image/dish.jpg")

    assert response.status_code == 200
    assert response.content == b"fake-jpg"


def test_get_reference_image_resolves_prefixed_filename(client, tmp_path, monkeypatch):
    image_dir = tmp_path / "Food Images"
    image_dir.mkdir()
    image_path = image_dir / "my-mothers-butter-tomato-and-onion-sauce-395730.jpg"
    image_path.write_bytes(b"prefixed-jpg")

    from app.api.routes import recipes as recipes_module

    monkeypatch.setattr(
        recipes_module.settings.__class__,
        "food_images_dir",
        property(lambda self: image_dir),
    )

    response = client.get("/recipes/reference-image/my-mothers-butter-tomato-and-onion-sauce")

    assert response.status_code == 200
    assert response.content == b"prefixed-jpg"


def test_get_reference_image_rejects_path_traversal(client):
    response = client.get("/recipes/reference-image/%2E%2E/secret.jpg")

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid image path."
