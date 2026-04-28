#common fixtures

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.recipe_service import recipe_service


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(recipe_service, "load", lambda: None)

    with TestClient(app) as test_client:
        yield test_client
