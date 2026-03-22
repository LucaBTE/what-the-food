#simple endpoint test

#it makes sure that '/health' endpoint will answer with status code 200 body {"status": "ok"}

def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
