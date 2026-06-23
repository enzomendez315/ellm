from fastapi.testclient import TestClient

from ellm.main import app


def test_generate_returns_completion():
    # Activate lifespan to load model
    with TestClient(app) as client:
        response = client.post(
            "/generate", json={"prompt": "The capital of Italy is", "max_tokens": 10}
        )

    assert response.status_code == 200
    body = response.json()
    assert "completion" in body
    assert isinstance(body["completion"], str)
    assert len(body["completion"]) > 0


def test_generate_rejects_missing_prompt():
    # Activate lifespan to load model
    with TestClient(app) as client:
        response = client.post("/generate", json={"max_tokens": 10})
    assert response.status_code == 422
