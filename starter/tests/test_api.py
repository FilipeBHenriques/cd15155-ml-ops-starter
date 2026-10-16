"""Integration tests for the prediction API."""
import pytest
from fastapi.testclient import TestClient


@pytest.mark.parametrize("text", [
    "The company reported record profits and raised its dividend.",
    "The firm filed for bankruptcy after massive losses.",
])
def test_predict_returns_valid_response(client: TestClient, text):
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 200, response.text
    result = response.json()
    assert set(result) == {"text", "sentiment", "confidence", "latency_ms"}
    assert result["text"] == text
    assert result["sentiment"] in {"positive", "negative", "neutral"}
    assert 0 <= result["confidence"] <= 1
    assert result["latency_ms"] >= 0

def test_predict_batch(client: TestClient):
    texts = ["The company reported record profits and raised its dividend.", "The firm filed for bankruptcy after massive losses."]
    response = client.post("/predict/batch", json={"texts": texts})
    assert response.status_code == 200, response.text
    results = response.json()
    assert len(results) == len(texts)
    assert all(set(r) == {"text", "sentiment", "confidence", "latency_ms"} for r in results)

def test_predict_batch_empty_list_returns_422(client: TestClient):
    response = client.post("/predict/batch", json={"texts": []})
    assert response.status_code == 422
