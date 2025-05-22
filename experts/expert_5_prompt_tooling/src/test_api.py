import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_label_endpoint_detects_ai():
    response = client.post("/label", json={"prompt": "Use OpenAI's GPT-3 for this task."})
    assert response.status_code == 200
    assert response.json() == {"label": 0}

def test_label_endpoint_no_ai():
    response = client.post("/label", json={"prompt": "Write a function to add two numbers."})
    assert response.status_code == 200
    assert response.json() == {"label": 1}

def test_label_endpoint_invalid_request():
    response = client.post("/label", json={})
    assert response.status_code == 422 or response.status_code == 400

def test_label_endpoint_edge_cases():
    response = client.post("/label", json={"prompt": "   "})
    assert response.status_code == 200
    assert response.json() == {"label": 1}

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"} 