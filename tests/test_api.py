import os
from fastapi.testclient import TestClient

os.environ["API_KEY"] = "test-api-key"

from app.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"

def test_model_info_endpoint():
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["detector_model"] == "face_detection_yunet_2026may.onnx"
    assert data["recognizer_model"] == "face_recognition_sface_2021dec.onnx"
    assert data["embedding_model_version"] == "sface_v1"
    assert data["embedding_dimension"] == 128

def test_people_without_api_key_fails():
    response = client.get("/people")

    assert response.status_code == 401
    
    data = response.json()

    assert data["detail"]["success"] is False
    assert data["detail"]["error_code"] == "UNAUTHORIZED"

def test_people_with_api_key_passes_auth():
    response = client.get(
        "/people",
        headers={"x-api-key": "test-api-key"}
    )

    assert response.status_code in [200, 500]