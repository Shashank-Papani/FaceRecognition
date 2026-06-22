from app.main import repo


def test_root_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "status": "running",
        "message": "Face Recognition API is working",
    }


def test_model_info_endpoint(client):
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert (
        data["detector_model"]
        == "face_detection_yunet_2026may.onnx"
    )
    assert (
        data["recognizer_model"]
        == "face_recognition_sface_2021dec.onnx"
    )
    assert data["embedding_model_version"] == "sface_v1"
    assert data["embedding_dimension"] == 128


def test_collections_without_api_key_fails(auth_client):
    response = auth_client.get("/collections")

    assert response.status_code == 401

    detail = response.json()["detail"]

    assert detail["success"] is False
    assert detail["error_code"] == "UNAUTHORIZED"


def test_create_collection_with_valid_api_key(
    auth_client,
    monkeypatch,
):
    monkeypatch.setattr(
        repo,
        "create_collection",
        lambda collection_id: {
            "success": True,
            "statusCode": 200,
            "collectionArn": (
                "arn:local:face-recognition:"
                f"collection/{collection_id}"
            ),
            "faceModelVersion": "sface_v1",
        },
    )

    response = auth_client.post(
        "/collections",
        headers={"x-api-key": "test-api-key"},
        json={"collectionId": "AUTH_TEST_COLLECTION"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["statusCode"] == 200
    assert data["faceModelVersion"] == "sface_v1"


def test_create_collection_with_wrong_api_key_fails(
    auth_client,
):
    response = auth_client.post(
        "/collections",
        headers={"x-api-key": "wrong-key"},
        json={"collectionId": "AUTH_TEST_COLLECTION"},
    )

    assert response.status_code == 401

    detail = response.json()["detail"]

    assert detail["success"] is False
    assert detail["error_code"] == "UNAUTHORIZED"
