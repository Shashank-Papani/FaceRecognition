from app.main import engine, repo

def fake_image(
    filename: str = "face.jpg",
    content_type: str = "image/jpeg",
):
    return {
        "image": (
            filename,
            b"fake-image-content",
            content_type,
        )
    }

def test_index_face_success(client, monkeypatch):
    monkeypatch.setattr(
        engine,
        "index_face",
        lambda collection_id, image_path, external_image_id: {
            "success": True,
            "faceRecords": [
                {
                    "face": {
                        "faceId": "11111111-1111-1111-1111-111111111111",
                        "imageId": "22222222-2222-2222-2222-222222222222",
                        "externalImageId": external_image_id,
                        "confidence": 99.2,
                        "boundingBox": {
                            "width": 0.4,
                            "height": 0.5,
                            "left": 0.2,
                            "top": 0.1,
                        },
                    }
                }
            ],
            "faceModelVersion": "sface_v1",
        },
    )

    response = client.post(
        "/collections/TEST_COLLECTION/faces",
        files=fake_image(),
        data={"externalImageId": "person-123"},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["faceModelVersion"] == "sface_v1"
    assert len(body["faceRecords"]) == 1
    assert (
        body["faceRecords"][0]["face"]["externalImageId"]
        == "person-123"
    )


def test_index_face_missing_collection_returns_404(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        engine,
        "index_face",
        lambda collection_id, image_path, external_image_id: {
            "success": False,
            "error_code": "ResourceNotFoundException",
            "message": "Collection does not exist",
        },
    )

    response = client.post(
        "/collections/MISSING_COLLECTION/faces",
        files=fake_image(),
    )

    assert response.status_code == 404

    detail = response.json()["detail"]

    assert (
        detail["error_code"]
        == "ResourceNotFoundException"
    )


def test_list_faces_success(client, monkeypatch):
    def fake_list_faces(
        collection_id,
        max_results,
        next_token,
        face_ids,
    ):
        assert collection_id == "TEST_COLLECTION"
        assert max_results == 10
        assert next_token is None
        assert face_ids is None

        return {
            "success": True,
            "faces": [
                {
                    "faceId": "11111111-1111-1111-1111-111111111111",
                    "imageId": "22222222-2222-2222-2222-222222222222",
                    "externalImageId": "person-123",
                    "confidence": 99.2,
                    "boundingBox": {
                        "width": 0.4,
                        "height": 0.5,
                        "left": 0.2,
                        "top": 0.1,
                    },
                }
            ],
            "nextToken": None,
            "faceModelVersion": "sface_v1",
        }

    monkeypatch.setattr(
        repo,
        "list_faces",
        fake_list_faces,
    )

    response = client.get(
        "/collections/TEST_COLLECTION/faces",
        params={"maxResults": 10},
    )

    assert response.status_code == 200

    body = response.json()

    assert len(body["faces"]) == 1
    assert body["nextToken"] is None
    assert body["faceModelVersion"] == "sface_v1"


def test_list_faces_missing_collection_returns_404(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        repo,
        "list_faces",
        lambda **kwargs: {
            "success": False,
            "error_code": "ResourceNotFoundException",
            "message": "Collection does not exist",
        },
    )

    response = client.get(
        "/collections/MISSING_COLLECTION/faces"
    )

    assert response.status_code == 404

    detail = response.json()["detail"]

    assert (
        detail["error_code"]
        == "ResourceNotFoundException"
    )


def test_list_faces_rejects_invalid_max_results(client):
    response = client.get(
        "/collections/TEST_COLLECTION/faces",
        params={"maxResults": 4097},
    )

    assert response.status_code == 422


def test_delete_faces_success(client, monkeypatch):
    face_id = "11111111-1111-1111-1111-111111111111"

    monkeypatch.setattr(
        repo,
        "delete_faces",
        lambda collection_id, face_ids: {
            "success": True,
            "deletedFaces": face_ids,
        },
    )

    response = client.request(
        "DELETE",
        "/collections/TEST_COLLECTION/faces",
        json={"faceIds": [face_id]},
    )

    assert response.status_code == 200
    assert response.json() == {
        "deletedFaces": [face_id]
    }


def test_delete_faces_invalid_uuid_returns_400(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        repo,
        "delete_faces",
        lambda collection_id, face_ids: {
            "success": False,
            "error_code": "InvalidParameterException",
            "message": "Each faceId must be a valid UUID",
        },
    )

    response = client.request(
        "DELETE",
        "/collections/TEST_COLLECTION/faces",
        json={"faceIds": ["not-a-uuid"]},
    )

    assert response.status_code == 400

    detail = response.json()["detail"]

    assert (
        detail["error_code"]
        == "InvalidParameterException"
    )


def test_search_faces_by_image_success(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        engine,
        "search_faces_by_image",
        lambda collection_id,
        image_path,
        face_match_threshold,
        max_faces: {
            "success": True,
            "faceMatches": [
                {
                    "face": {
                        "faceId": "11111111-1111-1111-1111-111111111111",
                        "imageId": "22222222-2222-2222-2222-222222222222",
                        "externalImageId": "person-123",
                        "confidence": 99.2,
                        "boundingBox": {
                            "width": 0.4,
                            "height": 0.5,
                            "left": 0.2,
                            "top": 0.1,
                        },
                    },
                    "similarity": 94.5,
                }
            ],
            "searchedFaceBoundingBox": {
                "width": 0.4,
                "height": 0.5,
                "left": 0.2,
                "top": 0.1,
            },
            "searchedFaceConfidence": 98.7,
            "faceModelVersion": "sface_v1",
        },
    )

    response = client.post(
        "/collections/TEST_COLLECTION/search",
        files=fake_image(),
        data={
            "faceMatchThreshold": "80",
            "maxFaces": "1",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert len(body["faceMatches"]) == 1
    assert body["faceMatches"][0]["similarity"] == 94.5
    assert body["searchedFaceConfidence"] == 98.7
    assert body["faceModelVersion"] == "sface_v1"


def test_search_missing_collection_returns_404(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        engine,
        "search_faces_by_image",
        lambda collection_id,
        image_path,
        face_match_threshold,
        max_faces: {
            "success": False,
            "error_code": "ResourceNotFoundException",
            "message": "Collection does not exist",
        },
    )

    response = client.post(
        "/collections/MISSING_COLLECTION/search",
        files=fake_image(),
    )

    assert response.status_code == 404

    detail = response.json()["detail"]

    assert (
        detail["error_code"]
        == "ResourceNotFoundException"
    )


def test_search_rejects_invalid_threshold(client):
    response = client.post(
        "/collections/TEST_COLLECTION/search",
        files=fake_image(),
        data={"faceMatchThreshold": "101"},
    )

    assert response.status_code == 422


def test_detect_faces_success(client, monkeypatch):
    expected_result = {
        "faceDetails": [
            {
                "confidence": 88.94,
                "boundingBox": {
                    "width": 0.36,
                    "height": 0.40,
                    "left": 0.33,
                    "top": 0.33,
                },
                "landmarks": [
                    {
                        "type": "eyeRight",
                        "x": 0.40,
                        "y": 0.49,
                    }
                ],
                "quality": {
                    "brightness": 41.72,
                    "sharpness": 16.75,
                },
            }
        ],
        "orientationCorrection": None,
    }

    monkeypatch.setattr(
        engine,
        "detect_faces_info",
        lambda image_path, attributes: expected_result,
    )

    response = client.post(
        "/faces/detect",
        files=fake_image(),
        data={"attributes": "DEFAULT"},
    )

    assert response.status_code == 200
    assert response.json() == expected_result


def test_detect_faces_rejects_invalid_extension(client):
    response = client.post(
        "/faces/detect",
        files=fake_image(
            filename="face.gif",
            content_type="image/gif",
        ),
        data={"attributes": "DEFAULT"},
    )

    assert response.status_code == 400

    detail = response.json()["detail"]

    assert (
        detail["error_code"]
        == "InvalidImageFormatException"
    )
