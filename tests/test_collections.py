from app.main import repo


def test_create_collection_success(client, monkeypatch):
    def fake_create_collection(collection_id: str):
        assert collection_id == "TEST_COLLECTION"

        return {
            "success": True,
            "statusCode": 200,
            "collectionArn": (
                "arn:local:face-recognition:"
                "collection/TEST_COLLECTION"
            ),
            "faceModelVersion": "sface_v1",
        }

    monkeypatch.setattr(
        repo,
        "create_collection",
        fake_create_collection,
    )

    response = client.post(
        "/collections",
        json={"collectionId": "TEST_COLLECTION"},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["statusCode"] == 200
    assert body["faceModelVersion"] == "sface_v1"
    assert body["collectionArn"].endswith(
        "/TEST_COLLECTION"
    )

    assert "x-request-id" in response.headers
    assert "x-process-time-ms" in response.headers


def test_create_duplicate_collection_returns_409(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        repo,
        "create_collection",
        lambda collection_id: {
            "success": False,
            "error_code": "ResourceAlreadyExistsException",
            "message": "Collection with this ID already exists",
        },
    )

    response = client.post(
        "/collections",
        json={"collectionId": "TEST_COLLECTION"},
    )

    assert response.status_code == 409

    detail = response.json()["detail"]

    assert detail["success"] is False
    assert (
        detail["error_code"]
        == "ResourceAlreadyExistsException"
    )


def test_list_collections_success(client, monkeypatch):
    def fake_list_collections(
        max_results: int,
        next_token: str | None,
    ):
        assert max_results == 10
        assert next_token is None

        return {
            "collectionIds": [
                "COLLECTION_A",
                "COLLECTION_B",
            ],
            "nextToken": None,
            "faceModelVersions": [
                "sface_v1",
                "sface_v1",
            ],
        }

    monkeypatch.setattr(
        repo,
        "list_collections",
        fake_list_collections,
    )

    response = client.get(
        "/collections",
        params={"maxResults": 10},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["collectionIds"] == [
        "COLLECTION_A",
        "COLLECTION_B",
    ]
    assert body["nextToken"] is None


def test_describe_missing_collection_returns_404(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        repo,
        "describe_collection",
        lambda collection_id: {
            "success": False,
            "error_code": "ResourceNotFoundException",
            "message": "Collection does not exist",
        },
    )

    response = client.get(
        "/collections/MISSING_COLLECTION"
    )

    assert response.status_code == 404

    detail = response.json()["detail"]

    assert (
        detail["error_code"]
        == "ResourceNotFoundException"
    )


def test_delete_collection_success(client, monkeypatch):
    monkeypatch.setattr(
        repo,
        "delete_collection",
        lambda collection_id: {
            "success": True,
            "statusCode": 200,
        },
    )

    response = client.delete(
        "/collections/TEST_COLLECTION"
    )

    assert response.status_code == 200
    assert response.json() == {"statusCode": 200}


def test_delete_missing_collection_returns_404(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        repo,
        "delete_collection",
        lambda collection_id: {
            "success": False,
            "statusCode": 404,
            "error_code": "ResourceNotFoundException",
            "message": "Collection does not exist",
        },
    )

    response = client.delete(
        "/collections/MISSING_COLLECTION"
    )

    assert response.status_code == 404

    detail = response.json()["detail"]

    assert (
        detail["error_code"]
        == "ResourceNotFoundException"
    )


def test_list_collections_rejects_zero_max_results(client):
    response = client.get(
        "/collections",
        params={"maxResults": 0},
    )

    assert response.status_code == 422


def test_list_collections_rejects_more_than_1000(client):
    response = client.get(
        "/collections",
        params={"maxResults": 1001},
    )

    assert response.status_code == 422