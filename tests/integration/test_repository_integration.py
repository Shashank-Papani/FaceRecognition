from uuid import uuid4

import pytest
from sqlalchemy import text

from app import face_repository as repo
from app.db import SessionLocal


EXPECTED_TEST_DATABASE = "face_recognition_test"

BOUNDING_BOX = {
    "width": 0.40,
    "height": 0.50,
    "left": 0.20,
    "top": 0.10,
}

QUALITY = {
    "face_confidence": 0.99,
    "face_confidence_percent": 99.0,
    "face_width": 200.0,
    "face_height": 240.0,
    "bounding_box": BOUNDING_BOX,
}


def make_unit_embedding(index: int) -> list[float]:
    """Create a normalized 128-dimensional test embedding."""
    embedding = [0.0] * 128
    embedding[index] = 1.0
    return embedding


def clear_test_database() -> None:
    with SessionLocal() as db:
        db.execute(
            text("""
                TRUNCATE TABLE
                    face_embeddings,
                    face_collections
                RESTART IDENTITY CASCADE
            """)
        )
        db.commit()


@pytest.fixture(scope="session", autouse=True)
def verify_test_database():
    """
    Refuse to run destructive integration tests against
    a development or production database.
    """
    with SessionLocal() as db:
        current_database = db.execute(
            text("SELECT current_database()")
        ).scalar_one()

    assert current_database == EXPECTED_TEST_DATABASE, (
        "Integration tests connected to the wrong database. "
        f"Expected '{EXPECTED_TEST_DATABASE}', "
        f"but connected to '{current_database}'. "
        "Tests stopped to protect your data."
    )


@pytest.fixture(autouse=True)
def clean_database(verify_test_database):
    clear_test_database()

    yield

    clear_test_database()


def save_test_face(
    collection_id: str,
    embedding: list[float],
    external_image_id: str,
) -> str:
    face_id = str(uuid4())
    image_id = str(uuid4())

    repo.save_indexed_face(
        collection_id=collection_id,
        face_id=face_id,
        image_id=image_id,
        external_image_id=external_image_id,
        embedding=embedding,
        confidence=99.0,
        bounding_box=BOUNDING_BOX,
        detector_model="test_yunet.onnx",
        recognizer_model="test_sface.onnx",
        embedding_model_version="sface_v1",
        quality=QUALITY,
    )

    return face_id


def test_collection_crud_uses_real_database():
    collection_id = "INTEGRATION_COLLECTION"

    create_result = repo.create_collection(collection_id)

    assert create_result["success"] is True
    assert create_result["statusCode"] == 200
    assert create_result["faceModelVersion"] == "sface_v1"

    describe_result = repo.describe_collection(collection_id)

    assert describe_result["success"] is True
    assert describe_result["faceCount"] == 0
    assert describe_result["faceModelVersion"] == "sface_v1"

    list_result = repo.list_collections()

    assert collection_id in list_result["collectionIds"]

    delete_result = repo.delete_collection(collection_id)

    assert delete_result["success"] is True

    missing_result = repo.describe_collection(collection_id)

    assert missing_result["success"] is False
    assert (
        missing_result["error_code"]
        == "ResourceNotFoundException"
    )


def test_save_face_updates_list_and_face_count():
    collection_id = "FACE_COUNT_COLLECTION"

    repo.create_collection(collection_id)

    face_id = save_test_face(
        collection_id=collection_id,
        embedding=make_unit_embedding(0),
        external_image_id="person-a",
    )

    list_result = repo.list_faces(collection_id)

    assert list_result["success"] is True
    assert len(list_result["faces"]) == 1
    assert list_result["faces"][0]["faceId"] == face_id
    assert (
        list_result["faces"][0]["externalImageId"]
        == "person-a"
    )

    describe_result = repo.describe_collection(collection_id)

    assert describe_result["faceCount"] == 1


def test_delete_face_removes_only_requested_face():
    collection_id = "DELETE_FACE_COLLECTION"

    repo.create_collection(collection_id)

    face_id_1 = save_test_face(
        collection_id=collection_id,
        embedding=make_unit_embedding(0),
        external_image_id="person-a",
    )

    face_id_2 = save_test_face(
        collection_id=collection_id,
        embedding=make_unit_embedding(1),
        external_image_id="person-b",
    )

    delete_result = repo.delete_faces(
        collection_id=collection_id,
        face_ids=[face_id_1],
    )

    assert delete_result["success"] is True
    assert delete_result["deletedFaces"] == [face_id_1]

    list_result = repo.list_faces(collection_id)

    remaining_ids = {
        face["faceId"]
        for face in list_result["faces"]
    }

    assert face_id_1 not in remaining_ids
    assert face_id_2 in remaining_ids

    describe_result = repo.describe_collection(collection_id)

    assert describe_result["faceCount"] == 1


def test_vector_search_returns_correct_face():
    collection_id = "VECTOR_SEARCH_COLLECTION"

    repo.create_collection(collection_id)

    matching_face_id = save_test_face(
        collection_id=collection_id,
        embedding=make_unit_embedding(0),
        external_image_id="matching-person",
    )

    save_test_face(
        collection_id=collection_id,
        embedding=make_unit_embedding(1),
        external_image_id="different-person",
    )

    result = repo.search_faces_by_embedding(
        collection_id=collection_id,
        query_embedding=make_unit_embedding(0),
        face_match_threshold=99.0,
        max_faces=10,
    )

    assert result["success"] is True
    assert len(result["faceMatches"]) == 1

    match = result["faceMatches"][0]

    assert match["face"]["faceId"] == matching_face_id
    assert match["face"]["externalImageId"] == "matching-person"
    assert match["similarity"] == pytest.approx(
        100.0,
        abs=0.01,
    )


def test_vector_search_isolated_by_collection():
    collection_a = "COLLECTION_A"
    collection_b = "COLLECTION_B"

    repo.create_collection(collection_a)
    repo.create_collection(collection_b)

    face_a = save_test_face(
        collection_id=collection_a,
        embedding=make_unit_embedding(0),
        external_image_id="person-a",
    )

    face_b = save_test_face(
        collection_id=collection_b,
        embedding=make_unit_embedding(0),
        external_image_id="person-b",
    )

    result = repo.search_faces_by_embedding(
        collection_id=collection_a,
        query_embedding=make_unit_embedding(0),
        face_match_threshold=99.0,
        max_faces=10,
    )

    returned_ids = {
        match["face"]["faceId"]
        for match in result["faceMatches"]
    }

    assert face_a in returned_ids
    assert face_b not in returned_ids


def test_delete_collection_cascades_to_faces():
    collection_id = "CASCADE_COLLECTION"

    repo.create_collection(collection_id)

    save_test_face(
        collection_id=collection_id,
        embedding=make_unit_embedding(0),
        external_image_id="person-a",
    )

    repo.delete_collection(collection_id)

    with SessionLocal() as db:
        remaining_faces = db.execute(
            text("""
                SELECT COUNT(*)
                FROM face_embeddings
                WHERE collection_id = :collection_id
            """),
            {"collection_id": collection_id},
        ).scalar_one()

    assert remaining_faces == 0

