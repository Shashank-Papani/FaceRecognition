from pathlib import Path

import cv2
import numpy as np
import pytest
from sqlalchemy import text

from app.db import SessionLocal
from app.main import engine


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = PROJECT_ROOT / "tests" / "private_assets"

PERSON_A_ENROLL = ASSET_DIR / "person_a" / "enroll.jpg"
PERSON_A_QUERY = ASSET_DIR / "person_a" / "query.jpg"
PERSON_B_QUERY = ASSET_DIR / "person_b" / "query.jpg"

EXPECTED_TEST_DATABASE = "face_recognition_test"
COLLECTION_ID = "ML_PIPELINE_TEST"


def image_upload(image_path: Path) -> dict:
    if image_path.suffix.lower() == ".png":
        content_type = "image/png"
    else:
        content_type = "image/jpeg"

    return {
        "image": (
            image_path.name,
            image_path.read_bytes(),
            content_type,
        )
    }


def require_image(image_path: Path) -> None:
    assert image_path.exists(), (
        f"Missing test image: {image_path}\n"
        "Add the required private test image before running "
        "the ML integration tests."
    )


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
def verify_ml_test_database():
    with SessionLocal() as db:
        current_database = db.execute(
            text("SELECT current_database()")
        ).scalar_one()

    assert current_database == EXPECTED_TEST_DATABASE, (
        "ML integration tests connected to the wrong database. "
        f"Expected '{EXPECTED_TEST_DATABASE}', "
        f"but connected to '{current_database}'."
    )


@pytest.fixture(autouse=True)
def clean_ml_database(verify_ml_test_database):
    clear_test_database()

    yield

    clear_test_database()


@pytest.fixture(scope="session")
def face_images():
    required_images = {
        "person_a_enroll": PERSON_A_ENROLL,
        "person_a_query": PERSON_A_QUERY,
        "person_b_query": PERSON_B_QUERY,
    }

    for image_path in required_images.values():
        require_image(image_path)

    return required_images


def test_real_yunet_detects_face(client, face_images):
    response = client.post(
        "/faces/detect",
        files=image_upload(
            face_images["person_a_enroll"]
        ),
        data={"attributes": "DEFAULT"},
    )

    assert response.status_code == 200

    body = response.json()

    assert "faceDetails" in body
    assert len(body["faceDetails"]) == 1

    face = body["faceDetails"][0]

    assert 0.0 <= face["confidence"] <= 100.0
    assert face["confidence"] >= 60.0

    bounding_box = face["boundingBox"]

    assert 0.0 <= bounding_box["left"] <= 1.0
    assert 0.0 <= bounding_box["top"] <= 1.0
    assert 0.0 < bounding_box["width"] <= 1.0
    assert 0.0 < bounding_box["height"] <= 1.0

    assert len(face["landmarks"]) == 5
    assert "brightness" in face["quality"]
    assert "sharpness" in face["quality"]


def test_real_sface_embedding_has_128_dimensions(
    face_images,
):
    embedding = engine.get_embedding(
        str(face_images["person_a_enroll"])
    )

    embedding_array = np.asarray(
        embedding,
        dtype=np.float32,
    ).reshape(-1)

    assert embedding_array.shape == (128,)
    assert np.isfinite(embedding_array).all()
    assert np.linalg.norm(embedding_array) > 0


def test_real_detector_rejects_image_without_face(client):
    blank_image = np.full(
        (600, 600, 3),
        255,
        dtype=np.uint8,
    )

    encoded_successfully, encoded_image = cv2.imencode(
        ".jpg",
        blank_image,
    )

    assert encoded_successfully is True

    response = client.post(
        "/faces/detect",
        files={
            "image": (
                "blank.jpg",
                encoded_image.tobytes(),
                "image/jpeg",
            )
        },
        data={"attributes": "DEFAULT"},
    )

    assert response.status_code == 400

    detail = response.json()["detail"]

    assert detail["success"] is False
    assert (
        detail["error_code"]
        == "InvalidParameterException"
    )


def test_real_index_and_search_pipeline(
    client,
    face_images,
):
    create_response = client.post(
        "/collections",
        json={"collectionId": COLLECTION_ID},
    )

    assert create_response.status_code == 200

    index_response = client.post(
        f"/collections/{COLLECTION_ID}/faces",
        files=image_upload(
            face_images["person_a_enroll"]
        ),
        data={"externalImageId": "person-a"},
    )

    assert index_response.status_code == 200

    index_body = index_response.json()

    assert len(index_body["faceRecords"]) == 1

    indexed_face_id = (
        index_body["faceRecords"][0]["face"]["faceId"]
    )

    exact_search_response = client.post(
        f"/collections/{COLLECTION_ID}/search",
        files=image_upload(
            face_images["person_a_enroll"]
        ),
        data={
            "faceMatchThreshold": "99",
            "maxFaces": "1",
        },
    )

    assert exact_search_response.status_code == 200

    exact_matches = exact_search_response.json()[
        "faceMatches"
    ]

    assert len(exact_matches) == 1
    assert (
        exact_matches[0]["face"]["faceId"]
        == indexed_face_id
    )
    assert exact_matches[0]["similarity"] >= 99.0

    same_person_response = client.post(
        f"/collections/{COLLECTION_ID}/search",
        files=image_upload(
            face_images["person_a_query"]
        ),
        data={
            "faceMatchThreshold": "0",
            "maxFaces": "1",
        },
    )

    assert same_person_response.status_code == 200

    same_person_matches = same_person_response.json()[
        "faceMatches"
    ]

    assert len(same_person_matches) == 1
    assert (
        same_person_matches[0]["face"]["faceId"]
        == indexed_face_id
    )

    same_person_similarity = same_person_matches[0][
        "similarity"
    ]

    different_person_response = client.post(
        f"/collections/{COLLECTION_ID}/search",
        files=image_upload(
            face_images["person_b_query"]
        ),
        data={
            "faceMatchThreshold": "0",
            "maxFaces": "1",
        },
    )

    assert different_person_response.status_code == 200

    different_person_matches = (
        different_person_response.json()[
            "faceMatches"
        ]
    )

    if different_person_matches:
        different_person_similarity = (
            different_person_matches[0]["similarity"]
        )
    else:
        different_person_similarity = 0.0

    assert (
        same_person_similarity
        > different_person_similarity
    )

    strict_different_person_response = client.post(
        f"/collections/{COLLECTION_ID}/search",
        files=image_upload(
            face_images["person_b_query"]
        ),
        data={
            "faceMatchThreshold": "99",
            "maxFaces": "1",
        },
    )

    assert strict_different_person_response.status_code == 200
    assert (
        strict_different_person_response.json()[
            "faceMatches"
        ]
        == []
    )

    describe_response = client.get(
        f"/collections/{COLLECTION_ID}"
    )

    assert describe_response.status_code == 200
    assert describe_response.json()["faceCount"] == 1
