import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.auth import verify_api_key
from app.main import app, get_text_engine
from app.text_engine import TextEngine

class FakeCorruptedImageEngine:
    def detect_text(
        self,
        image_path: str,
        min_confidence: float = 0.0,
        regions_of_interest: list[dict] | None = None,
        min_bounding_box_width: float = 0.0,
        min_bounding_box_height: float = 0.0,
    ) -> dict:
        raise ValueError(
            "InvalidImageFormatException: "
            "The uploaded image could not be decoded."
        )

class FakeTextEngine:
    def __init__(self):
        self.calls = []
        self.file_existed_during_call = False

    def detect_text(
        self,
        image_path: str,
        min_confidence: float = 0.0,
        regions_of_interest: list[dict] | None = None,
        min_bounding_box_width: float = 0.0,
        min_bounding_box_height: float = 0.0,
    ) -> dict:
        image = Path(image_path)

        self.file_existed_during_call = image.exists()

        self.calls.append(
            {
                "image_path": image_path,
                "min_confidence": min_confidence,
                "regions_of_interest": regions_of_interest,
                "min_bounding_box_width": min_bounding_box_width,
                "min_bounding_box_height": min_bounding_box_height,
            }
        )

        return {
            "textDetections": [
                {
                    "id": 0,
                    "type": "LINE",
                    "detectedText": "091308",
                    "confidence": 99.58,
                    "geometry": {
                        "boundingBox": {
                            "width": 0.22875,
                            "height": 0.08126,
                            "left": 0.34875,
                            "top": 0.70812,
                        },
                        "polygon": [],
                    },
                },
                {
                    "id": 1,
                    "parentId": 0,
                    "type": "WORD",
                    "detectedText": "091308",
                    "confidence": 99.58,
                    "geometry": {
                        "boundingBox": {
                            "width": 0.19375,
                            "height": 0.08126,
                            "left": 0.36875,
                            "top": 0.70812,
                        },
                        "polygon": [],
                    },
                },
            ]
        }


@pytest.fixture
def client_and_engine():
    fake_engine = FakeTextEngine()

    app.dependency_overrides[
        get_text_engine
    ] = lambda: fake_engine

    app.dependency_overrides[
        verify_api_key
    ] = lambda: True

    with TestClient(app) as client:
        yield client, fake_engine

    app.dependency_overrides.pop(
        get_text_engine,
        None,
    )

    app.dependency_overrides.pop(
        verify_api_key,
        None,
    )


def upload_image():
    return {
        "image": (
            "odometer.jpg",
            b"fake-image-content",
            "image/jpeg",
        )
    }


def test_detect_text_returns_ocr_response(
    client_and_engine,
):
    client, fake_engine = client_and_engine

    response = client.post(
        "/text/detect",
        files=upload_image(),
    )

    assert response.status_code == 200

    body = response.json()

    assert len(body["textDetections"]) == 2
    assert body["textDetections"][0]["type"] == "LINE"
    assert (
        body["textDetections"][0]["detectedText"]
        == "091308"
    )

    assert body["textDetections"][1]["type"] == "WORD"
    assert body["textDetections"][1]["parentId"] == 0

    assert len(fake_engine.calls) == 1
    assert fake_engine.file_existed_during_call is True


def test_detect_text_uses_default_filters(
    client_and_engine,
):
    client, fake_engine = client_and_engine

    response = client.post(
        "/text/detect",
        files=upload_image(),
    )

    assert response.status_code == 200

    call = fake_engine.calls[0]

    assert call["min_confidence"] == 0.0
    assert call["regions_of_interest"] == []
    assert call["min_bounding_box_width"] == 0.0
    assert call["min_bounding_box_height"] == 0.0


def test_detect_text_passes_confidence_and_region(
    client_and_engine,
):
    client, fake_engine = client_and_engine

    region = {
        "boundingBox": {
            "left": 0.30,
            "top": 0.65,
            "width": 0.35,
            "height": 0.20,
        }
    }

    filters = {
        "minConfidence": 50,
        "regionsOfInterest": [region],
        "minBoundingBoxWidth": 0.02,
        "minBoundingBoxHeight": 0.03,
    }

    response = client.post(
        "/text/detect",
        files=upload_image(),
        data={
            "filters": json.dumps(filters)
        },
    )

    assert response.status_code == 200

    call = fake_engine.calls[0]

    assert call["min_confidence"] == 50.0
    assert call["regions_of_interest"] == [region]
    assert call["min_bounding_box_width"] == 0.02
    assert call["min_bounding_box_height"] == 0.03


def test_detect_text_deletes_temporary_upload(
    client_and_engine,
):
    client, fake_engine = client_and_engine

    response = client.post(
        "/text/detect",
        files=upload_image(),
    )

    assert response.status_code == 200

    uploaded_path = Path(
        fake_engine.calls[0]["image_path"]
    )

    assert fake_engine.file_existed_during_call is True
    assert uploaded_path.exists() is False


def test_detect_text_rejects_invalid_json(
    client_and_engine,
):
    client, fake_engine = client_and_engine

    response = client.post(
        "/text/detect",
        files=upload_image(),
        data={
            "filters": "{invalid-json}"
        },
    )

    assert response.status_code == 400

    detail = response.json()["detail"]

    assert (
        detail["error_code"]
        == "InvalidParameterException"
    )

    assert detail["message"] == (
        "filters must be valid JSON."
    )

    assert fake_engine.calls == []


def test_detect_text_rejects_non_object_filters(
    client_and_engine,
):
    client, fake_engine = client_and_engine

    response = client.post(
        "/text/detect",
        files=upload_image(),
        data={
            "filters": json.dumps(
                ["not", "an", "object"]
            )
        },
    )

    assert response.status_code == 400

    detail = response.json()["detail"]

    assert detail["message"] == (
        "filters must be a JSON object."
    )

    assert fake_engine.calls == []


def test_detect_text_rejects_non_numeric_confidence(
    client_and_engine,
):
    client, fake_engine = client_and_engine

    response = client.post(
        "/text/detect",
        files=upload_image(),
        data={
            "filters": json.dumps(
                {
                    "minConfidence": "high"
                }
            )
        },
    )

    assert response.status_code == 400

    detail = response.json()["detail"]

    assert detail["message"] == (
        "minConfidence must be a number."
    )

    assert fake_engine.calls == []


def test_region_validation_rejects_region_outside_image():
    regions = [
        {
            "boundingBox": {
                "left": 0.80,
                "top": 0.20,
                "width": 0.30,
                "height": 0.20,
            }
        }
    ]

    with pytest.raises(
        ValueError,
        match=r"left \+ width cannot exceed 1",
    ):
        TextEngine._validate_regions_of_interest(
            regions
        )


def test_geometry_matches_region_using_center_point():
    geometry = {
        "boundingBox": {
            "left": 0.35,
            "top": 0.70,
            "width": 0.20,
            "height": 0.08,
        }
    }

    matching_region = [
        {
            "left": 0.30,
            "top": 0.65,
            "width": 0.35,
            "height": 0.20,
        }
    ]

    non_matching_region = [
        {
            "left": 0.00,
            "top": 0.00,
            "width": 0.20,
            "height": 0.20,
        }
    ]

    assert TextEngine._geometry_matches_regions(
        geometry,
        matching_region,
    ) is True

    assert TextEngine._geometry_matches_regions(
        geometry,
        non_matching_region,
    ) is False

def test_detect_text_rejects_image_larger_than_15_mb(
    client_and_engine,
):
    client, fake_engine = client_and_engine

    oversized_image = (
        b"x" * (15 * 1024 * 1024 + 1)
    )

    response = client.post(
        "/text/detect",
        files={
            "image": (
                "large-image.jpg",
                oversized_image,
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 413

    detail = response.json()["detail"]

    assert detail["error_code"] == (
        "IMAGE_TOO_LARGE"
    )

    assert detail["message"] == (
        "Image size cannot exceed 15 MB."
    )

    assert fake_engine.calls == []

def test_detect_text_rejects_unsupported_image_extension(
    client_and_engine,
):
    client, fake_engine = client_and_engine

    response = client.post(
        "/text/detect",
        files={
            "image": (
                "document.pdf",
                b"fake-pdf-content",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 415

    detail = response.json()["detail"]

    assert detail["error_code"] == (
        "InvalidImageFormatException"
    )

    assert detail["message"] == (
        "Only JPG, JPEG, and PNG images are supported."
    )

    assert fake_engine.calls == []

def test_detect_text_maps_corrupted_image_to_415():
    fake_engine = FakeCorruptedImageEngine()

    app.dependency_overrides[
        get_text_engine
    ] = lambda: fake_engine

    app.dependency_overrides[
        verify_api_key
    ] = lambda: True

    try:
        with TestClient(app) as client:
            response = client.post(
                "/text/detect",
                files={
                    "image": (
                        "corrupted.jpg",
                        b"not-a-real-image",
                        "image/jpeg",
                    )
                },
            )

        assert response.status_code == 415

        detail = response.json()["detail"]

        assert detail["error_code"] == (
            "InvalidImageFormatException"
        )

        assert detail["message"] == (
            "The uploaded image could not be decoded."
        )

    finally:
        app.dependency_overrides.pop(
            get_text_engine,
            None,
        )

        app.dependency_overrides.pop(
            verify_api_key,
            None,
        )

def test_detect_text_rejects_non_numeric_min_bounding_box_width(
    client_and_engine,
):
    client, fake_engine = client_and_engine

    response = client.post(
        "/text/detect",
        files=upload_image(),
        data={
            "filters": json.dumps(
                {
                    "minBoundingBoxWidth": "wide"
                }
            )
        },
    )

    assert response.status_code == 400

    detail = response.json()["detail"]

    assert detail["error_code"] == (
        "InvalidParameterException"
    )

    assert detail["message"] == (
        "minBoundingBoxWidth must be a number."
    )

    assert fake_engine.calls == []


def test_detect_text_rejects_non_numeric_min_bounding_box_height(
    client_and_engine,
):
    client, fake_engine = client_and_engine

    response = client.post(
        "/text/detect",
        files=upload_image(),
        data={
            "filters": json.dumps(
                {
                    "minBoundingBoxHeight": "tall"
                }
            )
        },
    )

    assert response.status_code == 400

    detail = response.json()["detail"]

    assert detail["error_code"] == (
        "InvalidParameterException"
    )

    assert detail["message"] == (
        "minBoundingBoxHeight must be a number."
    )

    assert fake_engine.calls == []