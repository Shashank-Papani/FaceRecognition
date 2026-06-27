from pathlib import Path

import pytest

from app.text_engine import TextEngine


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ODOMETER_IMAGE = (
    PROJECT_ROOT
    / "tests"
    / "private_assets"
    /"text"
    / "odometer.jpg"
)


def test_paddleocr_detects_odometer_reading():
    if not ODOMETER_IMAGE.exists():
        pytest.skip(
            "Private odometer test image is unavailable."
        )

    engine = TextEngine()

    result = engine.detect_text(
        image_path=str(ODOMETER_IMAGE),
        min_confidence=50.0,
        regions_of_interest=[
            {
                "boundingBox": {
                    "left": 0.30,
                    "top": 0.65,
                    "width": 0.35,
                    "height": 0.20,
                }
            }
        ],
    )

    detections = result["textDetections"]

    line_detections = [
        detection
        for detection in detections
        if detection["type"] == "LINE"
    ]

    word_detections = [
        detection
        for detection in detections
        if detection["type"] == "WORD"
    ]

    normalized_lines = {
        detection["detectedText"].replace(" ", "")
        for detection in line_detections
    }

    assert "091308" in normalized_lines

    odometer_line = next(
        detection
        for detection in line_detections
        if (
            detection["detectedText"]
            .replace(" ", "")
            == "091308"
        )
    )

    assert 50.0 <= odometer_line["confidence"] <= 100.0

    bounding_box = odometer_line[
        "geometry"
    ]["boundingBox"]

    assert 0.0 <= bounding_box["left"] <= 1.0
    assert 0.0 <= bounding_box["top"] <= 1.0
    assert 0.0 < bounding_box["width"] <= 1.0
    assert 0.0 < bounding_box["height"] <= 1.0

    child_words = [
        detection
        for detection in word_detections
        if detection.get("parentId")
        == odometer_line["id"]
    ]

    assert child_words

    combined_words = "".join(
        detection["detectedText"].replace(" ", "")
        for detection in child_words
    )

    assert combined_words == "091308"