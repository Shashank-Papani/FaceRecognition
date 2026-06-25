import json
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_IMAGE = PROJECT_ROOT / "test_images" / "odometer.jpg"
PROCESSED_IMAGE = (
    PROJECT_ROOT
    / "test_images"
    / "odometer_preprocessed.png"
)

MINIMUM_WIDTH = 800


def preprocess_image(image_path: Path) -> Path:
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    height, width = image.shape[:2]

    # Maintain aspect ratio and ensure width is at least 800 px.
    if width < MINIMUM_WIDTH:
        scale = MINIMUM_WIDTH / width

        resized_width = MINIMUM_WIDTH
        resized_height = int(height * scale)

        image = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_CUBIC,
        )

    # Convert to grayscale.
    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    # Apply the documented 3x3 sharpen kernel.
    sharpen_kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )

    sharpened = cv2.filter2D(
        grayscale,
        ddepth=-1,
        kernel=sharpen_kernel,
    )

    # Save as PNG to match the documented preprocessing.
    saved = cv2.imwrite(
        str(PROCESSED_IMAGE),
        sharpened,
    )

    if not saved:
        raise RuntimeError(
            "Failed to save preprocessed PNG image."
        )

    return PROCESSED_IMAGE


def make_json_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    raise TypeError(
        f"Unsupported JSON value: {type(value)}"
    )


def main() -> None:
    if not INPUT_IMAGE.exists():
        raise FileNotFoundError(
            f"Missing test image: {INPUT_IMAGE}"
        )

    processed_path = preprocess_image(INPUT_IMAGE)

    print(f"Original image:     {INPUT_IMAGE}")
    print(f"Preprocessed image: {processed_path}")

    ocr = PaddleOCR(
        lang="en",
        ocr_version="PP-OCRv5",
        device="cpu",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        return_word_box=True,
    )

    results = ocr.predict(str(processed_path))

    result_count = 0

    for result_count, result in enumerate(
        results,
        start=1,
    ):
        payload = result.json

        print("\nFULL PADDLEOCR RESULT")
        print("---------------------")

        print(
            json.dumps(
                payload,
                indent=2,
                default=make_json_serializable,
            )
        )

        result_data = payload.get("res", payload)

        print("\nAVAILABLE RESULT KEYS")
        print("---------------------")

        for key in sorted(result_data.keys()):
            print(key)

        print("\nRECOGNIZED LINES")
        print("---------------------")

        texts = result_data.get("rec_texts", [])
        scores = result_data.get("rec_scores", [])

        for index, text in enumerate(texts):
            score = (
                float(scores[index])
                if index < len(scores)
                else 0.0
            )

            print(
                f"{index}: "
                f"{text!r} "
                f"confidence={score * 100:.2f}"
            )

        print("\nWORD-RELATED FIELDS")
        print("---------------------")

        word_keys = [
            key
            for key in result_data.keys()
            if "word" in key.lower()
        ]

        if not word_keys:
            print(
                "No word-related keys were returned."
            )
        else:
            for key in word_keys:
                print(f"{key}: {result_data[key]}")

    if result_count == 0:
        print("PaddleOCR returned no result objects.")


if __name__ == "__main__":
    main()