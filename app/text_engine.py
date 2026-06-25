from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from paddleocr import PaddleOCR


MINIMUM_WIDTH = 800


class TextEngine:
    def __init__(self):
        self.ocr = PaddleOCR(
            lang="en",
            ocr_version="PP-OCRv5",
            device="cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            return_word_box=True,
        )

        # Avoid running the same PaddleOCR instance concurrently.
        self._ocr_lock = Lock()

    def preprocess_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(str(Path(image_path)))

        if image is None:
            raise ValueError(
                "InvalidImageFormatException: "
                "The uploaded image could not be decoded."
            )

        height, width = image.shape[:2]

        # Maintain aspect ratio and ensure a minimum width of 800 px.
        if width < MINIMUM_WIDTH:
            scale = MINIMUM_WIDTH / width

            resized_width = MINIMUM_WIDTH
            resized_height = max(1, round(height * scale))

            image = cv2.resize(
                image,
                (resized_width, resized_height),
                interpolation=cv2.INTER_CUBIC,
            )

        grayscale = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

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

        # Encode as PNG in memory to match the documented preprocessing.
        encoded_successfully, encoded_png = cv2.imencode(
            ".png",
            sharpened,
        )

        if not encoded_successfully:
            raise RuntimeError(
                "Failed to encode the preprocessed image as PNG."
            )

        # Decode into a 3-channel image PaddleOCR can process directly.
        processed_image = cv2.imdecode(
            encoded_png,
            cv2.IMREAD_COLOR,
        )

        if processed_image is None:
            raise RuntimeError(
                "Failed to decode the preprocessed PNG image."
            )

        return processed_image

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @classmethod
    def _geometry_from_polygon(
        cls,
        polygon,
        image_width: int,
        image_height: int,
    ) -> dict:
        points = np.asarray(
            polygon,
            dtype=np.float32,
        ).reshape(-1, 2)

        x_values = points[:, 0]
        y_values = points[:, 1]

        x_min = float(np.min(x_values))
        x_max = float(np.max(x_values))
        y_min = float(np.min(y_values))
        y_max = float(np.max(y_values))

        return {
            "boundingBox": {
                "width": cls._clamp(
                    (x_max - x_min) / image_width
                ),
                "height": cls._clamp(
                    (y_max - y_min) / image_height
                ),
                "left": cls._clamp(
                    x_min / image_width
                ),
                "top": cls._clamp(
                    y_min / image_height
                ),
            },
            "polygon": [
                {
                    "x": cls._clamp(
                        float(x) / image_width
                    ),
                    "y": cls._clamp(
                        float(y) / image_height
                    ),
                }
                for x, y in points
            ],
        }

    @classmethod
    def _geometry_from_box(
        cls,
        box,
        image_width: int,
        image_height: int,
    ) -> dict:
        x_min, y_min, x_max, y_max = [
            float(value)
            for value in box
        ]

        polygon = [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ]

        return cls._geometry_from_polygon(
            polygon=polygon,
            image_width=image_width,
            image_height=image_height,
        )

    @staticmethod
    def _merge_word_tokens(
        tokens,
        token_boxes,
    ) -> list[dict]:
        """
        Merge PaddleOCR tokens separated by whitespace.

        Example:
        ["MPH", " ", "km", "/", "h"]
        becomes:
        ["MPH", "km/h"]
        """
        merged_words = []

        current_text = ""
        current_boxes = []

        def flush_current_word():
            nonlocal current_text, current_boxes

            if not current_text or not current_boxes:
                current_text = ""
                current_boxes = []
                return

            boxes_array = np.asarray(
                current_boxes,
                dtype=np.float32,
            ).reshape(-1, 4)

            merged_box = [
                float(np.min(boxes_array[:, 0])),
                float(np.min(boxes_array[:, 1])),
                float(np.max(boxes_array[:, 2])),
                float(np.max(boxes_array[:, 3])),
            ]

            merged_words.append(
                {
                    "text": current_text,
                    "box": merged_box,
                }
            )

            current_text = ""
            current_boxes = []

        for token, box in zip(tokens, token_boxes):
            token = str(token)

            if token.isspace():
                flush_current_word()
                continue

            current_text += token
            current_boxes.append(box)

        flush_current_word()

        return merged_words

    def detect_text(
        self,
        image_path: str,
        min_confidence: float = 0.0,
        regions_of_interest: list[dict] | None = None,
    ) -> dict:
        if not 0.0 <= min_confidence <= 100.0:
            raise ValueError(
                "InvalidParameterException: "
                "minConfidence must be between "
                "0 and 100."
            )

        regions = (
            self._validate_regions_of_interest(
                regions_of_interest
            )
        )

        processed_image = self.preprocess_image(
            image_path
        )

        image_height, image_width = (
            processed_image.shape[:2]
        )

        with self._ocr_lock:
            results = self.ocr.predict(
                processed_image
            )

        text_detections = []
        next_id = 0

        for result in results:
            payload = result.json

            result_data = payload.get(
                "res",
                payload,
            )

            texts = result_data.get(
                "rec_texts",
                [],
            )

            scores = result_data.get(
                "rec_scores",
                [],
            )

            polygons = result_data.get(
                "rec_polys",
                [],
            )

            boxes = result_data.get(
                "rec_boxes",
                [],
            )

            word_tokens = result_data.get(
                "text_word",
                [],
            )

            word_boxes = result_data.get(
                "text_word_boxes",
                [],
            )

            for index, detected_text in enumerate(
                texts
            ):
                detected_text = str(
                    detected_text
                ).strip()

                if not detected_text:
                    continue

                score = (
                    float(scores[index]) * 100.0
                    if index < len(scores)
                    else 0.0
                )

                if score < min_confidence:
                    continue

                if index < len(polygons):
                    line_geometry = (
                        self._geometry_from_polygon(
                            polygon=polygons[index],
                            image_width=image_width,
                            image_height=image_height,
                        )
                    )
                elif index < len(boxes):
                    line_geometry = (
                        self._geometry_from_box(
                            box=boxes[index],
                            image_width=image_width,
                            image_height=image_height,
                        )
                    )
                else:
                    continue

                if not self._geometry_matches_regions(
                    geometry=line_geometry,
                    regions=regions,
                ):
                    continue

                line_id = next_id
                next_id += 1

                text_detections.append(
                    {
                        "id": line_id,
                        "type": "LINE",
                        "detectedText": detected_text,
                        "confidence": round(
                            score,
                            2,
                        ),
                        "geometry": line_geometry,
                    }
                )

                if (
                    index >= len(word_tokens)
                    or index >= len(word_boxes)
                ):
                    continue

                words = self._merge_word_tokens(
                    tokens=word_tokens[index],
                    token_boxes=word_boxes[index],
                )

                for word in words:
                    word_id = next_id
                    next_id += 1

                    word_geometry = (
                        self._geometry_from_box(
                            box=word["box"],
                            image_width=image_width,
                            image_height=image_height,
                        )
                    )

                    text_detections.append(
                        {
                            "id": word_id,
                            "parentId": line_id,
                            "type": "WORD",
                            "detectedText": word[
                                "text"
                            ],
                            "confidence": round(
                                score,
                                2,
                            ),
                            "geometry": (
                                word_geometry
                            ),
                        }
                    )

        return {
            "textDetections": text_detections
        }

    @staticmethod
    def _validate_regions_of_interest(
        regions_of_interest: list[dict] | None,
    ) -> list[dict]:
        if regions_of_interest is None:
            return []

        if not isinstance(regions_of_interest, list):
            raise ValueError(
                "InvalidParameterException: "
                "regionsOfInterest must be a list."
            )

        validated_regions = []

        for index, region in enumerate(regions_of_interest):
            if not isinstance(region, dict):
                raise ValueError(
                    "InvalidParameterException: "
                    f"regionsOfInterest[{index}] must be an object."
                )

            bounding_box = region.get("boundingBox")

            if not isinstance(bounding_box, dict):
                raise ValueError(
                    "InvalidParameterException: "
                    f"regionsOfInterest[{index}].boundingBox "
                    "must be an object."
                )

            required_fields = {
                "left",
                "top",
                "width",
                "height",
            }

            if not required_fields.issubset(bounding_box):
                raise ValueError(
                    "InvalidParameterException: "
                    f"regionsOfInterest[{index}].boundingBox "
                    "must contain left, top, width, and height."
                )

            try:
                left = float(bounding_box["left"])
                top = float(bounding_box["top"])
                width = float(bounding_box["width"])
                height = float(bounding_box["height"])
            except (TypeError, ValueError):
                raise ValueError(
                    "InvalidParameterException: "
                    "Region bounding-box values must be numbers."
                )

            if not 0.0 <= left <= 1.0:
                raise ValueError(
                    "InvalidParameterException: "
                    "Region left must be between 0 and 1."
                )

            if not 0.0 <= top <= 1.0:
                raise ValueError(
                    "InvalidParameterException: "
                    "Region top must be between 0 and 1."
                )

            if not 0.0 < width <= 1.0:
                raise ValueError(
                    "InvalidParameterException: "
                    "Region width must be greater than 0 "
                    "and no greater than 1."
                )

            if not 0.0 < height <= 1.0:
                raise ValueError(
                    "InvalidParameterException: "
                    "Region height must be greater than 0 "
                    "and no greater than 1."
                )

            if left + width > 1.0:
                raise ValueError(
                    "InvalidParameterException: "
                    "Region left + width cannot exceed 1."
                )

            if top + height > 1.0:
                raise ValueError(
                    "InvalidParameterException: "
                    "Region top + height cannot exceed 1."
                )

            validated_regions.append(
                {
                    "left": left,
                    "top": top,
                    "width": width,
                    "height": height,
                }
            )

        return validated_regions

    @staticmethod
    def _geometry_matches_regions(
        geometry: dict,
        regions: list[dict],
    ) -> bool:
        if not regions:
            return True

        bounding_box = geometry["boundingBox"]

        center_x = (
            bounding_box["left"]
            + bounding_box["width"] / 2
        )

        center_y = (
            bounding_box["top"]
            + bounding_box["height"] / 2
        )

        for region in regions:
            right = region["left"] + region["width"]
            bottom = region["top"] + region["height"]

            inside_horizontal = (
                region["left"] <= center_x <= right
            )

            inside_vertical = (
                region["top"] <= center_y <= bottom
            )

            if inside_horizontal and inside_vertical:
                return True

        return False
