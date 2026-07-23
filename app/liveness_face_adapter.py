from copy import deepcopy
from dataclasses import dataclass
from threading import Lock
from typing import Any

import cv2
import numpy as np

from app.face_engine import FaceEngine


@dataclass(slots=True)
class PreparedLivenessFace:
    """
    Contains everything MiniFASNet needs after face detection.
    """

    face_crop: np.ndarray
    face_quality: dict[str, Any]


class LivenessFaceAdapter:
    """
    Reuses FaceEngine's established YuNet detection and quality
    validation without modifying FaceEngine or existing APIs.
    """

    def __init__(
        self,
        face_engine: FaceEngine | None = None,
        crop_scale: float = 2.7,
    ):
        if crop_scale <= 0.0:
            raise ValueError(
                "crop_scale must be greater than zero."
            )

        self.face_engine = (
            face_engine
            if face_engine is not None
            else FaceEngine()
        )

        self.crop_scale = crop_scale
        self._detection_lock = Lock()

    def _load_image(
        self,
        image_path: str,
    ) -> np.ndarray:
        """
        Loads the original uploaded image.

        FaceEngine.detect_single_face expects an OpenCV image array,
        rather than a path.
        """

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(
                "InvalidImageFormatException: "
                "The uploaded image could not be decoded."
            )

        return image

    def _translate_detection_error(
        self,
        error: ValueError,
    ) -> ValueError:
        """
        Converts FaceEngine's existing messages into the structured
        liveness error codes expected by the liveness API.
        """

        message = str(error)

        if (
            message.startswith("No face detected")
            or message.startswith("No valid face detected")
        ):
            return ValueError(
                f"NO_FACE_DETECTED: {message}"
            )

        if message.startswith("Multiple faces detected"):
            return ValueError(
                f"MULTIPLE_FACES_DETECTED: {message}"
            )

        if message.startswith(
            "Detected face bounding box is invalid"
        ):
            return ValueError(
                f"LOW_QUALITY_FACE: {message}"
            )

        return ValueError(
            f"LOW_QUALITY_FACE: {message}"
        )

    def _get_scaled_crop_box(
        self,
        image: np.ndarray,
        face: np.ndarray,
    ) -> tuple[int, int, int, int]:
        image_height, image_width = image.shape[:2]

        x, y, face_width, face_height = [
            float(value)
            for value in face[:4]
        ]

        if face_width <= 0.0 or face_height <= 0.0:
            raise ValueError(
                "LOW_QUALITY_FACE: "
                "The detected face dimensions are invalid."
            )

        effective_scale = min(
            (image_height - 1) / face_height,
            (image_width - 1) / face_width,
            self.crop_scale,
        )

        crop_width = face_width * effective_scale
        crop_height = face_height * effective_scale

        center_x = x + face_width / 2.0
        center_y = y + face_height / 2.0

        left = center_x - crop_width / 2.0
        top = center_y - crop_height / 2.0
        right = center_x + crop_width / 2.0
        bottom = center_y + crop_height / 2.0

        if left < 0.0:
            right -= left
            left = 0.0

        if top < 0.0:
            bottom -= top
            top = 0.0

        if right > image_width - 1:
            left -= right - image_width + 1
            right = image_width - 1

        if bottom > image_height - 1:
            top -= bottom - image_height + 1
            bottom = image_height - 1

        left = max(0, int(left))
        top = max(0, int(top))
        right = min(image_width - 1, int(right))
        bottom = min(image_height - 1, int(bottom))

        if right <= left or bottom <= top:
            raise ValueError(
                "LOW_QUALITY_FACE: "
                "The scaled face crop is invalid."
            )

        return left, top, right, bottom


    def _crop_face(
        self,
        image: np.ndarray,
        face: np.ndarray,
    ) -> np.ndarray:
        """
        Creates the scaled face crop expected by the
        MiniFASNet 2.7_80x80 model.
        """

        left, top, right, bottom = (
            self._get_scaled_crop_box(
                image=image,
                face=face,
            )
        )

        face_crop = image[
            top:bottom + 1,
            left:right + 1,
        ]

        if face_crop.size == 0:
            raise ValueError(
                "LOW_QUALITY_FACE: "
                "The detected face crop is empty."
            )

        return face_crop

    def prepare_face(
        self,
        image_path: str,
    ) -> PreparedLivenessFace:
        """
        Main adapter function.

        1. Loads the uploaded image.
        2. Calls the existing FaceEngine detection pipeline.
        3. Copies FaceEngine's quality metadata safely.
        4. Crops the face for MiniFASNet.
        """

        image = self._load_image(image_path)

        with self._detection_lock:
            try:
                face = self.face_engine.detect_single_face(
                    image
                )
            except ValueError as error:
                raise self._translate_detection_error(
                    error
                ) from error

            face_quality = deepcopy(
                self.face_engine.last_face_quality
            )

        face_crop = self._crop_face(
            image=image,
            face=face,
        )

        return PreparedLivenessFace(
            face_crop=face_crop,
            face_quality=face_quality,
        )