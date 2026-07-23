from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import mediapipe as mp
import numpy as np


class ActiveLivenessEngine:
    """
    Extracts temporal liveness signals using MediaPipe.

    This class does not create sessions or make the final
    multi-frame decision. It analyzes one frame at a time.
    """

    model_version = "mediapipe_face_landmarker_v1"

    def __init__(
        self,
        model_path: str = (
            "models/liveness/face_landmarker.task"
        ),
        blink_closed_threshold: float = 0.55,
        centered_yaw_threshold: float = 8.0,
        turn_yaw_threshold: float = 15.0,
    ):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                "MediaPipe Face Landmarker model not found: "
                f"{self.model_path}"
            )

        if not 0.0 <= blink_closed_threshold <= 1.0:
            raise ValueError(
                "blink_closed_threshold must be between 0 and 1."
            )

        if centered_yaw_threshold < 0.0:
            raise ValueError(
                "centered_yaw_threshold cannot be negative."
            )

        if turn_yaw_threshold <= centered_yaw_threshold:
            raise ValueError(
                "turn_yaw_threshold must be greater than "
                "centered_yaw_threshold."
            )

        self.blink_closed_threshold = (
            blink_closed_threshold
        )
        self.centered_yaw_threshold = (
            centered_yaw_threshold
        )
        self.turn_yaw_threshold = turn_yaw_threshold

        options = (
            mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(
                        self.model_path
                    )
                ),
                running_mode=(
                    mp.tasks.vision.RunningMode.IMAGE
                ),
                # Allow detection of a second face so that
                # multi-face frames can be rejected.
                num_faces=2,
                min_face_detection_confidence=0.6,
                min_face_presence_confidence=0.6,
                min_tracking_confidence=0.6,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
        )

        self.landmarker = (
            mp.tasks.vision.FaceLandmarker
            .create_from_options(options)
        )

        # Protect the shared MediaPipe task when FastAPI
        # processes concurrent requests.
        self._inference_lock = Lock()

    def _extract_blendshapes(
        self,
        result: Any,
    ) -> dict[str, float]:
        if not result.face_blendshapes:
            raise RuntimeError(
                "MediaPipe did not return face blendshapes."
            )

        blendshapes: dict[str, float] = {}

        for category in result.face_blendshapes[0]:
            category_name = category.category_name

            if not category_name:
                continue

            blendshapes[category_name] = float(
                category.score
            )

        return blendshapes

    def _extract_head_pose(
        self,
        result: Any,
    ) -> tuple[float, float, float]:
        if not result.facial_transformation_matrixes:
            raise RuntimeError(
                "MediaPipe did not return a facial "
                "transformation matrix."
            )

        transformation = np.asarray(
            result.facial_transformation_matrixes[0],
            dtype=np.float64,
        )

        if transformation.shape != (4, 4):
            raise RuntimeError(
                "Unexpected facial transformation matrix "
                f"shape: {transformation.shape}"
            )

        rotation_with_scale = transformation[:3, :3]

        # Remove any scale component and recover the closest
        # proper rotation matrix.
        u, _, vt = np.linalg.svd(
            rotation_with_scale
        )

        rotation = u @ vt

        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt

        euler_angles = cv2.RQDecomp3x3(
            rotation
        )[0]

        pitch, yaw, roll = (
            float(angle)
            for angle in euler_angles
        )

        return pitch, yaw, roll

    def analyze_image(
        self,
        image: np.ndarray,
    ) -> dict[str, Any]:
        """
        Analyzes one BGR OpenCV image.

        A temporal service will later combine several of these
        frame-level results to verify a blink or head turn.
        """

        if image is None or image.size == 0:
            raise ValueError(
                "InvalidImageFormatException: "
                "The image could not be decoded."
            )

        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        rgb_image = np.ascontiguousarray(
            rgb_image
        )

        media_pipe_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image,
        )

        with self._inference_lock:
            result = self.landmarker.detect(
                media_pipe_image
            )

        face_count = len(result.face_landmarks)

        if face_count == 0:
            raise ValueError(
                "NO_FACE_DETECTED: "
                "No face landmarks were detected."
            )

        if face_count > 1:
            raise ValueError(
                "MULTIPLE_FACES_DETECTED: "
                "Multiple faces were detected."
            )

        blendshapes = self._extract_blendshapes(
            result
        )

        blink_left = blendshapes.get(
            "eyeBlinkLeft",
            0.0,
        )

        blink_right = blendshapes.get(
            "eyeBlinkRight",
            0.0,
        )

        # Both eyes must be closed to count as a blink frame.
        blink_score = min(
            blink_left,
            blink_right,
        )

        pitch, yaw, roll = (
            self._extract_head_pose(result)
        )

        if abs(yaw) <= self.centered_yaw_threshold:
            yaw_position = "CENTER"
        elif yaw <= -self.turn_yaw_threshold:
            yaw_position = "NEGATIVE"
        elif yaw >= self.turn_yaw_threshold:
            yaw_position = "POSITIVE"
        else:
            yaw_position = "TRANSITION"

        return {
            "status": "SUCCEEDED",
            "modelVersion": self.model_version,
            "faceCount": face_count,
            "blinkLeft": round(blink_left, 4),
            "blinkRight": round(blink_right, 4),
            "blinkScore": round(blink_score, 4),
            "eyesClosed": (
                blink_score
                >= self.blink_closed_threshold
            ),
            "pitchDegrees": round(pitch, 2),
            "yawDegrees": round(yaw, 2),
            "rollDegrees": round(roll, 2),
            "yawPosition": yaw_position,
        }

    def analyze_file(
        self,
        image_path: str,
    ) -> dict[str, Any]:
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(
                "InvalidImageFormatException: "
                "The image could not be decoded."
            )

        return self.analyze_image(image)

    def close(self) -> None:
        self.landmarker.close()