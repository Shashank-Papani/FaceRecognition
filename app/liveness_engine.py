import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

class LivenessEngine:
    model_version = "minifasnet_onnx_v1"

    def __init__(
        self,
        model_path: str = (
            "models/liveness/minifasnet.onnx"
        ),
        face_detector_path: str = (
            "models/face_detection_yunet_2026may.onnx"
        ),
    ):
        self.model_path = Path(model_path)
        self.face_detector_path = Path(face_detector_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Liveness model not found: {self.model_path}"
            )

        if not self.face_detector_path.exists():
            raise FileNotFoundError(
                "YuNet face detector model not found: "
                f"{self.face_detector_path}"
            )

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name

        self.live_class_index = int(
            os.getenv("LIVENESS_LIVE_CLASS_INDEX", "0")
        )

        self.face_detector = cv2.FaceDetectorYN.create(
            str(self.face_detector_path),
            "",
            (320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000,
        )

    def _load_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(
                "InvalidImageFormatException: "
                "The uploaded image could not be decoded."
            )

        return image
    
    def _detect_single_face(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        height, width = image.shape[:2]

        self.face_detector.setInputSize(
            (width, height)
        )

        _, faces = self.face_detector.detect(image)

        if faces is None or len(faces) == 0:
            raise ValueError(
                "NO_FACE_DETECTED: "
                "No face detected. Please upload a clear "
                "front-facing image."
            )

        if len(faces) > 1:
            raise ValueError(
                "MULTIPLE_FACES_DETECTED: "
                "Multiple faces detected. Please upload an image "
                "with exactly one face."
            )

        face = faces[0]

        x, y, face_width, face_height = face[:4]
        confidence = float(face[-1])

        if confidence < 0.6:
            raise ValueError(
                "LOW_QUALITY_FACE: "
                "Face confidence is too low for liveness detection."
            )

        if min(face_width, face_height) < 80:
            raise ValueError(
                "LOW_QUALITY_FACE: "
                "Face is too small for liveness detection."
            )

        face_quality = {
            "face_confidence": confidence,
            "face_confidence_percent": confidence * 100,
            "face_width": float(face_width),
            "face_height": float(face_height),
            "bounding_box": {
                "Width": float(face_width / width),
                "Height": float(face_height / height),
                "Left": float(x / width),
                "Top": float(y / height),
            },
        }

        return face, face_quality
    
    def _crop_face(
        self,
        image: np.ndarray,
        face: np.ndarray,
        margin_ratio: float = 0.4,
    ) -> np.ndarray:
        image_height, image_width = image.shape[:2]

        x, y, face_width, face_height = face[:4]

        margin_x = face_width * margin_ratio / 2
        margin_y = face_height * margin_ratio / 2

        left = int(max(0, x - margin_x))
        top = int(max(0, y - margin_y))
        right = int(
            min(
                image_width,
                x + face_width + margin_x,
            )
        )
        bottom = int(
            min(
                image_height,
                y + face_height + margin_y,
            )
        )

        face_crop = image[top:bottom, left:right]

        if face_crop.size == 0:
            raise ValueError(
                "LOW_QUALITY_FACE: "
                "Face crop could not be generated."
            )

        return face_crop

    def _get_model_input_size(self) -> tuple[int, int]:
        shape = self.input_shape

        if len(shape) != 4:
            return 80, 80

        if isinstance(shape[2], int) and isinstance(shape[3], int):
            return shape[3], shape[2]

        return 80, 80

    def _preprocess_face(
        self,
        face_crop: np.ndarray,
    ) -> np.ndarray:
        input_width, input_height = self._get_model_input_size()

        resized = cv2.resize(
            face_crop,
            (input_width, input_height),
        )

        tensor = resized.astype(np.float32) / 255.0

        tensor = np.transpose(
            tensor,
            (2, 0, 1),
        )

        tensor = np.expand_dims(
            tensor,
            axis=0,
        )

        return tensor
    
    def _softmax(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        values = values.astype(np.float32)
        values = values - np.max(values)

        exp_values = np.exp(values)

        return exp_values / np.sum(exp_values)
    
    def check_liveness(
        self,
        image_path: str,
        threshold: float = 80.0,
    ) -> dict[str, Any]:
        if not 0.0 <= threshold <= 100.0:
            raise ValueError(
                "InvalidParameterException: "
                "threshold must be between 0 and 100."
            )

        image = self._load_image(image_path)

        face, face_quality = self._detect_single_face(image)

        face_crop = self._crop_face(
            image=image,
            face=face,
        )

        model_input = self._preprocess_face(
            face_crop
        )

        output = self.session.run(
            [self.output_name],
            {
                self.input_name: model_input,
            },
        )[0]

        output = np.squeeze(output)

        probabilities = self._softmax(output)

        live_probability = float(
            probabilities[self.live_class_index]
        )

        confidence = live_probability * 100.0

        return {
            "status": "SUCCEEDED",
            "live": confidence >= threshold,
            "confidence": round(confidence, 2),
            "threshold": threshold,
            "modelVersion": self.model_version,
            "faceQuality": face_quality,
        }