import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from app.liveness_face_adapter import (
    LivenessFaceAdapter,
)


class LivenessEngine:
    model_version = "minifasnet_onnx_v1"

    def __init__(
        self,
        model_path: str = (
            "models/liveness/minifasnet.onnx"
        ),
        face_adapter: LivenessFaceAdapter | None = None,
    ):
        """
        Loads MiniFASNet and receives the face-preparation adapter.

        It no longer creates its own YuNet detector.
        """

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Liveness model not found: "
                f"{self.model_path}"
            )

        self.face_adapter = (
            face_adapter
            if face_adapter is not None
            else LivenessFaceAdapter()
        )

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )

        self.input_name = (
            self.session.get_inputs()[0].name
        )

        self.input_shape = (
            self.session.get_inputs()[0].shape
        )

        self.output_name = (
            self.session.get_outputs()[0].name
        )

        self.live_class_index = int(
            os.getenv(
                "LIVENESS_LIVE_CLASS_INDEX",
                "1",
            )
        )

        self.color_mode = os.getenv(
            "LIVENESS_COLOR_MODE",
            "BGR",
        ).upper()

        self.normalize_input = (
            os.getenv(
                "LIVENESS_NORMALIZE_INPUT",
                "true",
            ).lower()
            == "true"
        )

    def _get_model_input_size(
        self,
    ) -> tuple[int, int]:
        """
        Reads the model tensor dimensions.

        Your model input is:
        ['batch', 3, 80, 80]

        ONNX NCHW order:
        batch, channels, height, width
        """

        shape = self.input_shape

        if len(shape) != 4:
            return 80, 80

        input_height = shape[2]
        input_width = shape[3]

        if (
            isinstance(input_height, int)
            and isinstance(input_width, int)
        ):
            return input_width, input_height

        return 80, 80

    def _preprocess_face(
        self,
        face_crop: np.ndarray,
    ) -> np.ndarray:
        """
        Converts the face crop into MiniFASNet's expected input.

        1. Resize to 80 × 80.
        2. Optionally convert BGR to RGB.
        3. Convert to float32.
        4. Optionally normalize from 0–255 to 0–1.
        5. Convert HWC to NCHW.
        6. Add batch dimension.
        """

        input_width, input_height = (
            self._get_model_input_size()
        )

        resized = cv2.resize(
            face_crop,
            (input_width, input_height),
            interpolation=cv2.INTER_LINEAR,
        )

        if self.color_mode == "RGB":
            resized = cv2.cvtColor(
                resized,
                cv2.COLOR_BGR2RGB,
            )
        elif self.color_mode != "BGR":
            raise ValueError(
                "InvalidParameterException: "
                "LIVENESS_COLOR_MODE must be BGR or RGB."
            )

        tensor = resized.astype(np.float32)

        if self.normalize_input:
            tensor = tensor / 255.0

        tensor = np.transpose(
            tensor,
            (2, 0, 1),
        )

        tensor = np.expand_dims(
            tensor,
            axis=0,
        )

        return np.ascontiguousarray(
            tensor,
            dtype=np.float32,
        )

    def _softmax(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        """
        Converts raw logits into probabilities.
        """

        values = values.astype(np.float32)

        values = values - np.max(values)

        exp_values = np.exp(values)

        denominator = np.sum(exp_values)

        if denominator <= 0.0:
            raise RuntimeError(
                "MiniFASNet returned invalid output."
            )

        return exp_values / denominator

    def _to_probabilities(
        self,
        output: np.ndarray,
    ) -> np.ndarray:
        """
        Handles models that return either:

        - raw logits, or
        - already-normalized probabilities.
        """

        values = np.asarray(
            output,
            dtype=np.float32,
        )

        values = np.squeeze(values)

        if values.ndim != 1:
            raise RuntimeError(
                "Unexpected MiniFASNet output shape: "
                f"{values.shape}"
            )

        if values.size != 3:
            raise RuntimeError(
                "Expected three MiniFASNet classes, "
                f"received {values.size}."
            )

        already_probabilities = (
            np.all(values >= 0.0)
            and np.all(values <= 1.0)
            and np.isclose(
                np.sum(values),
                1.0,
                atol=1e-3,
            )
        )

        if already_probabilities:
            return values

        return self._softmax(values)

    def check_liveness(
        self,
        image_path: str,
        threshold: float = 80.0,
    ) -> dict[str, Any]:
        """
        Runs the complete passive liveness flow.

        Face preparation is delegated to LivenessFaceAdapter.
        MiniFASNet inference remains inside this class.
        """

        if not 0.0 <= threshold <= 100.0:
            raise ValueError(
                "InvalidParameterException: "
                "threshold must be between 0 and 100."
            )

        prepared_face = (
            self.face_adapter.prepare_face(
                image_path=image_path
            )
        )

        model_input = self._preprocess_face(
            prepared_face.face_crop
        )

        output = self.session.run(
            [self.output_name],
            {
                self.input_name: model_input,
            },
        )[0]

        probabilities = self._to_probabilities(
            output
        )

        if not (
            0
            <= self.live_class_index
            < probabilities.size
        ):
            raise RuntimeError(
                "LIVENESS_LIVE_CLASS_INDEX is outside "
                "the model output range."
            )

        live_probability = float(
            probabilities[
                self.live_class_index
            ]
        )

        confidence = live_probability * 100.0

        return {
            "status": "SUCCEEDED",
            "live": confidence >= threshold,
            "confidence": round(
                confidence,
                2,
            ),
            "threshold": threshold,
            "modelVersion": self.model_version,
            "faceQuality": (
                prepared_face.face_quality
            ),
        }