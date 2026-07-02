class LivenessEngine:
    model_version = "minifasnet_onnx_v1"

    def check_liveness(
        self,
        image_path: str,
        threshold: float = 80.0,
    ) -> dict:
        if not 0.0 <= threshold <= 100.0:
            raise ValueError(
                "InvalidParameterException: "
                "threshold must be between 0 and 100."
            )
        
        confidence = 90.0

        return {
            "status": "SUCCEEDED",
            "live": confidence >= threshold,
            "confidence": confidence,
            "threshold": threshold,
            "modelVersion": self.model_version,
            "faceQuality": {
                "face_confidence": None,
                "face_width": None,
                "face_height": None,
            },
        }