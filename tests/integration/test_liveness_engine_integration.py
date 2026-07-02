from pathlib import Path

import pytest

from app.liveness_engine import LivenessEngine


MODEL_PATH = Path("models/liveness/minifasnet.onnx")
LIVE_IMAGE = Path(
    "tests/private_assets/liveness/live.jpg"
)


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="MiniFASNet ONNX model is not available.",
)
@pytest.mark.skipif(
    not LIVE_IMAGE.exists(),
    reason="Private liveness test image is not available.",
)
def test_liveness_engine_returns_real_score():
    engine = LivenessEngine(
        model_path=str(MODEL_PATH)
    )

    result = engine.check_liveness(
        image_path=str(LIVE_IMAGE),
        threshold=80.0,
    )

    assert result["status"] == "SUCCEEDED"
    assert isinstance(result["live"], bool)
    assert 0.0 <= result["confidence"] <= 100.0
    assert result["threshold"] == 80.0
    assert result["modelVersion"] == "minifasnet_onnx_v1"
    assert "faceQuality" in result