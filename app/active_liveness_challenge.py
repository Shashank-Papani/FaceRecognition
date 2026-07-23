from time import monotonic
from typing import Any


class ActiveLivenessChallenge:
    """
    Verifies a temporal liveness sequence:

    1. Eyes open
    2. Eyes closed
    3. Eyes open again
    4. Face centered
    5. Head turned
    6. Face centered again
    """

    def __init__(
        self,
        timeout_seconds: float = 20.0,
        eyes_open_threshold: float = 0.25,
        eyes_closed_threshold: float = 0.55,
        center_yaw_threshold: float = 8.0,
        turn_yaw_threshold: float = 15.0,
    ):
        if timeout_seconds <= 0.0:
            raise ValueError(
                "timeout_seconds must be greater than zero."
            )

        if not (
            0.0
            <= eyes_open_threshold
            < eyes_closed_threshold
            <= 1.0
        ):
            raise ValueError(
                "Eye thresholds must satisfy: "
                "0 <= open < closed <= 1."
            )

        if turn_yaw_threshold <= center_yaw_threshold:
            raise ValueError(
                "turn_yaw_threshold must be greater than "
                "center_yaw_threshold."
            )

        self.timeout_seconds = timeout_seconds
        self.eyes_open_threshold = eyes_open_threshold
        self.eyes_closed_threshold = eyes_closed_threshold
        self.center_yaw_threshold = center_yaw_threshold
        self.turn_yaw_threshold = turn_yaw_threshold

        self.started_at = monotonic()
        self.stage = "WAITING_FOR_OPEN_EYES"

    def _instruction(self) -> str:
        instructions = {
            "WAITING_FOR_OPEN_EYES": (
                "Look at the camera with your eyes open"
            ),
            "WAITING_FOR_CLOSED_EYES": (
                "Blink now"
            ),
            "WAITING_FOR_REOPENED_EYES": (
                "Open your eyes"
            ),
            "WAITING_FOR_CENTER": (
                "Look directly at the camera"
            ),
            "WAITING_FOR_HEAD_TURN": (
                "Turn your head to either side"
            ),
            "WAITING_FOR_RETURN_TO_CENTER": (
                "Look directly at the camera again"
            ),
            "PASSED": (
                "Liveness challenge passed"
            ),
            "FAILED": (
                "Liveness challenge timed out"
            ),
        }

        return instructions[self.stage]

    def _progress(self) -> int:
        progress = {
            "WAITING_FOR_OPEN_EYES": 0,
            "WAITING_FOR_CLOSED_EYES": 15,
            "WAITING_FOR_REOPENED_EYES": 35,
            "WAITING_FOR_CENTER": 50,
            "WAITING_FOR_HEAD_TURN": 65,
            "WAITING_FOR_RETURN_TO_CENTER": 85,
            "PASSED": 100,
            "FAILED": 0,
        }

        return progress[self.stage]

    def _response(
        self,
        signal: dict[str, Any],
    ) -> dict[str, Any]:
        elapsed_seconds = monotonic() - self.started_at

        return {
            "status": self.stage,
            "passed": self.stage == "PASSED",
            "failed": self.stage == "FAILED",
            "instruction": self._instruction(),
            "progress": self._progress(),
            "elapsedSeconds": round(
                elapsed_seconds,
                2,
            ),
            "blinkScore": signal.get(
                "blinkScore"
            ),
            "yawDegrees": signal.get(
                "yawDegrees"
            ),
        }

    def update(
        self,
        signal: dict[str, Any],
    ) -> dict[str, Any]:
        if self.stage in {"PASSED", "FAILED"}:
            return self._response(signal)

        elapsed_seconds = monotonic() - self.started_at

        if elapsed_seconds > self.timeout_seconds:
            self.stage = "FAILED"
            return self._response(signal)

        blink_score = float(
            signal["blinkScore"]
        )

        yaw_degrees = float(
            signal["yawDegrees"]
        )

        eyes_open = (
            blink_score
            <= self.eyes_open_threshold
        )

        eyes_closed = (
            blink_score
            >= self.eyes_closed_threshold
        )

        face_centered = (
            abs(yaw_degrees)
            <= self.center_yaw_threshold
        )

        head_turned = (
            abs(yaw_degrees)
            >= self.turn_yaw_threshold
        )

        if self.stage == "WAITING_FOR_OPEN_EYES":
            if eyes_open:
                self.stage = (
                    "WAITING_FOR_CLOSED_EYES"
                )

        elif self.stage == "WAITING_FOR_CLOSED_EYES":
            if eyes_closed:
                self.stage = (
                    "WAITING_FOR_REOPENED_EYES"
                )

        elif self.stage == "WAITING_FOR_REOPENED_EYES":
            if eyes_open:
                self.stage = "WAITING_FOR_CENTER"

        elif self.stage == "WAITING_FOR_CENTER":
            if face_centered:
                self.stage = (
                    "WAITING_FOR_HEAD_TURN"
                )

        elif self.stage == "WAITING_FOR_HEAD_TURN":
            if head_turned:
                self.stage = (
                    "WAITING_FOR_RETURN_TO_CENTER"
                )

        elif self.stage == "WAITING_FOR_RETURN_TO_CENTER":
            if face_centered:
                self.stage = "PASSED"

        return self._response(signal)