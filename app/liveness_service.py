from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from app.active_liveness_engine import (
    ActiveLivenessEngine,
)
from app.liveness_face_adapter import (
    LivenessFaceAdapter,
)
from app.liveness_repository import (
    LivenessRepository,
)


class LivenessService:
    INSTRUCTIONS = {
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
            "Liveness challenge failed"
        ),
    }

    PROGRESS = {
        "WAITING_FOR_OPEN_EYES": 0,
        "WAITING_FOR_CLOSED_EYES": 15,
        "WAITING_FOR_REOPENED_EYES": 35,
        "WAITING_FOR_CENTER": 50,
        "WAITING_FOR_HEAD_TURN": 65,
        "WAITING_FOR_RETURN_TO_CENTER": 85,
        "PASSED": 100,
        "FAILED": 0,
    }

    TERMINAL_STATUSES = {
        "SUCCEEDED",
        "FAILED",
        "EXPIRED",
    }

    def __init__(
        self,
        repository: LivenessRepository | None = None,
        active_engine: ActiveLivenessEngine | None = None,
        face_adapter: LivenessFaceAdapter | None = None,
        eyes_open_threshold: float = 0.25,
        eyes_closed_threshold: float = 0.55,
        center_yaw_threshold: float = 8.0,
        turn_yaw_threshold: float = 15.0,
    ):
        self.repository = (
            repository
            if repository is not None
            else LivenessRepository()
        )

        self.active_engine = (
            active_engine
            if active_engine is not None
            else ActiveLivenessEngine()
        )

        self.face_adapter = (
            face_adapter
            if face_adapter is not None
            else LivenessFaceAdapter()
        )

        self.eyes_open_threshold = (
            eyes_open_threshold
        )

        self.eyes_closed_threshold = (
            eyes_closed_threshold
        )

        self.center_yaw_threshold = (
            center_yaw_threshold
        )

        self.turn_yaw_threshold = (
            turn_yaw_threshold
        )

    def _instruction(
        self,
        stage: str,
    ) -> str:
        return self.INSTRUCTIONS[stage]

    def _progress(
        self,
        stage: str,
    ) -> int:
        return self.PROGRESS[stage]

    def _advance_stage(
        self,
        current_stage: str,
        signal: dict[str, Any],
        direction: str,
    ) -> str:
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

        if direction == "NEGATIVE":
            head_turned = (
                yaw_degrees
                <= -self.turn_yaw_threshold
            )

        elif direction == "POSITIVE":
            head_turned = (
                yaw_degrees
                >= self.turn_yaw_threshold
            )

        else:
            head_turned = (
                abs(yaw_degrees)
                >= self.turn_yaw_threshold
            )

        if current_stage == "WAITING_FOR_OPEN_EYES":
            if eyes_open:
                return "WAITING_FOR_CLOSED_EYES"

        elif current_stage == "WAITING_FOR_CLOSED_EYES":
            if eyes_closed:
                return "WAITING_FOR_REOPENED_EYES"

        elif current_stage == "WAITING_FOR_REOPENED_EYES":
            if eyes_open:
                return "WAITING_FOR_CENTER"

        elif current_stage == "WAITING_FOR_CENTER":
            if face_centered:
                return "WAITING_FOR_HEAD_TURN"

        elif current_stage == "WAITING_FOR_HEAD_TURN":
            if head_turned:
                return "WAITING_FOR_RETURN_TO_CENTER"

        elif current_stage == (
            "WAITING_FOR_RETURN_TO_CENTER"
        ):
            if face_centered:
                return "PASSED"

        return current_stage

    def _expire_session(
        self,
        db: Session,
        session: dict[str, Any],
        threshold: float,
    ) -> dict[str, Any]:
        return self.repository.save_active_state(
            db=db,
            session_id=session["session_id"],
            status="EXPIRED",
            threshold=threshold,
            live=False,
            confidence=0.0,
            model_version=(
                self.active_engine.model_version
            ),
            face_quality=session.get("face_quality"),
            challenge_stage="FAILED",
            challenge_progress=0,
            challenge_instruction=(
                "Liveness challenge expired"
            ),
            active_liveness_passed=False,
            last_active_signal=(
                session.get("last_active_signal")
            ),
            completed_at=datetime.utcnow(),
        )

    def process_frame(
        self,
        db: Session,
        session_id: str,
        image_path: str,
        threshold: float = 80.0,
    ) -> dict[str, Any]:
        session = self.repository.get_session(
            db=db,
            session_id=session_id,
        )

        if session is None:
            raise ValueError(
                "ResourceNotFoundException: "
                "Liveness session not found."
            )

        if session["status"] in self.TERMINAL_STATUSES:
            raise ValueError(
                "SESSION_NOT_ACTIVE: "
                "The liveness session has already ended."
            )

        expires_at = session[
            "challenge_expires_at"
        ]

        if (
            expires_at is not None
            and datetime.utcnow() >= expires_at
        ):
            return self._expire_session(
                db=db,
                session=session,
                threshold=threshold,
            )

        # Reuse the established FaceEngine detector and
        # quality validation.
        prepared_face = (
            self.face_adapter.prepare_face(
                image_path=image_path
            )
        )

        # MediaPipe needs the full frame for head pose and
        # blink measurements.
        signal = self.active_engine.analyze_file(
            image_path=image_path
        )

        current_stage = session[
            "challenge_stage"
        ]

        next_stage = self._advance_stage(
            current_stage=current_stage,
            signal=signal,
            direction=session[
                "challenge_direction"
            ],
        )

        passed = next_stage == "PASSED"

        status = (
            "SUCCEEDED"
            if passed
            else "IN_PROGRESS"
        )

        return self.repository.save_active_state(
            db=db,
            session_id=session_id,
            status=status,
            threshold=threshold,
            live=True if passed else None,
            confidence=100.0 if passed else None,
            model_version=(
                self.active_engine.model_version
            ),
            face_quality=(
                prepared_face.face_quality
            ),
            challenge_stage=next_stage,
            challenge_progress=self._progress(
                next_stage
            ),
            challenge_instruction=self._instruction(
                next_stage
            ),
            active_liveness_passed=(
                True if passed else None
            ),
            last_active_signal=signal,
            completed_at=(
                datetime.utcnow()
                if passed
                else None
            ),
        )