import json
from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.orm import Session


class LivenessRepository:
    def create_session(
        self,
        db: Session,
    ) -> dict[str, Any]:
        session_id = str(uuid4())

        query = text(
            """
            INSERT INTO liveness_sessions (
                session_id,
                status
            )
            VALUES (
                :session_id,
                'CREATED'
            )
            RETURNING
                session_id,
                status,
                challenge_type,
                challenge_direction,
                challenge_stage,
                challenge_progress,
                challenge_instruction,
                active_liveness_passed,
                challenge_started_at,
                challenge_expires_at,
                created_at
            """
        )

        try:
            row = db.execute(
                query,
                {
                    "session_id": session_id,
                },
            ).mappings().one()

            db.commit()

            return dict(row)

        except Exception:
            db.rollback()
            raise

    def get_session(
        self,
        db: Session,
        session_id: str,
    ) -> dict[str, Any] | None:
        query = text(
            """
            SELECT
                session_id,
                status,
                confidence,
                threshold,
                live,
                model_version,
                face_quality,
                challenge_type,
                challenge_direction,
                challenge_stage,
                challenge_progress,
                challenge_instruction,
                active_liveness_passed,
                challenge_started_at,
                challenge_expires_at,
                last_active_signal,
                completed_at,
                created_at,
                updated_at
            FROM liveness_sessions
            WHERE session_id = :session_id
            """
        )

        row = db.execute(
            query,
            {
                "session_id": session_id,
            },
        ).mappings().first()

        if row is None:
            return None

        return dict(row)

    def save_result(
        self,
        db: Session,
        session_id: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Retained for the existing passive MiniFASNet workflow.
        """

        query = text(
            """
            UPDATE liveness_sessions
            SET
                status = :status,
                confidence = :confidence,
                threshold = :threshold,
                live = :live,
                model_version = :model_version,
                face_quality = CAST(:face_quality AS jsonb),
                updated_at = :updated_at
            WHERE session_id = :session_id
            RETURNING
                session_id,
                status,
                confidence,
                threshold,
                live,
                model_version,
                face_quality,
                created_at,
                updated_at
            """
        )

        try:
            row = db.execute(
                query,
                {
                    "session_id": session_id,
                    "status": result["status"],
                    "confidence": result["confidence"],
                    "threshold": result["threshold"],
                    "live": result["live"],
                    "model_version": result["modelVersion"],
                    "face_quality": json.dumps(
                        result["faceQuality"]
                    ),
                    "updated_at": datetime.utcnow(),
                },
            ).mappings().one()

            db.commit()

            return dict(row)

        except Exception:
            db.rollback()
            raise

    def save_active_state(
        self,
        db: Session,
        session_id: str,
        *,
        status: str,
        threshold: float,
        live: bool | None,
        confidence: float | None,
        model_version: str,
        face_quality: dict[str, Any] | None,
        challenge_stage: str,
        challenge_progress: int,
        challenge_instruction: str,
        active_liveness_passed: bool | None,
        last_active_signal: dict[str, Any] | None,
        completed_at: datetime | None,
    ) -> dict[str, Any]:
        query = text(
            """
            UPDATE liveness_sessions
            SET
                status = :status,
                confidence = :confidence,
                threshold = :threshold,
                live = :live,
                model_version = :model_version,
                face_quality = CAST(:face_quality AS jsonb),
                challenge_stage = :challenge_stage,
                challenge_progress = :challenge_progress,
                challenge_instruction = :challenge_instruction,
                active_liveness_passed =
                    :active_liveness_passed,
                last_active_signal =
                    CAST(:last_active_signal AS jsonb),
                completed_at = :completed_at,
                updated_at = :updated_at
            WHERE session_id = :session_id
            RETURNING
                session_id,
                status,
                confidence,
                threshold,
                live,
                model_version,
                face_quality,
                challenge_type,
                challenge_direction,
                challenge_stage,
                challenge_progress,
                challenge_instruction,
                active_liveness_passed,
                challenge_started_at,
                challenge_expires_at,
                last_active_signal,
                completed_at,
                created_at,
                updated_at
            """
        )

        parameters = {
            "session_id": session_id,
            "status": status,
            "confidence": confidence,
            "threshold": threshold,
            "live": live,
            "model_version": model_version,
            "face_quality": (
                json.dumps(face_quality)
                if face_quality is not None
                else None
            ),
            "challenge_stage": challenge_stage,
            "challenge_progress": challenge_progress,
            "challenge_instruction": (
                challenge_instruction
            ),
            "active_liveness_passed": (
                active_liveness_passed
            ),
            "last_active_signal": (
                json.dumps(last_active_signal)
                if last_active_signal is not None
                else None
            ),
            "completed_at": completed_at,
            "updated_at": datetime.utcnow(),
        }

        try:
            row = db.execute(
                query,
                parameters,
            ).mappings().one()

            db.commit()

            return dict(row)

        except Exception:
            db.rollback()
            raise