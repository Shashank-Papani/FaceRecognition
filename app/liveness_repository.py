import json
from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.orm import Session

class LivenessRepository:
    def create_session(self, db: Session) -> dict[str, Any]:
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
                created_at
            """
        )

        row = db.execute(
            query,
            {
                "session_id": session_id,
            },
        ).mappings().one()

        db.commit()

        return dict(row)
    
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