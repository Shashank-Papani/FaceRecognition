import json
from sqlalchemy import text
from app.db import SessionLocal

def database_health_check():
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))

        return {
            "connected": True,
            "status": "success",
            "error": None
        }
    
    except Exception as e:
        return {
            "connected": False,
            "status": "failed",
            "error": str(e)
        }
    
def create_person_if_not_exists(person_id: str):
    with SessionLocal() as db:
        db.execute(
            text("""
                INSERT INTO people (person_id)
                VALUES (:person_id)
                ON CONFLICT (person_id) DO NOTHING
            """),
            {"person_id": person_id}
        )
        db.commit()


def save_face_embedding(
    person_id: str,
    embedding: list[float],
    detector_model: str,
    recognizer_model: str,
    embedding_model_version: str,
    quality: dict
):
    create_person_if_not_exists(person_id)

    embedding_str = "[" + ",".join(map(str, embedding)) + "]"

    with SessionLocal() as db:
        db.execute(
            text("""
                INSERT INTO face_embeddings (
                    person_id,
                    embedding,
                    detector_model,
                    recognizer_model,
                    embedding_model_version,
                    quality
                )
                VALUES (
                    :person_id,
                    :embedding,
                    :detector_model,
                    :recognizer_model,
                    :embedding_model_version,
                    CAST(:quality AS jsonb)
                )
            """),
            {
                "person_id": person_id,
                "embedding": embedding_str,
                "detector_model": detector_model,
                "recognizer_model": recognizer_model,
                "embedding_model_version": embedding_model_version,
                "quality": json.dumps(quality),
            }
        )
        db.commit()


def get_all_embeddings():
    with SessionLocal() as db:
        result = db.execute(
            text("""
                SELECT
                    person_id,
                    embedding::text AS embedding,
                    quality,
                    created_at
                FROM face_embeddings
            """)
        )

        return result.mappings().all()


def list_people():
    with SessionLocal() as db:
        result = db.execute(
            text("""
                SELECT
                    p.person_id,
                    COUNT(fe.id) AS embedding_count,
                    MAX(fe.embedding_model_version) AS model_version
                FROM people p
                LEFT JOIN face_embeddings fe
                    ON p.person_id = fe.person_id
                GROUP BY p.person_id
                ORDER BY p.person_id
            """)
        )

        return result.mappings().all()


def delete_person(person_id: str):
    with SessionLocal() as db:
        result = db.execute(
            text("""
                DELETE FROM people
                WHERE person_id = :person_id
            """),
            {"person_id": person_id}
        )
        db.commit()

        return result.rowcount > 0


def log_verification(
    matched_person_id: str | None,
    similarity: float | None,
    threshold: float,
    verified: bool,
    quality: dict
):
    with SessionLocal() as db:
        db.execute(
            text("""
                INSERT INTO verification_logs (
                    matched_person_id,
                    similarity,
                    threshold,
                    verified,
                    quality
                )
                VALUES (
                    :matched_person_id,
                    :similarity,
                    :threshold,
                    :verified,
                    CAST(:quality AS jsonb)
                )
            """),
            {
                "matched_person_id": matched_person_id,
                "similarity": similarity,
                "threshold": threshold,
                "verified": verified,
                "quality": json.dumps(quality),
            }
        )
        db.commit()


def find_best_match(query_embedding: list[float]):
    embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

    with SessionLocal() as db:
        result = db.execute(
            text("""
                SELECT
                    id,
                    person_id,
                    1 - (embedding <=> CAST(:query_embedding AS vector)) AS similarity,
                    quality,
                    created_at
                FROM face_embeddings
                ORDER BY embedding <=> CAST(:query_embedding AS vector)
                LIMIT 1
            """),
            {
                "query_embedding": embedding_str
            }
        )

        return result.mappings().first()