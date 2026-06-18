import json
import re
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
    

COLLECTION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,255}$")
FACE_MODEL_VERSION = "sface_v1"

def is_valid_collection_id(collection_id: str) -> bool:
    return bool(COLLECTION_ID_PATTERN.fullmatch(collection_id))

def build_collection_arn(collection_id: str) -> str:
    return f"arn:local:face-recognition:collection/{collection_id}"

def create_collection(collection_id: str):
    if not is_valid_collection_id(collection_id):
        return {
            "success": False,
            "error_code": "InvalidParameterException",
            "message": "collectionId contains invalid characters. Allowed: letters, digits, underscores, hyphens. Max 255 chars."
        }
    collection_arn = build_collection_arn(collection_id)

    with SessionLocal() as db:
        existing = db.execute(
            text("""
                SELECT collection_id
                FROM face_collections
                WHERE collection_id = :collection_id     
            """),
            {"collection_id": collection_id}
        ).first()
        
        if existing:
            return{
                "success": False,
                "error_code": "ResourceAlreadyExistsException",
                "message": "Collection with this ID already exists"
            }
        
        db.execute(
            text("""
                INSERT INTO face_collections (
                    collection_id,
                    collection_arn,
                    face_model_version
                )
                VALUES(
                    :collection_id,
                    :collection_arn,
                    :face_model_version
                )
        """),
        {
            "collection_id": collection_id,
            "collection_arn": collection_arn,
            "face_model_version": FACE_MODEL_VERSION,
        }
        )
        db.commit

        return {
            "success": True,
            "statusCode": 200,
            "collectionArn": collection_arn,
            "faceModelVersion": FACE_MODEL_VERSION
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