import json
import re
from sqlalchemy import text, bindparam
from app.db import SessionLocal
from uuid import UUID

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
            return {
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
                VALUES (
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

        db.commit()

    return {
        "success": True,
        "statusCode": 200,
        "collectionArn": collection_arn,
        "faceModelVersion": FACE_MODEL_VERSION
    }


def describe_collection(collection_id: str):
    with SessionLocal() as db:
        result = db.execute(
            text("""
                SELECT
                    collection_arn,
                    face_count,
                    face_model_version,
                    creation_timestamp
                FROM face_collections
                WHERE collection_id = :collection_id
            """),
            {"collection_id": collection_id}
        )

        row = result.mappings().first()

        if not row:
            return {
                "success": False,
                "error_code": "ResourceNotFoundException",
                "message": "Collection does not exist"
            }

        return {
            "success": True,
            "collectionARN": row["collection_arn"],
            "faceCount": row["face_count"],
            "faceModelVersion": row["face_model_version"],
            "creationTimestamp": row["creation_timestamp"].isoformat()
        }


def list_collections(max_results: int = 1000, next_token: str | None = None):
    if max_results < 1:
        max_results = 1

    if max_results > 1000:
        max_results = 1000

    offset = 0

    if next_token:
        try:
            offset = int(next_token)
        except ValueError:
            offset = 0

    with SessionLocal() as db:
        result = db.execute(
            text("""
                SELECT
                    collection_id,
                    face_model_version
                FROM face_collections
                ORDER BY collection_id
                LIMIT :limit
                OFFSET :offset
            """),
            {
                "limit": max_results + 1,
                "offset": offset
            }
        )

        rows = result.mappings().all()

    has_more = len(rows) > max_results
    visible_rows = rows[:max_results]

    new_next_token = str(offset + max_results) if has_more else None

    return {
        "collectionIds": [row["collection_id"] for row in visible_rows],
        "nextToken": new_next_token,
        "faceModelVersions": [row["face_model_version"] for row in visible_rows]
    }


def delete_collection(collection_id: str):
    with SessionLocal() as db:
        result = db.execute(
            text("""
                DELETE FROM face_collections
                WHERE collection_id = :collection_id
            """),
            {"collection_id": collection_id}
        )

        db.commit()

        if result.rowcount == 0:
            return {
                "success": False,
                "statusCode": 404,
                "error_code": "ResourceNotFoundException",
                "message": "Collection does not exist"
            }

        return {
            "success": True,
            "statusCode": 200
        }
    
def collection_exists(collection_id: str) -> bool:
    with SessionLocal() as db:
        result = db.execute(
            text("""
                SELECT collection_id
                FROM face_collections
                WHERE collection_id = :collection_id
            """),
            {"collection_id": collection_id}
        )

        return result.first() is not None
    
def save_indexed_face(
        collection_id: str,
        face_id: str,
        image_id: str,
        external_image_id: str| None,
        embedding: list[float],
        confidence: float,
        bounding_box: dict,
        detector_model: str,
        recognizer_model: str,
        embedding_model_version: str,
        quality: dict
):
    embedding_str = "[" + ",".join(map(str, embedding)) + "]"
    with SessionLocal() as db:
        db.execute(
            text("""
                INSERT INTO face_embeddings(
                    face_id,
                    image_id,
                    collection_id,
                    external_image_id,
                    embedding,
                    confidence,
                    bounding_box,
                    detector_model,
                    recognizer_model,
                    embedding_model_version,
                    quality
                )
                VALUES(
                    CAST(:face_id AS uuid),
                    CAST(:image_id AS uuid),
                    :collection_id,
                    :external_image_id,
                    :embedding,
                    :confidence,
                    CAST(:bounding_box AS jsonb),
                    :detector_model,
                    :recognizer_model,
                    :embedding_model_version,
                    CAST(:quality AS jsonb)
                )
            """),
            {
                "face_id": face_id,
                "image_id": image_id,
                "collection_id": collection_id,
                "external_image_id": external_image_id,
                "embedding": embedding_str,
                "confidence": confidence,
                "bounding_box": json.dumps(bounding_box),
                "detector_model": detector_model,
                "recognizer_model": recognizer_model,
                "embedding_model_version": embedding_model_version,
                "quality": json.dumps(quality),
            }
        )

        db.execute(
            text("""
                UPDATE face_collections
                SET face_count = (
                    SELECT COUNT(*)
                    FROM face_embeddings
                    WHERE collection_id = :collection_id
                )
                WHERE collection_id = :collection_id
            """),
            {"collection_id": collection_id}
        )
        db.commit()

def list_faces(
    collection_id: str,
    max_results: int = 4096,
    next_token: str | None = None,
    face_ids: list[str] | None = None
):
    if not collection_exists(collection_id):
        return {
            "success": False,
            "error_code": "ResourceNotFoundException",
            "message": "Collection does not exist"
        }

    if max_results < 1:
        max_results = 1

    if max_results > 4096:
        max_results = 4096

    offset = 0

    if next_token:
        try:
            offset = int(next_token)
        except ValueError:
            offset = 0

    params = {
        "collection_id": collection_id,
        "limit": max_results + 1,
        "offset": offset
    }

    face_filter_sql = ""

    if face_ids:
        face_filter_sql = "AND CAST(face_id AS TEXT) IN :face_ids"
        params["face_ids"] = face_ids

    query = text(f"""
        SELECT
            face_id,
            image_id,
            external_image_id,
            confidence,
            bounding_box
        FROM face_embeddings
        WHERE collection_id = :collection_id
        {face_filter_sql}
        ORDER BY created_at
        LIMIT :limit
        OFFSET :offset
    """)

    if face_ids:
        query = query.bindparams(bindparam("face_ids", expanding=True))

    with SessionLocal() as db:
        result = db.execute(query, params)

        rows = result.mappings().all()

        version_result = db.execute(
            text("""
                SELECT face_model_version
                FROM face_collections
                WHERE collection_id = :collection_id
            """),
            {"collection_id": collection_id}
        )

        version_row = version_result.mappings().first()

    has_more = len(rows) > max_results
    visible_rows = rows[:max_results]

    return {
        "success": True,
        "faces": [
            {
                "faceId": str(row["face_id"]),
                "imageId": str(row["image_id"]),
                "externalImageId": row["external_image_id"],
                "confidence": row["confidence"],
                "boundingBox": row["bounding_box"]
            }
            for row in visible_rows
        ],
        "nextToken": str(offset + max_results) if has_more else None,
        "faceModelVersion": version_row["face_model_version"] if version_row else FACE_MODEL_VERSION
    }

def delete_faces(
    collection_id: str,
    face_ids: list[str]
):
    if not collection_exists(collection_id):
        return {
            "success": False,
            "error_code": "ResourceNotFoundException",
            "message": "Collection does not exist"
        }
    
    if not face_ids:
        return {
            "success": False,
            "error_code": "InvalidParameterException",
            "message": "No face IDs provided, or invalid UUID format"
        }
    
    if len(face_ids) > 4096:
        return {
            "success": False,
            "error_code": "InvalidParameterException",
            "message": "Maximum 4096 face IDs allowed per call"
        }
    
    try:
        validated_face_ids = [
            str(UUID(face_id))
            for face_id in face_ids
        ]
    except (ValueError, TypeError, AttributeError):
        return {
            "success": False,
            "error_code": "InvalidParameterException",
            "message": "Each faceId must be a valid UUID"
        }

    params = {
        "collection_id": collection_id,
        "face_ids": validated_face_ids
    }

    delete_query = text("""
        DELETE FROM face_embeddings
        WHERE collection_id = :collection_id
        AND CAST(face_id AS TEXT) IN :face_ids
        RETURNING CAST(face_id AS TEXT) AS face_id
    """).bindparams(bindparam("face_ids", expanding=True))

    with SessionLocal() as db:
        result = db.execute(delete_query, params)
        deleted_rows = result.mappings().all()

        db.execute(
            text("""
                UPDATE face_collections
                SET face_count = (
                    SELECT COUNT(*)
                    FROM face_embeddings
                    WHERE collection_id = :collection_id
                )
                WHERE collection_id = :collection_id
            """),
            {"collection_id": collection_id}
        )

        db.commit()

    return {
        "success": True,
        "deletedFaces": [row["face_id"] for row in deleted_rows]
    }

def search_faces_by_embedding(
    collection_id: str,
    query_embedding: list[float],
    face_match_threshold: float = 80.0,
    max_faces: int = 1
):
    if face_match_threshold < 0 or face_match_threshold > 100:
        return {
            "success": False,
            "error_code": "InvalidParameterException",
            "message": "faceMatchThreshold must be between 0 and 100"
        }
    
    if max_faces < 1 or max_faces > 4096:
        return {
            "success": False,
            "error_code": "InvalidParameterException",
            "message": "maxFaces must be between 1 and 4096"
        }
    
    embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

    with SessionLocal() as db:
        collection_row = db.execute(
            text("""
                SELECT face_model_version
                FROM face_collections
                WHERE collection_id = :collection_id
            """),
            {"collection_id": collection_id}
        ).mappings().first()

        if not collection_row:
            return {
                "success": False,
                "error_code": "ResourceNotFoundException",
                "message": "Collection does not exist"
            }
        
        result = db.execute(
            text("""
                SELECT
                    CAST(face_id AS TEXT) AS face_id,
                    CAST(image_id AS TEXT) AS image_id,
                    external_image_id,
                    confidence,
                    bounding_box,
                    (
                        1 - (
                            embedding <=>
                            CAST(:query_embedding AS vector)
                        )
                    ) * 100 AS similarity
                FROM face_embeddings
                WHERE collection_id = :collection_id
                  AND face_id IS NOT NULL
                  AND image_id IS NOT NULL
                  AND (
                        1 - (
                            embedding <=>
                            CAST(:query_embedding AS vector)
                        )
                      ) * 100 >= :face_match_threshold
                ORDER BY
                    embedding <=> CAST(:query_embedding AS vector)
                LIMIT :max_faces
            """),
            {
                "collection_id": collection_id,
                "query_embedding": embedding_str,
                "face_match_threshold": face_match_threshold,
                "max_faces": max_faces
            }
        )

        rows = result.mappings().all()

    face_matches = []

    for row in rows:
        similarity = float(row["similarity"])
        similarity = max(0.0, min(100.0, similarity))

        face_matches.append({
            "face": {
                "faceId": row["face_id"],
                "imageId": row["image_id"],
                "externalImageId": row["external_image_id"],
                "confidence": row["confidence"],
                "boundingBox": row["bounding_box"]
            },
            "similarity": round(similarity, 2)
        })

    return {
        "success": True,
        "faceMatches": face_matches,
        "faceModelVersion": collection_row["face_model_version"]
    }