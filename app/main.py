import time
import logging
import shutil
import json
from pathlib import Path
from pydantic import BaseModel
from app.face_engine import FaceEngine
from app.auth import verify_api_key
from uuid import uuid4
from app.errors import raise_api_error
from app import face_repository as repo
from app.text_engine import TextEngine
from functools import lru_cache

from fastapi import (
    FastAPI,
    Request,
    UploadFile,
    File,
    Form,
    HTTPException,
    Depends,
    Query,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Face Recognition API")

class CreateCollectionRequest(BaseModel):
    collectionId: str

class DeleteFacesRequest(BaseModel):
    faceIds: list[str]

@app.middleware("http")
async def log_requests(request: Request, call_next):

    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    request.state.request_id = request_id
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.exception(
            "request_id=%s method=%s path=%s latency_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            latency_ms,
        )
        raise

    latency_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "request_id=%s method=%s path=%s status=%s latency_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-ms"] = f"{latency_ms:.2f}"

    return response

engine = FaceEngine()

@lru_cache(maxsize=1) 
def get_text_engine() -> TextEngine: 
    return TextEngine()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def save_upload_file(image: UploadFile) -> Path:
    filename = image.filename or ""
    file_extension = Path(filename).suffix.lower()

    if file_extension not in {".jpg", ".jpeg", ".png"}:
        raise ValueError(
            "InvalidImageFormatException: "
            "Only JPG, JPEG, and PNG images are supported."
        )

    image_path = UPLOAD_DIR / f"{uuid4()}{file_extension}"

    image.file.seek(0)

    with image_path.open("wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    if image_path.stat().st_size == 0:
        image_path.unlink(missing_ok=True)

        raise ValueError(
            "InvalidParameterException: Uploaded image is empty."
        )

    return image_path

@app.get("/")
def home():
    return {
        "status": "running",
        "message": "Face Recognition API is working"
    }

@app.get("/health")
def health_check():
    db_status = engine.database_health_check()

    return {
        "api": "healthy" if db_status["connected"] else "unhealthy",
        "database": db_status
    }

@app.get("/model-info")
def model_info():
    return engine.model_info()

@app.post("/faces/detect")
def detect_faces(
    image: UploadFile = File(...),
    attributes: str = Form("DEFAULT"),
    authenticated: bool = Depends(verify_api_key)
):
    image_path = None

    try:
        image_path = save_upload_file(image)

        return engine.detect_faces_info(
            image_path=str(image_path),
            attributes=attributes
        )

    except ValueError as e:
        message = str(e)

        if message.startswith("InvalidImageFormatException"):
            error_code = "InvalidImageFormatException"
        else:
            error_code = "InvalidParameterException"

        clean_message = (
            message.split(":", 1)[1].strip()
            if ":" in message
            else message
        )

        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error_code": error_code,
                "message": clean_message
            }
        )

    except HTTPException:
        raise

    except Exception as e:
        raise_api_error(e)

    finally:
        if image_path and image_path.exists():
            image_path.unlink()
            
@app.post("/text/detect")
def detect_text(
    image: UploadFile = File(...),
    filters: str | None = Form(None),
    authenticated: bool = Depends(verify_api_key),
    ocr_engine: TextEngine = Depends(get_text_engine),
):
    image_path = None

    try:
        min_confidence = 0.0
        regions_of_interest = []

        if filters:
            try:
                filter_data = json.loads(
                    filters
                )
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "success": False,
                        "error_code": (
                            "InvalidParameterException"
                        ),
                        "message": (
                            "filters must be valid JSON."
                        ),
                    },
                )

            if not isinstance(
                filter_data,
                dict,
            ):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "success": False,
                        "error_code": (
                            "InvalidParameterException"
                        ),
                        "message": (
                            "filters must be "
                            "a JSON object."
                        ),
                    },
                )

            min_confidence = filter_data.get(
                "minConfidence",
                0.0,
            )

            regions_of_interest = (
                filter_data.get(
                    "regionsOfInterest",
                    [],
                )
            )

            try:
                min_confidence = float(
                    min_confidence
                )
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "success": False,
                        "error_code": (
                            "InvalidParameterException"
                        ),
                        "message": (
                            "minConfidence "
                            "must be a number."
                        ),
                    },
                )

        image_path = save_upload_file(
            image
        )

        return ocr_engine.detect_text(
            image_path=str(image_path),
            min_confidence=min_confidence,
            regions_of_interest=regions_of_interest,
        )

    except HTTPException:
        raise

    except ValueError as error:
        message = str(error)

        if message.startswith(
            "InvalidImageFormatException"
        ):
            error_code = (
                "InvalidImageFormatException"
            )
        else:
            error_code = (
                "InvalidParameterException"
            )

        clean_message = (
            message.split(":", 1)[1].strip()
            if ":" in message
            else message
        )

        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error_code": error_code,
                "message": clean_message,
            },
        )

    except Exception as error:
        raise_api_error(error)

    finally:
        if (
            image_path
            and image_path.exists()
        ):
            image_path.unlink()

@app.post("/collections")
def create_collection(
    request: CreateCollectionRequest,
    authenticated: bool = Depends(verify_api_key)
):
    result = repo.create_collection(request.collectionId)

    if not result.get("success"):
        status_code = 409 if result["error_code"] == "ResourceAlreadyExistsException" else 400

        raise HTTPException(
            status_code = status_code,
            detail = {
                "success": False,
                "error_code": result["error_code"],
                "message": result["message"]
            }
        )
    
    return {
        "statusCode": result["statusCode"],
        "collectionArn": result["collectionArn"],
        "faceModelVersion": result["faceModelVersion"]
    }



@app.get("/collections")
def list_collections(
    maxResults: int = Query(1000, ge=1, le=1000),
    nextToken: str | None = Query(None),
    authenticated: bool = Depends(verify_api_key)
):
    return repo.list_collections(
        max_results=maxResults,
        next_token=nextToken
    )

@app.get("/collections/{collection_id}")
def describe_collection(
    collection_id: str,
    authenticated: bool = Depends(verify_api_key)
):
    result = repo.describe_collection(collection_id)

    if not result.get("success"):
        raise HTTPException(
            status_code = 404,
            detail ={
                "success": False,
                "error_code": result["error_code"],
                "message": result["message"]
            }
        )
    
    return {
        "collectionARN": result["collectionARN"],
        "faceCount": result["faceCount"],
        "faceModelVersion": result["faceModelVersion"],
        "creationTimestamp": result["creationTimestamp"]
    }

@app.delete("/collections/{collection_id}")
def delete_collection(
    collection_id: str,
    authenticated: bool = Depends(verify_api_key)
):
    result = repo.delete_collection(collection_id)

    if not result.get("success"):
        raise HTTPException(
            status_code=404,
            detail={
                "success": False,
                "error_code": result["error_code"],
                "message": result["message"]
            }
        )

    return {
        "statusCode": result["statusCode"]
    }

@app.post("/collections/{collection_id}/faces")
def index_faces(
    collection_id: str,
    image: UploadFile = File(...),
    externalImageId: str | None = Form(None),
    authenticated: bool = Depends(verify_api_key)
):
    image_path = None
    try:
        image_path = save_upload_file(image)

        result = engine.index_face(
            collection_id = collection_id,
            image_path = str(image_path),
            external_image_id = externalImageId
        )

        if not result.get("success"):
            raise HTTPException(
                status_code = 404,
                detail = {
                    "success": False,
                    "error_code": result["error_code"],
                    "message": result["message"]
                }
            )
        
        return {
            "faceRecords": result["faceRecords"],
            "faceModelVersion": result["faceModelVersion"]
        }
    
    except HTTPException:
        raise

    except Exception as e:
        raise_api_error(e)

    finally:
        if image_path and image_path.exists():
            image_path.unlink()

@app.get("/collections/{collection_id}/faces")
def list_faces(
    collection_id: str,
    maxResults: int = Query(4096, ge=1, le=4096),
    nextToken: str | None = Query(None),
    faceIds: list[str] | None = Query(None),
    authenticated: bool = Depends(verify_api_key)
):
    result = repo.list_faces(
        collection_id=collection_id,
        max_results=maxResults,
        next_token=nextToken,
        face_ids=faceIds
    )

    if not result.get("success"):
        raise HTTPException(
            status_code=404,
            detail={
                "success": False,
                "error_code": result["error_code"],
                "message": result["message"]
            }
        )

    return {
        "faces": result["faces"],
        "nextToken": result["nextToken"],
        "faceModelVersion": result["faceModelVersion"]
    }

@app.delete("/collections/{collection_id}/faces")
def delete_faces(
    collection_id: str,
    request: DeleteFacesRequest,
    authenticated: bool = Depends(verify_api_key)
):
    result = repo.delete_faces(
        collection_id=collection_id,
        face_ids=request.faceIds
    )

    if not result.get("success"):
        status_code = 404 if result["error_code"] == "ResourceNotFoundException" else 400

        raise HTTPException(
            status_code=status_code,
            detail={
                "success": False,
                "error_code": result["error_code"],
                "message": result["message"]
            }
        )

    return {
        "deletedFaces": result["deletedFaces"]
    }

@app.post("/collections/{collection_id}/search")
def search_faces_by_image(
    collection_id: str,
    image: UploadFile = File(...),
    faceMatchThreshold: float = Form(80.0, ge=0.0, le=100.0),
    maxFaces: int = Form(1, ge=1, le=4096),
    authenticated: bool = Depends(verify_api_key)
):
    image_path = None

    try:
        image_path = save_upload_file(image)

        result = engine.search_faces_by_image(
            collection_id=collection_id,
            image_path=str(image_path),
            face_match_threshold=faceMatchThreshold,
            max_faces=maxFaces
        )

        if not result.get("success"):
            status_code = (
                404
                if result["error_code"] == "ResourceNotFoundException"
                else 400
            )

            raise HTTPException(
                status_code=status_code,
                detail={
                    "success": False,
                    "error_code": result["error_code"],
                    "message": result["message"]
                }
            )
        
        return {
            "faceMatches": result["faceMatches"],
            "searchedFaceBoundingBox": result[
                "searchedFaceBoundingBox"
            ],
            "searchedFaceConfidence": result[
                "searchedFaceConfidence"
            ],
            "faceModelVersion": result["faceModelVersion"]
        }
    
    except HTTPException:
        raise

    except Exception as e:
        raise_api_error(e)

    finally:
        if image_path and image_path.exists():
            image_path.unlink()

