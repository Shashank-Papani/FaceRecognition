import time
import logging
from fastapi import Request
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Query
from pydantic import BaseModel
from app.face_engine import FaceEngine
from app.auth import verify_api_key
from uuid import uuid4
from app.errors import raise_api_error
from app import face_repository as repo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Face Recognition API")

class CreateCollectionRequest(BaseModel):
    collectionId: str

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

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def save_upload_file(image: UploadFile) -> Path:
    file_extension = Path(image.filename).suffix.lower()

    if file_extension not in [".jpg", ".jpeg", ".png"]:
        raise ValueError("Only JPG, JPEG, and PNG images are supported.")
    
    image_path = UPLOAD_DIR / f"{uuid4()}{file_extension}"

    image.file.seek(0)

    with image_path.open("wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    if image_path.stat().st_size == 0:
        raise ValueError("Uploaded image is empty.")

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

@app.post("/detect")
def detect_face(
    image: UploadFile = File(...),
    authenticated: bool = Depends(verify_api_key)
):
    image_path = None

    try:
        image_path = save_upload_file(image)

        result = engine.detect_face_info(
            image_path=str(image_path)
        )

        return result
    
    except Exception as e:
        raise_api_error(e)

    finally:
        if image_path and image_path.exists():
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
    maxResults: int = Query(1000),
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
        return {
            "statusCode": 404
        }

    return {
        "statusCode": 200
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

@app.post("/enroll")
def enroll_face(
    person_id: str = Form(...),
    image: UploadFile = File(...),
    authenticated: bool = Depends(verify_api_key)
):
    image_path = None

    try:
        image_path = save_upload_file(image)

        result = engine.enroll_face(
            person_id=person_id,
            image_path=str(image_path)
        )

        return result
    
    except Exception as e:
        raise_api_error(e)
    
    finally:
        if image_path and image_path.exists():
            image_path.unlink()
    
@app.post("/verify")
def verify_face(
    image: UploadFile = File(...),
    threshold: float = Form(0.70),
    authenticated: bool = Depends(verify_api_key)
):
    image_path = None

    try:
        image_path = save_upload_file(image)

        result = engine.verify_face(
            image_path=str(image_path),
            threshold=threshold
        )

        return result
    
    except Exception as e:
        raise_api_error(e)
    
    finally:
        if image_path and image_path.exists():
            image_path.unlink()
    
@app.get("/people")
def list_people(
    authenticated: bool = Depends(verify_api_key)
):
    return engine.list_people()

@app.delete("/people/{person_id}")
def delete_person(
    person_id: str,
    authenticated: bool = Depends(verify_api_key)
):
    result = engine.delete_person(person_id)

    if not result["deleted"]:
        raise HTTPException(
            status_code=404,
            detail={
                "success": False,
                "error_code": "PERSON_NOT_FOUND",
                "message": result["message"]
            }
        )
    
    return result