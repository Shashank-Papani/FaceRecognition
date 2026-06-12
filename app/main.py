import time
import logging
from fastapi import Request
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from app.face_engine import FaceEngine
from app.auth import verify_api_key
from uuid import uuid4
from app.errors import raise_api_error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Face Recognition API")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "%s %s %s %.2fms",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms
    )

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