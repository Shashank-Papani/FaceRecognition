import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from app.face_engine import FaceEngine
from app.auth import verify_api_key

app = FastAPI(title="Face Recognition API")
engine = FaceEngine()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

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

@app.post("/enroll")
def enroll_face(
    person_id: str = Form(...),
    image: UploadFile = File(...),
    authenticated: bool = Depends(verify_api_key)
):
    try:
        image_path = UPLOAD_DIR / image.filename

        with image_path.open("wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        result = engine.enroll_face(
            person_id=person_id,
            image_path=str(image_path)
        )
        return result
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/verify")
def verify_face(
    image: UploadFile = File(...),
    threshold: float = Form(0.70),
    authenticated: bool = Depends(verify_api_key)
):
    try:
        image_path = UPLOAD_DIR / image.filename

        with image_path.open("wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        result = engine.verify_face(
            image_path=str(image_path),
            threshold=threshold
        )
        return result
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
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
        raise HTTPException(status_code=404, detail=result["message"])
    
    return result