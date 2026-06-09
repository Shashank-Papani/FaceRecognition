import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent

DETECTOR_MODEL = str(BASE_DIR / "models" / "face_detection_yunet_2026may.onnx")
RECOGNIZER_MODEL = str(BASE_DIR / "models" / "face_recognition_sface_2021dec.onnx")
DB_PATH = BASE_DIR / "database" / "face_db.json"

DETECTOR_MODEL_NAME = "face_detection_yunet_2026may.onnx"
RECOGNIZER_MODEL_NAME = "face_recognition_sface_2021dec.onnx"
EMBEDDING_MODEL_VERSION = "sface_v1"

MIN_FACE_CONFIDENCE = 0.6
MIN_FACE_SIZE = 80
MIN_BLUR_SCORE = 80.0

class FaceEngine:
    def __init__(self):
        self.detector = cv2.FaceDetectorYN.create(
            DETECTOR_MODEL,
            "",
            (320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000
        )

        self.recognizer = cv2.FaceRecognizerSF.create(
            RECOGNIZER_MODEL,
            ""
        )

    def detect_single_face(self, image):
        original_h, original_w = image.shape[:2]

        # Resize large images for detection
        max_size = 1280
        scale = 1.0

        if max(original_w, original_h) > max_size:
            scale = max_size / max(original_w, original_h)
            new_w = int(original_w * scale)
            new_h = int(original_h * scale)
            resized = cv2.resize(image, (new_w, new_h))
        else:
            resized = image

        h, w = resized.shape[:2]    
        self.detector.setInputSize((w, h))

        _, faces = self.detector.detect(resized)

        if faces is None or len(faces) == 0:
            raise ValueError(
                "No face detected. Please upload a clear front-facing image."
            )
        
        valid_faces = []

        for face in faces:
            x, y, face_w, face_h = face[:4]
            confidence = face[-1]

            # Reject weak fake detections
            if confidence < MIN_FACE_CONFIDENCE:
                continue

            # Reject tiny detections
            if face_w < MIN_FACE_SIZE or face_h < MIN_FACE_SIZE:
                continue

            valid_faces.append(face)

        if len(valid_faces) == 0:
            raise ValueError(
                "No valid face detected. The face may be too small, unclear, or low confidence."
            )
        
        if len(valid_faces) > 1:
            raise ValueError(
                "Multiple faces detected. Please upload an image with only one face."
            )

        face = valid_faces[0].copy()

        self.last_face_quality = {
            "face_confidence": float(face[-1]),
            "face_width": float(face[2]),
            "face_height": float(face[3])
        }

        # Scale face coordinates back to original image size
        if scale != 1.0:
            face[:14] = face[:14] / scale
        
        return face

    def get_embedding(self, image_path: str):
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
    
        face = self.detect_single_face(image)

        aligned_face = self.recognizer.alignCrop(image, face)
        embedding = self.recognizer.feature(aligned_face)

        embedding = embedding.flatten()
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def compare_faces(self, image_path_1: str, image_path_2: str):
        emb1 = self.get_embedding(image_path_1)
        emb2 = self.get_embedding(image_path_2)

        similarity = float(np.dot(emb1, emb2))

        return {
            "similarity": similarity,
            "match": similarity >= 0.70
        }
    
    def load_database(self):
        if not DB_PATH.exists():
            return {}

        with open(DB_PATH, "r") as f:
            content = f.read().strip()

            if not content:
                return {}

            return json.loads(content)
        
    def save_database(self, db):
        with open(DB_PATH, "w") as f:
            json.dump(db, f, indent=4)

    def enroll_face(self, person_id: str, image_path: str):
        embedding = self.get_embedding(image_path)

        db = self.load_database()

        new_embedding_record = {
            "embedding": embedding.tolist(),
            "quality": self.last_face_quality,
            "created_at": datetime.utcnow().isoformat()
        }

        if person_id not in db:
            db[person_id] = {
                "embeddings": [],
                "detector_model": DETECTOR_MODEL_NAME,
                "recognizer_model": RECOGNIZER_MODEL_NAME,
                "embedding_model_version": EMBEDDING_MODEL_VERSION
            }
        
        db[person_id]["embeddings"].append(new_embedding_record)

        self.save_database(db)

        return {
            "status": "enrolled",
            "person_id": person_id,
            "embedding_count": len(db[person_id]["embeddings"]),
            "model_version": EMBEDDING_MODEL_VERSION,
            "quality": self.last_face_quality
        }

    def verify_face(self, image_path: str, threshold: float = 0.70):
        query_embedding = self.get_embedding(image_path)

        db = self.load_database()

        if not db:
            return {
                "match": False,
                "message": "No enrolled faces found"
            }

        best_person_id = None
        best_similarity = -1.0

        for person_id, record in db.items():
            for embedding_record in record["embeddings"]:
                stored_embedding = np.array(embedding_record["embedding"], dtype=np.float32)

                similarity = float(np.dot(query_embedding, stored_embedding))

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_person_id = person_id

        is_match = best_similarity >= threshold

        return {
            "verified": is_match,
            "person_id": best_person_id if is_match else None,
            "similarity": best_similarity,
            "threshold": threshold,
            "quality": self.last_face_quality,
            "message": "Face verified successfully" if is_match else "No matching face found"
        }

    def delete_person(self, person_id: str):
        db = self.load_database()

        if person_id not in db:
            return {
                "deleted": False,
                "person_id": person_id,
                "message": "Person not found"
            }

        del db[person_id]
        self.save_database(db)

        return {
            "deleted": True,
            "person_id": person_id,
            "message": "Person deleted successfully"
        }

    def is_image_blurry(self, image, threshold: float = MIN_BLUR_SCORE):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        return {
            "is_blurry": blur_score < threshold,
            "blur_score": float(blur_score),
            "threshold": threshold
        }