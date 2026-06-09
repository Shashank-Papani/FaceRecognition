import cv2
import numpy as np
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent

DETECTOR_MODEL = str(BASE_DIR / "models" / "face_detection_yunet_2026may.onnx")
RECOGNIZER_MODEL = str(BASE_DIR / "models" / "face_recognition_sface_2021dec.onnx")
DB_PATH = BASE_DIR / "database" / "face_db.json"

class FaceEngine:
    def __init__(self):
        self.detector = cv2.FaceDetectorYN.create(
            DETECTOR_MODEL,
            "",
            (320, 320),
            score_threshold=0.3,
            nms_threshold=0.3,
            top_k=5000
        )

        self.recognizer = cv2.FaceRecognizerSF.create(
            RECOGNIZER_MODEL,
            ""
        )

    def detect_largest_face(self, image):
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
            raise ValueError("No face detected. Please upload a clear front-facing image.")
        
        if len(faces) > 1:
            raise ValueError("Multiple faces detected. Please upload an image with only one face.")

        face = faces[0].copy()

        # Scale face coordinates back to original image size
        if scale != 1.0:
            face[:14] = face[:14] / scale
        
        return face

    def get_embedding(self, image_path: str):
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        face = self.detect_largest_face(image)

        if face is None:
            raise ValueError("No face detected. Please upload a clear front-facing image.")

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
            return json.load(f)
        
    def save_database(self, db):
        with open(DB_PATH, "w") as f:
            json.dump(db, f, indent=4)

    def enroll_face(self, person_id: str, image_path: str):
        embedding = self.get_embedding(image_path)

        db = self.load_database()
        db[person_id] = embedding.tolist()
        self.save_database(db)

        return {
            "status": "enrolled",
            "person_id": person_id
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

        for person_id, stored_embedding in db.items():
            stored_embedding = np.array(stored_embedding, dtype=np.float32)

            similarity = float(np.dot(query_embedding, stored_embedding))

            if similarity > best_similarity:
                best_similarity = similarity
                best_person_id = person_id

        is_match = best_similarity >= threshold

        return {
            "verified": is_match,
            "person_id": best_person_id if best_similarity >= threshold else None,
            "similarity": best_similarity,
            "threshold": threshold,
            "message": "Face verified successfully" if is_match else "No matching face found"
        }   