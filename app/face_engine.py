import cv2
import numpy as np
from pathlib import Path

from app.face_repository import (
    save_face_embedding,
    get_all_embeddings,
    list_people as repo_list_people,
    delete_person as repo_delete_person,
    log_verification,
    find_best_match
)

BASE_DIR = Path(__file__).resolve().parent.parent

DETECTOR_MODEL = str(BASE_DIR / "models" / "face_detection_yunet_2026may.onnx")
RECOGNIZER_MODEL = str(BASE_DIR / "models" / "face_recognition_sface_2021dec.onnx")

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

    def model_info(self):
        return{
            "detector_model": DETECTOR_MODEL_NAME,
            "recognizer_model": RECOGNIZER_MODEL_NAME,
            "embedding_model_version": EMBEDDING_MODEL_VERSION,
            "embedding_dimension": 128,
            "similarity_metric": "cosine_similarity",
            "defualt_threshold": 0.70,
            "min_face_confidence": MIN_FACE_CONFIDENCE,
            "min_face_size": MIN_FACE_SIZE,
        }

    def database_health_check(self):
        from app.face_repository import database_health_check
        return database_health_check()

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

    def enroll_face(self, person_id: str, image_path: str):
        embedding = self.get_embedding(image_path)

        existing_records = get_all_embeddings()

        for record in existing_records:
            if record["person_id"] != person_id:
                continue

            stored_embedding_text = record["embedding"]
            stored_embedding = np.array(
                [float(x) for x in stored_embedding_text.strip("[]").split(",")],
                dtype=np.float32
            )

            similarity = float(np.dot(embedding, stored_embedding))

            if similarity >= 0.95:
                return {
                    "status": "duplicate",
                    "person_id": person_id,
                    "message": "This face image is already very similar to an enrolled image.",
                    "similarity": similarity
                }

        save_face_embedding(
            person_id=person_id,
            embedding=embedding.tolist(),
            detector_model=DETECTOR_MODEL_NAME,
            recognizer_model=RECOGNIZER_MODEL_NAME,
            embedding_model_version=EMBEDDING_MODEL_VERSION,
            quality=self.last_face_quality
        )
        
        return {
            "status": "enrolled",
            "person_id": person_id,
            "model_version": EMBEDDING_MODEL_VERSION,
            "quality": self.last_face_quality
        }

    def verify_face(self, image_path: str, threshold: float = 0.70):
        query_embedding = self.get_embedding(image_path)

        best_match = find_best_match(query_embedding.tolist())

        if best_match is None:
            log_verification(
                matched_person_id=None,
                similarity=None,
                threshold=threshold,
                verified=False,
                quality=self.last_face_quality
            )

            return {
                "verified": False,
                "person_id": None,
                "similarity": None,
                "threshold": threshold,
                "quality": self.last_face_quality,
                "message": "No enrolled faces found"
            }
        
        best_person_id = best_match["person_id"]
        best_similarity = float(best_match["similarity"])

        is_match = best_similarity >= threshold
        matched_person_id = best_person_id if is_match else None

        log_verification(
            matched_person_id=matched_person_id,
            similarity=best_similarity,
            threshold=threshold,
            verified=is_match,
            quality=self.last_face_quality
        )

        return {
            "verified": is_match,
            "person_id": matched_person_id,
            "similarity": best_similarity,
            "threshold": threshold,
            "quality": self.last_face_quality,
            "message": "Face verified successfully" if is_match else "No matching face found"
        }

    def list_people(self):
        people = repo_list_people()

        return {
            "count": len(people),
            "people": [
                {
                    "person_id": person["person_id"],
                    "embedding_count": person["embedding_count"],
                    "model_version": person["model_version"]
                }
                for person in people
            ]
        }

    def delete_person(self, person_id: str):
        deleted = repo_delete_person(person_id)

        if not deleted:
            return {
                "deleted": False,
                "person_id": person_id,
                "message": "Person not found"
            }

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
    
    