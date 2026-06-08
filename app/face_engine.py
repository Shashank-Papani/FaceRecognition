import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DETECTOR_MODEL = str(BASE_DIR / "models" / "face_detection_yunet_2026may.onnx")
RECOGNIZER_MODEL = str(BASE_DIR / "models" / "face_recognition_sface_2021dec.onnx")


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

        retval, faces = self.detector.detect(resized)

        if faces is None or len(faces) == 0:
            return None

        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        face = faces[0].copy()

        # Scale face coordinates back to original image size
        if scale != 1.0:
            face[:14] = face[:14] / scale
        
        return face

    def get_embedding(self, image_path: str):
        image = cv2.imread(image_path)

        print("Reading:", image_path)
        print("Image loaded:", image is not None)
        if image is not None:
            print("Image shape:", image.shape)

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        face = self.detect_largest_face(image)

        if face is None:
            raise ValueError("No face detected")

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