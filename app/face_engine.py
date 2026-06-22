import cv2
import numpy as np
from pathlib import Path
from uuid import uuid4

from app.face_repository import (
    collection_exists,
    save_indexed_face,
    search_faces_by_embedding
)

BASE_DIR = Path(__file__).resolve().parent.parent

DETECTOR_MODEL = str(BASE_DIR / "models" / "face_detection_yunet_2026may.onnx")
RECOGNIZER_MODEL = str(BASE_DIR / "models" / "face_recognition_sface_2021dec.onnx")

DETECTOR_MODEL_NAME = "face_detection_yunet_2026may.onnx"
RECOGNIZER_MODEL_NAME = "face_recognition_sface_2021dec.onnx"
EMBEDDING_MODEL_VERSION = "sface_v1"

MIN_FACE_CONFIDENCE = 0.6
MIN_FACE_SIZE = 80

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
            "default_threshold": 0.70,
            "min_face_confidence": MIN_FACE_CONFIDENCE,
            "min_face_size": MIN_FACE_SIZE,
        }

    def database_health_check(self):
        from app.face_repository import database_health_check
        return database_health_check()

    def detect_single_face(self, image):
        original_h, original_w = image.shape[:2]

        max_size = 1280
        scale = 1.0

        if max(original_w, original_h) > max_size:
            scale = max_size / max(original_w, original_h)
            new_w = int(original_w * scale)
            new_h = int(original_h * scale)

            resized = cv2.resize(
                image,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA
            )
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

        for detected_face in faces:
            candidate = detected_face.copy()

            # Convert detection coordinates back to the original image size.
            if scale != 1.0:
                candidate[:14] = candidate[:14] / scale

            x, y, face_w, face_h = candidate[:4]
            confidence = float(candidate[-1])

            if confidence < MIN_FACE_CONFIDENCE:
                continue

            if face_w < MIN_FACE_SIZE or face_h < MIN_FACE_SIZE:
                continue

            valid_faces.append(candidate)

        if len(valid_faces) == 0:
            raise ValueError(
                "No valid face detected. The face may be too small, unclear, or low confidence."
            )

        if len(valid_faces) > 1:
            raise ValueError(
                "Multiple faces detected. Please upload an image with only one face."
            )

        face = valid_faces[0]

        x, y, face_w, face_h = face[:4]
        confidence = float(face[-1])

        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(original_w, int(x + face_w))
        y2 = min(original_h, int(y + face_h))

        if x2 <= x1 or y2 <= y1:
            raise ValueError("Detected face bounding box is invalid.")

        bounding_box = {
            "width": float((x2 - x1) / original_w),
            "height": float((y2 - y1) / original_h),
            "left": float(x1 / original_w),
            "top": float(y1 / original_h)
        }

        self.last_face_quality = {
            "face_confidence": confidence,
            "face_confidence_percent": confidence * 100,
            "face_width": float(face_w),
            "face_height": float(face_h),
            "bounding_box": bounding_box
        }

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
    
    def index_face(
        self,
        collection_id: str,
        image_path: str,
        external_image_id: str | None = None
    ):
        if not collection_exists(collection_id):
            return{
                "success": False,
                "error_code": "ResourceNotFoundException",
                "message": "Collection does not exist"
            }
        
        embedding = self.get_embedding(image_path)

        face_id = str(uuid4())
        image_id = str(uuid4())

        confidence = self.last_face_quality["face_confidence_percent"]
        bounding_box = self.last_face_quality["bounding_box"]

        save_indexed_face(
            collection_id = collection_id,
            face_id = face_id,
            image_id = image_id,
            external_image_id = external_image_id,
            embedding = embedding.tolist(),
            confidence = confidence,
            bounding_box = bounding_box,
            detector_model = DETECTOR_MODEL_NAME,
            recognizer_model = RECOGNIZER_MODEL_NAME,
            embedding_model_version = EMBEDDING_MODEL_VERSION,
            quality = self.last_face_quality
        )

        return {
            "success": True,
            "faceRecords": [
                {
                    "face": {
                        "faceId": face_id,
                        "imageId": image_id,
                        "externalImageId": external_image_id,
                        "confidence": confidence,
                        "boundingBox": bounding_box 
                    }
                }
            ],
            "faceModelVersion": EMBEDDING_MODEL_VERSION
        }
    
    def search_faces_by_image(
        self,
        collection_id: str,
        image_path: str,
        face_match_threshold: float = 80.0,
        max_faces: int = 1
    ):
        if not collection_exists(collection_id):
            return {
                "success": False,
                "error_code": "ResourceNotFoundException",
                "message": "Collection does not exist"
            }
        
        embedding = self.get_embedding(image_path)

        searched_face_bounding_box = self.last_face_quality["bounding_box"]
        searched_face_confidence = self.last_face_quality[
            "face_confidence_percent"
        ]

        result = search_faces_by_embedding(
            collection_id=collection_id,
            query_embedding=embedding.tolist(),
            face_match_threshold=face_match_threshold,
            max_faces=max_faces
        )

        if not result.get("success"):
            return result

        return {
            "success": True,
            "faceMatches": result["faceMatches"],
            "searchedFaceBoundingBox": searched_face_bounding_box,
            "searchedFaceConfidence": searched_face_confidence,
            "faceModelVersion": result["faceModelVersion"]
        }
    
    def detect_faces_info(
        self,
        image_path: str,
        attributes: str = "DEFAULT"
    ):
            
        attributes = attributes.upper().strip()

        if attributes not in {"DEFAULT", "ALL"}:
            raise ValueError("attributes must be DEFAULT or ALL")

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(
                "InvalidImageFormatException: Unsupported or corrupted image"
            )

        original_h, original_w = image.shape[:2]

        max_size = 1280
        scale = min(1.0, max_size / max(original_w, original_h))

        if scale < 1.0:
            resized_w = int(original_w * scale)
            resized_h = int(original_h * scale)

            detection_image = cv2.resize(
                image,
                (resized_w, resized_h),
                interpolation=cv2.INTER_AREA
            )
        else:
            detection_image = image

        detection_h, detection_w = detection_image.shape[:2]

        self.detector.setInputSize((detection_w, detection_h))

        _, faces = self.detector.detect(detection_image)

        if faces is None or len(faces) == 0:
            raise ValueError(
                "InvalidParameterException: No face detected"
            )

        faces = faces.copy()

        if scale != 1.0:
            faces[:, :14] = faces[:, :14] / scale

        face_details = []

        for face in faces:
            x, y, face_w, face_h = face[:4]

            raw_confidence = float(face[-1])

            if raw_confidence < MIN_FACE_CONFIDENCE:
                continue

            if face_w < MIN_FACE_SIZE or face_h < MIN_FACE_SIZE:
                continue

            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(original_w, int(x + face_w))
            y2 = min(original_h, int(y + face_h))

            if x2 <= x1 or y2 <= y1:
                continue

            face_crop = image[y1:y2, x1:x2]
            gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

            brightness = float(np.mean(gray_crop) / 255.0 * 100.0)

            raw_sharpness = float(
                cv2.Laplacian(
                    gray_crop,
                    cv2.CV_64F
                ).var()
            )

            # Local 0–100 sharpness score.
            # This is not calibrated to AWS Rekognition.
            sharpness = min(100.0, raw_sharpness)

            confidence = raw_confidence * 100.0

            bounding_box = {
                "width": float((x2 - x1) / original_w),
                "height": float((y2 - y1) / original_h),
                "left": float(x1 / original_w),
                "top": float(y1 / original_h)
            }

            # YuNet landmark order:
            # right eye, left eye, nose, right mouth, left mouth
            landmarks = [
                {
                    "type": "eyeRight",
                    "x": float(face[4] / original_w),
                    "y": float(face[5] / original_h)
                },
                {
                    "type": "eyeLeft",
                    "x": float(face[6] / original_w),
                    "y": float(face[7] / original_h)
                },
                {
                    "type": "nose",
                    "x": float(face[8] / original_w),
                    "y": float(face[9] / original_h)
                },
                {
                    "type": "mouthRight",
                    "x": float(face[10] / original_w),
                    "y": float(face[11] / original_h)
                },
                {
                    "type": "mouthLeft",
                    "x": float(face[12] / original_w),
                    "y": float(face[13] / original_h)
                }
            ]

            detail = {
                "confidence": round(confidence, 2),
                "boundingBox": bounding_box,
                "landmarks": landmarks,
                "quality": {
                    "brightness": round(brightness, 2),
                    "sharpness": round(sharpness, 2)
                }
            }

            face_details.append(detail)

        if not face_details:
            raise ValueError(
                "InvalidParameterException: No valid faces detected"
            )

        response = {
            "faceDetails": face_details,
            "orientationCorrection": None
        }

        if attributes == "ALL":
            response["unavailableAttributes"] = [
                "pose",
                "faceOccluded",
                "eyesOpen",
                "mouthOpen",
                "eyeglasses",
                "sunglasses",
                "beard",
                "mustache",
                "gender",
                "ageRange",
                "smile",
                "emotions"
            ]

        return response
