# Face Recognition API

A FastAPI-based face detection and face recognition system built with OpenCV YuNet and SFace.

This project allows users to enroll faces, verify faces against enrolled identities, manage enrolled people, and store multiple embeddings per person. It is designed as a starting point for a SaaS-style face recognition backend.

## Features

* Face detection using OpenCV YuNet
* Face recognition using OpenCV SFace
* FastAPI backend
* Enroll a person using face images
* Store multiple face embeddings per person
* Verify a new face against enrolled identities
* Prevent duplicate or near-duplicate enrollment images
* Reject images with no face
* Reject images with multiple faces
* Reject very small or low-confidence face detections
* Return face quality metadata
* List enrolled people
* Delete enrolled people
* Local JSON database for MVP testing
* Ready to migrate to PostgreSQL + pgvector

## Tech Stack

* Python
* FastAPI
* OpenCV
* NumPy
* YuNet face detector
* SFace face recognizer
* Uvicorn
* JSON file storage for local development

## Project Structure

```text
FaceRecognition/
  app/
    __init__.py
    main.py
    face_engine.py

  models/
    face_detection_yunet_2026may.onnx
    face_recognition_sface_2021dec.onnx

  database/
    face_db.json

  uploads/
    uploaded images

  test_images/
    local test images

  requirements.txt
  README.md
  .gitignore
```

## How It Works

The system follows this pipeline:

```text
Input Image
   ↓
Face Detection using YuNet
   ↓
Single-face validation
   ↓
Face Alignment
   ↓
Embedding generation using SFace
   ↓
Embedding comparison
   ↓
Verification result
```

During enrollment, the system generates an embedding from the uploaded face image and stores it under the provided `person_id`.

During verification, the system generates an embedding from the uploaded image and compares it against all stored embeddings. The closest match is returned if the similarity score is above the threshold.

## Model Details

### Face Detector

This project uses:

```text
face_detection_yunet_2026may.onnx
```

YuNet detects face bounding boxes, landmarks, and confidence scores.

### Face Recognizer

This project uses:

```text
face_recognition_sface_2021dec.onnx
```

SFace generates face embeddings that can be compared using cosine similarity.

## Licensing Notes

The OpenCV YuNet model files are distributed under the MIT License in the official OpenCV Zoo model directory.

The OpenCV library is distributed under the Apache 2.0 License.

For commercial or SaaS usage, make sure to include third-party license notices in your repository.

Recommended files:

```text
THIRD_PARTY_LICENSES.md
OPENCV_APACHE_2.0_LICENSE.txt
YUNET_MIT_LICENSE.txt
SFACE_LICENSE.txt
```

Do not remove license or copyright notices from third-party models or libraries.

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install fastapi uvicorn opencv-python numpy python-multipart
```

### 4. Add model files

Create a `models/` folder and add:

```text
face_detection_yunet_2026may.onnx
face_recognition_sface_2021dec.onnx
```

These can be downloaded from the official OpenCV Zoo model folders.

### 5. Create local database file

Create the database folder:

```bash
mkdir database
```

Create:

```text
database/face_db.json
```

Put this inside the file:

```json
{}
```

### 6. Create uploads folder

```bash
mkdir uploads
```

## Running the API

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

Open the API docs in your browser:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

### Health Check

```http
GET /
```

Example response:

```json
{
  "status": "running",
  "message": "Face Recognition API is working"
}
```

## Enroll Face

```http
POST /enroll
```

Enrolls a person using an uploaded face image.

### Form Data

| Field     | Type   | Required | Description              |
| --------- | ------ | -------- | ------------------------ |
| person_id | string | Yes      | Unique ID for the person |
| image     | file   | Yes      | Face image to enroll     |

### Example response

```json
{
  "status": "enrolled",
  "person_id": "person_001",
  "embedding_count": 3,
  "model_version": "sface_v1",
  "quality": {
    "face_confidence": 0.91,
    "face_width": 220.0,
    "face_height": 220.0
  }
}
```

### Duplicate enrollment response

If the same or very similar face image is enrolled again:

```json
{
  "status": "duplicate",
  "person_id": "person_001",
  "message": "This face image is already very similar to an enrolled image.",
  "similarity": 0.98,
  "embedding_count": 3
}
```

## Verify Face

```http
POST /verify
```

Verifies an uploaded face against all enrolled people.

### Form Data

| Field     | Type  | Required | Description                             |
| --------- | ----- | -------- | --------------------------------------- |
| image     | file  | Yes      | Face image to verify                    |
| threshold | float | No       | Similarity threshold, default is `0.70` |

### Example successful response

```json
{
  "verified": true,
  "person_id": "person_001",
  "similarity": 0.91,
  "threshold": 0.7,
  "quality": {
    "face_confidence": 0.92,
    "face_width": 210.0,
    "face_height": 210.0
  },
  "message": "Face verified successfully"
}
```

### Example failed response

```json
{
  "verified": false,
  "person_id": null,
  "similarity": 0.42,
  "threshold": 0.7,
  "quality": {
    "face_confidence": 0.89,
    "face_width": 190.0,
    "face_height": 190.0
  },
  "message": "No matching face found"
}
```

## List Enrolled People

```http
GET /people
```

Returns all enrolled people and their embedding counts.

### Example response

```json
{
  "count": 2,
  "people": [
    {
      "person_id": "person_001",
      "embedding_count": 3,
      "model_version": "sface_v1"
    },
    {
      "person_id": "person_002",
      "embedding_count": 2,
      "model_version": "sface_v1"
    }
  ]
}
```

## Delete Person

```http
DELETE /people/{person_id}
```

Deletes a person and all their stored embeddings.

### Example response

```json
{
  "deleted": true,
  "person_id": "person_001",
  "message": "Person deleted successfully"
}
```

## Face Validation Rules

The system currently applies these validation rules:

```text
0 faces detected      → reject
1 valid face detected → continue
2+ faces detected     → reject
low confidence face   → reject
tiny face             → reject
```

This is important for SaaS and attendance-style systems because the API should not accidentally verify the wrong person in a group image.

## Similarity Threshold

The verification threshold controls how strict the match is.

Example:

```text
similarity >= threshold → verified
similarity < threshold  → not verified
```

The default threshold is:

```text
0.70
```

For testing, a lower threshold like `0.60` may help when images have different lighting, expressions, or camera angles.

For production, the threshold should be tuned using real test data.

Recommended testing:

```text
same person / same person
same person / different lighting
same person / different angle
different person / similar-looking person
different person / random person
```

## Current Database Format

The local JSON database stores multiple embeddings per person.

Example:

```json
{
  "person_001": {
    "embeddings": [
      {
        "embedding": [0.01, -0.04, 0.22],
        "quality": {
          "face_confidence": 0.91,
          "face_width": 220.0,
          "face_height": 220.0
        },
        "created_at": "2026-06-08T12:30:00"
      }
    ],
    "detector_model": "face_detection_yunet_2026may.onnx",
    "recognizer_model": "face_recognition_sface_2021dec.onnx",
    "embedding_model_version": "sface_v1"
  }
}
```

## Important Security Notes

Do not commit real face images or embeddings to GitHub.

Add these to `.gitignore`:

```text
.venv/
__pycache__/
*.pyc

uploads/
test_images/
database/face_db.json

.env
```

Face embeddings are biometric data. Treat them as sensitive information.

For production, add:

```text
authentication
authorization
encryption at rest
secure database access
audit logs
rate limiting
liveness detection
consent and privacy policy
```

## Limitations

This is an MVP, not a full production biometric system.

Current limitations:

* Uses local JSON storage
* No authentication
* No user roles
* No liveness detection
* No database encryption
* No audit logging
* No production-grade threshold tuning
* No PostgreSQL/pgvector integration yet
* No face anti-spoofing
* No model monitoring

## Roadmap

Planned improvements:

* Move from JSON to PostgreSQL
* Add pgvector for embedding search
* Add verification logs
* Add liveness detection
* Add API authentication
* Add rate limiting
* Add model version migration support
* Add Docker setup
* Add unit tests
* Add confidence/quality dashboard
* Add `/detect` endpoint for detecting all faces
* Add annotated image output with bounding boxes and landmarks

## Future Database Design

For production, the project should move to PostgreSQL with pgvector.

Suggested tables:

```text
people
face_embeddings
verification_logs
```

Example:

```text
people
- id
- person_id
- created_at

face_embeddings
- id
- person_id
- embedding
- model_version
- quality
- created_at

verification_logs
- id
- uploaded_image_id
- matched_person_id
- similarity
- threshold
- verified
- created_at
```

## Development Status

Current status:

```text
✅ Face detection working
✅ Face recognition working
✅ Enroll endpoint working
✅ Verify endpoint working
✅ Multiple embeddings per person
✅ Duplicate enrollment protection
✅ People list endpoint
✅ Delete person endpoint
✅ Basic quality metadata
✅ Local JSON database
```

Next major step:

```text
PostgreSQL + pgvector migration
```

## Disclaimer

This project is for development and research purposes. If used in production, especially for attendance, access control, or identity verification, proper privacy, security, compliance, consent, and liveness detection measures must be added.
