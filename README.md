# Face Recognition API

A Dockerized FastAPI-based face detection and face recognition system built with **OpenCV YuNet**, **OpenCV SFace**, **PostgreSQL**, and **pgvector**.

This project allows users to detect faces, enroll faces, verify faces against enrolled identities, manage enrolled people, and store multiple embeddings per person. It is designed as a starting point for a SaaS-style face recognition backend.

---

## Features

* Face detection using OpenCV YuNet
* Face recognition using OpenCV SFace
* FastAPI backend
* API key authentication for protected endpoints
* Enroll a person using face images
* Store multiple face embeddings per person
* Verify a new face against enrolled identities
* Standalone face detection endpoint
* Model information endpoint
* Store face embeddings in PostgreSQL
* Use pgvector for vector similarity search
* HNSW index for faster embedding search
* Prevent duplicate or near-duplicate enrollment images
* Reject images with no face
* Reject images with multiple faces
* Reject very small or low-confidence face detections
* Return face quality metadata
* List enrolled people
* Delete enrolled people
* Store verification logs
* Structured API error responses
* Request logging and latency tracking
* `X-Process-Time-ms` response header
* Dockerized API and database using Docker Compose
* Swagger UI for API testing
* Basic API tests
* Folder-based threshold evaluation script

---

## Tech Stack

* Python
* FastAPI
* OpenCV
* NumPy
* YuNet face detector
* SFace face recognizer
* PostgreSQL
* pgvector
* SQLAlchemy
* Uvicorn
* Docker
* Docker Compose
* Pytest

---

## Project Structure

```text
FaceRecognition/
  app/
    __init__.py
    auth.py
    db.py
    errors.py
    face_engine.py
    face_repository.py
    main.py

  db/
    init.sql

  docs/
    CheckList.md
    setup.md

  licenses/

  models/
    face_detection_yunet_2026may.onnx
    face_recognition_sface_2021dec.onnx

  tests/
    __init__.py
    test_api.py
    test_db.py
    test_repo.py
    threshold_test.py

  uploads/

  Dockerfile
  docker-compose.yml
  requirements.txt
  README.md
  .gitignore
  .dockerignore
```

---

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
Embedding storage/search using PostgreSQL + pgvector
   ↓
Verification result
```

During enrollment, the system generates an embedding from the uploaded face image and stores it under the provided `person_id`.

During verification, the system generates an embedding from the uploaded image and searches for the closest stored embedding using pgvector cosine similarity.

Uploaded images are saved temporarily for processing and deleted after the request completes.

---

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

Current embedding model version:

```text
sface_v1
```

Current embedding dimension:

```text
128
```

Default verification threshold:

```text
0.70
```

---

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

---

## Requirements

Before running the project, install:

* Docker Desktop
* Git

No manual PostgreSQL setup is required. Docker Compose starts both the FastAPI API and PostgreSQL + pgvector database.

---

## Environment Variables

The API uses an API key for protected endpoints.

Create a local `.env` file if running outside Docker:

```env
API_KEY=dev-secret-key
DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5433/face_recognition
```

For Docker, the API key can be set in `docker-compose.yml`:

```yaml
environment:
  DATABASE_URL: postgresql://postgres:postgres@db:5432/face_recognition
  API_KEY: dev-secret-key
```

For public GitHub repositories, do not commit `.env`.

Recommended:

```text
.env
.env.example
```

Example `.env.example`:

```env
API_KEY=change-me
```

---

## Running with Docker

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

Start the API and database:

```bash
docker compose up --build -d
```

Check running containers:

```bash
docker ps
```

Expected containers:

```text
face-recognition-api
face-postgres-db
```

The API runs on:

```text
http://127.0.0.1:8080
```

Open Swagger UI:

```text
http://127.0.0.1:8080/docs
```

Health check:

```bash
curl http://127.0.0.1:8080/health
```

Expected response:

```json
{
  "api": "healthy",
  "database": {
    "connected": true,
    "status": "success",
    "error": null
  }
}
```

---

## Stop the App

```bash
docker compose down
```

---

## Rebuild After Code Changes

```bash
docker compose up --build -d
```

---

## Docker Services

### API Service

The FastAPI application runs inside the `face-recognition-api` container.

Host URL:

```text
http://127.0.0.1:8080
```

Host-to-container port mapping:

```text
8080 -> 8000
```

### Database Service

PostgreSQL with pgvector runs inside the `face-postgres-db` container.

Database name:

```text
face_recognition
```

Database user:

```text
postgres
```

Host-to-container port mapping:

```text
5433 -> 5432
```

The API connects to the database internally using:

```text
postgresql://postgres:postgres@db:5432/face_recognition
```

---

## Authentication

Protected endpoints require an API key in the request header:

```text
x-api-key: dev-secret-key
```

Public endpoints:

```text
GET /
GET /health
GET /model-info
```

Protected endpoints:

```text
POST /detect
POST /enroll
POST /verify
GET /people
DELETE /people/{person_id}
```

Example request:

```bash
curl -H "x-api-key: dev-secret-key" http://127.0.0.1:8080/people
```

Unauthorized response:

```json
{
  "detail": {
    "success": false,
    "error_code": "UNAUTHORIZED",
    "message": "Invalid or missing API key"
  }
}
```

---

## API Endpoints

### Root

```http
GET /
```

Checks whether the API is running.

Example response:

```json
{
  "status": "running",
  "message": "Face Recognition API is working"
}
```

---

### Health Check

```http
GET /health
```

Checks API and database health.

Example response:

```json
{
  "api": "healthy",
  "database": {
    "connected": true,
    "status": "success",
    "error": null
  }
}
```

---

### Model Info

```http
GET /model-info
```

Returns detector, recognizer, embedding, and threshold configuration.

Example response:

```json
{
  "detector_model": "face_detection_yunet_2026may.onnx",
  "recognizer_model": "face_recognition_sface_2021dec.onnx",
  "embedding_model_version": "sface_v1",
  "embedding_dimension": 128,
  "similarity_metric": "cosine_similarity",
  "default_threshold": 0.7,
  "min_face_confidence": 0.6,
  "min_face_size": 80
}
```

---

## Detect Face

```http
POST /detect
```

Detects whether an uploaded image contains one valid face.

Requires API key.

### Form Data

| Field   | Type | Required | Description         |
| ------- | ---- | -------- | ------------------- |
| `image` | file | Yes      | Face image to check |

### Example Response

```json
{
  "face_detected": true,
  "quality": {
    "face_confidence": 0.99,
    "face_width": 220.0,
    "face_height": 220.0
  },
  "message": "One valid face detected"
}
```

---

## Enroll Face

```http
POST /enroll
```

Enrolls a person using an uploaded face image.

Requires API key.

### Form Data

| Field       | Type   | Required | Description              |
| ----------- | ------ | -------- | ------------------------ |
| `person_id` | string | Yes      | Unique ID for the person |
| `image`     | file   | Yes      | Face image to enroll     |

### Example Response

```json
{
  "status": "enrolled",
  "person_id": "person_001",
  "model_version": "sface_v1",
  "quality": {
    "face_confidence": 0.91,
    "face_width": 220.0,
    "face_height": 220.0
  }
}
```

### Duplicate Enrollment Response

If the same or very similar face image is enrolled again:

```json
{
  "status": "duplicate",
  "person_id": "person_001",
  "message": "This face image is already very similar to an enrolled image.",
  "similarity": 0.98
}
```

---

## Verify Face

```http
POST /verify
```

Verifies an uploaded face against enrolled people.

Requires API key.

### Form Data

| Field       | Type  | Required | Description                             |
| ----------- | ----- | -------- | --------------------------------------- |
| `image`     | file  | Yes      | Face image to verify                    |
| `threshold` | float | No       | Similarity threshold, default is `0.70` |

### Example Successful Response

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

### Example Failed Response

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

---

## List Enrolled People

```http
GET /people
```

Returns all enrolled people and their embedding counts.

Requires API key.

### Example Response

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

---

## Delete Person

```http
DELETE /people/{person_id}
```

Deletes a person and all their stored embeddings.

Requires API key.

### Example Response

```json
{
  "deleted": true,
  "person_id": "person_001",
  "message": "Person deleted successfully"
}
```

### Not Found Response

```json
{
  "detail": {
    "success": false,
    "error_code": "PERSON_NOT_FOUND",
    "message": "Person not found"
  }
}
```

---

## Structured Error Responses

The API returns structured error responses for common failures.

Example:

```json
{
  "detail": {
    "success": false,
    "error_code": "NO_FACE_DETECTED",
    "message": "No face detected. Please upload a clear front-facing image."
  }
}
```

Common error codes:

```text
UNAUTHORIZED
API_KEY_NOT_CONFIGURED
NO_FACE_DETECTED
LOW_QUALITY_FACE
MULTIPLE_FACES_DETECTED
INVALID_IMAGE
EMPTY_IMAGE
UNSUPPORTED_IMAGE_TYPE
PERSON_NOT_FOUND
INTERNAL_ERROR
```

---

## Request Logging and Latency Tracking

Each request is logged with:

```text
HTTP method
endpoint path
status code
latency in milliseconds
```

Example log:

```text
GET /health 200 64.34ms
```

Each response also includes:

```text
X-Process-Time-ms
```

Example:

```text
X-Process-Time-ms: 64.34
```

---

## Face Validation Rules

The system currently applies these validation rules:

```text
0 faces detected      -> reject
1 valid face detected -> continue
2+ faces detected     -> reject
low confidence face   -> reject
tiny face             -> reject
unsupported file type -> reject
empty upload          -> reject
```

This is important for SaaS and attendance-style systems because the API should not accidentally verify the wrong person in a group image.

---

## Similarity Threshold

The verification threshold controls how strict the match is.

Example:

```text
similarity >= threshold -> verified
similarity < threshold  -> not verified
```

The default threshold is:

```text
0.70
```

Current threshold testing showed clean separation on the local test set:

```text
Same-person comparisons:      16
Different-person comparisons: 39

Same-person score range:      0.7704 to 0.8771
Different-person score range: -0.0845 to 0.4479

Threshold 0.70:
False rejects: 0/16
False accepts: 0/39
```

Based on this test, `0.70` is used as the default threshold.

For production, the threshold should be tuned using more real-world test data.

Recommended testing:

```text
same person / same person
same person / different lighting
same person / different angle
same person / different camera
different person / similar-looking person
different person / random person
bad lighting
side angle
glasses / no glasses
distance changes
```

---

## Threshold Evaluation Script

The project includes a local threshold evaluation script:

```text
tests/threshold_test.py
```

It expects this local folder structure:

```text
threshold_tests/
  people/
    person_1/
      image1.jpg
      image2.jpg

    person_2/
      image1.jpg
      image2.jpg

    person_3/
      image1.jpg
```

The script automatically calculates:

```text
same-person scores
different-person scores
false reject rate
false accept rate
recommended threshold range
```

Run:

```bash
python -m tests.threshold_test
```

Important:

```text
threshold_tests/
```

should not be committed to GitHub because it may contain real face images.

---

## Database

The project uses PostgreSQL with pgvector.

The database is initialized automatically from:

```text
db/init.sql
```

It creates:

```text
people
face_embeddings
verification_logs
```

### people

Stores enrolled identities.

```text
id
person_id
created_at
```

### face_embeddings

Stores face embeddings for each enrolled person.

```text
id
person_id
embedding
detector_model
recognizer_model
embedding_model_version
quality
created_at
```

### verification_logs

Stores verification attempts.

```text
id
matched_person_id
similarity
threshold
verified
quality
created_at
```

The vector index is created automatically:

```sql
CREATE INDEX IF NOT EXISTS face_embeddings_embedding_hnsw_idx
ON face_embeddings
USING hnsw (embedding vector_cosine_ops);
```

---

## Useful Database Commands

Open PostgreSQL inside Docker:

```bash
docker exec -it face-postgres-db psql -U postgres -d face_recognition
```

Show tables:

```sql
\dt
```

Show enrolled people:

```sql
SELECT * FROM people;
```

Show embeddings:

```sql
SELECT
    id,
    person_id,
    detector_model,
    recognizer_model,
    embedding_model_version,
    quality,
    created_at
FROM face_embeddings;
```

Show verification logs:

```sql
SELECT
    id,
    matched_person_id,
    similarity,
    threshold,
    verified,
    created_at
FROM verification_logs
ORDER BY created_at DESC;
```

---

## Running Tests

Install test dependencies if needed:

```bash
pip install pytest httpx
```

Run basic API tests:

```bash
python -m pytest tests/test_api.py
```

Run threshold evaluation:

```bash
python -m tests.threshold_test
```

---

## Important Security Notes

Do not commit real face images, uploaded images, test images, threshold test images, API keys, `.env` files, or face embeddings to GitHub.

Recommended `.gitignore` entries:

```text
.venv/
__pycache__/
*.pyc
.pytest_cache/

.env
.DS_Store

uploads/*
!uploads/.gitkeep

test_images/
threshold_tests/

database/face_db.json
db/face_db.json
```

Face images and face embeddings are biometric data. Treat them as sensitive information.

For production, add:

```text
organization / tenant separation
per-organization API keys
hashed API keys
authorization / roles
encryption at rest
secure database access
audit logs
rate limiting
liveness detection
consent and privacy policy
```

---

## Limitations

This is an MVP, not a full production biometric system.

Current limitations:

* Uses a single global API key
* No user roles
* No tenant or organization separation
* No liveness detection
* No face anti-spoofing
* No database encryption
* No rate limiting
* No production-scale threshold testing
* No model monitoring
* Uploaded images are processed locally and deleted after request completion
* Current model stack is optimized for MVP simplicity, not maximum biometric accuracy

---

## Roadmap

Planned improvements:

* Add organization / tenant support
* Add per-organization API keys
* Store hashed API keys instead of raw keys
* Add structured request IDs
* Add rate limiting
* Add liveness detection
* Add face anti-spoofing
* Add cloud deployment
* Add stronger integration tests
* Add confidence/quality dashboard
* Add annotated image output with bounding boxes and landmarks
* Add admin dashboard
* Evaluate stronger detector/recognizer models for production use

---

## Development Status

Current status:

```text
✅ Face detection working
✅ Face recognition working
✅ Detect endpoint working
✅ Enroll endpoint working
✅ Verify endpoint working
✅ Multiple embeddings per person
✅ Duplicate enrollment protection
✅ People list endpoint
✅ Delete person endpoint
✅ Basic quality metadata
✅ PostgreSQL database
✅ pgvector similarity search
✅ HNSW vector index
✅ Verification logs
✅ Dockerized FastAPI app
✅ Dockerized PostgreSQL + pgvector database
✅ API key authentication
✅ Structured API error responses
✅ Request logging and latency tracking
✅ Model info endpoint
✅ Basic API tests
✅ Folder-based threshold evaluation script
```

Next major step:

```text
SaaS-style tenant support and liveness detection planning
```

---

## Disclaimer

This project is for development and research purposes.

If used in production, especially for attendance, access control, or identity verification, proper privacy, security, compliance, consent, and liveness detection measures must be added.
