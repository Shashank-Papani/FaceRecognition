````markdown
# Face Recognition and Text Detection API

A Dockerized FastAPI backend for face recognition, face search, and OCR-based text detection.

This project implements an AWS Rekognition-style face recognition workflow using **OpenCV YuNet**, **OpenCV SFace**, **PostgreSQL**, and **pgvector**, with an additional **PaddleOCR-powered text detection API** for extracting printed text such as odometer readings, labels, and document text.

---

## Overview

This API provides:

- Face detection
- Face indexing into named collections
- Face search by image
- Face listing and deletion
- PostgreSQL-backed vector storage using pgvector
- OCR text detection with line-level and word-level results
- Confidence, region, and bounding-box filters
- API key authentication
- Dockerized local development
- Automated API, repository, database, and integration tests

The project is designed as a practical backend system for SaaS-style face recognition and OCR workflows.

---

## Licensing and Third-Party Notices

This project uses third-party open-source libraries and model files. License and attribution files are included near the top level of the repository so the project can be reviewed safely for portfolio, research, and commercial-evaluation use.

### Included license and notice files

```text
licenses/
  OPENCV_APACHE_2.0_LICENSE.txt
  YUNET_MIT_LICENSE.txt
  PADDLEOCR_APACHE_2.0_LICENSE.txt
  PADDLEPADDLE_APACHE_2.0_LICENSE.txt

THIRD_PARTY_NOTICES.md
````

If generated locally, dependency-level license reports may also be stored as:

```text
THIRD_PARTY_DEPENDENCIES.md
```

### Main third-party components

| Component               | Purpose                                                                                                                        | License / Notice                     |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------ |
| OpenCV                  | Computer vision library used for face detection, face alignment, image loading, resizing, grayscale conversion, and sharpening | Apache 2.0 for OpenCV 4.5.0+         |
| OpenCV YuNet            | Face detection model                                                                                                           | See included YuNet license notice    |
| OpenCV SFace            | Face recognition model                                                                                                         | See included model/license notice    |
| PaddleOCR               | OCR text detection and recognition                                                                                             | Apache 2.0                           |
| PaddlePaddle            | Deep learning runtime used by PaddleOCR                                                                                        | Apache 2.0                           |
| PostgreSQL              | Relational database                                                                                                            | PostgreSQL License                   |
| pgvector                | Vector similarity search extension for PostgreSQL                                                                              | PostgreSQL-style open-source license |
| pgvector Python package | Python integration for pgvector                                                                                                | MIT License                          |

Do not remove copyright notices, license files, model notices, or third-party attributions.

This README is not legal advice. Before production or commercial deployment, review all third-party licenses and dependency licenses for the exact versions used in `requirements.txt`.

---

## Current Status

```text
✅ Dockerized FastAPI API
✅ Dockerized PostgreSQL + pgvector database
✅ OpenCV YuNet face detection
✅ OpenCV SFace face recognition
✅ Collection-based face indexing
✅ Face search using vector similarity
✅ Face listing and deletion
✅ PaddleOCR text detection
✅ OCR preprocessing pipeline
✅ OCR confidence filtering
✅ OCR region filtering
✅ OCR bounding-box width/height filtering
✅ 15 MB text-image upload limit
✅ Structured error handling
✅ API key authentication
✅ Request logging and latency headers
✅ GitHub Actions test workflow
✅ Mocked API tests
✅ Real PostgreSQL integration tests
✅ Real face ML integration tests with private assets
✅ Real PaddleOCR integration test with private assets
```

---

## Features

### Face Recognition

* Detect faces using OpenCV YuNet.
* Align faces before recognition.
* Generate 128-dimensional SFace embeddings.
* Store embeddings in PostgreSQL using pgvector.
* Search embeddings using cosine similarity.
* Support multiple collections.
* Index faces into a collection.
* Search a collection using an input image.
* List faces in a collection.
* Delete faces from a collection.
* Delete entire collections.
* Return normalized face bounding boxes.
* Return face confidence and quality metadata.
* Reject images with no face.
* Reject images with multiple faces.
* Reject small or low-confidence face detections.

### Text Detection

* Detect printed text using PaddleOCR.
* Return `LINE` and `WORD` detections.
* Preserve parent-child relationships using `parentId`.
* Return confidence scores from `0` to `100`.
* Return normalized bounding boxes and polygons.
* Filter results by minimum confidence.
* Filter results by normalized region of interest.
* Filter results by minimum bounding-box width and height.
* Enforce a 15 MB image-size limit for text detection.
* Return documented HTTP status codes for invalid image and oversized image errors.

### API and Infrastructure

* FastAPI backend.
* Swagger UI for manual testing.
* API key authentication.
* Docker Compose for API and database.
* PostgreSQL with pgvector.
* Request ID middleware.
* Request latency logging.
* `X-Request-ID` response header.
* `X-Process-Time-ms` response header.
* Structured JSON error responses.
* Automated tests with pytest.
* GitHub Actions CI.

---

## Tech Stack

* Python
* FastAPI
* OpenCV
* YuNet
* SFace
* PaddleOCR
* PaddlePaddle
* NumPy
* PostgreSQL
* pgvector
* SQLAlchemy
* Uvicorn
* Docker
* Docker Compose
* Pytest
* GitHub Actions

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
    text_engine.py

  db/
    init.sql

  docs/
    CheckList.md
    setup.md

  licenses/
    OPENCV_APACHE_2.0_LICENSE.txt
    YUNET_MIT_LICENSE.txt
    PADDLEOCR_APACHE_2.0_LICENSE.txt
    PADDLEPADDLE_APACHE_2.0_LICENSE.txt

  models/
    face_detection_yunet_2026may.onnx
    face_recognition_sface_2021dec.onnx

  scripts/
    evaluate_threshold.py
    test_paddleocr.py

  tests/
    api/
      test_text_api.py

    integration/
      test_text_engine_integration.py

    private_assets/
      # Local-only private test images.
      # This folder is ignored by Git.

    sql/
      init_test.sql

    test_api.py
    test_db.py
    test_repo.py

  uploads/

  .dockerignore
  .gitignore
  docker-compose.yml
  docker-compose.test.yml
  Dockerfile
  pytest.ini
  requirements.txt
  README.md
  THIRD_PARTY_NOTICES.md
```

---

## How the Face Recognition Pipeline Works

```text
Input image
   ↓
Face detection with YuNet
   ↓
Single-face validation
   ↓
Face alignment
   ↓
Embedding generation with SFace
   ↓
Embedding storage in PostgreSQL + pgvector
   ↓
Cosine-similarity search
   ↓
Face match result
```

During indexing, the API detects and validates the face, generates a face embedding, and stores the embedding inside a selected collection.

During search, the API generates an embedding from the query image and compares it against stored embeddings in the requested collection using pgvector cosine similarity.

Uploaded images are saved temporarily for request processing and deleted after the request completes.

---

## How the Text Detection Pipeline Works

```text
Input image
   ↓
Resize while preserving aspect ratio
   ↓
Ensure minimum width of 800 px
   ↓
Convert to grayscale
   ↓
Apply sharpening kernel
   ↓
Encode as PNG in memory
   ↓
Run PaddleOCR
   ↓
Convert OCR output into LINE and WORD detections
   ↓
Apply confidence, region, and bounding-box filters
   ↓
Return normalized OCR results
```

The OCR preprocessing uses this sharpening kernel:

```text
[ 0, -1,  0]
[-1,  5, -1]
[ 0, -1,  0]
```

---

## Models

### Face Detector

```text
face_detection_yunet_2026may.onnx
```

YuNet is used for face detection, bounding boxes, landmarks, and confidence scores.

### Face Recognizer

```text
face_recognition_sface_2021dec.onnx
```

SFace is used to generate face embeddings.

Current recognizer settings:

```text
Embedding model version: sface_v1
Embedding dimension:    128
Similarity metric:      cosine similarity
```

### Text Detector / Recognizer

```text
PaddleOCR PP-OCRv5
```

PaddleOCR is used for text detection and recognition.

The API returns both:

```text
LINE
WORD
```

detections.

---

## Requirements

Install:

* Docker Desktop
* Git

No manual PostgreSQL installation is required. Docker Compose starts both the API and PostgreSQL + pgvector database.

---

## Environment Variables

The API uses an API key for protected endpoints.

Create a local `.env` file if running outside Docker:

```env
API_KEY=change-me
DATABASE_URL=postgresql+psycopg2://postgres:postgres@127.0.0.1:5433/face_recognition
```

For Docker Compose, configure environment variables in `docker-compose.yml`:

```yaml
environment:
  DATABASE_URL: postgresql+psycopg2://postgres:postgres@db:5432/face_recognition
  API_KEY: change-me
```

Do not commit `.env`.

A safe `.env.example` can contain:

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
docker compose ps
```

Expected services:

```text
api
db
```

The API runs on:

```text
http://127.0.0.1:8080
```

Swagger UI:

```text
http://127.0.0.1:8080/docs
```

Health check:

```bash
curl http://127.0.0.1:8080/health
```

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

## Stop the App

```bash
docker compose down
```

---

## Rebuild After Code Changes

```bash
docker compose down && \
docker compose build api && \
docker compose up -d
```

---

## Docker Services

### API Service

The FastAPI application runs inside the API container.

Host URL:

```text
http://127.0.0.1:8080
```

Port mapping:

```text
8080 -> 8000
```

### Database Service

PostgreSQL with pgvector runs inside the database container.

Database name:

```text
face_recognition
```

Database user:

```text
postgres
```

Port mapping:

```text
5433 -> 5432
```

The API connects to the database internally using:

```text
postgresql+psycopg2://postgres:postgres@db:5432/face_recognition
```

---

## Authentication

Protected endpoints require an API key:

```text
x-api-key: change-me
```

Public endpoints:

```text
GET /
GET /health
GET /model-info
```

Protected endpoints:

```text
POST /faces/detect
POST /collections
GET /collections
GET /collections/{collection_id}
DELETE /collections/{collection_id}
POST /collections/{collection_id}/faces
GET /collections/{collection_id}/faces
DELETE /collections/{collection_id}/faces
POST /collections/{collection_id}/search
POST /text/detect
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

# API Endpoints

## Root

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

## Health Check

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

## Model Info

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

# Face Detection API

## Detect Faces

```http
POST /faces/detect
```

Detects faces in an uploaded image and returns normalized bounding boxes, landmarks, confidence, and quality metadata.

Requires API key.

### Form Data

| Field        |   Type | Required | Description                        |
| ------------ | -----: | -------: | ---------------------------------- |
| `image`      |   File |      Yes | JPEG or PNG image                  |
| `attributes` | String |       No | Attribute mode. Default: `DEFAULT` |

### Example Request

```bash
curl -X POST "http://127.0.0.1:8080/faces/detect" \
  -H "x-api-key: change-me" \
  -F "image=@test_images/person.jpg" \
  -F "attributes=DEFAULT"
```

### Example Response

```json
{
  "faceDetails": [
    {
      "boundingBox": {
        "width": 0.24,
        "height": 0.31,
        "left": 0.38,
        "top": 0.21
      },
      "confidence": 99.8,
      "landmarks": [
        {
          "type": "eyeLeft",
          "x": 0.44,
          "y": 0.32
        },
        {
          "type": "eyeRight",
          "x": 0.55,
          "y": 0.32
        }
      ],
      "quality": {
        "brightness": 86.4,
        "sharpness": 91.2
      }
    }
  ],
  "orientationCorrection": null
}
```

---

# Collection APIs

## Create Collection

```http
POST /collections
```

Creates a new face collection.

Requires API key.

### JSON Body

```json
{
  "collectionId": "employees"
}
```

### Example Request

```bash
curl -X POST "http://127.0.0.1:8080/collections" \
  -H "x-api-key: change-me" \
  -H "Content-Type: application/json" \
  -d '{"collectionId":"employees"}'
```

### Example Response

```json
{
  "statusCode": 200,
  "collectionArn": "local:rekognition:collection/employees",
  "faceModelVersion": "sface_v1"
}
```

---

## List Collections

```http
GET /collections
```

Lists existing collections.

Requires API key.

### Query Parameters

| Parameter    |    Type | Required | Description                   |
| ------------ | ------: | -------: | ----------------------------- |
| `maxResults` | Integer |       No | Maximum collections to return |
| `nextToken`  |  String |       No | Pagination token              |

### Example Request

```bash
curl -H "x-api-key: change-me" \
  "http://127.0.0.1:8080/collections"
```

### Example Response

```json
{
  "collectionIds": [
    "employees",
    "visitors"
  ],
  "nextToken": null
}
```

---

## Describe Collection

```http
GET /collections/{collection_id}
```

Returns metadata for a collection.

Requires API key.

### Example Request

```bash
curl -H "x-api-key: change-me" \
  "http://127.0.0.1:8080/collections/employees"
```

### Example Response

```json
{
  "collectionARN": "local:rekognition:collection/employees",
  "faceCount": 3,
  "faceModelVersion": "sface_v1",
  "creationTimestamp": "2026-06-01T12:30:00"
}
```

---

## Delete Collection

```http
DELETE /collections/{collection_id}
```

Deletes a collection and its stored face embeddings.

Requires API key.

### Example Request

```bash
curl -X DELETE "http://127.0.0.1:8080/collections/employees" \
  -H "x-api-key: change-me"
```

### Example Response

```json
{
  "statusCode": 200
}
```

---

## Index Face

```http
POST /collections/{collection_id}/faces
```

Indexes one face into a collection.

Requires API key.

### Form Data

| Field             |   Type | Required | Description                            |
| ----------------- | -----: | -------: | -------------------------------------- |
| `image`           |   File |      Yes | JPEG or PNG face image                 |
| `externalImageId` | String |       No | Caller-provided image/person reference |

### Example Request

```bash
curl -X POST "http://127.0.0.1:8080/collections/employees/faces" \
  -H "x-api-key: change-me" \
  -F "image=@test_images/person_001.jpg" \
  -F "externalImageId=person_001"
```

### Example Response

```json
{
  "faceRecords": [
    {
      "face": {
        "faceId": "c9f70b8f-3d3f-4f18-9871-1bc4c70a9b22",
        "boundingBox": {
          "width": 0.24,
          "height": 0.31,
          "left": 0.38,
          "top": 0.21
        },
        "confidence": 99.8,
        "externalImageId": "person_001",
        "imageId": "7bc67c86-cb8f-49ec-a83e-fb53a45763bb"
      },
      "faceDetail": {
        "boundingBox": {
          "width": 0.24,
          "height": 0.31,
          "left": 0.38,
          "top": 0.21
        },
        "confidence": 99.8
      }
    }
  ],
  "faceModelVersion": "sface_v1"
}
```

---

## List Faces

```http
GET /collections/{collection_id}/faces
```

Lists faces stored in a collection.

Requires API key.

### Query Parameters

| Parameter    |    Type | Required | Description                 |
| ------------ | ------: | -------: | --------------------------- |
| `maxResults` | Integer |       No | Maximum faces to return     |
| `nextToken`  |  String |       No | Pagination token            |
| `faceIds`    |    List |       No | Optional face IDs to filter |

### Example Request

```bash
curl -H "x-api-key: change-me" \
  "http://127.0.0.1:8080/collections/employees/faces"
```

### Example Response

```json
{
  "faces": [
    {
      "faceId": "c9f70b8f-3d3f-4f18-9871-1bc4c70a9b22",
      "imageId": "7bc67c86-cb8f-49ec-a83e-fb53a45763bb",
      "externalImageId": "person_001",
      "boundingBox": {
        "width": 0.24,
        "height": 0.31,
        "left": 0.38,
        "top": 0.21
      },
      "confidence": 99.8,
      "createdAt": "2026-06-01T12:30:00"
    }
  ],
  "nextToken": null,
  "faceModelVersion": "sface_v1"
}
```

---

## Delete Faces

```http
DELETE /collections/{collection_id}/faces
```

Deletes one or more face records from a collection.

Requires API key.

### JSON Body

```json
{
  "faceIds": [
    "c9f70b8f-3d3f-4f18-9871-1bc4c70a9b22"
  ]
}
```

### Example Request

```bash
curl -X DELETE "http://127.0.0.1:8080/collections/employees/faces" \
  -H "x-api-key: change-me" \
  -H "Content-Type: application/json" \
  -d '{"faceIds":["c9f70b8f-3d3f-4f18-9871-1bc4c70a9b22"]}'
```

### Example Response

```json
{
  "deletedFaces": [
    "c9f70b8f-3d3f-4f18-9871-1bc4c70a9b22"
  ]
}
```

---

## Search Faces by Image

```http
POST /collections/{collection_id}/search
```

Searches a collection for the closest matching face using an uploaded image.

Requires API key.

### Form Data

| Field                |    Type | Required | Description                       |
| -------------------- | ------: | -------: | --------------------------------- |
| `image`              |    File |      Yes | Query face image                  |
| `faceMatchThreshold` |   Float |       No | Match threshold from `0` to `100` |
| `maxFaces`           | Integer |       No | Maximum matches to return         |

### Example Request

```bash
curl -X POST "http://127.0.0.1:8080/collections/employees/search" \
  -H "x-api-key: change-me" \
  -F "image=@test_images/query.jpg" \
  -F "faceMatchThreshold=80" \
  -F "maxFaces=1"
```

### Example Response

```json
{
  "faceMatches": [
    {
      "similarity": 92.35,
      "face": {
        "faceId": "c9f70b8f-3d3f-4f18-9871-1bc4c70a9b22",
        "externalImageId": "person_001",
        "imageId": "7bc67c86-cb8f-49ec-a83e-fb53a45763bb",
        "boundingBox": {
          "width": 0.24,
          "height": 0.31,
          "left": 0.38,
          "top": 0.21
        },
        "confidence": 99.8
      }
    }
  ],
  "searchedFaceBoundingBox": {
    "width": 0.24,
    "height": 0.31,
    "left": 0.38,
    "top": 0.21
  },
  "searchedFaceConfidence": 99.8,
  "faceModelVersion": "sface_v1"
}
```

---

# Text Detection API

## Detect Text

```http
POST /text/detect
```

Detects text in an uploaded image and returns line-level and word-level OCR results.

Requires API key.

### Form Data

| Field     |        Type | Required | Description                                           |
| --------- | ----------: | -------: | ----------------------------------------------------- |
| `image`   |        File |      Yes | JPEG or PNG image                                     |
| `filters` | JSON string |       No | Optional confidence, region, and bounding-box filters |

### Image Limits

| Format   | Maximum Size |
| -------- | -----------: |
| JPEG/JPG |        15 MB |
| PNG      |        15 MB |

### Filter Options

| Field                  |  Type | Description                                                                     |
| ---------------------- | ----: | ------------------------------------------------------------------------------- |
| `minConfidence`        | Float | Minimum confidence from `0` to `100`                                            |
| `minBoundingBoxWidth`  | Float | Minimum normalized bounding-box width from `0.0` to `1.0`                       |
| `minBoundingBoxHeight` | Float | Minimum normalized bounding-box height from `0.0` to `1.0`                      |
| `regionsOfInterest`    |  List | Optional normalized regions used to keep detections inside selected image areas |

### Region Filter Format

```json
{
  "regionsOfInterest": [
    {
      "boundingBox": {
        "left": 0.30,
        "top": 0.65,
        "width": 0.35,
        "height": 0.20
      }
    }
  ]
}
```

A line detection is kept when the center point of its bounding box falls inside one of the requested regions. Child `WORD` detections are included only when their parent `LINE` detection is included.

### Example Request

```bash
curl -X POST "http://127.0.0.1:8080/text/detect" \
  -H "x-api-key: change-me" \
  -F "image=@test_images/odometer.jpg" \
  -F 'filters={"minConfidence":50,"minBoundingBoxWidth":0.02,"minBoundingBoxHeight":0.02,"regionsOfInterest":[{"boundingBox":{"left":0.30,"top":0.65,"width":0.35,"height":0.20}}]}'
```

### Example Response

```json
{
  "textDetections": [
    {
      "id": 0,
      "type": "LINE",
      "detectedText": "091308",
      "confidence": 99.58,
      "geometry": {
        "boundingBox": {
          "width": 0.22875,
          "height": 0.08126,
          "left": 0.34875,
          "top": 0.70812
        },
        "polygon": [
          {
            "x": 0.34875,
            "y": 0.70812
          },
          {
            "x": 0.5775,
            "y": 0.70812
          },
          {
            "x": 0.5775,
            "y": 0.78938
          },
          {
            "x": 0.34875,
            "y": 0.78938
          }
        ]
      }
    },
    {
      "id": 1,
      "parentId": 0,
      "type": "WORD",
      "detectedText": "091308",
      "confidence": 99.58,
      "geometry": {
        "boundingBox": {
          "width": 0.19375,
          "height": 0.08126,
          "left": 0.36875,
          "top": 0.70812
        },
        "polygon": [
          {
            "x": 0.36875,
            "y": 0.70812
          },
          {
            "x": 0.5625,
            "y": 0.70812
          },
          {
            "x": 0.5625,
            "y": 0.78938
          },
          {
            "x": 0.36875,
            "y": 0.78938
          }
        ]
      }
    }
  ]
}
```

### Text Detection Errors

| Error Code                    | HTTP Status | Description                                   |
| ----------------------------- | ----------: | --------------------------------------------- |
| `InvalidParameterException`   |         400 | Invalid filter value or malformed filter JSON |
| `InvalidImageFormatException` |         415 | Unsupported or corrupted image                |
| `ImageTooLargeException`      |         413 | Image exceeds the 15 MB limit                 |
| `InternalServerError`         |         500 | Unexpected service-side error                 |

### Example Invalid Filter Response

```json
{
  "detail": {
    "success": false,
    "error_code": "InvalidParameterException",
    "message": "filters must be valid JSON."
  }
}
```

### Example Image Too Large Response

```json
{
  "detail": {
    "success": false,
    "error_code": "ImageTooLargeException",
    "message": "Image size cannot exceed 15 MB."
  }
}
```

---

## Database

The project uses PostgreSQL with pgvector.

The database is initialized automatically from:

```text
db/init.sql
```

Main tables:

```text
collections
face_embeddings
verification_logs
```

### collections

Stores collection metadata.

```text
id
collection_id
collection_arn
face_model_version
created_at
```

### face_embeddings

Stores face embeddings for indexed faces.

```text
id
collection_id
face_id
image_id
external_image_id
embedding
detector_model
recognizer_model
embedding_model_version
bounding_box
confidence
quality
created_at
```

### verification_logs

Stores search and verification attempts.

```text
id
collection_id
matched_face_id
matched_external_image_id
similarity
threshold
verified
quality
created_at
```

The vector index is created automatically using pgvector.

Example:

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

Show collections:

```sql
SELECT * FROM collections;
```

Show stored face records:

```sql
SELECT
    id,
    collection_id,
    face_id,
    image_id,
    external_image_id,
    detector_model,
    recognizer_model,
    embedding_model_version,
    confidence,
    created_at
FROM face_embeddings
ORDER BY created_at DESC;
```

Show verification/search logs:

```sql
SELECT
    id,
    collection_id,
    matched_face_id,
    similarity,
    threshold,
    verified,
    created_at
FROM verification_logs
ORDER BY created_at DESC;
```

---

## Testing

### Run Text API Tests

```bash
pytest -v tests/api/test_text_api.py
```

### Run Real Text OCR Integration Test

This test uses a private local odometer image under:

```text
tests/private_assets/text/odometer.jpg
```

That image is intentionally excluded from Git.

```bash
pytest -v tests/integration/test_text_engine_integration.py
```

### Run Full Test Suite with Test Database

Start the test database:

```bash
docker compose -f docker-compose.test.yml up -d --wait
```

Run tests:

```bash
DATABASE_URL="postgresql+psycopg2://postgres:test-password@localhost:5434/face_recognition_test" \
pytest -v
```

Stop the test database:

```bash
docker compose -f docker-compose.test.yml down
```

### Private Asset Tests

Some integration tests use local private images from:

```text
tests/private_assets/
```

This folder is ignored by Git.

If the private images are not present, those tests are skipped safely.

---

## Threshold Evaluation

The project includes a local threshold evaluation script:

```text
scripts/evaluate_threshold.py
```

It expects a local folder structure such as:

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
      image2.jpg
```

The script calculates:

```text
same-person scores
different-person scores
false reject rate
false accept rate
recommended threshold range
```

Run:

```bash
python scripts/evaluate_threshold.py
```

Important:

```text
threshold_tests/
```

should not be committed to GitHub because it may contain real face images.

---

## Request Logging and Latency Tracking

Each request is logged with:

```text
request_id
HTTP method
endpoint path
status code
latency in milliseconds
```

Each response includes:

```text
X-Request-ID
X-Process-Time-ms
```

Example:

```text
X-Request-ID: 8e6d7c83-b3d8-495d-b2fe-9f6decb35c23
X-Process-Time-ms: 64.34
```

---

## Error Handling

The API returns structured JSON error responses.

Example:

```json
{
  "detail": {
    "success": false,
    "error_code": "InvalidParameterException",
    "message": "minConfidence must be a number."
  }
}
```

Common error codes include:

```text
UNAUTHORIZED
API_KEY_NOT_CONFIGURED
InvalidParameterException
InvalidImageFormatException
ImageTooLargeException
ResourceAlreadyExistsException
ResourceNotFoundException
InternalServerError
```

---

## Security Notes

Do not commit:

```text
.env
API keys
private face images
private OCR images
customer images
uploaded images
threshold test images
PaddleOCR model cache
face embeddings
database dumps
```

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
tests/private_assets/

database/
*.db
```

Face images and face embeddings are biometric data. Treat them as sensitive information.

For production usage, add:

```text
tenant separation
per-tenant API keys or JWT auth
hashed API keys
authorization roles
encryption at rest
secure database access
audit logs
rate limiting
liveness detection
anti-spoofing
consent and privacy policy
monitoring and alerting
```

---

## Limitations

This project is an MVP-style backend, not a full production biometric identity platform.

Current limitations:

* Uses a single API key.
* Does not include tenant-level authorization.
* Does not include liveness detection.
* Does not include face anti-spoofing.
* Does not include rate limiting.
* Does not include database encryption.
* Does not include production monitoring.
* Does not include model drift monitoring.
* Uses local file uploads temporarily during request processing.
* OCR model loading may increase first-request latency.
* OCR region filtering is implemented as an API extension.
* Thresholds should be tuned on larger real-world datasets before production use.

---

## Roadmap

Planned improvements:

* Add tenant support.
* Add per-tenant API keys or JWT authentication.
* Store hashed API keys instead of raw keys.
* Add liveness detection.
* Add face anti-spoofing.
* Add rate limiting.
* Add production deployment configuration.
* Add observability and dashboards.
* Add OCR model cache packaging.
* Add stronger Swagger/Pydantic response models.
* Add more real-world OCR and face-recognition integration tests.
* Add annotated image output for debugging.
* Add admin dashboard.
* Evaluate stronger detector and recognizer models for production accuracy.

---

## Development Status

```text
✅ Face detection
✅ Face indexing
✅ Face search
✅ Collection management
✅ Face listing
✅ Face deletion
✅ PostgreSQL database
✅ pgvector similarity search
✅ Dockerized API
✅ Dockerized database
✅ API key authentication
✅ Structured error responses
✅ Request logging
✅ Latency headers
✅ PaddleOCR text detection
✅ OCR preprocessing
✅ OCR confidence filter
✅ OCR region filter
✅ OCR bounding-box filters
✅ 15 MB OCR upload limit
✅ 415 invalid-image handling
✅ 413 image-too-large handling
✅ Mocked API tests
✅ Real database integration tests
✅ Real ML integration tests with private assets
✅ GitHub Actions CI
```

Next major step:

```text
Production hardening: tenant auth, rate limiting, liveness detection, and deployment.
```

---

## Disclaimer

This project is for development, research, and portfolio purposes.

If used in production, especially for attendance, access control, financial workflows, or identity verification, proper privacy, security, compliance, consent, anti-spoofing, and liveness-detection measures must be added.

```
```
