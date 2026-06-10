Ignore below set up instructions:
    To run this program download docker and set up
    Run: docker compose up -d

    To stop
    Run: docker compose down

clone the repo and activate virtual env
    python -m venv .venv
    .venv\Scripts\activate

Install packages
    pip install opencv-python numpy fastapi uvicorn python-multipart

Correct loading of model files check:
    python -c "import cv2; print(hasattr(cv2, 'FaceDetectorYN')); print(hasattr(cv2, 'FaceRecognizerSF'))"

Add two images of yourself and other one of you and other in test_images and run:
    python -m app.main
        example output: {'similarity': 0.71, 'match': True}
    
    **Make sure the images you add are jpg and if there's any errors
    ** If you get error: frozen runpy, result = engine.compare_faces change score_threshold to lower for the start in face_engine.py

To run API endpoints:
    run, uvicorn app.main:app --reload

for API endpoint checks, use swagger
    http://127.0.0.1:8000/docs

To setup pgvector I used psql docker with command
    docker run --name face-postgres `
        -e POSTGRES_USER=postgres `
        -e POSTGRES_PASSWORD=postgres `
        -e POSTGRES_DB=face_recognition `
        -p 5432:5432 `
        -d pgvector/pgvector:pg17

To open Postgres shell for SQL code:
    docker exec -it face-postgres psql -U postgres -d face_recognition

    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS people (
        id BIGSERIAL PRIMARY KEY,
        person_id TEXT UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS face_embeddings (
        id BIGSERIAL PRIMARY KEY,
        person_id TEXT NOT NULL REFERENCES people(person_id) ON DELETE CASCADE,
        embedding vector(128) NOT NULL,
        detector_model TEXT NOT NULL,
        recognizer_model TEXT NOT NULL,
        embedding_model_version TEXT NOT NULL,
        quality JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS verification_logs (
        id BIGSERIAL PRIMARY KEY,
        matched_person_id TEXT,
        similarity FLOAT,
        threshold FLOAT,
        verified BOOLEAN,
        quality JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

Connecting FastAPI app to PostgreSQL:
    pip install sqlalchemy psycopg2-binary pgvector python-dotenv

How to check your DB content

    Open Postgres inside Docker:

        docker exec -it face-postgres psql -U postgres -d face_recognition

    Then run these.

    See all tables
        \dt

    See all enrolled people
        SELECT * FROM people;

    See embeddings without printing the huge vector:
        SELECT
            id,
            person_id,
            detector_model,
            recognizer_model,
            embedding_model_version,
            quality,
            created_at
        FROM face_embeddings;

    Count embeddings per person:
        SELECT
            person_id,
            COUNT(*) AS embedding_count
        FROM face_embeddings
        GROUP BY person_id;