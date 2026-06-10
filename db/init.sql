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

CREATE INDEX IF NOT EXISTS face_embeddings_embedding_hnsw_idx
ON face_embeddings
USING hnsw (embedding vector_cosine_ops);