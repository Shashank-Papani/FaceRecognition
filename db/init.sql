CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS face_collections (
    id BIGSERIAL PRIMARY KEY,
    collection_id TEXT UNIQUE NOT NULL,
    collection_arn TEXT UNIQUE NOT NULL,
    face_count BIGINT DEFAULT 0,
    face_model_version TEXT NOT NULL,
    creation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP 
);

CREATE TABLE IF NOT EXISTS people (
    id BIGSERIAL PRIMARY KEY,
    person_id TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    id BIGSERIAL PRIMARY KEY,

    face_id UUID UNIQUE NOT NULL,
    image_id UUID UNIQUE NOT NULL,
    collection_id text NOT NULL REFERENCES face_collections(collection_id) ON DELETE CASCADE,
    external_image_id TEXT,

    embedding vector(128) NOT NULL,

    confidence FLOAT,
    bounding_box JSNOB,

    detector_model TEXT NOT NULL,
    recognizer_model TEXT NOT NULL,
    embedding_model_version TEXT NOT NULL,
    quality JSONOB,

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

CREATE INDEX IF NOT EXISTS face_embeddings_collection_id_idx
ON face_embeddings (collection_id);

CREATE INDEX IF NOT EXISTS face_embeddings_face_id_idx
ON face_embeddings (face_id);

CREATE INDEX IF NOT EXISTS face_embeddings_external_image_id_idx
ON face_embeddings (external_image_id);

CREATE TABLE IF NOT EXISTS liveness_sessions (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT UNIQUE NOT NULL,
    status TEXT NOT NULL DEFAULT 'CREATED',
    confidence FLOAT,
    threshold FLOAT,
    live BOOLEAN,
    model_version TEXT,
    face_quality JSONB,
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now()
);