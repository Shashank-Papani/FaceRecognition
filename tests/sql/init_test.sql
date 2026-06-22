CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS face_collections (
    id BIGSERIAL PRIMARY KEY,
    collection_id TEXT UNIQUE NOT NULL,
    collection_arn TEXT UNIQUE NOT NULL,
    face_count INTEGER NOT NULL DEFAULT 0,
    face_model_version TEXT NOT NULL,
    creation_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    id BIGSERIAL PRIMARY KEY,
    face_id UUID UNIQUE NOT NULL,
    image_id UUID NOT NULL,
    collection_id TEXT NOT NULL
        REFERENCES face_collections(collection_id)
        ON DELETE CASCADE,
    external_image_id TEXT,
    embedding vector(128) NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    bounding_box JSONB NOT NULL,
    detector_model TEXT NOT NULL,
    recognizer_model TEXT NOT NULL,
    embedding_model_version TEXT NOT NULL,
    quality JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS face_embeddings_collection_id_idx
ON face_embeddings (collection_id);

CREATE INDEX IF NOT EXISTS face_embeddings_face_id_idx
ON face_embeddings (face_id);

CREATE INDEX IF NOT EXISTS face_embeddings_external_image_id_idx
ON face_embeddings (external_image_id);

CREATE INDEX IF NOT EXISTS face_embeddings_embedding_hnsw_idx
ON face_embeddings
USING hnsw (embedding vector_cosine_ops);
