
## Current status

Already have:

* YuNet face detection
* SFace recognition
* PostgreSQL + pgvector
* HNSW vector index
* Multiple embeddings per person
* Enroll, verify, detect, list, delete endpoints
* API key protection
* Structured errors
* Health and model info endpoints
* Docker Compose
* Request latency logging
* Basic API tests
* Threshold evaluation
* Temporary upload cleanup
* README and architecture docs

# Phase 1 — Finish observability

## 1. Add request IDs

Add:

```text
X-Request-ID
```

Log it with every request.

## 2. Improve logs

Log:

```text
request_id
endpoint
status_code
latency
organization_id later
error_code
```

Do not log:

```text
raw images
embeddings
API keys
```

## 3. Add global exception handling

Right now your structured errors mainly cover expected errors.

Add handlers for:

```text
HTTPException
RequestValidationError
unexpected server errors
```

Goal:

```json
{
  "success": false,
  "error_code": "INTERNAL_ERROR",
  "message": "An unexpected error occurred",
  "request_id": "..."
}
```

### Commit ideas

```bash
git commit -m "Add request ID tracking"
git commit -m "Add global exception handlers"
```

---

# Phase 2 — Improve test coverage

## 4. Expand API tests

Test:

```text
GET /
GET /health
GET /model-info
POST /detect
POST /enroll
POST /verify
GET /people
DELETE /people/{person_id}
```

Also test:

```text
missing API key
wrong API key
unsupported file type
empty image
no face
multiple faces
person not found
duplicate enrollment
invalid threshold
```

## 5. Add repository tests

Test database behavior:

```text
create person
save embedding
retrieve embeddings
find nearest match
delete person
verification logging
```

## 6. Add integration tests

Run the API against a real test database.

Ideal flow:

```text
start test database
enroll test identity
verify matching face
verify different face
delete identity
confirm identity is gone
```

## 7. Add tenant-isolation tests later

After tenant support:

```text
Organization A cannot see Organization B
Organization A cannot verify against Organization B
same person_id can exist in separate organizations
```

### Completion target

Aim for:

```text
core backend coverage: 70%+
critical enrollment/verification paths: fully tested
```

Do not fake a high coverage number. Test the important flows first.

---

# Phase 3 — SaaS tenant implementation

This is the biggest backend upgrade.

## 8. Add organizations table

```text
organizations
- id
- name
- is_active
- created_at
```

## 9. Add API keys table

```text
api_keys
- id
- organization_id
- key_hash
- name
- is_active
- created_at
- last_used_at
```

Never store raw API keys.

Store:

```text
SHA-256 or HMAC-based API key hash
```

For a stronger production design, split the key into:

```text
key_id.secret
```

Use `key_id` for lookup and hash only the secret.

## 10. Add organization_id everywhere

Add to:

```text
people
face_embeddings
verification_logs
```

Use:

```sql
UNIQUE (organization_id, person_id)
```

## 11. Update authentication

Instead of returning:

```python
True
```

return an authenticated context:

```python
{
    "organization_id": 1,
    "api_key_id": 4
}
```

Better as a typed object:

```python
AuthContext
```

## 12. Scope every query

All operations must filter by:

```text
organization_id
```

Including vector search.

Wrong:

```sql
SELECT ... FROM face_embeddings
ORDER BY embedding <=> :query_embedding
LIMIT 1;
```

Correct conceptually:

```sql
SELECT ...
FROM face_embeddings
WHERE organization_id = :organization_id
ORDER BY embedding <=> :query_embedding
LIMIT 1;
```

## 13. Add tenant-management commands

You do not need public admin endpoints yet.

Create scripts such as:

```text
scripts/create_organization.py
scripts/create_api_key.py
scripts/revoke_api_key.py
```

That is safer for the MVP than exposing admin endpoints.

### Commit ideas

```bash
git commit -m "Add organization schema"
git commit -m "Add hashed tenant API keys"
git commit -m "Scope face operations by organization"
git commit -m "Add tenant isolation tests"
```

---

# Phase 4 — Database migrations

## 14. Add Alembic

Right now `init.sql` works for a fresh database, but once the schema changes, you need versioned migrations.

Add:

```text
Alembic
```

Use migrations for:

```text
organizations table
api_keys table
organization_id columns
indexes
constraints
future schema updates
```

Keep `init.sql` only if you still want an easy fresh-start path, but make Alembic the real schema history.

### Goal

A developer should be able to run:

```bash
alembic upgrade head
```

and get the latest schema.

### Commit

```bash
git commit -m "Add Alembic database migrations"
```

---

# Phase 5 — Security hardening

## 15. Validate threshold limits

Do not let a user send arbitrary values.

Use something like:

```text
minimum: 0.0
maximum: 1.0
```

You may also restrict production use to:

```text
0.50–0.90
```

depending on your intended API behavior.

## 16. Validate person IDs

Add rules:

```text
minimum length
maximum length
allowed characters
trim whitespace
```

Example:

```text
letters
numbers
underscore
hyphen
```

## 17. Add upload limits

Protect against oversized files.

Example:

```text
maximum image size: 5 MB or 10 MB
```

Also validate:

```text
actual image decoding
MIME type
extension
dimensions
```

Do not trust only the filename.

## 18. Add rate limiting

Apply limits by API key or organization.

Example:

```text
/enroll: 20 requests/minute
/verify: 100 requests/minute
/detect: 100 requests/minute
```

For a small implementation:

```text
Redis + rate limiter
```

## 19. Add safer secrets management

For local development:

```text
.env
```

For deployment:

```text
AWS Secrets Manager
Google Secret Manager
Azure Key Vault
```

## 20. Restrict database access

Production database should not be publicly exposed.

Use:

```text
private network
strong password
TLS
limited database user
```

## 21. Add secure headers

Consider:

```text
TrustedHostMiddleware
HTTPS redirect in production
CORS configuration
```

Do not allow:

```text
CORS *
```

in production unless you truly need it.

---

# Phase 6 — Liveness detection

This is the biggest computer-vision feature still missing.

## 22. Choose Phase 1 liveness approach

For a free prototype:

```text
MediaPipe Face Mesh
blink challenge
head turn challenge
smile challenge
randomized action order
```

Example session:

```text
1. Start session
2. Server randomly selects challenge
3. Client sends frames
4. Server validates challenge
5. Face verification runs
6. Final result requires both checks
```

## 23. Add liveness session endpoints

Example:

```http
POST /liveness/session/start
POST /liveness/session/frame
POST /liveness/session/complete
```

## 24. Store liveness results

Add:

```text
liveness_sessions
- id
- organization_id
- challenge_type
- status
- score
- created_at
- expires_at
```

## 25. Connect liveness to verification

Final rule:

```text
verified = face_match AND liveness_passed
```

## 26. Add replay protection

Each session should have:

```text
unique session ID
expiration time
single-use completion
random challenge
```

## 27. Understand the limitation

Active liveness is a strong prototype, but it is not equivalent to a certified commercial anti-spoofing system.

For serious production use, you would likely evaluate:

```text
commercial liveness SDK
passive anti-spoofing model
ISO/IEC 30107-3 evaluated solution
```

---

# Phase 7 — Recognition quality improvements

## 28. Expand threshold testing

Your current results are promising, but the dataset is small.

Test:

```text
more people
more age groups
different skin tones
different cameras
indoor and outdoor lighting
side angles
glasses
facial hair
distance
low resolution
motion blur
partial occlusion
```

## 29. Separate validation and threshold datasets

Do not tune and report on the same images.

Use:

```text
development set
validation set
final test set
```

## 30. Calculate proper metrics

Add:

```text
FAR
FRR
TAR
ROC curve
EER
precision
recall
confusion counts
```

Most important for face verification:

```text
FAR at selected threshold
FRR at selected threshold
TAR at a fixed FAR
```

## 31. Save anonymous test reports

Do not commit face images.

You can commit:

```text
docs/threshold_report.md
test result summaries
charts
aggregate metrics
```

Avoid putting personal names in public test reports.

Use labels like:

```text
subject_01
subject_02
```

## 32. Test model alternatives

Current stack:

```text
YuNet + SFace
```

Good for the MVP.

Later compare:

```text
detector accuracy
recognition accuracy
CPU latency
memory usage
model size
license suitability
```

Possible future production stack:

```text
SCRFD + ArcFace
```

only after confirming licensing for the exact weights and intended commercial use.

---

# Phase 8 — Performance and scalability

## 33. Benchmark latency

Measure:

```text
face detection latency
embedding latency
vector search latency
total API latency
```

Report:

```text
average
p50
p95
p99
```

Test on:

```text
CPU
different image sizes
different database sizes
```

## 34. Add load testing

Use:

```text
Locust
k6
```

Test:

```text
10 concurrent users
50 concurrent users
100 concurrent users
```

Endpoints:

```text
/detect
/verify
/enroll
```

## 35. Improve model loading

Make sure models load once when the application starts, not per request.

You already mostly follow this pattern through the shared engine instance.

## 36. Control worker count carefully

Multiple Uvicorn workers mean multiple copies of the models in memory.

Benchmark:

```text
1 worker
2 workers
4 workers
```

## 37. Add connection pooling

Configure SQLAlchemy pool values:

```text
pool_size
max_overflow
pool_timeout
pool_recycle
```

## 38. Improve vector search at scale

For the current project:

```text
PostgreSQL + pgvector is enough
```

Later consider:

```text
partitioning by organization
index tuning
higher ef_search
separate vector database
```

Only do this when real scale requires it.

---

# Phase 9 — API quality

## 39. Add Pydantic response models

Currently many responses may be plain dictionaries.

Create models such as:

```text
HealthResponse
ModelInfoResponse
DetectResponse
EnrollResponse
VerifyResponse
PeopleResponse
DeleteResponse
ErrorResponse
```

Benefits:

```text
better Swagger docs
validation
consistent responses
clearer code
```

## 40. Add API versioning

Move to:

```text
/api/v1/detect
/api/v1/enroll
/api/v1/verify
/api/v1/people
```

You can keep `/health` unversioned.

## 41. Add pagination

For:

```text
GET /people
GET /verification-logs
```

Use:

```text
limit
offset
```

or cursor-based pagination.

## 42. Add verification history endpoint

Protected and tenant-scoped:

```http
GET /verification-logs
```

Include:

```text
verified
matched_person_id
similarity
threshold
timestamp
```

Do not expose embeddings.

## 43. Add API documentation examples

Show:

```text
curl
Python requests
Postman
Swagger
```

---

# Phase 10 — Deployment

## 44. Choose a simple production architecture

A reasonable first cloud deployment:

```text
FastAPI container
AWS ECS Fargate
RDS PostgreSQL with pgvector
Redis
S3 only if required
CloudWatch
Application Load Balancer
Secrets Manager
```

For your portfolio, you do not need Kubernetes yet.

## 45. Add CI

GitHub Actions should run:

```text
lint
tests
Docker build
```

On every pull request and push.

## 46. Add code quality tools

Use:

```text
Ruff
Black
Pytest
mypy optionally
```

Example checks:

```bash
ruff check .
black --check .
pytest
```

## 47. Add production Docker improvements

Use:

```text
non-root user
pinned dependency versions
health check
smaller image
no development files
```

## 48. Add deployment configuration

Create:

```text
docker-compose.dev.yml
docker-compose.prod.yml
```

or keep one Compose file for local development and use cloud-specific infrastructure separately.

## 49. Add monitoring

Track:

```text
request count
error rate
latency
database connection failures
verification success/failure count
model inference latency
```

---

# Phase 11 — Privacy and biometric safety

## 50. Add retention rules

Define:

```text
how long verification logs are stored
whether images are stored
how embeddings are deleted
what happens when a person is removed
```

## 51. Add consent and deletion flow

A production system needs:

```text
consent
data deletion request
organization ownership
auditability
retention policy
```

## 52. Avoid storing raw images by default

Your current approach of temporary processing and deletion is good.

Document that clearly.

## 53. Add audit logs

Track sensitive operations:

```text
person enrolled
person deleted
API key created
API key revoked
verification attempted
```

Do not store secrets in audit logs.

---

# Phase 12 — Final portfolio polish

## 54. Improve README

Add:

```text
architecture diagram
API flow diagram
database diagram
benchmark summary
threshold results
security decisions
limitations
future work
```

## 55. Add screenshots

Good screenshots:

```text
Swagger UI
successful enrollment
successful verification
structured error
Docker containers running
database tables
test results
```

Do not show:

```text
real API keys
personal face images
private person names
database passwords
```

## 56. Add architecture diagram

Show:

```text
Client
  ↓
FastAPI
  ↓
Authentication
  ↓
Face Engine
  ↓
YuNet + SFace
  ↓
Repository
  ↓
PostgreSQL + pgvector
```

Later tenant version:

```text
API key → organization → scoped vector search
```

## 57. Add project demo video

A short demo:

```text
start Docker
open Swagger
enroll person
verify person
show database result
show error handling
show threshold testing
```

## 58. Add a final project report

Create:

```text
docs/project_report.md
```

Include:

```text
problem
architecture
technology choices
implementation
testing
results
limitations
future work
```

---

# Recommended order from today

Do not build everything randomly. Use this exact order:

```text
1. Request IDs
2. Global exception handling
3. More API tests
4. Alembic migrations
5. Organizations table
6. Hashed per-tenant API keys
7. Scope all operations by organization
8. Tenant isolation tests
9. Upload size and input validation
10. Rate limiting
11. Threshold metrics report
12. Latency benchmarks
13. Load testing
14. Active liveness prototype
15. CI with GitHub Actions
16. Cloud deployment
17. Final documentation and demo
```

# Minimum “finished” version

You can call the project finished as a strong portfolio project when you have:

```text
✅ Core face pipeline
✅ PostgreSQL + pgvector
✅ Docker
✅ API authentication
✅ Structured errors
✅ Automated API tests
✅ Threshold testing
✅ Request IDs
✅ Tenant isolation
✅ Hashed API keys
✅ Database migrations
✅ Input/file-size validation
✅ Rate limiting
✅ CI pipeline
✅ Benchmark results
✅ Clean documentation
✅ Cloud deployment or reproducible local deployment
```

# Advanced version

These are valuable but not required before calling it finished:

```text
liveness detection
passive anti-spoofing
admin dashboard
mobile/web frontend
Kubernetes
dedicated vector database
GPU inference
commercial-grade liveness certification
```

Your best finish line is: **multi-tenant authenticated face-recognition API with tested thresholds, tenant-isolated vector search, CI, security controls, documented benchmarks, and Docker/cloud deployment.**
