from io import BytesIO

from fastapi.testclient import TestClient

from app.auth import verify_api_key
from app.main import app, get_db, get_liveness_engine


class FakeLivenessEngine:
    def __init__(self):
        self.calls = []

    def check_liveness(
        self,
        image_path: str,
        threshold: float = 80.0,
    ) -> dict:
        self.calls.append(
            {
                "image_path": image_path,
                "threshold": threshold,
            }
        )

        return {
            "status": "SUCCEEDED",
            "live": True,
            "confidence": 95.0,
            "threshold": threshold,
            "modelVersion": "fake_liveness_v1",
            "faceQuality": {
                "face_confidence": 0.99,
                "face_width": 220.0,
                "face_height": 220.0,
            },
        }


class FakeDb:
    pass


class FakeLivenessRepository:
    def __init__(self):
        self.sessions = {}

    def create_session(self, db):
        session = {
            "session_id": "session-123",
            "status": "CREATED",
            "created_at": "2026-01-01T00:00:00",
        }

        self.sessions["session-123"] = {
            **session,
            "confidence": None,
            "threshold": None,
            "live": None,
            "model_version": None,
            "face_quality": None,
            "updated_at": "2026-01-01T00:00:00",
        }

        return session

    def get_session(self, db, session_id):
        return self.sessions.get(session_id)

    def save_result(self, db, session_id, result):
        session = self.sessions[session_id]

        session.update(
            {
                "status": result["status"],
                "confidence": result["confidence"],
                "threshold": result["threshold"],
                "live": result["live"],
                "model_version": result["modelVersion"],
                "face_quality": result["faceQuality"],
                "updated_at": "2026-01-01T00:00:01",
            }
        )

        return session


def upload_image():
    return {
        "image": (
            "face.jpg",
            BytesIO(b"fake-image"),
            "image/jpeg",
        )
    }


def fake_get_db():
    yield FakeDb()


def test_create_liveness_session(monkeypatch):
    fake_repo = FakeLivenessRepository()

    monkeypatch.setattr(
        "app.main.liveness_repo",
        fake_repo,
    )

    app.dependency_overrides[get_db] = fake_get_db
    app.dependency_overrides[verify_api_key] = lambda: True

    client = TestClient(app)

    response = client.post(
        "/liveness/sessions",
        headers={"x-api-key": "dev-secret-key"},
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200

    body = response.json()

    assert body["sessionId"] == "session-123"
    assert body["status"] == "CREATED"


def test_upload_liveness_frame(monkeypatch):
    fake_repo = FakeLivenessRepository()
    fake_repo.create_session(FakeDb())

    fake_engine = FakeLivenessEngine()

    monkeypatch.setattr(
        "app.main.liveness_repo",
        fake_repo,
    )

    app.dependency_overrides[get_db] = fake_get_db
    app.dependency_overrides[verify_api_key] = lambda: True
    app.dependency_overrides[get_liveness_engine] = (
        lambda: fake_engine
    )

    client = TestClient(app)

    response = client.post(
        "/liveness/sessions/session-123/frames",
        headers={"x-api-key": "dev-secret-key"},
        files=upload_image(),
        data={"threshold": "80"},
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200

    body = response.json()

    assert body["sessionId"] == "session-123"
    assert body["status"] == "SUCCEEDED"
    assert body["live"] is True
    assert body["confidence"] == 95.0
    assert body["threshold"] == 80.0
    assert body["modelVersion"] == "fake_liveness_v1"

    assert len(fake_engine.calls) == 1
    assert fake_engine.calls[0]["threshold"] == 80.0


def test_upload_liveness_frame_session_not_found(
    monkeypatch,
):
    fake_repo = FakeLivenessRepository()

    monkeypatch.setattr(
        "app.main.liveness_repo",
        fake_repo,
    )

    app.dependency_overrides[get_db] = fake_get_db
    app.dependency_overrides[verify_api_key] = lambda: True
    app.dependency_overrides[get_liveness_engine] = (
        lambda: FakeLivenessEngine()
    )

    client = TestClient(app)

    response = client.post(
        "/liveness/sessions/missing-session/frames",
        headers={"x-api-key": "dev-secret-key"},
        files=upload_image(),
    )

    app.dependency_overrides.clear()

    assert response.status_code == 404

    detail = response.json()["detail"]

    assert detail["error_code"] == (
        "ResourceNotFoundException"
    )


def test_get_liveness_session_results(monkeypatch):
    fake_repo = FakeLivenessRepository()
    fake_repo.create_session(FakeDb())

    fake_repo.save_result(
        FakeDb(),
        "session-123",
        {
            "status": "SUCCEEDED",
            "live": True,
            "confidence": 95.0,
            "threshold": 80.0,
            "modelVersion": "fake_liveness_v1",
            "faceQuality": {
                "face_confidence": 0.99,
                "face_width": 220.0,
                "face_height": 220.0,
            },
        },
    )

    monkeypatch.setattr(
        "app.main.liveness_repo",
        fake_repo,
    )

    app.dependency_overrides[get_db] = fake_get_db
    app.dependency_overrides[verify_api_key] = lambda: True
    client = TestClient(app)

    response = client.get(
        "/liveness/sessions/session-123/results",
        headers={"x-api-key": "dev-secret-key"},
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200

    body = response.json()

    assert body["sessionId"] == "session-123"
    assert body["status"] == "SUCCEEDED"
    assert body["live"] is True
    assert body["confidence"] == 95.0


def test_get_liveness_session_results_not_found(
    monkeypatch,
):
    fake_repo = FakeLivenessRepository()

    monkeypatch.setattr(
        "app.main.liveness_repo",
        fake_repo,
    )

    app.dependency_overrides[get_db] = fake_get_db
    app.dependency_overrides[verify_api_key] = lambda: True
    
    client = TestClient(app)

    response = client.get(
        "/liveness/sessions/missing-session/results",
        headers={"x-api-key": "dev-secret-key"},
    )

    app.dependency_overrides.clear()

    assert response.status_code == 404

    detail = response.json()["detail"]

    assert detail["error_code"] == (
        "ResourceNotFoundException"
    )