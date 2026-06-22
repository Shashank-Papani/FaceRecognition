import os

import pytest
from fastapi.testclient import TestClient


# Must be set before importing app.auth or app.main.
os.environ.setdefault("API_KEY", "test-api-key")

from app.auth import verify_api_key
from app.main import app


def override_verify_api_key() -> bool:
    return True


@pytest.fixture
def client():
    """Client that bypasses API-key authentication."""
    app.dependency_overrides[verify_api_key] = (
        override_verify_api_key
    )

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def auth_client():
    """Client that uses the real API-key dependency."""
    app.dependency_overrides.clear()

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()

