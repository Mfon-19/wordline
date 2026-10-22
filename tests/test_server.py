"""The web API's guards: refusals for bad uploads and questions, and readiness."""
import pytest
from fastapi.testclient import TestClient

from worldline import server


@pytest.fixture
def api():
    server.state["ready"] = False
    return TestClient(server.app)


def test_health_reports_loading(api):
    body = api.get("/api/health").json()
    assert body["ready"] is False and body["model"] == "google/embeddinggemma-2"


def test_search_waits_for_the_model(api):
    assert api.get("/api/search", params={"q": "badger"}).status_code == 503


def test_upload_needs_rights_confirmation(api):
    r = api.post("/api/upload", content=b"x" * 10)
    assert r.status_code == 400 and "right to share" in r.json()["detail"]


def test_upload_rejects_large_and_non_video_files(api):
    r = api.post("/api/upload?agree=1", content=b"x", headers={"content-length": str(60 * 2**20)})
    assert r.status_code == 413
    r = api.post("/api/upload?agree=1", content=b"definitely not a video", headers={"x-filename": "a.mp4",
                                                                                   "x-forwarded-for": "198.51.100.7"})
    assert r.status_code == 400 and "couldn't be read as a video" in r.json()["detail"]


def test_unknown_upload_and_run_ids(api):
    assert api.get("/api/upload/doesnotexist1").status_code == 404
    assert api.get("/api/upload/../../etc").status_code == 404
    assert api.get("/api/agent/nope").status_code == 404


def test_agent_question_validation(api):
    assert api.post("/api/agent", json={"question": "x"}).status_code == 400
    assert api.post("/api/agent", json={"question": "When did the fox arrive?"}).status_code == 503


def test_visitor_is_the_address_cloudfront_adds():
    req = type("R", (), {"headers": {"x-forwarded-for": "6.6.6.6, 203.0.113.5"}, "client": None})()
    assert server.visitor(req) == "203.0.113.5"
