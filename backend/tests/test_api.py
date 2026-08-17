"""
API-level tests. Heavy model calls (predict_fake_probability,
get_face_embedding, get_model_info) are monkeypatched onto the `main`
module so this suite runs fast and deterministically without needing the
GPU, the multi-hundred-MB checkpoints, or a network connection. See
test_model_integration.py for a real, end-to-end model-loading test.
"""
import numpy as np
import pytest

import main
from errors import ModelLoadError
from tests.conftest import jpeg_bytes


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info_route_returns_predictor_metadata(client, monkeypatch):
    monkeypatch.setattr(
        main,
        "get_model_info",
        lambda: {"model_version": "test-model", "device": "cpu", "status": "loaded"},
    )
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json() == {"model_version": "test-model", "device": "cpu", "status": "loaded"}


def test_predict_rejects_unsupported_content_type(client):
    response = client.post(
        "/predict", files={"file": ("f.gif", b"not an image", "image/gif")}
    )
    assert response.status_code == 415


def test_predict_rejects_oversized_upload(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_IMAGE_UPLOAD_BYTES", 100)
    response = client.post(
        "/predict", files={"file": ("f.jpg", jpeg_bytes(), "image/jpeg")}
    )
    assert response.status_code == 413


def test_predict_rejects_corrupt_image(client):
    response = client.post(
        "/predict", files={"file": ("f.jpg", b"\xff\xd8\xff\x00garbage", "image/jpeg")}
    )
    assert response.status_code == 400
    assert "decode" in response.json()["detail"].lower()


def test_predict_flags_undersized_crop_as_insufficient_quality(client, monkeypatch):
    calls = []
    monkeypatch.setattr(main, "predict_fake_probability", lambda img: calls.append(img) or 0.9)

    response = client.post(
        "/predict", files={"file": ("f.jpg", jpeg_bytes(size=(16, 16)), "image/jpeg")}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["fake_probability"] is None
    assert body["classification"] == "insufficient_quality"
    assert body["quality_issue"] is not None
    assert calls == []  # model must never be called on a rejected crop


def test_predict_happy_path_returns_classification_and_metadata(client, monkeypatch):
    monkeypatch.setattr(main, "predict_fake_probability", lambda img: 0.9)
    monkeypatch.setattr(main, "get_model_info", lambda: {"model_version": "test-model"})

    response = client.post(
        "/predict", files={"file": ("f.jpg", jpeg_bytes(), "image/jpeg")}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["fake_probability"] == 0.9
    assert body["classification"] == "fake"
    assert body["quality_issue"] is None
    assert body["model_version"] == "test-model"
    assert body["processing_time_ms"] >= 0
    assert body["request_id"]
    assert response.headers["X-Request-ID"] == body["request_id"]


def test_predict_real_classification_for_low_probability(client, monkeypatch):
    monkeypatch.setattr(main, "predict_fake_probability", lambda img: 0.05)
    monkeypatch.setattr(main, "get_model_info", lambda: {"model_version": "test-model"})

    response = client.post("/predict", files={"file": ("f.jpg", jpeg_bytes(), "image/jpeg")})
    assert response.json()["classification"] == "real"


def test_predict_returns_503_when_model_fails_to_load(client, monkeypatch):
    def raise_load_error(img):
        raise ModelLoadError("checkpoint missing")

    monkeypatch.setattr(main, "predict_fake_probability", raise_load_error)

    response = client.post("/predict", files={"file": ("f.jpg", jpeg_bytes(), "image/jpeg")})
    assert response.status_code == 503


def test_predict_unhandled_exception_returns_generic_500_without_leaking_details(monkeypatch, isolated_dbs):
    # A real deployment (uvicorn) only ever sees the JSON response our
    # exception_handler builds -- Starlette's ServerErrorMiddleware re-raises
    # the original exception *after* sending that response, purely so an
    # ASGI server can log it; httpx's TestClient surfaces that re-raise by
    # default, which would fail this test even though the actual HTTP
    # response was correct. raise_server_exceptions=False observes what an
    # HTTP client actually receives, matching production behavior.
    from fastapi.testclient import TestClient

    import main

    client = TestClient(main.app, raise_server_exceptions=False)

    def explode(img):
        raise ValueError("some internal CUDA detail that should not reach the client")

    monkeypatch.setattr(main, "predict_fake_probability", explode)

    response = client.post("/predict", files={"file": ("f.jpg", jpeg_bytes(), "image/jpeg")})
    assert response.status_code == 500
    body = response.json()
    assert "CUDA" not in body["detail"]
    assert body["detail"] == "Internal server error"
    assert "request_id" in body


def test_contacts_rejects_when_no_face_detected(client, monkeypatch):
    monkeypatch.setattr(main, "get_face_embedding", lambda img: None)

    response = client.post(
        "/contacts",
        data={"name": "Alice"},
        files={"file": ("f.jpg", jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 400


def test_contacts_rejects_blank_name(client):
    response = client.post(
        "/contacts",
        data={"name": "   "},
        files={"file": ("f.jpg", jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 400


def test_contacts_enroll_and_list(client, monkeypatch):
    monkeypatch.setattr(main, "get_face_embedding", lambda img: np.ones(512, dtype=np.float32))

    response = client.post(
        "/contacts",
        data={"name": "Alice"},
        files={"file": ("f.jpg", jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["name"] == "Alice"

    listed = client.get("/contacts").json()
    assert len(listed) == 1
    assert listed[0]["name"] == "Alice"


@pytest.mark.parametrize("bad_bytes", [b"", b"\x00\x01\x02"])
def test_predict_rejects_empty_or_garbage_upload_gracefully(client, bad_bytes):
    response = client.post(
        "/predict", files={"file": ("f.jpg", bad_bytes, "image/jpeg")}
    )
    assert response.status_code in (400, 422)


# ---------------------------------------------------------------------------
# /analyze-image, /analyze-video (upload dashboard)
# ---------------------------------------------------------------------------

def test_analyze_image_rejects_unsupported_content_type(client):
    response = client.post(
        "/analyze-image", files={"file": ("f.gif", b"not an image", "image/gif")}
    )
    assert response.status_code == 415


def test_analyze_image_rejects_corrupt_image(client):
    response = client.post(
        "/analyze-image", files={"file": ("f.jpg", b"\xff\xd8\xff\x00garbage", "image/jpeg")}
    )
    assert response.status_code == 400


def test_analyze_image_happy_path(client, monkeypatch):
    fake_result = {
        "faces": [
            {
                "bbox": {"x": 0.1, "y": 0.1, "width": 0.3, "height": 0.3},
                "detection_confidence": 0.98,
                "fake_probability": 0.87,
                "classification": "fake",
                "quality_issue": None,
            }
        ],
        "overall_classification": "fake",
        "overall_confidence": 0.87,
    }
    monkeypatch.setattr(main, "analyze_image", lambda img: fake_result)
    monkeypatch.setattr(main, "get_model_info", lambda: {"model_version": "test-model"})

    response = client.post(
        "/analyze-image", files={"file": ("f.jpg", jpeg_bytes(), "image/jpeg")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["overall_classification"] == "fake"
    assert len(body["faces"]) == 1
    assert body["model_version"] == "test-model"
    assert body["processing_time_ms"] >= 0
    assert body["request_id"]


def test_analyze_image_returns_503_when_model_fails_to_load(client, monkeypatch):
    def raise_load_error(img):
        raise ModelLoadError("checkpoint missing")

    monkeypatch.setattr(main, "analyze_image", raise_load_error)
    response = client.post(
        "/analyze-image", files={"file": ("f.jpg", jpeg_bytes(), "image/jpeg")}
    )
    assert response.status_code == 503


def test_analyze_video_rejects_unsupported_content_type(client):
    response = client.post(
        "/analyze-video", files={"file": ("f.avi", b"not a video", "video/x-msvideo")}
    )
    assert response.status_code == 415


def test_analyze_video_rejects_oversized_upload(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_VIDEO_UPLOAD_BYTES", 10)
    response = client.post(
        "/analyze-video", files={"file": ("f.mp4", b"x" * 100, "video/mp4")}
    )
    assert response.status_code == 413


def test_analyze_video_rejects_empty_upload(client):
    response = client.post(
        "/analyze-video", files={"file": ("f.mp4", b"", "video/mp4")}
    )
    assert response.status_code == 400


def test_analyze_video_happy_path(client, monkeypatch):
    fake_result = {
        "metadata": {
            "duration_seconds": 5.0, "fps": 24.0, "width": 640, "height": 480,
            "total_frames": 120, "frames_analyzed": 5,
        },
        "frames": [{"timestamp": 0.0, "fake_probability": 0.2, "classification": "real"}],
        "overall_classification": "real",
        "overall_confidence": 0.2,
        "suspicious_segments": [],
    }
    monkeypatch.setattr(main, "analyze_video", lambda path: fake_result)
    monkeypatch.setattr(main, "get_model_info", lambda: {"model_version": "test-model"})

    response = client.post(
        "/analyze-video", files={"file": ("f.mp4", b"fake mp4 bytes", "video/mp4")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["overall_classification"] == "real"
    assert body["metadata"]["frames_analyzed"] == 5
    assert body["model_version"] == "test-model"
    assert body["request_id"]


def test_analyze_video_decode_error_returns_400(client, monkeypatch):
    from video_pipeline import VideoDecodeError

    def raise_decode_error(path):
        raise VideoDecodeError("Could not open video file")

    monkeypatch.setattr(main, "analyze_video", raise_decode_error)
    response = client.post(
        "/analyze-video", files={"file": ("f.mp4", b"not really a video", "video/mp4")}
    )
    assert response.status_code == 400


def test_analyze_video_too_long_returns_413(client, monkeypatch):
    from video_pipeline import VideoTooLongError

    def raise_too_long(path):
        raise VideoTooLongError("video exceeds the 300s duration limit")

    monkeypatch.setattr(main, "analyze_video", raise_too_long)
    response = client.post(
        "/analyze-video", files={"file": ("f.mp4", b"not really a video", "video/mp4")}
    )
    assert response.status_code == 413


def test_analyze_ui_route_serves_html(client):
    response = client.get("/analyze-ui")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
