"""
image_pipeline aggregation/orchestration logic, with face detection and the
model itself mocked out (already covered for real by test_face_detection.py's
crop math, test_model_integration.py's real inference, and manual
verification against a real synthetic non-face image -- see the session that
added this module)."""
from PIL import Image

import image_pipeline


def _face(bbox=(10, 10, 110, 110), det_score=0.9):
    return {"bbox": bbox, "det_score": det_score}


def test_no_faces_detected_returns_no_face_detected(monkeypatch):
    monkeypatch.setattr(image_pipeline, "detect_faces", lambda img: [])
    result = image_pipeline.analyze_image(Image.new("RGB", (200, 200)))
    assert result["faces"] == []
    assert result["overall_classification"] == "no_face_detected"
    assert result["overall_confidence"] is None


def test_single_confidently_fake_face(monkeypatch):
    monkeypatch.setattr(image_pipeline, "detect_faces", lambda img: [_face()])
    monkeypatch.setattr(image_pipeline, "assess_face_crop_quality", lambda img: None)
    monkeypatch.setattr(image_pipeline, "predict_fake_probability", lambda crop: 0.92)

    result = image_pipeline.analyze_image(Image.new("RGB", (200, 200)))
    assert len(result["faces"]) == 1
    assert result["faces"][0]["classification"] == "fake"
    assert result["overall_classification"] == "fake"
    assert result["overall_confidence"] == 0.92


def test_worst_case_face_determines_overall_classification(monkeypatch):
    # Two faces: one clearly real, one clearly fake -- the photo as a whole
    # must be flagged fake (a manipulated face in a group photo is the
    # finding, not something to average away).
    faces = [_face(bbox=(0, 0, 50, 50)), _face(bbox=(100, 100, 150, 150))]
    monkeypatch.setattr(image_pipeline, "detect_faces", lambda img: faces)
    monkeypatch.setattr(image_pipeline, "assess_face_crop_quality", lambda img: None)

    probs = iter([0.05, 0.95])
    monkeypatch.setattr(image_pipeline, "predict_fake_probability", lambda crop: next(probs))

    result = image_pipeline.analyze_image(Image.new("RGB", (300, 300)))
    assert len(result["faces"]) == 2
    assert result["overall_classification"] == "fake"
    assert result["overall_confidence"] == 0.95


def test_quality_rejected_face_is_not_scored(monkeypatch):
    monkeypatch.setattr(image_pipeline, "detect_faces", lambda img: [_face()])
    monkeypatch.setattr(image_pipeline, "assess_face_crop_quality", lambda img: "too blurry")

    calls = []
    monkeypatch.setattr(image_pipeline, "predict_fake_probability", lambda crop: calls.append(crop) or 0.5)

    result = image_pipeline.analyze_image(Image.new("RGB", (200, 200)))
    assert calls == []  # model must never run on a quality-rejected face
    assert result["faces"][0]["fake_probability"] is None
    assert result["faces"][0]["classification"] == "insufficient_quality"
    assert result["overall_classification"] == "insufficient_quality"


def test_all_faces_quality_rejected_gives_insufficient_quality_overall(monkeypatch):
    monkeypatch.setattr(image_pipeline, "detect_faces", lambda img: [_face(), _face(bbox=(150, 150, 160, 160))])
    monkeypatch.setattr(image_pipeline, "assess_face_crop_quality", lambda img: "too small")
    monkeypatch.setattr(image_pipeline, "predict_fake_probability", lambda crop: 0.5)

    result = image_pipeline.analyze_image(Image.new("RGB", (300, 300)))
    assert result["overall_classification"] == "insufficient_quality"
    assert result["overall_confidence"] is None
