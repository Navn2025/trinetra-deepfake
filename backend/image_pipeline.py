"""
Standalone image analysis for the upload dashboard: unlike /predict (which
only ever receives one pre-cropped face from the real-time clients), this
takes an arbitrary uncropped photo, detects every face in it, and scores
each one independently.

Synchronous end to end (detection + one model call per face) -- callers on
an asyncio event loop must run analyze_image via run_in_threadpool, same as
predictor.predict_fake_probability itself.
"""
from PIL import Image

from config import classify_fake_probability
from face_detection import crop_face, detect_faces, normalized_bbox
from predictor import predict_fake_probability
from preprocessing import assess_face_crop_quality


def analyze_image(image: Image.Image) -> dict:
    faces = detect_faces(image)
    if not faces:
        return {
            "faces": [],
            "overall_classification": "no_face_detected",
            "overall_confidence": None,
        }

    face_results = [_analyze_face(image, face) for face in faces]
    overall = _aggregate(face_results)
    return {"faces": face_results, **overall}


def _analyze_face(image: Image.Image, face: dict) -> dict:
    x1, y1, x2, y2 = face["bbox"]
    # Quality-gate on the true, un-resized source region first -- this is
    # the one caller in the whole system where assess_face_crop_quality's
    # pixel-dimension check can actually fire (the real-time /predict path
    # only ever sees an already-256x256-resized crop; see that function's
    # docstring). The blur check still runs on the final model input too,
    # since resizing itself can introduce/hide blur.
    source_region = image.crop(
        (max(0, int(x1)), max(0, int(y1)), max(int(x1) + 1, int(x2)), max(int(y1) + 1, int(y2)))
    )
    crop = crop_face(image, face["bbox"])

    quality_issue = assess_face_crop_quality(source_region) or assess_face_crop_quality(crop)
    fake_probability = None if quality_issue is not None else predict_fake_probability(crop)

    return {
        "bbox": normalized_bbox(face["bbox"], image.size),
        "detection_confidence": round(face["det_score"], 3),
        "fake_probability": fake_probability,
        "classification": classify_fake_probability(fake_probability),
        "quality_issue": quality_issue,
    }


def _aggregate(face_results: list[dict]) -> dict:
    scored = [f["fake_probability"] for f in face_results if f["fake_probability"] is not None]
    if not scored:
        return {"overall_classification": "insufficient_quality", "overall_confidence": None}

    # Worst-case-wins: for a forensic/security tool, one confidently-fake
    # face in a group photo is the finding -- it shouldn't get diluted by
    # averaging against other real faces in the same frame.
    worst = max(scored)
    return {
        "overall_classification": classify_fake_probability(worst),
        "overall_confidence": round(worst, 3),
    }
