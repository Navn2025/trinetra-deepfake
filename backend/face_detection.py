"""
Multi-face detection for the upload dashboard (image/video analysis).

The real-time extension/desktop clients do their own face detection
client-side (MediaPipe) and only ever send an already-cropped face to
/predict. The dashboard accepts raw, uncropped photos/video frames, so the
backend needs its own detector here.

Uses its own detection-only InsightFace FaceAnalysis app (buffalo_l,
allowed_modules=["detection"]) rather than sharing face_embedding.py's
full app: that one also runs the landmark/genderage/recognition heads on
every detected face because it needs the recognition embedding for contact
matching. Dashboard analysis only ever reads bbox/det_score, so running
those extra heads on every sampled video frame would be pure waste --
restricting modules here noticeably cuts per-frame detection time (video
analysis calls this once per sampled frame).
"""
import logging
import threading
import time

import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image

logger = logging.getLogger(__name__)

FACE_CROP_MARGIN_RATIO = 0.3  # matches extension/frame-capture.js cropFaceFromCanvas

_detector_app = None
_load_lock = threading.Lock()


def _get_detector_app() -> FaceAnalysis:
    global _detector_app
    if _detector_app is not None:
        return _detector_app

    with _load_lock:
        if _detector_app is None:
            logger.info("loading InsightFace detection-only model (buffalo_l/det_10g)")
            start = time.monotonic()
            app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
            app.prepare(ctx_id=-1)  # -1 = CPU
            _detector_app = app
            logger.info("face detector ready in %.1fs", time.monotonic() - start)
        return _detector_app


def detect_faces(image: Image.Image) -> list[dict]:
    """Returns every detected face as
    {"bbox": (x1, y1, x2, y2) in pixel coords, "det_score": float},
    largest first. Empty list if no face is found."""
    app = _get_detector_app()
    bgr = np.array(image.convert("RGB"))[:, :, ::-1]  # PIL RGB -> OpenCV BGR

    faces = app.get(bgr)
    results = [
        {
            "bbox": tuple(float(v) for v in face.bbox),
            "det_score": float(face.det_score),
        }
        for face in faces
    ]
    results.sort(key=lambda f: _area(f["bbox"]), reverse=True)
    return results


def _area(bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def crop_face(image: Image.Image, bbox: tuple[float, float, float, float], output_size: int = 256) -> Image.Image:
    """Square, margin-padded crop around `bbox`, resized to output_size --
    the same shape DeepfakeBench's own preprocessing uses (see
    preprocessing.py), mirroring extension/frame-capture.js's
    cropFaceFromCanvas so a dashboard-uploaded photo is treated the same way
    a live video-call frame would be."""
    sw, sh = image.size
    x1, y1, x2, y2 = bbox
    box_w, box_h = x2 - x1, y2 - y1
    cx, cy = x1 + box_w / 2, y1 + box_h / 2
    side = max(box_w, box_h) * (1 + FACE_CROP_MARGIN_RATIO * 2)

    sx = int(round(cx - side / 2))
    sy = int(round(cy - side / 2))
    s = int(round(side))

    sx = max(0, min(sx, sw - 1))
    sy = max(0, min(sy, sh - 1))
    s = max(1, min(s, sw - sx, sh - sy))

    cropped = image.crop((sx, sy, sx + s, sy + s))
    return cropped.resize((output_size, output_size))


def normalized_bbox(bbox: tuple[float, float, float, float], image_size: tuple[int, int]) -> dict:
    """Pixel bbox -> {x, y, width, height} normalized 0->1, the same
    contract detectFaces() returns client-side (face-detector.js) so the
    dashboard can draw boxes with the same math the extension uses."""
    width, height = image_size
    x1, y1, x2, y2 = bbox
    return {
        "x": max(0.0, x1 / width),
        "y": max(0.0, y1 / height),
        "width": min(1.0, (x2 - x1) / width),
        "height": min(1.0, (y2 - y1) / height),
    }
