"""
Face identity embeddings for contact verification.

Uses InsightFace's ArcFace recognition model (the "buffalo_l" model pack)
via onnxruntime -- pure Python + prebuilt wheels, no C++ compiler needed
(dlib was tried first and hit the same pip-build problems seen earlier in
this project on Windows; InsightFace's onnxruntime-based models sidestep
that entirely). Runs on CPU: this only executes once per ~2s prediction
cycle, so GPU isn't needed here the way it is for the main Xception
classifier.

Model weights (~350MB) auto-download on first use, cached under
~/.insightface/.
"""
import logging
import threading
import time

import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image

logger = logging.getLogger(__name__)

_face_app = None
_load_lock = threading.Lock()


def _get_face_app() -> FaceAnalysis:
    global _face_app
    if _face_app is not None:
        return _face_app

    with _load_lock:
        if _face_app is None:
            logger.info(
                "loading InsightFace buffalo_l model pack (first use only; "
                "downloads ~350MB to ~/.insightface/ if not already cached -- "
                "this can take a while on a slow connection)"
            )
            start = time.monotonic()
            app = FaceAnalysis(name="buffalo_l")
            app.prepare(ctx_id=-1)  # -1 = CPU
            _face_app = app
            logger.info("InsightFace buffalo_l ready in %.1fs", time.monotonic() - start)
        return _face_app


def get_face_embedding(image: Image.Image) -> np.ndarray | None:
    """Returns a 512-d embedding for the largest face in `image`, or None
    if no face could be detected/aligned."""
    app = _get_face_app()
    bgr = np.array(image.convert("RGB"))[:, :, ::-1]  # PIL RGB -> OpenCV BGR

    faces = app.get(bgr)
    if not faces:
        return None

    largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return largest.embedding


def average_embeddings(embeddings: list[np.ndarray]) -> np.ndarray:
    """Combines multiple enrollment-photo embeddings (e.g. a short burst of
    frames capturing slightly different angles/expressions) into one more
    robust reference embedding: L2-normalize each, average, re-normalize."""
    normalized = [e / np.linalg.norm(e) for e in embeddings]
    mean = np.mean(normalized, axis=0)
    return mean / np.linalg.norm(mean)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
