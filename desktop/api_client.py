"""
Talks to the same backend the browser extension uses
(meet-deepfake-detector/backend/main.py) -- no backend changes needed for
the desktop app, it's just another client of the same POST /predict
endpoint.
"""
import cv2
import numpy as np
import requests

API_URL = "http://127.0.0.1:8000"
# 5s was too tight for the very first identity-match request: the backend
# cold-loads InsightFace's full recognition model pack (detection +
# landmark + genderage + recognition) on first use, which alone measured
# 6-10s in testing and blew past the old timeout as a ReadTimeout. Raised
# with headroom above that; every request after the first one stays fast
# since the model's cached in the backend process.
TIMEOUT_SECONDS = 15


def predict_face(face_bgr: np.ndarray, platform: str | None = None) -> dict | None:
    ok, encoded = cv2.imencode(".jpg", face_bgr)
    if not ok:
        return None

    try:
        response = requests.post(
            f"{API_URL}/predict",
            files={"file": ("face.jpg", encoded.tobytes(), "image/jpeg")},
            data={"platform": platform} if platform else {},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as err:
        print(f"[API] Prediction failed: {err}")
        return None
