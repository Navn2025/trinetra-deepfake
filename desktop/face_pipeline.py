"""
Face detection on a captured desktop frame -- MediaPipe Tasks Vision
(BlazeFace, short-range), the same model used by the browser extension
(extension/models/blaze_face_short_range.tflite, copied into
desktop/models/ here).

Unlike the browser extension (one <video> element = one participant tile,
so "detect the single largest face" was the right rule), a captured desktop
window can show a full call layout with several participants visible at
once (e.g. Zoom gallery view) -- so this returns every detected face, not
just the largest.
"""
from pathlib import Path

import cv2
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat

MODEL_PATH = Path(__file__).resolve().parent / "models" / "blaze_face_short_range.tflite"

FACE_CROP_SIZE = 256  # matches DeepfakeBench's face-crop resolution
FACE_MARGIN_RATIO = 0.3  # padding added around the detected face box
CAPTURE_IS_MIRRORED = True


def create_face_detector() -> vision.FaceDetector:
    options = vision.FaceDetectorOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH),
            # CPU delegate for reliability across Windows/driver setups --
            # detection itself is cheap; the heavy model (Xception) already
            # runs on GPU in the backend. Switch to BaseOptions.Delegate.GPU
            # here if you want to try GPU-accelerated face detection too.
            delegate=BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=0.5,
    )
    return vision.FaceDetector.create_from_options(options)


def detect_faces(detector: vision.FaceDetector, frame_bgra: np.ndarray) -> list[dict]:
    """
    Runs face detection on a BGRA frame (as produced by windows_capture's
    Frame.frame_buffer) and returns every detected face as a dict with:
      - bbox_fraction: (x, y, w, h) normalized 0->1 relative to the frame,
        for mapping onto the target window's on-screen rect later
      - crop: an outputSize x outputSize BGR numpy array ready to send to
        the backend
    """
    height, width = frame_bgra.shape[:2]
    analysis_frame = cv2.flip(frame_bgra, 1) if CAPTURE_IS_MIRRORED else frame_bgra
    rgb = cv2.cvtColor(analysis_frame, cv2.COLOR_BGRA2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)

    faces = []
    for detection in result.detections:
        box = detection.bounding_box

        cx = box.origin_x + box.width / 2
        cy = box.origin_y + box.height / 2
        side = max(box.width, box.height) * (1 + FACE_MARGIN_RATIO * 2)

        sx = int(round(cx - side / 2))
        sy = int(round(cy - side / 2))
        s = int(round(side))

        sx = max(0, min(sx, width - 1))
        sy = max(0, min(sy, height - 1))
        s = min(s, width - sx, height - sy)
        if s <= 0:
            continue

        bgr_frame = cv2.cvtColor(analysis_frame, cv2.COLOR_BGRA2BGR)
        cropped = bgr_frame[sy : sy + s, sx : sx + s]
        crop = cv2.resize(cropped, (FACE_CROP_SIZE, FACE_CROP_SIZE))

        faces.append(
            {
                "bbox_fraction": (
                    box.origin_x / width,
                    box.origin_y / height,
                    box.width / width,
                    box.height / height,
                ),
                "crop": crop,
            }
        )

    return faces
