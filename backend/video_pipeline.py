"""
Standalone video analysis for the upload dashboard.

Pipeline: decode -> sample frames evenly across the timeline (not every
frame -- see config.VIDEO_SAMPLE_FPS/MAX_VIDEO_SAMPLE_FRAMES) -> detect the
largest face per sampled frame (same one-primary-subject-per-frame
convention the real-time extension uses for a video tile) -> score it ->
aggregate across time with outlier rejection (median, not mean) and
minimum-run-length suspicious-segment detection, so one anomalous frame
can't swing the whole video's classification.

Synchronous end to end -- callers on an asyncio event loop must run
analyze_video via run_in_threadpool.

Each sampled frame also keeps a downsized snapshot + the detected face's
normalized bbox around just for evidence purposes: the response surfaces a
JPEG + bbox for the single most suspicious frame overall (most_suspicious_frame)
and for the peak frame of each suspicious_segments entry, so the caller can
show *which* frame and *where in it* the model reacted to, not just a
timestamp and a number.
"""
import base64
import io
import logging
import time
from pathlib import Path

import cv2
from PIL import Image

from config import (
    FAKE_THRESHOLD,
    FRAME_SNAPSHOT_MAX_SIDE,
    MAX_VIDEO_DURATION_SECONDS,
    MAX_VIDEO_SAMPLE_FRAMES,
    MIN_SUSPICIOUS_RUN_FRAMES,
    MIN_VIDEO_SAMPLE_FRAMES,
    VIDEO_SAMPLE_FPS,
    classify_fake_probability,
)
from face_detection import crop_face, detect_faces, normalized_bbox
from predictor import predict_fake_probabilities_batch
from preprocessing import assess_face_crop_quality

logger = logging.getLogger(__name__)


class VideoDecodeError(RuntimeError):
    """Raised when the file isn't a decodable video (corrupt, unsupported
    codec, or not actually a video)."""


class VideoTooLongError(VideoDecodeError):
    """Raised when the video decodes fine but exceeds MAX_VIDEO_DURATION_SECONDS."""


def analyze_video(video_path: Path) -> dict:
    overall_start = time.monotonic()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise VideoDecodeError("Could not open video file (corrupt or unsupported codec)")

    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        if total_frames <= 0 or fps <= 0:
            raise VideoDecodeError("Video has no readable frames (corrupt or empty)")

        duration_seconds = total_frames / fps
        if duration_seconds > MAX_VIDEO_DURATION_SECONDS:
            # Checked before any frame is actually decoded/scored -- no
            # point spending GPU time on a video we're going to reject.
            raise VideoTooLongError(
                f"video exceeds the {MAX_VIDEO_DURATION_SECONDS:.0f}s duration limit "
                f"(got {duration_seconds:.0f}s)"
            )

        frame_indices = _sample_frame_indices(total_frames, duration_seconds)
        logger.info(
            "analyzing video: %.1fs duration, %dx%d, sampling %d of %d frames "
            "(first request also pays a one-time model-load cost -- see logs above)",
            duration_seconds, width, height, len(frame_indices), total_frames,
        )

        # Stage 1: decode + detect + quality-gate every sampled frame. Kept
        # sequential (cv2.VideoCapture isn't thread-safe) and fast per frame
        # now that detect_faces() only runs InsightFace's detection head
        # (see face_detection.py) instead of the full landmark/recognition
        # pipeline.
        pending = []  # [{"timestamp", "crop": Image|None, "classification": str|None, "bbox": dict|None, "snapshot": Image|None}]
        for i, frame_index in enumerate(frame_indices, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, bgr_frame = capture.read()
            if not ok:
                logger.warning("frame %d/%d (index %d) unreadable, skipping", i, len(frame_indices), frame_index)
                continue  # a single unreadable frame doesn't abort the whole analysis

            timestamp = round(frame_index / fps, 2)
            stage_start = time.monotonic()
            crop, forced_classification, bbox, snapshot = _detect_and_crop(bgr_frame)
            logger.info(
                "frame %d/%d @ %.2fs: %s (%.2fs)",
                i, len(frame_indices), timestamp,
                forced_classification or "face found, queued for scoring",
                time.monotonic() - stage_start,
            )
            pending.append({
                "timestamp": timestamp,
                "crop": crop,
                "classification": forced_classification,
                "bbox": bbox,
                "snapshot": snapshot,
            })
    finally:
        capture.release()

    # Stage 2: score every queued crop in one batched ensemble call instead
    # of one model-forward round trip per frame (predictor.py logs its own
    # timing for this).
    scorable = [p for p in pending if p["crop"] is not None]
    probabilities = predict_fake_probabilities_batch([p["crop"] for p in scorable])
    for p, probability in zip(scorable, probabilities):
        p["fake_probability"] = probability
        p["classification"] = classify_fake_probability(probability)

    frame_results = [
        {
            "timestamp": p["timestamp"],
            "fake_probability": p.get("fake_probability"),
            "classification": p["classification"],
            "bbox": p["bbox"],
        }
        for p in pending
    ]

    aggregate = _aggregate(frame_results)

    # Evidence: the single most suspicious sampled frame (highest
    # fake_probability, regardless of whether it actually crossed
    # FAKE_THRESHOLD -- like the video-level score itself, this is "the
    # frame the model reacted to most," not a certified verdict), plus a
    # peak-frame snapshot for each already-detected suspicious segment.
    scored_pending = [p for p in pending if p.get("fake_probability") is not None]
    most_suspicious_frame = None
    if scored_pending:
        peak = max(scored_pending, key=lambda p: p["fake_probability"])
        most_suspicious_frame = _make_frame_evidence(peak)

    for segment in aggregate["suspicious_segments"]:
        run = [
            p for p in scored_pending
            if segment["start_timestamp"] <= p["timestamp"] <= segment["end_timestamp"]
        ]
        peak = max(run, key=lambda p: p["fake_probability"]) if run else None
        segment["peak_frame"] = _make_frame_evidence(peak) if peak else None

    logger.info(
        "video analysis done in %.1fs: overall=%s confidence=%s",
        time.monotonic() - overall_start, aggregate["overall_classification"], aggregate["overall_confidence"],
    )
    return {
        "metadata": {
            "duration_seconds": round(duration_seconds, 2),
            "fps": round(fps, 2),
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "frames_analyzed": len(frame_results),
        },
        "frames": frame_results,
        "most_suspicious_frame": most_suspicious_frame,
        **aggregate,
    }


def _make_frame_evidence(pending_item: dict) -> dict | None:
    """pending_item -> {"timestamp", "fake_probability", "bbox", "thumbnail"}
    (base64 JPEG of the downsized full frame), or None if no snapshot was
    captured for it (e.g. no face detected on that frame)."""
    snapshot = pending_item.get("snapshot")
    if snapshot is None:
        return None
    buffer = io.BytesIO()
    snapshot.convert("RGB").save(buffer, format="JPEG", quality=80)
    return {
        "timestamp": pending_item["timestamp"],
        "fake_probability": pending_item.get("fake_probability"),
        "bbox": pending_item["bbox"],
        "thumbnail": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


def _sample_frame_indices(total_frames: int, duration_seconds: float) -> list[int]:
    target_count = round(duration_seconds * VIDEO_SAMPLE_FPS)
    target_count = max(MIN_VIDEO_SAMPLE_FRAMES, min(target_count, MAX_VIDEO_SAMPLE_FRAMES))
    target_count = min(target_count, total_frames)

    if target_count <= 1:
        return [0]

    step = (total_frames - 1) / (target_count - 1)
    return sorted({round(i * step) for i in range(target_count)})


def _detect_and_crop(
    bgr_frame,
) -> tuple[Image.Image | None, str | None, dict | None, Image.Image | None]:
    """Face-detection + quality-gate stage for one frame -- everything
    _analyze_frame used to do except the actual model scoring, which the
    caller now batches across every sampled frame. Returns
    (crop, classification, bbox, snapshot):
      - crop: the model-input face crop, or None if this frame won't be scored
      - classification: forced classification (no_face_detected /
        insufficient_quality) when crop is None, else None (caller fills
        this in after scoring)
      - bbox: normalized {x,y,width,height} of the detected face in the
        *full frame* (not the crop), or None if no face was found -- used
        for the evidence thumbnails' bounding box
      - snapshot: a downsized copy of the full frame (not just the face
        crop), or None if no face was found -- used as the evidence
        thumbnail image itself, encoded lazily (only for frames actually
        selected as most-suspicious/peak) to avoid holding/encoding a JPEG
        for every sampled frame
    """
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame)

    faces = detect_faces(image)
    if not faces:
        return None, "no_face_detected", None, None

    largest = faces[0]  # detect_faces() returns largest-first
    x1, y1, x2, y2 = largest["bbox"]
    source_region = image.crop(
        (max(0, int(x1)), max(0, int(y1)), max(int(x1) + 1, int(x2)), max(int(y1) + 1, int(y2)))
    )
    crop = crop_face(image, largest["bbox"])
    bbox = normalized_bbox(largest["bbox"], image.size)

    snapshot = image.copy()
    snapshot.thumbnail((FRAME_SNAPSHOT_MAX_SIDE, FRAME_SNAPSHOT_MAX_SIDE))

    quality_issue = assess_face_crop_quality(source_region) or assess_face_crop_quality(crop)
    if quality_issue is not None:
        logger.info(
            "  rejected: bbox=%dx%d (det_score=%.2f) source_region=%s crop=%s -> %s",
            int(x2 - x1), int(y2 - y1), largest["det_score"], source_region.size, crop.size, quality_issue,
        )
        return None, "insufficient_quality", bbox, snapshot
    return crop, None, bbox, snapshot


def _aggregate(frame_results: list[dict]) -> dict:
    scored = [f for f in frame_results if f["fake_probability"] is not None]
    if not scored:
        return {
            "overall_classification": "no_face_detected" if not frame_results or all(
                f["classification"] == "no_face_detected" for f in frame_results
            ) else "insufficient_quality",
            "overall_confidence": None,
            "suspicious_segments": [],
        }

    probabilities = sorted(f["fake_probability"] for f in scored)
    n = len(probabilities)
    median = (
        probabilities[n // 2]
        if n % 2 == 1
        else (probabilities[n // 2 - 1] + probabilities[n // 2]) / 2
    )

    return {
        "overall_classification": classify_fake_probability(median),
        "overall_confidence": round(median, 3),
        "suspicious_segments": _find_suspicious_segments(frame_results),
    }


def _find_suspicious_segments(frame_results: list[dict]) -> list[dict]:
    """Contiguous runs (in sampled-frame order) of fake_probability >=
    FAKE_THRESHOLD, at least MIN_SUSPICIOUS_RUN_FRAMES long -- a single
    spiking frame (motion blur, a hand passing in front of the face) is
    deliberately not enough to report a segment."""
    segments = []
    run: list[dict] = []

    def flush():
        if len(run) >= MIN_SUSPICIOUS_RUN_FRAMES:
            segments.append(
                {
                    "start_timestamp": run[0]["timestamp"],
                    "end_timestamp": run[-1]["timestamp"],
                    "peak_fake_probability": round(max(f["fake_probability"] for f in run), 3),
                    "frame_count": len(run),
                }
            )

    for frame in frame_results:
        prob = frame["fake_probability"]
        if prob is not None and prob >= FAKE_THRESHOLD:
            run.append(frame)
        else:
            flush()
            run = []
    flush()

    return segments
