"""
Centralized configuration for the backend. Values that were previously
hardcoded/duplicated across main.py, predictor.py, preprocessing.py, etc.
live here so they can be tuned or overridden via environment variables in
one place.

Nothing here changes model behavior by itself -- INPUT_SIZE/MEAN/STD still
come from preprocessing.py (tied directly to the DeepfakeBench xception.yaml
that trained the checkpoint) and must not be edited casually; see that
file's docstring.
"""
import os


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value else default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


# -- Upload limits / validation ---------------------------------------------
# The extension only ever sends small (256x256-ish) JPEG face crops -- this
# cap is generous headroom above real usage, meant to stop abuse/DoS, not to
# constrain normal use.
MAX_IMAGE_UPLOAD_BYTES = _env_int("MAX_IMAGE_UPLOAD_BYTES", 8 * 1024 * 1024)  # 8 MB
# Video is the one upload type where "a real user's phone-recorded clip"
# and "someone trying to make the server churn for minutes" are hard to
# tell apart from size/duration alone -- these caps bound the worst case
# for a single-machine local tool, not a tuned production limit.
MAX_VIDEO_UPLOAD_BYTES = _env_int("MAX_VIDEO_UPLOAD_BYTES", 200 * 1024 * 1024)  # 200 MB
MAX_VIDEO_DURATION_SECONDS = _env_float("MAX_VIDEO_DURATION_SECONDS", 300.0)  # 5 min

ALLOWED_IMAGE_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}
# Maps an allowed video content-type to a fixed, safe file extension for the
# temp file cv2.VideoCapture reads from -- never derived from the
# client-supplied filename.
ALLOWED_VIDEO_CONTENT_TYPES = {
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
    "video/x-matroska": ".mkv",
}

# -- Face-crop quality gate ---------------------------------------------
# Below this many source pixels on a side, a crop is mostly upsampling
# noise by the time it's resized to the model's 256x256 input -- the
# model will still happily emit a confident-looking number for it, which
# is worse than not answering. This is a coarse, deliberately conservative
# floor (not derived from a validation sweep -- there is no labeled
# dataset of "real crops at various resolutions" in this project to tune
# it against), not a calibrated quality score.
MIN_FACE_CROP_PX = _env_int("MIN_FACE_CROP_PX", 64)

# Laplacian-variance sharpness floor (see preprocessing.assess_face_crop_quality
# for why this -- not the pixel-dimension check above -- is what actually
# catches an upsampled/blurry crop on the real-time path, where both clients
# always resize to a fixed 256x256 before sending). Originally calibrated
# (scripts/calibrate_sharpness.py) only against synthetic upsampling cases:
# a 16x16 source face upsampled 16x to 256px measures ~5, a 32x32 source
# upsampled 8x measures ~25, a solid/near-blank crop measures ~0, while
# genuine sharp photo content measures in the hundreds to tens of thousands.
# That calibration never accounted for ordinary compressed video, though --
# the dashboard's own video pipeline (video_pipeline.py) logs showed
# legitimate, clearly-a-face frames from real H.264 news clips measuring
# 26-40, i.e. sitting right on top of the "bad" synthetic-upsampling range
# above and getting rejected as insufficient_quality on every single sampled
# frame. Lowered to sit below that observed real-video range while still
# well above the near-zero/blank-crop floor -- still a coarse, unswept
# floor, just no longer tuned exclusively for sharp photos.
MIN_SHARPNESS_VARIANCE = _env_float("MIN_SHARPNESS_VARIANCE", 15.0)

# -- Classification thresholds -------------------------------------------
# Mirrors the thresholds baked into extension/overlay.js and
# static/dashboard.html (both do their own client-side smoothing/labeling
# over the raw fake_probability this backend returns). Kept here too so the
# API response's own `classification` field is computed consistently
# server-side, and so a single place documents where "0.65 / 0.35" comes
# from: informal manual testing during development, not a validated
# operating point derived from a labeled precision/recall sweep -- there is
# no such held-out evaluation set in this project. Treat these as a
# reasonable starting point to tune, not a proven threshold.
#
# When predictor.py was swapped from plain Xception to DeepfakeBench's UCF
# detector, and later to a UCF+SPSL+Xception ensemble (see predictor.py's
# module docstring for why -- no single published detector generalizes to
# unseen manipulation methods, ensembling narrows but does not close that
# gap), these were re-checked each time against the same ~120 labeled
# FaceForensics++ frames (real + 5 manipulation methods, including
# FaceShifter, held out of UCF's own training set as an unseen-method check)
# and left unchanged both times -- sweeps found no clearly better operating
# point. At 0.65/0.35 the ensemble keeps the same ~5% real false-positive
# rate UCF alone had, while cutting FaceShifter false negatives (fakes
# silently scored "real") from 9/12 to 3/12 -- most of the rest now land in
# "uncertain" instead of being wrong with confidence, which is the outcome
# that actually matters for a security tool. This is still a small
# single-dataset sample, not a proper precision/recall calibration.
FAKE_THRESHOLD = _env_float("FAKE_THRESHOLD", 0.65)
REAL_THRESHOLD = _env_float("REAL_THRESHOLD", 0.35)

# -- Identity-match thresholds -------------------------------------------
IDENTITY_MATCH_THRESHOLD = _env_float("IDENTITY_MATCH_THRESHOLD", 0.45)

THUMBNAIL_SIZE = _env_int("THUMBNAIL_SIZE", 96)

# -- Video sampling ---------------------------------------------------
# Offline uploaded videos can afford heavier sampling than the live
# real-time path (which is capped by per-frame latency, not total budget) --
# but still bounded, since this runs synchronously behind one HTTP request.
# Target ~1 sampled frame/sec of video, evenly spaced, clamped to
# [MIN, MAX] regardless of the video's native frame rate or duration.
VIDEO_SAMPLE_FPS = _env_float("VIDEO_SAMPLE_FPS", 1.0)
MIN_VIDEO_SAMPLE_FRAMES = _env_int("MIN_VIDEO_SAMPLE_FRAMES", 5)
MAX_VIDEO_SAMPLE_FRAMES = _env_int("MAX_VIDEO_SAMPLE_FRAMES", 40)

# A single sampled frame scoring above FAKE_THRESHOLD doesn't make a
# "suspicious segment" on its own (camera motion/blur can spike one frame) --
# require this many consecutive sampled frames before reporting a segment.
MIN_SUSPICIOUS_RUN_FRAMES = _env_int("MIN_SUSPICIOUS_RUN_FRAMES", 2)


def classify_fake_probability(fake_probability: float | None) -> str:
    """Maps a raw fake_probability to one of the four decision states the
    system commits to: never force an ambiguous score into REAL or FAKE."""
    if fake_probability is None:
        return "insufficient_quality"
    if fake_probability >= FAKE_THRESHOLD:
        return "fake"
    if fake_probability <= REAL_THRESHOLD:
        return "real"
    return "uncertain"
