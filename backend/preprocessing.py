"""
Face-crop quality gate shared by main.py/image_pipeline.py/video_pipeline.py.

Model-input resizing/normalization is no longer done here -- predictor.py's
CLIPBackbone (clip_backbone.py) takes raw uint8 RGB crops and handles its own
resize + normalization internally, reading the normalization constants from
the checkpoint's own open_clip transform rather than a hardcoded config.
"""
import cv2
import numpy as np
from PIL import Image

from config import MIN_FACE_CROP_PX, MIN_SHARPNESS_VARIANCE


def sharpness_variance(image: Image.Image) -> float:
    """Variance of the Laplacian -- a standard, cheap focus/blur measure:
    a sharp image has lots of high-frequency edge content (high variance),
    a blurry/upsampled one is smoothed out (low variance)."""
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def assess_face_crop_quality(image: Image.Image) -> str | None:
    """Pre-inference quality gate. Returns None if the crop is fine to run
    through the model, or a short reason string if it should be rejected as
    insufficient quality.

    Two independent checks, because neither alone is reliable here:
      - Pixel dimensions (config.MIN_FACE_CROP_PX): catches a genuinely tiny
        source crop -- but both real-time clients (extension, desktop) always
        resize their crop to a fixed 256x256 before sending it, so by the
        time this function sees it in that path, .size is always 256x256
        regardless of how small the actual source face was. This check only
        does real work for callers that send an unresized crop (e.g. the
        dashboard's own pipeline, which checks the source bbox before
        cropping/resizing -- see image_pipeline.py).
      - Sharpness (config.MIN_SHARPNESS_VARIANCE): a face upsampled from a
        tiny source region looks blurry regardless of the final pixel
        dimensions, so this is what actually protects the fixed-256x256
        real-time path. Threshold is a coarse, deliberately conservative
        floor from manual testing, not a validated operating point -- there
        is no labeled sharp/blurry dataset in this project to tune it
        against.
    """
    width, height = image.size
    if width < MIN_FACE_CROP_PX or height < MIN_FACE_CROP_PX:
        return f"face crop too small ({width}x{height}px, need >= {MIN_FACE_CROP_PX}px)"

    sharpness = sharpness_variance(image)
    if sharpness < MIN_SHARPNESS_VARIANCE:
        return f"face crop too blurry (sharpness={sharpness:.1f}, need >= {MIN_SHARPNESS_VARIANCE})"

    return None
