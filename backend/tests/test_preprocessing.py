import numpy as np
from PIL import Image

from config import MIN_SHARPNESS_VARIANCE
from preprocessing import sharpness_variance, assess_face_crop_quality


def _solid_image(size, color=(128, 64, 200)):
    return Image.new("RGB", size, color)


def _textured_image(size, seed=0):
    # Real photo content has edges/texture and measures well above the
    # sharpness floor; a solid color measures exactly 0 and would always
    # fail assess_face_crop_quality's blur check regardless of size, which
    # is correct behavior but not what most of these tests are exercising.
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(array)


def test_assess_face_crop_quality_rejects_tiny_crop():
    tiny = _textured_image((16, 16))
    reason = assess_face_crop_quality(tiny)
    assert reason is not None
    assert "16x16" in reason


def test_assess_face_crop_quality_accepts_normal_sharp_crop():
    normal = _textured_image((256, 256))
    assert assess_face_crop_quality(normal) is None


def test_assess_face_crop_quality_rejects_when_only_one_dimension_is_small():
    thin = _textured_image((300, 20))
    assert assess_face_crop_quality(thin) is not None


def test_assess_face_crop_quality_rejects_blurry_crop_at_full_resolution():
    # Full 256x256 size (passes the dimension check) but flat/blurry content
    # -- this is the case that actually matters for the real-time path,
    # where the client always resizes to 256x256 before sending, so the
    # dimension check alone can never catch a low-quality source crop.
    blurry = _solid_image((256, 256))
    reason = assess_face_crop_quality(blurry)
    assert reason is not None
    assert "blurry" in reason


def test_assess_face_crop_quality_rejects_upsampled_tiny_source():
    # Simulates what the extension/desktop actually send for a genuinely
    # tiny source face: a low-detail crop upsampled (bilinear) to 256x256.
    # Dimension check can't see this (already 256x256); sharpness must.
    rng = np.random.default_rng(1)
    tiny_source = Image.fromarray(rng.integers(0, 255, (20, 20, 3), dtype=np.uint8))
    upsampled = tiny_source.resize((256, 256), Image.BILINEAR)
    assert assess_face_crop_quality(upsampled) is not None


def test_sharpness_variance_orders_solid_below_threshold_below_textured():
    solid = sharpness_variance(_solid_image((256, 256)))
    textured = sharpness_variance(_textured_image((256, 256)))
    assert solid == 0.0
    assert solid < MIN_SHARPNESS_VARIANCE < textured
