from PIL import Image

from face_detection import crop_face, normalized_bbox


def test_crop_face_returns_requested_output_size():
    image = Image.new("RGB", (640, 480), (100, 100, 100))
    crop = crop_face(image, (100, 100, 200, 220), output_size=256)
    assert crop.size == (256, 256)


def test_crop_face_is_square_even_for_non_square_bbox():
    image = Image.new("RGB", (640, 480), (100, 100, 100))
    # wide bbox: 200x40 -- the crop should still come out square (256x256),
    # padded by the larger dimension, not stretched to the bbox's aspect ratio
    crop = crop_face(image, (50, 50, 250, 90), output_size=128)
    assert crop.size == (128, 128)


def test_crop_face_clamps_to_image_bounds_near_edge():
    image = Image.new("RGB", (100, 100), (50, 50, 50))
    # bbox near the bottom-right corner with margin would overflow -- must
    # not raise, and must still return a valid non-empty crop
    crop = crop_face(image, (80, 80, 99, 99), output_size=64)
    assert crop.size == (64, 64)


def test_crop_face_clamps_near_top_left_corner():
    image = Image.new("RGB", (100, 100), (50, 50, 50))
    crop = crop_face(image, (0, 0, 15, 15), output_size=64)
    assert crop.size == (64, 64)


def test_normalized_bbox_maps_pixel_coords_to_unit_interval():
    result = normalized_bbox((100, 50, 300, 250), image_size=(1000, 500))
    assert result["x"] == 0.1
    assert result["y"] == 0.1
    assert result["width"] == 0.2
    assert result["height"] == 0.4


def test_normalized_bbox_never_goes_negative_or_over_one():
    result = normalized_bbox((-10, -5, 1200, 600), image_size=(1000, 500))
    assert 0.0 <= result["x"] <= 1.0
    assert 0.0 <= result["y"] <= 1.0
    assert 0.0 <= result["width"] <= 1.0
    assert 0.0 <= result["height"] <= 1.0
