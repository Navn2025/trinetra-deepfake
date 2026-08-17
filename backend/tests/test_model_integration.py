"""
Real end-to-end model tests -- no mocking. Loads the actual DeepfakeBench
UCF+SPSL+Xception ensemble checkpoints (from DeepfakeBench/training/weights/,
already present on disk in this environment, no network needed) onto the
real device (GPU if available) and runs real inference. Slower than the
mocked API tests; that's expected -- this is the test that actually proves
the model-loading path works end to end, not just that the API plumbing
around it is correct.
"""
import pytest
from PIL import Image

import predictor
from errors import ModelLoadError
from preprocessing import INPUT_SIZE


def test_model_loads_and_predicts_valid_probability():
    image = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (100, 120, 140))
    prob = predictor.predict_fake_probability(image)

    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


def test_model_load_is_cached_across_calls():
    predictor._models = None  # force a fresh load
    predictor._load_models()
    first = predictor._models
    predictor._load_models()
    assert predictor._models is first  # not rebuilt


def test_get_model_info_reports_loaded_status_and_real_device():
    info = predictor.get_model_info()
    assert info["status"] == "loaded"
    assert info["device"] in ("cuda", "cpu") or info["device"].startswith("cuda")
    assert info["model_version"] == predictor.MODEL_VERSION


def test_load_model_raises_model_load_error_on_missing_checkpoint(monkeypatch, tmp_path):
    predictor._models = None
    monkeypatch.setattr(predictor, "UCF_WEIGHTS", tmp_path / "does_not_exist.pth")

    try:
        with pytest.raises(ModelLoadError):
            predictor._load_models()
    finally:
        # Leave the module in a working state for any tests that run after.
        predictor._models = None
