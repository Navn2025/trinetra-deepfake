"""Shared exception types used across the model-loading modules
(predictor.py) so main.py can catch one type regardless of which model
failed to load."""


class ModelLoadError(RuntimeError):
    """Raised when a model checkpoint can't be loaded: missing/corrupt
    weights, unreachable model hub with nothing cached, or the target
    device (typically CUDA) failing to initialize/OOMing."""
