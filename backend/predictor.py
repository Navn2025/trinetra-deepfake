"""
Real model wired up: trinetra_v1_export's trained demo detector -- a frozen
CLIP ViT-B/16 (OpenCLIP, LAION-2B) image encoder with a linear classification
head trained on a FaceForensics++ subset (backend/models/demo_head.pt, see
demo_train_report.json alongside it for the full per-video breakdown). This
replaced an earlier DeepfakeBench UCF+SPSL+Xception ensemble; see git history
for that version.

HONESTY NOTES, carried over from trinetra_v1_export/src/inference.py because
they matter here too:
  * The checkpoint's own threshold (0.5) is an uncalibrated raw-sigmoid cutoff,
    not a probability of being fake. This module still runs every crop through
    it exactly as trained -- calibration is out of scope for this demo head,
    same as it was for the ensemble it replaced.
  * demo_train_report.json's numbers are blunt about this being a small,
    overfit demo model: train video_acc 0.95 / video_auc 0.99 vs. held-out
    val video_acc 0.54 / video_auc 0.67 and test video_acc 0.58 / video_auc
    0.72 -- barely above chance on data it wasn't trained on. Trained on only
    80 videos. Treat fake_probability as a weak signal, not a verdict, until
    it's retrained on more data with a proper held-out sweep.
  * trinetra_v1_export's own pipeline scores a *pooled* per-video embedding
    (mean of L2-normalized per-crop embeddings across a whole tracked face,
    aligned via RetinaFace + ArcFace landmarks). This backend's existing
    per-crop contract (predict_fake_probability(image) -> float, one
    already-detected-and-cropped face at a time -- see face_detection.py)
    doesn't have that per-video pooling or landmark alignment, so each crop
    here is scored independently instead: video_pipeline.py's own median +
    suspicious-segment aggregation across sampled frames is what stands in
    for trinetra's video-level pooling. This module only reuses the frozen
    CLIP encoder + trained linear head, not trinetra's own face-tracking
    pipeline.

Checkpoint layout (backend/models/demo_head.pt), from trinetra's own
scripts/train_demo_head.py:
    {"backbone": "clip_vit_b16", "checkpoint": "laion2b_s34b_b88k",
     "embed_dim": 512, "state_dict": {"weight": (1, 512), "bias": (1,)},
     "threshold": 0.5, ...}
The backbone (frozen CLIP ViT-B/16 weights) is not itself in the checkpoint
-- it's downloaded once via open_clip and cached locally, same as
trinetra_v1_export's README describes.
"""
import logging
import threading
import time
from pathlib import Path

import numpy as np
import torch

from clip_backbone import CLIPBackbone
from errors import ModelLoadError

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent / "models"
CHECKPOINT_PATH = MODEL_DIR / "demo_head.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_VERSION = "trinetra-clip-vit-b16-demo-head-v1"

_models = None  # {"backbone": CLIPBackbone, "head": torch.nn.Linear, "threshold": float}
_load_lock = threading.Lock()


def _load_models() -> dict:
    global _models
    # Fast path: no lock needed once loaded (module-level assignment of a
    # fully-built dict is atomic under the GIL). The lock only guards the
    # one-time build below, so concurrent first requests (this function runs
    # inside a threadpool -- see main.py) don't race to load the checkpoint
    # twice or hand back a half-built model.
    if _models is not None:
        return _models

    with _load_lock:
        if _models is not None:
            return _models

        logger.info("loading trinetra demo detector (CLIP ViT-B/16 + linear head) onto %s", DEVICE)
        start = time.monotonic()
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
            backbone = CLIPBackbone(checkpoint=ckpt.get("checkpoint", "laion2b_s34b_b88k"),
                                     device=str(DEVICE))
            head = torch.nn.Linear(ckpt["embed_dim"], 1)
            head.load_state_dict(ckpt["state_dict"])
            head.eval().to(DEVICE)
        except FileNotFoundError as exc:
            raise ModelLoadError(f"checkpoint file missing: {exc}") from exc
        except RuntimeError as exc:
            # Covers CUDA init failures (no driver, OOM building the model)
            # and state_dict shape mismatches from a swapped-in checkpoint.
            raise ModelLoadError(f"failed to load model onto {DEVICE}: {exc}") from exc

        _models = {
            "backbone": backbone,
            "head": head,
            "threshold": float(ckpt.get("threshold", 0.5)),
        }
        logger.info("model ready in %.1fs (threshold=%.2f)", time.monotonic() - start, _models["threshold"])
        return _models


def get_model_info() -> dict:
    """Metadata for the /model-info endpoint. Triggers a load if the model
    hasn't been used yet, so the reported device/status reflects reality."""
    try:
        models = _load_models()
        status = "loaded"
        threshold = models["threshold"]
    except ModelLoadError as exc:
        status = f"error: {exc}"
        threshold = None

    return {
        "model_version": MODEL_VERSION,
        "backbone": "CLIP ViT-B/16 (OpenCLIP, LAION-2B, frozen) + linear head",
        "checkpoint": CHECKPOINT_PATH.name,
        "input_size": 224,
        "raw_threshold": threshold,
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "status": status,
    }


def predict_fake_probabilities_batch(images: list) -> list[float]:
    """Scores every crop in one batched forward pass instead of one pass per
    crop -- see face_detection.py/video_pipeline.py for why (N sampled video
    frames scored in a single call). Returns [] for empty input (skips
    loading the model -- a video with no scorable frames shouldn't pay the
    load cost)."""
    if not images:
        return []

    start = time.monotonic()
    models = _load_models()

    arrays = np.stack([np.asarray(image.convert("RGB"), dtype=np.uint8) for image in images])  # NxHxWx3
    emb = models["backbone"].encode_image(arrays)  # (N, embed_dim) float32, CLIPBackbone handles resize+normalize
    emb = emb / np.clip(np.linalg.norm(emb, axis=-1, keepdims=True), 1e-12, None)  # matches training (l2_normalize)

    tensor = torch.from_numpy(emb).float().to(DEVICE)
    with torch.inference_mode():
        probs = torch.sigmoid(models["head"](tensor).squeeze(-1)).cpu().numpy()

    elapsed = time.monotonic() - start
    logger.info(
        "clip+head inference: %d crop(s) in %.2fs (%.3fs/crop)",
        len(images), elapsed, elapsed / len(images),
    )
    return probs.tolist()


def predict_fake_probability(image) -> float:
    """Synchronous, GPU/CPU-bound. Callers on an asyncio event loop (the
    FastAPI routes in main.py) must run this via run_in_threadpool rather
    than awaiting it directly -- it does not yield control on its own.

    Single-crop convenience wrapper around predict_fake_probabilities_batch,
    used by the live /predict endpoint (always exactly one crop per
    request, so batching buys nothing there -- see video_pipeline.py for
    the multi-frame case batching actually speeds up)."""
    return predict_fake_probabilities_batch([image])[0]
