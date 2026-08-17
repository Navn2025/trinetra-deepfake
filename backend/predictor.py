"""
Real model wired up: a 3-detector ensemble from DeepfakeBench, using the
official pretrained checkpoints trained on FaceForensics++
(https://github.com/SCLBD/DeepfakeBench/releases/tag/v1.0.1). No single
published deepfake detector generalizes reliably to manipulation methods
outside its training distribution (this is an open research problem, not
something a better checkpoint solves) -- so instead of ever chasing one
"best" model, this averages three architecturally different detectors and
returns their mean fake_probability. Different architectures tend to miss
different manipulation methods, so an ensemble covers more ground than any
single one -- concretely, on this project's own ~120-frame FF++ validation
sample (see git history / conversation for methodology, no held-out eval
set is checked into this repo), UCF alone collapsed on FaceShifter
(identity-swap) frames while plain Xception still showed rough separation
on the same images, which is exactly the kind of gap ensembling is meant to
close. This does NOT mean the ensemble is "perfect" -- it isn't, and
nothing published is. Still frame-by-frame, still trained on FF++'s 2023-era
manipulation methods, still no temporal video modeling, still no
adversarial robustness testing.

The three members:
  - UCF (ICCV'23, arxiv.org/abs/2304.13949): disentangles forgery-specific
    from "common forgery" features, DeepfakeBench's own top spatial-detector
    performer on cross-dataset AUC.
  - SPSL (CVPR'21, spatial-phase frequency-domain features): a phase
    spectrum computed from the image (see _phase_without_amplitude) is
    concatenated onto the RGB input as a 4th channel, giving this member a
    genuinely different signal from the two RGB-only ones.
  - Plain Xception (the single-detector baseline this project used before):
    kept in the ensemble specifically because it was empirically the best
    of the three at generalizing to the FaceShifter gap above.

All three share the same Xception backbone class (imported directly from
DeepfakeBench/training/networks/xception.py) with different `mode`/`inc`
configs, so there's no per-member vendored backbone code. UCF's two small
head modules are reimplemented in ucf_head.py. None of this goes through
DeepfakeBench/training/detectors/__init__.py, which eagerly imports every
detector (VideoMAE, CLIP, XCLIP, ...) and would otherwise pull in a pile of
unrelated heavy dependencies (transformers, timm, einops, fvcore, ...) just
to run three of its twenty-odd detectors.

Checkpoint layout for each (verified by inspection, not assumed):
  - ucf_best.pth: full UCFDetector state_dict (600 keys under encoder_f/
    encoder_c/block_spe/block_sha/head_spe/head_sha/con_gan). Only
    encoder_f + block_sha + head_sha are loaded -- the exact "shared" path
    UCFDetector.forward(inference=True) reads for pred_dict['cls']; see
    ucf_head.py's docstring for why the rest is skipped.
  - spsl_best.pth: SpslDetector's state_dict, single "backbone." prefix
    (283 keys, same shape as plain Xception's except conv1 is
    (32, 4, 3, 3) for the 4-channel RGB+phase input). Already the fully
    fine-tuned end-to-end weights -- no separate ImageNet-init step needed.
  - xception_best.pth: XceptionDetector's state_dict, single "backbone."
    prefix, standard 3-channel input.
"""
import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

from errors import ModelLoadError
from preprocessing import preprocess_face
from ucf_head import Conv2d1x1, Head

logger = logging.getLogger(__name__)

DEEPFAKEBENCH_TRAINING_DIR = Path(__file__).resolve().parents[2] / "DeepfakeBench" / "training"
if str(DEEPFAKEBENCH_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(DEEPFAKEBENCH_TRAINING_DIR))

from networks.xception import Xception  # noqa: E402  (path set up above)

WEIGHTS_DIR = DEEPFAKEBENCH_TRAINING_DIR / "weights"
UCF_WEIGHTS = WEIGHTS_DIR / "ucf_best.pth"
SPSL_WEIGHTS = WEIGHTS_DIR / "spsl_best.pth"
XCEPTION_WEIGHTS = WEIGHTS_DIR / "xception_best.pth"

# Matches DeepfakeBench/training/config/detector/ucf.yaml
UCF_BACKBONE_CONFIG = {"mode": "adjust_channel", "num_classes": 2, "inc": 3, "dropout": False}
UCF_ENCODER_FEAT_DIM = 512
UCF_HALF_FINGERPRINT_DIM = UCF_ENCODER_FEAT_DIM // 2  # 256, matches ucf_detector.py

# Matches DeepfakeBench/training/config/detector/spsl.yaml
SPSL_BACKBONE_CONFIG = {"mode": "original", "num_classes": 2, "inc": 4, "dropout": False}

# Matches DeepfakeBench/training/config/detector/xception.yaml
XCEPTION_BACKBONE_CONFIG = {"mode": "original", "num_classes": 2, "inc": 3, "dropout": False}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_VERSION = "deepfakebench-ensemble-ucf+spsl+xception-ffpp-v1"

_models = None  # dict of {"ucf": (encoder, block_sha, head_sha), "spsl": backbone, "xception": backbone}
_load_lock = threading.Lock()


def _strip_prefix(state_dict: dict, prefix: str) -> dict:
    return {k.removeprefix(prefix): v for k, v in state_dict.items() if k.startswith(prefix)}


def _load_ucf() -> tuple:
    state_dict = torch.load(UCF_WEIGHTS, map_location="cpu")

    encoder = Xception(UCF_BACKBONE_CONFIG)
    encoder.load_state_dict(_strip_prefix(state_dict, "encoder_f."), strict=True)

    block_sha = Conv2d1x1(in_f=UCF_ENCODER_FEAT_DIM, hidden_dim=UCF_HALF_FINGERPRINT_DIM, out_f=UCF_HALF_FINGERPRINT_DIM)
    block_sha.load_state_dict(_strip_prefix(state_dict, "block_sha."), strict=True)

    head_sha = Head(in_f=UCF_HALF_FINGERPRINT_DIM, hidden_dim=UCF_ENCODER_FEAT_DIM, out_f=UCF_BACKBONE_CONFIG["num_classes"])
    head_sha.load_state_dict(_strip_prefix(state_dict, "head_sha."), strict=True)

    for module in (encoder, block_sha, head_sha):
        module.eval()
        module.to(DEVICE)
    return encoder, block_sha, head_sha


def _load_plain_backbone(weights_path: Path, backbone_config: dict) -> Xception:
    """Shared loader for SPSL and plain Xception: both are DeepfakeBench's
    XceptionDetector-shaped wrapper (single "backbone." prefix, no auxiliary
    heads), differing only in backbone_config (SPSL's inc=4 for the extra
    phase channel)."""
    backbone = Xception(backbone_config)
    state_dict = torch.load(weights_path, map_location="cpu")
    backbone.load_state_dict(_strip_prefix(state_dict, "backbone."), strict=True)
    backbone.eval()
    backbone.to(DEVICE)
    return backbone


def _load_models() -> dict:
    global _models
    # Fast path: no lock needed once loaded (module-level assignment of a
    # fully-built dict is atomic under the GIL). The lock only guards the
    # one-time build below, so concurrent first requests (this function runs
    # inside a threadpool -- see main.py) don't race to load checkpoints
    # twice or hand back a half-built ensemble.
    if _models is not None:
        return _models

    with _load_lock:
        if _models is not None:
            return _models

        logger.info("loading deepfake detector ensemble (ucf+spsl+xception) onto %s", DEVICE)
        start = time.monotonic()
        try:
            models = {}
            for name, load in (
                ("ucf", _load_ucf),
                ("spsl", lambda: _load_plain_backbone(SPSL_WEIGHTS, SPSL_BACKBONE_CONFIG)),
                ("xception", lambda: _load_plain_backbone(XCEPTION_WEIGHTS, XCEPTION_BACKBONE_CONFIG)),
            ):
                member_start = time.monotonic()
                models[name] = load()
                logger.info("  %s checkpoint loaded in %.1fs", name, time.monotonic() - member_start)
        except FileNotFoundError as exc:
            raise ModelLoadError(f"checkpoint file missing: {exc}") from exc
        except RuntimeError as exc:
            # Covers CUDA init failures (no driver, OOM building the model)
            # and state_dict shape/key mismatches from a swapped-in checkpoint.
            raise ModelLoadError(f"failed to load model onto {DEVICE}: {exc}") from exc

        logger.info("ensemble ready in %.1fs", time.monotonic() - start)
        _models = models
        return _models


def get_model_info() -> dict:
    """Metadata for the /model-info endpoint. Triggers a load if the models
    haven't been used yet, so the reported device/status reflects reality."""
    try:
        _load_models()
        status = "loaded"
    except ModelLoadError as exc:
        status = f"error: {exc}"

    return {
        "model_version": MODEL_VERSION,
        "backbone": "ensemble: ucf + spsl + xception (all xception-based)",
        "checkpoint": f"{UCF_WEIGHTS.name}+{SPSL_WEIGHTS.name}+{XCEPTION_WEIGHTS.name}",
        "input_size": 256,
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "status": status,
    }


def _phase_without_amplitude(rgb_tensor: torch.Tensor) -> torch.Tensor:
    """SPSL's frequency-domain feature (SPSLDetector.phase_without_amplitude,
    reproduced verbatim): the image's phase spectrum with amplitude
    discarded, which is where forgery-related frequency artifacts tend to
    concentrate. Grayscale in, single-channel out, same HxW."""
    gray = torch.mean(rgb_tensor, dim=1, keepdim=True)
    spectrum = torch.fft.fftn(gray, dim=(-1, -2))
    phase_only = torch.exp(1j * torch.angle(spectrum))
    return torch.real(torch.fft.ifftn(phase_only, dim=(-1, -2)))


def _predict_ucf_batch(models: dict, tensor: torch.Tensor) -> torch.Tensor:
    encoder, block_sha, head_sha = models["ucf"]
    forgery_features = encoder.features(tensor)  # Nx512xHxW ("adjust_channel" mode)
    f_share = block_sha(forgery_features)
    logits, _feat = head_sha(f_share)
    return torch.softmax(logits, dim=1)[:, 1]


def _predict_plain_backbone_batch(backbone: Xception, tensor: torch.Tensor) -> torch.Tensor:
    features = backbone.features(tensor)
    logits = backbone.classifier(features)
    return torch.softmax(logits, dim=1)[:, 1]


def predict_fake_probabilities_batch(images: list) -> list[float]:
    """Same 3-model ensemble as predict_fake_probability, but scores every
    crop in one batched forward pass per model member instead of one pass
    per crop. Looping predict_fake_probability N times pays full
    model-forward call overhead N times; stacking N crops into a single
    NxCxHxW tensor amortizes that to exactly 3 forward passes (one per
    ensemble member) regardless of N -- the video pipeline's main lever for
    not scoring 9+ sampled frames one at a time. Returns [] for empty input
    (skips loading the models -- a video with no scorable frames shouldn't
    pay the load cost)."""
    if not images:
        return []

    start = time.monotonic()
    models = _load_models()

    arrays = np.stack([preprocess_face(image) for image in images])  # NxHxWx3, normalized
    tensor = torch.from_numpy(arrays).permute(0, 3, 1, 2).float().to(DEVICE)  # NxCxHxW

    with torch.inference_mode():
        ucf_probs = _predict_ucf_batch(models, tensor)
        xception_probs = _predict_plain_backbone_batch(models["xception"], tensor)

        phase = _phase_without_amplitude(tensor).to(tensor.dtype)
        spsl_input = torch.cat((tensor, phase), dim=1)  # NxCx+1xHxW
        spsl_probs = _predict_plain_backbone_batch(models["spsl"], spsl_input)

    mean_probs = ((ucf_probs + spsl_probs + xception_probs) / 3.0).tolist()
    elapsed = time.monotonic() - start
    logger.info(
        "batched ensemble inference: %d crop(s) in %.2fs (%.3fs/crop)",
        len(images), elapsed, elapsed / len(images),
    )
    if len(images) == 1:
        # Single-crop calls are almost always /predict (the real-time
        # extension/desktop path) -- log the per-member breakdown here since
        # that's exactly the case where "which model disagreed" matters most
        # for debugging a live-capture-specific misclassification (batched
        # video-frame calls skip this to avoid spamming N lines per video).
        logger.info(
            "  breakdown: ucf=%.3f spsl=%.3f xception=%.3f mean=%.3f",
            ucf_probs[0].item(), spsl_probs[0].item(), xception_probs[0].item(), mean_probs[0],
        )
    return mean_probs


def predict_fake_probability(image) -> float:
    """Synchronous, GPU/CPU-bound. Callers on an asyncio event loop (the
    FastAPI routes in main.py) must run this via run_in_threadpool rather
    than awaiting it directly -- it does not yield control on its own.

    Single-crop convenience wrapper around predict_fake_probabilities_batch,
    used by the live /predict endpoint (always exactly one crop per
    request, so batching buys nothing there -- see video_pipeline.py for
    the multi-frame case batching actually speeds up)."""
    return predict_fake_probabilities_batch([image])[0]
