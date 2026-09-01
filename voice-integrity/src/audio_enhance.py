"""
Optional "Denoise & Enhance" preprocessing for the /voice/check dashboard
flow (server.py's `enhance` form field) -- NOT used by the automatic
extension/desktop real-time voice checks, which stay on the raw captured
audio.

Two cascaded stages, both frozen/pretrained (nothing here is trained by
this project):
  1. reduce_noise() -- fast, non-neural spectral-gating noise reduction
     (the `noisereduce` package). Good at broadband hiss/hum/fan noise,
     no model download, runs first since it's cheap and removes the bulk
     of stationary background noise before the heavier stage.
  2. the MetricGAN+ enhancement stage in enhance_audio() below --
     speechbrain's pretrained speechbrain/metricgan-plus-voicebank model
     (a GAN trained to directly optimize the PESQ speech-quality metric),
     which speechbrain is already a dependency for (see
     speech_verification.py). Cleans up the more structured degradation
     spectral gating alone won't touch (compression artifacts, reverb,
     non-stationary noise).

HONESTY NOTE: this is a real, uncontrolled tradeoff for a spoof-detection
tool, not a strict improvement -- see server.py's `enhancement` response
block, which always reports the pre-enhancement ("raw") scores alongside
the enhanced ones rather than silently replacing them:
  - Noise/compression artifacts are part of what pushes xlsr_sls_detector's
    SLS head and Gustking's classifier toward false "spoofed" verdicts on
    genuine noisy real-world audio (see those modules' docstrings) --
    removing that noise can reduce those false positives, which is the
    point of adding this.
  - But a denoiser/enhancer is itself a neural reconstruction process that
    can just as easily smooth over the fine-grained vocoder/synthesis
    artifacts a spoof detector is trained to catch, *suppressing* a true
    "spoofed" signal instead. Neither model here was validated for that
    failure mode against a labeled dataset -- there is no ground-truth set
    of "genuine spoofed clips before/after enhancement" in this project to
    check it against.
Treat "enhanced" scores as a second opinion to compare against "raw", not
a strictly more trustworthy replacement for it -- which is why the API
always returns both rather than only the enhanced result.
"""
import logging
import threading
from pathlib import Path

import noisereduce as nr
import numpy as np
import soundfile as sf
import torch
import torchaudio

# Same torchaudio/speechbrain-lazy-import workarounds as
# speech_verification.py -- see that module's header comment for exactly
# why both are needed (harmless no-ops if a future torchaudio/speechbrain
# release fixes them upstream).
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

import importlib

importlib.import_module("speechbrain.dataio.audio_io")

from speechbrain.inference.enhancement import SpectralMaskEnhancement  # noqa: E402
from speechbrain.utils.fetching import LocalStrategy  # noqa: E402

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
ENHANCER_SAVE_DIR = REPO_ROOT / "pretrained_models" / "metricgan-plus-voicebank"

_enhancer = None
_load_lock = threading.Lock()


def _load_enhancer() -> SpectralMaskEnhancement:
    """Loaded once and reused -- see speech_verification.py's identical
    pattern for the other speechbrain model this service loads."""
    global _enhancer
    if _enhancer is None:
        with _load_lock:
            if _enhancer is None:
                logger.info("loading speechbrain MetricGAN+ speech enhancer (metricgan-plus-voicebank)")
                _enhancer = SpectralMaskEnhancement.from_hparams(
                    source="speechbrain/metricgan-plus-voicebank",
                    savedir=str(ENHANCER_SAVE_DIR),
                    local_strategy=LocalStrategy.COPY,
                )
    return _enhancer


def reduce_noise(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Stage 1: spectral-gating noise reduction. `stationary=False` (vs.
    noisereduce's stationary-noise default) tracks a moving noise profile
    across the clip instead of assuming one fixed profile for the whole
    clip -- a better fit for call audio, where background noise level
    isn't constant."""
    return nr.reduce_noise(y=samples, sr=sample_rate, stationary=False)


def enhance_audio(input_path: Path, out_dir: Path) -> Path:
    """Runs both stages on an already-16kHz-mono clip (see
    audio_utils.resample_to_16k_mono -- callers must resample first, same
    precondition the two anti-spoofing detectors have) and writes the
    final enhanced clip into out_dir. Returns the output path.

    Not cached: every /voice/check upload is a one-shot, unnamed clip
    (same reasoning as server.py's own resampling step) -- callers own a
    per-request temp dir and are expected to pass that as out_dir.

    Deliberately does NOT use SpectralMaskEnhancement.enhance_file(path):
    that method routes the path through speechbrain's split_path()/fetch()
    (built for HuggingFace-hub-style "repo + relative file" sources), which
    mangles a plain local Windows path -- "C:\\Users\\..." loses the
    backslash right after the drive letter and soundfile then fails to
    open it. This is the same class of Windows/speechbrain path bug
    speech_verification.py and xlsr_sls_detector.py's docstrings mention
    working around elsewhere; enhance_batch() (the tensor-in/tensor-out
    method enhance_file() itself calls after loading) has no such path
    handling, so this loads/saves the audio itself via soundfile -- the
    same approach every other module in this project already uses -- and
    calls enhance_batch() directly."""
    samples, sample_rate = sf.read(str(input_path))
    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    denoised = reduce_noise(samples.astype(np.float32), sample_rate)

    enhancer = _load_enhancer()
    noisy = torch.from_numpy(denoised.astype(np.float32)).unsqueeze(0).to(enhancer.device)
    with torch.no_grad():
        enhanced = enhancer.enhance_batch(noisy, lengths=torch.tensor([1.0], device=enhancer.device))
    enhanced = enhanced.squeeze(0).detach().cpu().numpy()

    enhanced_path = out_dir / f"{input_path.stem}_enhanced.wav"
    sf.write(enhanced_path, enhanced.astype(np.float32), sample_rate, subtype="PCM_16")
    return enhanced_path
