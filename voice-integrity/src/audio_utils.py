"""Shared audio loading/resampling for the voice-integrity pipeline.

Both anti-spoofing detectors (synthetic_detector.py, xlsr_sls_detector.py)
hard-require exactly 16kHz mono input and raise if given anything else --
that's what resample_to_16k_mono() is for. Speaker verification
(speech_verification.py) doesn't need this: speechbrain resamples
internally.
"""
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

# Extensions soundfile can never open (it only reads raw audio containers,
# not video) -- checked up front so a video upload goes straight to the
# ffmpeg fallback below instead of paying a doomed sf.info() probe first.
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}


def load_mono(path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr


def ensure_readable_audio(path, out_dir) -> Path:
    """soundfile (used by every detector here, and internally by speechbrain
    for identity verification) can only read raw audio containers -- not
    video. If `path` looks like a video file, or soundfile just can't open
    it regardless of extension (e.g. a video with a misleading .wav name),
    extract its audio track to a fresh 16kHz mono WAV via ffmpeg instead of
    failing outright with "Format not recognised". Returns `path` unchanged
    when it's already directly readable."""
    path = Path(path)
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        try:
            sf.info(str(path))
            return path
        except Exception:
            pass  # not a recognized video extension, but still unreadable -- fall through

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}_extracted.wav"
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(path), "-ar", "16000", "-ac", "1", str(out_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not out_path.exists():
        raise ValueError(f"Could not extract audio from {path.name} via ffmpeg: {result.stderr[-500:]}")
    return out_path


def resample_to_16k_mono(path, cache_dir) -> Path:
    """Returns a path to a 16kHz mono WAV version of `path`, writing it into
    `cache_dir` if a fresh cached copy doesn't already exist there (checked
    by source mtime, so repeat pipeline runs on the same file skip
    re-encoding). If `path` is already 16kHz mono, returns it unchanged --
    no cache copy needed."""
    path = Path(path)
    audio, sr = load_mono(path)
    if sr == 16000:
        return path

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{path.stem}_16k.wav"

    if out_path.exists() and out_path.stat().st_mtime >= path.stat().st_mtime:
        return out_path

    tensor = torch.from_numpy(audio).unsqueeze(0)
    tensor = torchaudio.functional.resample(tensor, sr, 16000)
    sf.write(out_path, tensor.squeeze(0).numpy(), 16000, subtype="PCM_16")
    return out_path
