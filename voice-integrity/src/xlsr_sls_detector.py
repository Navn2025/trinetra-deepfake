"""XLS-R + SLS deepfake/voice-clone detector.

Best cross-dataset generalization of the models evaluated for this project
(see conversation/commit history): 2.14% EER on ASVspoof2021 DF, 3.51% on
ASVspoof2021 LA, 7.84% on In-the-Wild (real-world audio, the closest proxy
to actual deployment conditions) -- from the model card at
https://huggingface.co/sukhdeveyash/XLS-R-SLS-Deepfake-Detection, itself
reproducing Zhang/Wen/Hu, "Audio Deepfake Detection with XLS-R and SLS
classifier," ACM Multimedia 2024.

Not a plug-and-play HF `transformers` model -- this is XLS-R 300M loaded via
fairseq (see model.py's SSLModel) with a custom SLS classification head
(models/sls_head_v1.pth, the "v1/epoch_2.pth" checkpoint -- the README notes
v1 generalizes better cross-domain than v2 despite less training, so v1 is
what's used here) on top. Requires:
  - models/xlsr2_300m.pt: the frozen XLS-R 300M backbone (fairseq format,
    downloaded from https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt,
    NOT the same file as the SLS checkpoint below)
  - models/sls_head_v1.pth: the trained SLS head, downloaded from
    huggingface_hub (sukhdeveyash/XLS-R-SLS-Deepfake-Detection, v1/epoch_2.pth)
  - fairseq itself, which needs _fairseq_py311_compat's monkeypatch applied
    first -- see that module's docstring for why.
"""
import argparse
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import _fairseq_py311_compat
_fairseq_py311_compat.apply()

from sls_asvspoof.model import Model  # noqa: E402  (must follow the compat patch)

REPO_ROOT = Path(__file__).resolve().parent.parent
XLSR_BACKBONE_PATH = REPO_ROOT / "models" / "xlsr2_300m.pt"
SLS_HEAD_PATH = REPO_ROOT / "models" / "sls_head_v1.pth"

CUT_LEN = 64600  # ~4.04s at 16kHz -- fixed input length the SLS head was trained on

_model = None
_device = None


def pad_or_tile(x: np.ndarray, max_len: int = CUT_LEN) -> np.ndarray:
    """Matches sls_asvspoof.data_utils.pad(): truncate if long enough,
    otherwise tile (repeat) the clip until it covers max_len, then trim."""
    if x.shape[0] >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x.shape[0]) + 1
    return np.tile(x, num_repeats)[:max_len]


def load_model(device: str | None = None):
    """Loaded once and reused across predict() calls. Slow (~seconds to
    tens of seconds on CPU) -- XLS-R 300M is a real transformer, not a
    lightweight head."""
    global _model, _device
    if _model is None:
        _device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        args = types.SimpleNamespace(xlsr_model=str(XLSR_BACKBONE_PATH))
        model = Model(args, _device)
        state_dict = torch.load(SLS_HEAD_PATH, map_location=_device)
        # Checkpoint was saved from a DataParallel-wrapped model -- every
        # key has a "module." prefix the plain (unwrapped) Model doesn't have.
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(_device)
        model.eval()
        _model = model
    return _model, _device


def predict(audio_path) -> dict[str, float]:
    """Returns {"bonafide": prob, "spoof": prob}. Input must be 16kHz mono."""
    audio, sample_rate = sf.read(str(audio_path))
    return predict_array(audio, sample_rate)


def predict_array(audio, sample_rate: int) -> dict[str, float]:
    """Same as predict(), but for audio already in memory -- see
    synthetic_detector.predict_array()'s docstring for why (server.py's
    segment scoring calls this directly)."""
    model, device = load_model()
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        raise ValueError(f"Expected sample rate of 16000 Hz, but got {sample_rate} Hz.")

    audio = pad_or_tile(audio.astype(np.float32))
    x = torch.from_numpy(audio).float().unsqueeze(0).to(device)  # (1, CUT_LEN)

    with torch.no_grad():
        log_probs = model(x)  # (1, 2) log-softmax; column 1 = bonafide, column 0 = spoof
        probs = log_probs.exp()[0]

    return {"bonafide": probs[1].item(), "spoof": probs[0].item()}


def predict_batch(windows: list[tuple[np.ndarray, int]]) -> list[dict[str, float]]:
    """Same result as calling predict_array() once per window, but as one
    stacked (N, CUT_LEN) forward pass instead of N sequential ones. This is
    the real fix for segment scoring's cost, not just overhead removal:
    XLS-R 300M is GPU-compute-bound, so N separate Python-level calls
    serialize on the GPU regardless of how they're scheduled (threads,
    asyncio.gather, ...) -- only an actual batched tensor lets the GPU
    process all N windows together. Empty list returns []."""
    if not windows:
        return []
    model, device = load_model()
    arrays = []
    for audio, sample_rate in windows:
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != 16000:
            raise ValueError(f"Expected sample rate of 16000 Hz, but got {sample_rate} Hz.")
        arrays.append(pad_or_tile(audio.astype(np.float32)))

    x = torch.from_numpy(np.stack(arrays)).float().to(device)  # (N, CUT_LEN)
    with torch.no_grad():
        log_probs = model(x)  # (N, 2)
        probs = log_probs.exp()

    return [{"bonafide": row[1].item(), "spoof": row[0].item()} for row in probs]


def main():
    parser = argparse.ArgumentParser(
        description="XLS-R + SLS deepfake/voice-clone detector (slower, best cross-dataset generalization)."
    )
    parser.add_argument("audio_file", help="Path to a 16kHz mono WAV file to analyze")
    args = parser.parse_args()

    print("Loading XLS-R backbone + SLS head...")
    _, device = load_model()
    print(f"Model loaded on {device}. Analyzing audio file: {args.audio_file}")

    result = predict(args.audio_file)

    print()
    print("=" * 50)
    print("XLS-R + SLS DEEPFAKE DETECTION RESULT")
    print("=" * 50)
    print(f"bonafide (real) : {result['bonafide']:.4f}")
    print(f"spoof (fake)    : {result['spoof']:.4f}")
    print(f"Verdict         : {'SPOOF/FAKE' if result['spoof'] > result['bonafide'] else 'BONAFIDE/REAL'}")
    print("=" * 50)


if __name__ == "__main__":
    main()
