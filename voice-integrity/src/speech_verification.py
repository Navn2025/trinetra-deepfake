import argparse
import importlib

import torch
import torchaudio

# torchaudio removed list_audio_backends() entirely in the version on PyPI,
# but speechbrain's check_torchaudio_backend() (run unconditionally at
# import time) still calls it as a sanity check. The actual audio loading
# speechbrain does for verify_files() goes through `soundfile` directly, not
# this torchaudio backend-dispatch API, so the check itself is vestigial --
# this just satisfies it without needing a torchaudio version that doesn't
# exist on PyPI yet.
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

# speechbrain.dataio.audio_io is normally accessed lazily (speechbrain's
# LazyModule machinery). Doing that lazily from inside verify_files() hits an
# upstream bug where the lazy import's own error-reporting path (inspecting
# the caller's stack frames) ends up eagerly resolving an unrelated lazy
# module -- speechbrain.integrations.k2_fsa -- which fails because k2 (an
# FST/alignment library this project doesn't use) isn't installed, and *that*
# failure is what surfaces as "Lazy import of ... k2_fsa failed" even though
# nothing here ever asked for k2. Importing audio_io directly, before
# speechbrain.inference.speaker touches it, replaces the lazy placeholder
# with the real module and avoids that code path entirely.
importlib.import_module("speechbrain.dataio.audio_io")

from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy

_verifier = None


def load_verifier() -> SpeakerRecognition:
    """Loaded once and reused -- ECAPA-TDNN, ~20MB, cheap to keep resident."""
    global _verifier
    if _verifier is None:
        _verifier = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            local_strategy=LocalStrategy.COPY,
        )
    return _verifier


def verify(reference_audio, incoming_audio) -> tuple[float, bool]:
    """Returns (cosine_similarity, same_speaker). Checks voice identity only
    -- a convincing clone of the reference voice WILL pass this; pair with
    an anti-spoofing check (synthetic_detector / xlsr_sls_detector) to catch
    that case."""
    verifier = load_verifier()
    score, prediction = verifier.verify_files(str(reference_audio), str(incoming_audio))
    return float(score), bool(prediction)


def main():
    parser = argparse.ArgumentParser(
        description="ECAPA speaker verification -- checks whether two audio files are the same speaker."
    )
    parser.add_argument("reference_audio", help="Path to the known-genuine reference audio")
    parser.add_argument("incoming_audio", help="Path to the audio being checked")
    args = parser.parse_args()

    print("Loading the pre-trained model...")
    load_verifier()
    print("Model Loaded.")

    score, same_speaker = verify(args.reference_audio, args.incoming_audio)

    print()
    print("=" * 50)
    print("VOICE VERIFICATION RESULT")
    print("=" * 50)
    print(f"Similarity score : {score:.4f}")
    print(f"Same speaker    : {'YES' if same_speaker else 'NO'}")
    print("=" * 50)


if __name__ == "__main__":
    main()
