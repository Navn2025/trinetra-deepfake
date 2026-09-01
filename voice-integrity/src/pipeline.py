"""End-to-end voice-integrity pipeline: given a reference (known-genuine)
clip and an incoming clip to check, runs all three detectors together and
prints one consolidated verdict.

    python src/pipeline.py --reference data/reference/ref.wav --incoming data/incoming/cloned_denoised_20s.wav

Checks run:
  1. Speaker verification (speech_verification.py, ECAPA) -- is this the
     same voice as the reference? Does NOT detect synthetic speech on its
     own -- a good clone WILL legitimately pass this. It's here to prove
     identity, not authenticity.
  2. Gustking wav2vec2-XLSR anti-spoofing (synthetic_detector.py) -- fast,
     ~93% in-domain accuracy.
  3. XLS-R + SLS anti-spoofing (xlsr_sls_detector.py) -- slower (real
     XLS-R 300M forward pass), best cross-dataset generalization of the
     three (see that module's docstring for the actual EER numbers).

The overall verdict is FAKE/SPOOFED if either anti-spoofing check flags the
incoming clip, regardless of what the identity check says -- a clone that
passes speaker verification but fails both spoofing checks is exactly the
attack this pipeline exists to catch (see conversation/commit history for a
worked example: 0.66 similarity + "same speaker: YES" alongside a 100%
"spoof" verdict from XLS-R+SLS on the same clip).

Both anti-spoofing checks need 16kHz mono input; audio_utils auto-resamples
the incoming clip into data/.cache/ if it isn't already (cached by source
mtime -- repeat runs on the same file skip re-encoding).
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "data" / ".cache"

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audio_utils  # noqa: E402
import speech_verification  # noqa: E402
import synthetic_detector  # noqa: E402
import xlsr_sls_detector  # noqa: E402

SPOOF_THRESHOLD = 0.5


def run_pipeline(reference_path, incoming_path) -> dict:
    incoming_16k = audio_utils.resample_to_16k_mono(incoming_path, CACHE_DIR)

    print("Loading models (speaker verification, Gustking, XLS-R+SLS)...")
    speech_verification.load_verifier()
    synthetic_detector.load_model()
    xlsr_sls_detector.load_model()
    print("All models loaded.\n")

    print(f"[1/3] Identity check: {reference_path}  vs  {incoming_path}")
    similarity, same_speaker = speech_verification.verify(reference_path, incoming_path)

    print(f"[2/3] Gustking anti-spoofing on {incoming_16k.name}...")
    gustking_labels = synthetic_detector.predict(incoming_16k)

    print(f"[3/3] XLS-R+SLS anti-spoofing on {incoming_16k.name}...")
    sls_result = xlsr_sls_detector.predict(incoming_16k)

    gustking_fake_prob = gustking_labels.get("fake", max(gustking_labels.values()))
    gustking_flagged = gustking_fake_prob >= SPOOF_THRESHOLD
    sls_flagged = sls_result["spoof"] >= SPOOF_THRESHOLD

    return {
        "reference": str(reference_path),
        "incoming": str(incoming_path),
        "identity": {"similarity": similarity, "same_speaker": same_speaker},
        "gustking": {**gustking_labels, "flagged_fake": gustking_flagged},
        "xlsr_sls": {**sls_result, "flagged_fake": sls_flagged},
        "overall_verdict": "FAKE/SPOOFED" if (gustking_flagged or sls_flagged) else "LIKELY GENUINE",
    }


def print_report(result: dict) -> None:
    print()
    print("=" * 60)
    print("VOICE INTEGRITY PIPELINE REPORT")
    print("=" * 60)
    print(f"Reference : {result['reference']}")
    print(f"Incoming  : {result['incoming']}")
    print("-" * 60)

    ident = result["identity"]
    print(f"[Identity]   similarity={ident['similarity']:.4f}  same_speaker={'YES' if ident['same_speaker'] else 'NO'}")
    print("             (voice match only -- a good clone WILL pass this)")

    g = result["gustking"]
    g_probs = "  ".join(f"{k}={v:.4f}" for k, v in g.items() if k != "flagged_fake")
    print(f"[Gustking]   {g_probs}  {'-> FLAGGED' if g['flagged_fake'] else ''}")

    s = result["xlsr_sls"]
    print(f"[XLS-R+SLS]  bonafide={s['bonafide']:.4f}  spoof={s['spoof']:.4f}  {'-> FLAGGED' if s['flagged_fake'] else ''}")

    print("-" * 60)
    print(f"OVERALL VERDICT: {result['overall_verdict']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run the full voice-integrity detection pipeline on one clip.")
    parser.add_argument("--reference", required=True, help="Known-genuine reference audio to check identity against")
    parser.add_argument("--incoming", required=True, help="Audio clip to check")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of the text report")
    args = parser.parse_args()

    start = time.monotonic()
    result = run_pipeline(args.reference, args.incoming)
    result["elapsed_s"] = round(time.monotonic() - start, 2)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_report(result)


if __name__ == "__main__":
    main()
