"""
Scam Likelihood Score -- a transparent, rule-based combination of signals,
not a trained fusion model. There's no labeled multimodal scam-call dataset
to train a real fusion classifier on, and a rule-based score is more
explainable anyway: the breakdown returned here is meant to be shown to the
user directly ("flagged because: face doesn't match contact + deepfake
artifacts detected"), not hidden behind a black-box number.

Combines up to two independent signals, whichever are available for a
given call:
  - face_fake_probability     : is the video face AI-generated (predictor.py)
  - face_identity_mismatch    : does the face NOT match the claimed contact

Base weights (used when both are present) sum to 0.6; when a signal is
missing, the remaining weight is renormalized so the score stays well-
scaled regardless of how many signals a given call actually has.

Voice signals (voice_fake_probability, voice_identity_mismatch) were removed
after voice_deepfake.py's 3-model ensemble was validated against known
synthetic samples and consistently failed to detect them (all three models
independently scored known-fake clips as 2-7% fake) -- see project history.
A confidently-wrong signal is worse than no signal for a scam-detection
score, so it was dropped rather than kept in as a weak/misleading input.
"""

BASE_WEIGHTS = {
    "face_fake_probability": 0.35,
    "face_identity_mismatch": 0.25,
}


def compute_scam_score(
    face_fake_probability: float | None = None,
    face_identity_similarity: float | None = None,
) -> dict:
    signals = {}

    if face_fake_probability is not None:
        signals["face_fake_probability"] = face_fake_probability
    if face_identity_similarity is not None:
        signals["face_identity_mismatch"] = 1.0 - face_identity_similarity

    if not signals:
        return {"scam_likelihood": None, "breakdown": {}}

    total_weight = sum(BASE_WEIGHTS[k] for k in signals)
    score = sum(signals[k] * BASE_WEIGHTS[k] for k in signals) / total_weight
    score = min(max(score, 0.0), 1.0)

    breakdown = {k: round(v, 3) for k, v in signals.items()}

    return {"scam_likelihood": round(score, 3), "breakdown": breakdown}
