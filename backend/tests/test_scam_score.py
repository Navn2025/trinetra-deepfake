import pytest

from scam_score import BASE_WEIGHTS, compute_scam_score


def test_no_signals_returns_none():
    result = compute_scam_score()
    assert result == {"scam_likelihood": None, "breakdown": {}}


def test_single_signal_equals_that_signal_after_renormalization():
    # With only one signal present, renormalizing by its own weight means
    # the score should equal the raw signal value.
    result = compute_scam_score(face_fake_probability=0.8)
    assert result["scam_likelihood"] == pytest.approx(0.8, abs=1e-3)
    assert result["breakdown"] == {"face_fake_probability": 0.8}


def test_identity_mismatch_signal_is_inverted_similarity():
    # similarity 0.9 (strong match) -> mismatch signal 0.1 (low risk)
    result = compute_scam_score(face_identity_similarity=0.9)
    assert result["breakdown"]["face_identity_mismatch"] == pytest.approx(0.1, abs=1e-3)


def test_both_signals_weighted_average_matches_base_weights():
    signals = {
        "face_fake_probability": 0.9,
        "face_identity_mismatch": 1.0 - 0.2,  # from similarity=0.2
    }
    result = compute_scam_score(
        face_fake_probability=0.9,
        face_identity_similarity=0.2,
    )
    expected = sum(signals[k] * BASE_WEIGHTS[k] for k in signals) / sum(BASE_WEIGHTS.values())
    assert result["scam_likelihood"] == pytest.approx(round(expected, 3), abs=1e-3)


def test_score_is_always_clamped_to_unit_interval():
    result = compute_scam_score(face_fake_probability=1.0, face_identity_similarity=0.0)
    assert 0.0 <= result["scam_likelihood"] <= 1.0


def test_missing_signals_renormalize_rather_than_deflate_score():
    # A single maximally-suspicious signal should still read as high risk,
    # not diluted toward zero just because other signals are unavailable.
    result = compute_scam_score(face_fake_probability=1.0)
    assert result["scam_likelihood"] == pytest.approx(1.0, abs=1e-3)
