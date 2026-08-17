from config import FAKE_THRESHOLD, REAL_THRESHOLD, classify_fake_probability


def test_classify_none_is_insufficient_quality():
    assert classify_fake_probability(None) == "insufficient_quality"


def test_classify_high_probability_is_fake():
    assert classify_fake_probability(FAKE_THRESHOLD) == "fake"
    assert classify_fake_probability(0.99) == "fake"


def test_classify_low_probability_is_real():
    assert classify_fake_probability(REAL_THRESHOLD) == "real"
    assert classify_fake_probability(0.01) == "real"


def test_classify_middle_probability_is_uncertain():
    midpoint = (FAKE_THRESHOLD + REAL_THRESHOLD) / 2
    assert classify_fake_probability(midpoint) == "uncertain"


def test_no_probability_is_ever_forced_into_real_or_fake_at_boundaries():
    # Just inside each boundary should NOT cross into the opposite class.
    assert classify_fake_probability(REAL_THRESHOLD + 0.001) == "uncertain"
    assert classify_fake_probability(FAKE_THRESHOLD - 0.001) == "uncertain"
