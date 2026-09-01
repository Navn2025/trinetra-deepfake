"""
video_pipeline aggregation logic (sampling, median outlier rejection,
suspicious-segment run-length filtering), with per-frame face
detection/scoring mocked out. Real decode + real model inference is covered
separately by scripts/make_test_video.py-based manual verification (see the
session that added this module) and test_model_integration.py.
"""
from video_pipeline import _aggregate, _find_suspicious_segments, _sample_frame_indices


def _frame(t, prob, classification=None):
    return {
        "timestamp": t,
        "fake_probability": prob,
        "classification": classification or ("no_face_detected" if prob is None else "x"),
    }


def test_sample_frame_indices_respects_min_and_max():
    # Very short video: still get at least MIN_VIDEO_SAMPLE_FRAMES (bounded
    # by total_frames itself, obviously can't sample more frames than exist)
    indices = _sample_frame_indices(total_frames=3, duration_seconds=0.5)
    assert indices == sorted(set(indices))
    assert len(indices) <= 3

    # Long video: capped at MAX_VIDEO_SAMPLE_FRAMES, not ~4 samples per second
    indices = _sample_frame_indices(total_frames=100000, duration_seconds=3600)
    assert len(indices) <= 120


def test_sample_frame_indices_are_evenly_spread_and_in_bounds():
    total_frames = 1000
    indices = _sample_frame_indices(total_frames=total_frames, duration_seconds=100)
    assert indices[0] >= 0
    assert indices[-1] <= total_frames - 1
    assert indices == sorted(indices)
    assert len(set(indices)) == len(indices)  # no duplicates


def test_aggregate_with_no_faces_at_all_reports_no_face_detected():
    frames = [_frame(0.0, None), _frame(1.0, None)]
    result = _aggregate(frames)
    assert result["overall_classification"] == "no_face_detected"
    assert result["overall_confidence"] is None
    assert result["suspicious_segments"] == []


def test_aggregate_distinguishes_no_face_from_quality_rejected():
    # Faces WERE detected in every sampled frame, but all were too
    # low-quality to score -- should not be conflated with "no face ever
    # appeared in this video" (a materially different finding for the user).
    frames = [_frame(0.0, None, "insufficient_quality"), _frame(1.0, None, "insufficient_quality")]
    result = _aggregate(frames)
    assert result["overall_classification"] == "insufficient_quality"
    assert result["overall_confidence"] is None


def test_aggregate_uses_median_not_mean_for_outlier_rejection():
    # One wildly anomalous frame (1.0) shouldn't drag the overall score as
    # far as a mean would -- median should sit near the cluster of normal
    # frames instead.
    frames = [_frame(i, p) for i, p in enumerate([0.1, 0.12, 0.11, 0.13, 1.0])]
    result = _aggregate(frames)
    mean = sum(f["fake_probability"] for f in frames) / len(frames)
    assert result["overall_confidence"] < mean
    assert result["overall_confidence"] == 0.12  # the true median


def test_single_spiking_frame_does_not_create_a_suspicious_segment():
    frames = [_frame(0, 0.1), _frame(1, 0.9), _frame(2, 0.1), _frame(3, 0.1)]
    segments = _find_suspicious_segments(frames)
    assert segments == []  # run length 1 < MIN_SUSPICIOUS_RUN_FRAMES (2)


def test_sustained_run_creates_a_suspicious_segment():
    frames = [_frame(0, 0.1), _frame(1, 0.9), _frame(2, 0.95), _frame(3, 0.1)]
    segments = _find_suspicious_segments(frames)
    assert len(segments) == 1
    assert segments[0]["start_timestamp"] == 1
    assert segments[0]["end_timestamp"] == 2
    assert segments[0]["frame_count"] == 2
    assert segments[0]["peak_fake_probability"] == 0.95


def test_multiple_separate_suspicious_segments_are_both_reported():
    frames = [
        _frame(0, 0.9), _frame(1, 0.9),  # segment 1
        _frame(2, 0.1),
        _frame(3, 0.8), _frame(4, 0.85), _frame(5, 0.7),  # segment 2
    ]
    segments = _find_suspicious_segments(frames)
    assert len(segments) == 2
    assert segments[0]["frame_count"] == 2
    assert segments[1]["frame_count"] == 3


def test_trailing_suspicious_run_is_still_flushed():
    # Regression check: a run that extends to the very last frame must
    # still be reported (no frame after it to trigger the flush).
    frames = [_frame(0, 0.1), _frame(1, 0.9), _frame(2, 0.9)]
    segments = _find_suspicious_segments(frames)
    assert len(segments) == 1
    assert segments[0]["frame_count"] == 2
