from __future__ import annotations

import numpy as np

from gui.core.evaluation.board_motion import (
    MARGINAL,
    MOVING,
    STABLE,
    BoardPoseSample,
    analyze_board_motion,
    classify_motion,
    export_stable_frames_csv,
    low_motion_frame_indices,
)


def _rz(deg: float) -> np.ndarray:
    th = np.radians(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _sample(i: int, t, yaw_deg: float = 0.0) -> BoardPoseSample:
    return BoardPoseSample(timestamp_ns=i * 100_000_000, frame_index=i, R=_rz(yaw_deg), t=np.asarray(t, float))


def test_static_board_has_zero_motion_after_first_frame():
    samples = [_sample(i, [1.0, 2.0, 3.0]) for i in range(5)]
    result = analyze_board_motion(samples, translation_threshold_mm=3.0, rotation_threshold_deg=0.1)

    assert result.total_detected == 5
    assert result.stable_frames == 4
    assert result.moving_frames == 0
    assert np.isclose(result.translation_max_mm, 0.0)
    assert np.isclose(result.rotation_max_deg, 0.0)


def test_translation_only_motion_is_measured_in_mm():
    samples = [
        _sample(0, [0.0, 0.0, 1.0]),
        _sample(1, [0.002, 0.0, 1.0]),
        _sample(2, [0.012, 0.0, 1.0]),
    ]
    result = analyze_board_motion(samples, translation_threshold_mm=5.0, rotation_threshold_deg=0.1)

    assert np.isclose(result.frames[1].translation_delta_mm, 2.0)
    assert np.isclose(result.frames[2].translation_delta_mm, 10.0)
    assert result.frames[1].status == STABLE
    assert result.frames[2].status == MOVING


def test_rotation_only_motion_is_measured_in_degrees():
    samples = [
        _sample(0, [0.0, 0.0, 1.0], yaw_deg=0.0),
        _sample(1, [0.0, 0.0, 1.0], yaw_deg=0.05),
        _sample(2, [0.0, 0.0, 1.0], yaw_deg=0.4),
    ]
    result = analyze_board_motion(samples, translation_threshold_mm=5.0, rotation_threshold_deg=0.1)

    assert np.isclose(result.frames[1].rotation_delta_deg, 0.05, atol=1e-6)
    assert np.isclose(result.frames[2].rotation_delta_deg, 0.35, atol=1e-6)
    assert result.frames[1].status == STABLE
    assert result.frames[2].status == MOVING


def test_stable_marginal_moving_classification():
    assert classify_motion(1.0, 0.02, 5.0, 0.1) == STABLE
    assert classify_motion(4.0, 0.02, 5.0, 0.1) == MARGINAL
    assert classify_motion(6.0, 0.02, 5.0, 0.1) == MOVING
    assert classify_motion(1.0, 0.2, 5.0, 0.1) == MOVING


def test_stable_segments_and_csv_export(tmp_path):
    samples = [
        _sample(0, [0.0, 0.0, 1.0]),
        _sample(1, [0.001, 0.0, 1.0]),
        _sample(2, [0.002, 0.0, 1.0]),
        _sample(3, [0.030, 0.0, 1.0]),
        _sample(4, [0.031, 0.0, 1.0]),
    ]
    result = analyze_board_motion(samples, translation_threshold_mm=5.0, rotation_threshold_deg=0.1)

    assert [(s.start_frame_index, s.end_frame_index) for s in result.stable_segments] == [(1, 2)]
    out = export_stable_frames_csv(result, tmp_path / "stable_board_frames.csv")
    text = out.read_text()
    assert "timestamp_ns,frame_index,translation_delta_mm,rotation_delta_deg,motion_score,status" in text
    assert ",STABLE" in text
    assert ",MOVING" not in text


def test_motion_distribution_and_low_motion_subset():
    samples = [_sample(0, [0.0, 0.0, 1.0])]
    total_mm = 0.0
    for i, delta_mm in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], start=1):
        total_mm += delta_mm
        samples.append(_sample(i, [total_mm / 1000.0, 0.0, 1.0]))
    result = analyze_board_motion(samples, translation_threshold_mm=10.0, rotation_threshold_deg=1.0)

    assert np.isclose(result.translation_p10_mm, 1.9)
    assert np.isclose(result.translation_p25_mm, 3.25)
    assert np.isfinite(result.rotation_p90_deg)
    assert low_motion_frame_indices(result, 0.2) == {1, 2}
