from __future__ import annotations

import numpy as np

from gui.core.evaluation.board_timestamp import (
    BoardTimestampResult,
    camera_pairing_for_board_time,
    export_frames_csv,
    frame_from_timestamps,
    summarize,
)


def test_board_timestamp_frame_uses_board_median_and_sign_convention():
    sweep_start = 1_000_000_000
    sweep_end = 1_100_000_000
    camera = 1_075_000_000
    frame = frame_from_timestamps(
        frame_index=3,
        lidar_record_ns=1_200_000_000,
        camera_header_ns=camera,
        sweep_start_ns=sweep_start,
        sweep_end_ns=sweep_end,
        board_point_timestamps_s=np.array([1.020, 1.030, 1.040]),
        status="STABLE",
    )

    assert frame.sweep_mid_ns == 1_050_000_000
    assert frame.board_median_offset_from_start_ms == 30.0
    assert frame.board_span_ms == 20.0
    assert frame.camera_to_board_median_ms == 45.0
    assert frame.camera_to_sweep_mid_ms == 25.0


def test_board_timestamp_summary_all_and_stable():
    frames = [
        frame_from_timestamps(0, 0, 1_050_000_000, 1_000_000_000, 1_100_000_000, [1.020, 1.030], "STABLE"),
        frame_from_timestamps(1, 0, 1_040_000_000, 1_000_000_000, 1_100_000_000, [1.060, 1.070], "MOVING"),
    ]

    all_stats = summarize(frames)
    stable_stats = summarize([f for f in frames if f.status == "STABLE"])

    assert all_stats["n"] == 2
    assert stable_stats["n"] == 1
    assert all_stats["camera_to_board_median"].median_ms == 0.0
    assert stable_stats["camera_to_board_median"].median_ms == 25.0


def test_board_timestamp_csv_export(tmp_path):
    frame = frame_from_timestamps(0, 0, 1_050_000_000, 1_000_000_000, 1_100_000_000, [1.020, 1.030], "STABLE")
    path = export_frames_csv(BoardTimestampResult(frames=[frame]), tmp_path / "board_times.csv")
    text = path.read_text()

    assert "board_median_offset_from_start_ms" in text
    assert "camera_to_board_median_ms" in text
    assert "STABLE" in text


def test_board_time_header_nearest_pairing_detects_previous_frame_shift():
    camera_headers = [
        (10, 1_000_000_000, 1_000_000_000),
        (11, 1_100_000_000, 1_100_000_000),
        (12, 1_200_000_000, 1_200_000_000),
    ]
    frame = frame_from_timestamps(
        frame_index=0,
        lidar_record_ns=1_115_000_000,
        camera_header_ns=1_100_000_000,
        sweep_start_ns=1_000_000_000,
        sweep_end_ns=1_100_000_000,
        board_point_timestamps_s=[1.085],
        current_camera_index=11,
        current_camera_record_ns=1_100_000_000,
        camera_headers=camera_headers,
    )

    assert frame.camera_to_board_median_ms == 15.0
    assert frame.nearest_camera_index == 11
    assert frame.nearest_offset_ms == 15.0
    assert frame.previous_offset_ms == -85.0
    assert frame.next_offset_ms == 115.0


def test_board_time_header_nearest_can_select_previous_camera():
    camera_headers = [
        (10, 1_000_000_000, 1_000_000_000),
        (11, 1_100_000_000, 1_100_000_000),
    ]
    pairing = camera_pairing_for_board_time(
        board_ns=1_015_000_000,
        current_camera_index=11,
        camera_headers=camera_headers,
    )

    assert pairing["nearest_camera_index"] == 10
    assert pairing["camera_index_shift"] == -1
    assert pairing["nearest_offset_ms"] == -15.0
