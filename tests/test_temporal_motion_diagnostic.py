from __future__ import annotations

import json

import numpy as np
import pytest

from gui.core.evaluation.temporal_motion_diagnostic import (
    CameraPose,
    DELTA_T_SIGN_CONVENTION,
    TemporalScene,
    classify_temporal_result,
    export_temporal_motion_diagnostic,
    interpolate_camera_pose,
    run_temporal_motion_diagnostic,
)
from gui.core.project import Target


def test_delta_t_sign_convention_is_explicit():
    assert "t_camera_effective - t_lidar_board" in DELTA_T_SIGN_CONVENTION
    assert "Positive means" in DELTA_T_SIGN_CONVENTION


def test_camera_pose_interpolation_linear_translation_and_slerp():
    R0 = np.eye(3)
    theta = np.deg2rad(90.0)
    R1 = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    poses = [
        CameraPose(0, 0, R0, np.array([0.0, 0.0, 0.0])),
        CameraPose(100_000_000, 1, R1, np.array([1.0, 0.0, 0.0])),
    ]

    R, t = interpolate_camera_pose(poses, 50_000_000)

    assert t == pytest.approx([0.5, 0.0, 0.0])
    assert np.degrees(np.arctan2(R[1, 0], R[0, 0])) == pytest.approx(45.0)


def test_classification_requires_multiple_valid_scenes():
    scene = TemporalScene(
        scene_id="407",
        frame_index=407,
        camera_header_ns=110,
        lidar_board_ns=100,
        original_lidar_timestamp_ns=100,
        timestamp_difference_ms=10.0,
        lidar_points_ordered=np.zeros((4, 3)),
        stored_camera_points=np.zeros((4, 3)),
    )

    status, reason = classify_temporal_result([scene], {"NORMAL": {}}, min_valid_scenes=3)

    assert status == "D. INSUFFICIENT_VALID_SCENES"
    assert "Only 1" in reason


def test_export_temporal_motion_diagnostic_writes_summary(tmp_path):
    result = run_temporal_motion_diagnostic(
        correspondence_dir=tmp_path,
        board_pairing_csv=_empty_csv(tmp_path),
        target=Target(),
        camera_poses=[],
        current_global_projection_json=None,
    )
    out = export_temporal_motion_diagnostic(result, tmp_path / "out")
    data = json.loads((out / "temporal_motion_diagnostic.json").read_text(encoding="utf-8"))

    assert data["classification"] == "D. INSUFFICIENT_VALID_SCENES"
    assert (out / "temporal_sweep.csv").exists()
    assert (out / "valid_scenes.csv").exists()
    assert (out / "summary.txt").exists()


def _empty_csv(tmp_path):
    path = tmp_path / "board.csv"
    path.write_text(
        "frame_index,lidar_record_ns,board_median_ns,board_timestamp_valid\n",
        encoding="utf-8",
    )
    return path
