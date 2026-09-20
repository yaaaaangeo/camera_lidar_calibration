from __future__ import annotations

import json

import numpy as np
import pytest

from gui.core.detect_camera import board_model
from gui.core.evaluation.temporal_calibration import (
    DELTA_T_CONVENTION,
    TemporalCalibrationConfig,
    TemporalCalibrationScene,
    estimate_temporal_offset,
    evaluate_delta_t,
    export_temporal_calibration_result,
)
from gui.core.evaluation.temporal_motion_diagnostic import CameraPose, interpolate_camera_pose
from gui.core.project import Target


def test_delta_t_convention_uses_header_minus_board_median():
    assert "t_camera - t_lidar_board" in DELTA_T_CONVENTION
    assert "median timestamp of board ROI points" in DELTA_T_CONVENTION


def test_interpolation_gap_rejects_sparse_camera_trajectory():
    poses = [
        CameraPose(0, 0, np.eye(3), np.zeros(3)),
        CameraPose(1_000_000_000, 1, np.eye(3), np.ones(3)),
    ]

    assert interpolate_camera_pose(poses, 500_000_000, max_gap_ms=100.0) is None


def test_synthetic_temporal_offset_recovers_plus_15ms():
    target = Target()
    model = board_model(target)[1]
    velocity_m_s = np.array([2.0, 0.0, 0.0])
    true_dt_ms = 15.0
    camera_poses = [
        CameraPose(ns, i, np.eye(3), _pose_t(ns, velocity_m_s))
        for i, ns in enumerate(range(0, 1_500_000_001, 50_000_000))
    ]
    scenes = []
    for i, lidar_ns in enumerate(range(200_000_000, 800_000_001, 100_000_000)):
        camera_time_ns = lidar_ns + int(true_dt_ms * 1e6)
        lidar_points = model + _pose_t(camera_time_ns, velocity_m_s)
        scenes.append(TemporalCalibrationScene(
            scene_id=f"s{i}",
            frame_index=i,
            lidar_board_ns=lidar_ns,
            camera_header_ns=camera_time_ns,
            lidar_points_xyz=lidar_points,
            board_point_count=120,
            board_min_ns=lidar_ns - 1_000_000,
            board_mean_ns=lidar_ns,
            board_max_ns=lidar_ns + 1_000_000,
            board_span_ms=2.0,
            motion_group="HIGH",
            translation_velocity_mm_s=2000.0,
        ))

    config = TemporalCalibrationConfig(
        coarse_min_ms=-30,
        coarse_max_ms=30,
        coarse_step_ms=5,
        fine_radius_ms=20,
        fine_step_ms=1,
        min_marginal_scenes=5,
        min_ready_scenes=5,
    )
    result = estimate_temporal_offset(scenes, camera_poses, np.eye(3), np.zeros(3), target, config=config)

    assert result.best_delta_t_ms == pytest.approx(15.0)
    assert result.classification in {"TEMPORAL_OFFSET_SUPPORTED", "TEMPORAL_OFFSET_WEAK"}
    best = min([r for r in result.sweep if np.isfinite(r.rmse_mm)], key=lambda r: r.rmse_mm)
    assert best.rmse_mm < 1e-6


def test_static_board_has_no_temporal_evidence():
    target = Target()
    model = board_model(target)[1]
    poses = [
        CameraPose(ns, i, np.eye(3), np.array([0.1, 0.2, 5.0]))
        for i, ns in enumerate(range(0, 1_000_000_001, 100_000_000))
    ]
    scenes = [
        TemporalCalibrationScene(f"s{i}", i, ns, ns, model + np.array([0.1, 0.2, 5.0]), motion_group="LOW")
        for i, ns in enumerate(range(200_000_000, 800_000_001, 100_000_000))
    ]
    config = TemporalCalibrationConfig(min_marginal_scenes=5, min_ready_scenes=5)

    result = estimate_temporal_offset(scenes, poses, np.eye(3), np.zeros(3), target, config=config)

    assert result.classification == "NO_TEMPORAL_EVIDENCE"
    assert result.best_delta_t_ms is not None


def test_insufficient_scene_count_blocks_estimate():
    target = Target()
    model = board_model(target)[1]
    poses = [
        CameraPose(0, 0, np.eye(3), np.zeros(3)),
        CameraPose(100_000_000, 1, np.eye(3), np.zeros(3)),
    ]
    scenes = [
        TemporalCalibrationScene("one", 1, 50_000_000, 50_000_000, model.copy())
    ]

    result = estimate_temporal_offset(scenes, poses, np.eye(3), np.zeros(3), target)

    assert result.classification == "INSUFFICIENT_VALID_SCENES"
    assert result.best_delta_t_ms is None
    assert result.metadata["record_timestamp_used_for_pairing"] is False
    assert result.metadata["spatial_extrinsic_fixed"] is True
    assert result.metadata["yaml_or_extrinsic_written"] is False


def test_export_temporal_outputs_timestamp_and_scene_csv(tmp_path):
    target = Target()
    model = board_model(target)[1]
    poses = [
        CameraPose(0, 0, np.eye(3), np.zeros(3)),
        CameraPose(100_000_000, 1, np.eye(3), np.zeros(3)),
    ]
    scenes = [
        TemporalCalibrationScene(
            "one",
            1,
            50_000_000,
            65_000_000,
            model.copy(),
            board_point_count=50,
            board_min_ns=49_000_000,
            board_mean_ns=50_000_000,
            board_max_ns=52_000_000,
            board_span_ms=3.0,
        )
    ]
    result = estimate_temporal_offset(scenes, poses, np.eye(3), np.zeros(3), target)
    out = export_temporal_calibration_result(result, tmp_path, camera_poses=poses)
    summary = json.loads((out / "temporal_summary.json").read_text(encoding="utf-8"))
    board_csv = (out / "board_timestamp_diagnostic.csv").read_text(encoding="utf-8")

    assert summary["metadata"]["record_timestamp_used_for_pairing"] is False
    assert "delta_t_observed_ms" in board_csv
    assert "15.0" in board_csv
    assert (out / "temporal_sweep.csv").exists()
    assert (out / "camera_pose_trajectory.csv").exists()


def test_evaluate_delta_t_reports_motion_group_rmse():
    target = Target()
    model = board_model(target)[1]
    poses = [
        CameraPose(0, 0, np.eye(3), np.zeros(3)),
        CameraPose(100_000_000, 1, np.eye(3), np.zeros(3)),
    ]
    scene = TemporalCalibrationScene(
        "high",
        1,
        50_000_000,
        50_000_000,
        model.copy(),
        motion_group="HIGH",
    )

    metric, residuals = evaluate_delta_t([scene], poses, np.eye(3), np.zeros(3), target, 0.0)

    assert metric.valid_scene_count == 1
    assert metric.high_motion_rmse_mm == pytest.approx(0.0)
    assert residuals[0].motion_group == "HIGH"


def _pose_t(timestamp_ns: int, velocity_m_s: np.ndarray) -> np.ndarray:
    return np.array([0.0, 0.0, 3.0]) + velocity_m_s * (timestamp_ns / 1e9)
