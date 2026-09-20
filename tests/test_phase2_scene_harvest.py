from __future__ import annotations

import json

import numpy as np

from gui.core.evaluation.phase2_scene_harvest import (
    HarvestConfig,
    HarvestResult,
    HarvestScene,
    export_harvest_result,
    motion_group,
    nearest_camera_sample,
    remove_duplicate_scenes,
    temporal_readiness,
    CameraDetectionSample,
)


def _scene(idx: int, t_ns: int, offset: float = 0.0) -> HarvestScene:
    lidar = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.4, 0.0],
        [0.0, 0.4, 0.0],
    ], dtype=float) + np.array([offset, 0.0, 0.0])
    camera = lidar + np.array([0.0, 0.0, 3.0])
    return HarvestScene(
        frame_index=idx,
        camera_index=idx,
        lidar_index=idx,
        camera_header_ns=t_ns,
        lidar_record_ns=t_ns,
        sweep_start_ns=t_ns,
        sweep_mid_ns=t_ns,
        sweep_end_ns=t_ns,
        board_median_ns=t_ns,
        board_span_ms=3.0,
        camera_minus_board_ms=0.0,
        camera_reproj_rms_px=0.5,
        camera_reproj_p95_px=1.0,
        lidar_centers_xyz=lidar,
        camera_centers_xyz=camera,
        lidar_plane_point_count=100,
        lidar_boundary_count=20,
        lidar_circle_count=4,
        lidar_cluster_count=4,
        lidar_plane_rms=0.003,
        pose_x_m=offset,
        pose_yaw_deg=offset * 10.0,
    )


def test_nearest_camera_sample_pairs_by_timestamp():
    samples = [
        CameraDetectionSample(100, 90, 0, True),
        CameraDetectionSample(250, 240, 1, True),
    ]

    assert nearest_camera_sample(samples, 180).header_ns == 250


def test_duplicate_removal_uses_time_and_pose_separation():
    scenes = [
        _scene(1, 1_000_000_000, 0.0),
        _scene(2, 1_100_000_000, 0.01),
        _scene(3, 2_000_000_000, 0.50),
    ]

    selected = remove_duplicate_scenes(scenes, HarvestConfig())

    assert [s.frame_index for s in selected] == [1, 3]


def test_motion_group_binning():
    assert motion_group(50.0, 0.2) == "LOW"
    assert motion_group(400.0, 1.0) == "MID"
    assert motion_group(900.0, 8.0) == "HIGH"


def test_readiness_thresholds():
    motion = {"groups": {"LOW": 5, "MID": 5, "HIGH": 0, "UNKNOWN": 0}}
    assert temporal_readiness([_scene(i, i) for i in range(10)], motion, {})[0] == "READY"
    assert temporal_readiness([_scene(i, i) for i in range(6)], motion, {})[0] == "MARGINAL"
    assert temporal_readiness([_scene(i, i) for i in range(2)], motion, {})[0] == "INSUFFICIENT"


def test_harvest_persistence_writes_summary_and_scene_files(tmp_path):
    scene = _scene(407, 10)
    result = HarvestResult(
        total_lidar_frames=1,
        funnel_counts={k: 1 for k in [
            "TOTAL_LIDAR_FRAMES", "CAMERA_PAIR_FOUND", "CAMERA_DETECTION_VALID",
            "CAMERA_PNP_VALID", "CAMERA_PNP_QUALITY_PASS", "LIDAR_ROI_VALID",
            "LIDAR_PLANE_VALID", "LIDAR_BOUNDARY_VALID", "LIDAR_CIRCLE_GE_1",
            "LIDAR_CIRCLE_GE_2", "LIDAR_CIRCLE_GE_3", "LIDAR_CIRCLE_GE_4",
            "LIDAR_GEOMETRY_PASS", "LIDAR_FULL_TARGET_VALID", "TIMESTAMP_VALID",
            "FOUR_CORRESPONDENCES_CREATED", "FINAL_VALID_SCENES",
        ]},
        failure_reasons={},
        raw_scenes=[scene],
        selected_scenes=[scene],
        motion_distribution={"groups": {"LOW": 1}},
        pose_diversity={},
        readiness="INSUFFICIENT",
        readiness_reason="test",
        metadata={},
    )

    out = export_harvest_result(result, tmp_path)
    summary = json.loads((out / "harvest_summary.json").read_text(encoding="utf-8"))

    assert summary["scene407_detected"] is True
    assert (out / "scene_00407.json").exists()
    assert (out / "valid_scenes.csv").exists()
    assert (out / "detection_funnel.csv").exists()
