from __future__ import annotations

import json

import numpy as np
import pytest

from gui.core.evaluation.calibration_correspondence import (
    SceneMetadata,
    analyze_correspondences,
    export_correspondence_diagnostics,
    geometry_condition,
    leave_one_out_pair_errors,
    swapped_ordering_warning,
)
from gui.core.detect_camera import CIRCLE_PHYSICAL_IDS, board_circle_metadata
from gui.core.project import Target
from gui.core.solve import solve, solve_rigid


def _rigid_transform():
    theta = np.deg2rad(12.0)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    t = np.array([0.3, -0.2, 1.1])
    return R, t


def _rectangle(z=3.0):
    return np.array([
        [-0.25, -0.20, z],
        [0.25, -0.20, z],
        [0.25, 0.20, z],
        [-0.25, 0.20, z],
    ], dtype=float)


def test_correspondence_json_serialization_and_residuals(tmp_path):
    R, t = _rigid_transform()
    lidar = _rectangle()
    camera = (R @ lidar.T).T + t
    sol = solve([("407", lidar, camera)])
    assert sol.ok

    scenes, summary, out_dir = export_correspondence_diagnostics(
        [("407", lidar, camera)],
        sol.R,
        sol.t,
        metadata={"407": SceneMetadata("407", camera_timestamp_ns=10, lidar_timestamp_ns=9)},
        out_dir=tmp_path,
    )

    assert out_dir == tmp_path
    assert len(scenes) == 1
    assert summary["pair_count"] == 4
    assert scenes[0].fit_stats_mm["rmse"] < 1e-6

    data = json.loads((tmp_path / "scene_00407.json").read_text(encoding="utf-8"))
    assert data["scene_index"] == "407"
    assert data["camera_timestamp_ns"] == 10
    assert len(data["pairs"]) == 4
    assert data["pairs"][0]["feature_id"] == "ordered_0"
    assert (tmp_path / "all_correspondences.csv").exists()
    assert (tmp_path / "scene_quality_summary.csv").exists()


def test_camera_aruco_physical_order_is_available_for_step6_metadata(tmp_path):
    target = Target()
    meta = board_circle_metadata(target)
    assert [m["marker_id"] for m in meta] == [1, 2, 4, 3]
    assert [m["physical_id"] for m in meta] == CIRCLE_PHYSICAL_IDS

    R, t = _rigid_transform()
    lidar = _rectangle()
    camera = (R @ lidar.T).T + t
    sol = solve([("407", lidar, camera)])
    export_correspondence_diagnostics(
        [("407", lidar, camera)],
        sol.R,
        sol.t,
        metadata={
            "407": SceneMetadata(
                "407",
                camera_physical_ids=[m["physical_id"] for m in meta],
                camera_marker_ids=[m["marker_id"] for m in meta],
                camera_board_xy=[m["board_xy"] for m in meta],
            )
        },
        out_dir=tmp_path,
    )

    data = json.loads((tmp_path / "scene_00407.json").read_text(encoding="utf-8"))
    assert data["camera_marker_ids"] == [1, 2, 4, 3]
    assert data["camera_physical_ids"] == [
        "BOARD_TOP_LEFT",
        "BOARD_TOP_RIGHT",
        "BOARD_BOTTOM_RIGHT",
        "BOARD_BOTTOM_LEFT",
    ]
    assert data["pairs"][0]["feature_id"] == "ordered_0"
    assert data["camera_physical_ids"][0] != data["pairs"][0]["feature_id"]


def test_known_synthetic_rigid_transform_round_trips():
    R, t = _rigid_transform()
    lidar = _rectangle()
    camera = (R @ lidar.T).T + t
    found_R, found_t = solve_rigid(lidar, camera)
    assert found_R == pytest.approx(R)
    assert found_t == pytest.approx(t)


def test_leave_one_out_is_small_for_consistent_rectangle():
    R, t = _rigid_transform()
    lidar = _rectangle()
    camera = (R @ lidar.T).T + t
    errors, warnings = leave_one_out_pair_errors(lidar, camera)
    assert max(errors) < 1e-6
    assert warnings == []


def test_degenerate_geometry_warning_for_collinear_points():
    pts = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [0.3, 0.0, 0.0],
    ])
    geom = geometry_condition(pts)
    assert "NEARLY_COLLINEAR" in geom["warnings"]


def test_intentionally_swapped_ordering_is_flagged():
    R, t = _rigid_transform()
    lidar = _rectangle()
    camera = (R @ lidar.T).T + t
    swapped_lidar = np.roll(lidar, 1, axis=0)
    warning = swapped_ordering_warning(swapped_lidar, camera, R, t)
    assert warning is not None
    assert "POSSIBLE_ORDERING_SWAP" in warning


def test_existing_calibration_regression_matches_solve_rmse():
    R, t = _rigid_transform()
    lidar1 = _rectangle(z=3.0)
    lidar2 = _rectangle(z=4.0) + np.array([0.4, 0.1, 0.0])
    camera1 = (R @ lidar1.T).T + t
    camera2 = (R @ lidar2.T).T + t
    scenes_in = [("s1", lidar1, camera1), ("s2", lidar2, camera2)]

    sol = solve(scenes_in)
    diags, summary = analyze_correspondences(scenes_in, sol.R, sol.t)

    assert sol.ok
    assert summary["fit_stats_mm"]["rmse"] == pytest.approx(sol.rmse * 1000.0)
    assert {d.scene_index for d in diags} == {"s1", "s2"}
