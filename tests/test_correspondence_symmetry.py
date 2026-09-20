from __future__ import annotations

import numpy as np
import pytest

from gui.core.evaluation.correspondence_symmetry import (
    SYMMETRY_CANDIDATES,
    PlausibilityThresholds,
    ProjectionProxyParams,
    analyze_calibration_symmetry,
    analyze_symmetry_candidates,
    candidate_permutations,
    decide_candidate,
    diagnostic_status,
    export_candidate_selection_evidence,
    lightweight_projection_proxy,
    pairwise_distance_matrix,
    pca_plane_xy,
)


def _rigid_transform():
    theta = np.deg2rad(18.0)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    t = np.array([0.5, -0.3, 2.0])
    return R, t


def _rectangle():
    return np.array([
        [-0.25, -0.20, 4.0],
        [0.25, -0.20, 4.0],
        [0.25, 0.20, 4.0],
        [-0.25, 0.20, 4.0],
    ])


def _mark_cross_scene_good(results):
    for r in results:
        r.cross_scene_summary = {
            "scene_count": 3,
            "rotation_deg": {"median": 0.5, "p95": 1.0, "max": 1.2},
            "translation_mm": {"median": 10.0, "p95": 20.0, "max": 25.0},
            "per_scene": [],
        }


def test_symmetric_rectangle_has_multiple_equal_rmse_candidates():
    R, t = _rigid_transform()
    lidar = _rectangle()
    camera = (R @ lidar.T).T + t

    results = analyze_symmetry_candidates(lidar, camera, include_all_permutations=True)
    named = {r.name: r for r in results if r.name in SYMMETRY_CANDIDATES}

    rmses = [named[name].fit_stats_mm["rmse"] for name in ("NORMAL", "REVERSED", "MIRROR", "HALF_TURN")]
    assert max(rmses) - min(rmses) < 1e-9
    assert diagnostic_status(results) == "MULTIPLE_SYMMETRIC_SOLUTIONS"
    decision = decide_candidate(results)
    assert decision.status == "CALIBRATION_AMBIGUOUS"
    assert decision.selection_source == "AMBIGUOUS"
    assert decision.preferred_candidate_name is None


def test_half_turn_case_has_same_rmse_as_normal():
    R, t = _rigid_transform()
    lidar = _rectangle()
    camera = (R @ lidar.T).T + t
    results = {r.name: r for r in analyze_symmetry_candidates(lidar, camera, include_all_permutations=True)}

    assert results["NORMAL"].fit_stats_mm["rmse"] == pytest.approx(
        results["HALF_TURN"].fit_stats_mm["rmse"], abs=1e-9
    )
    assert results["NORMAL"].loo_stats_mm["max"] == pytest.approx(
        results["HALF_TURN"].loo_stats_mm["max"], abs=1e-9
    )


def test_asymmetric_target_yields_unique_best_permutation():
    R, t = _rigid_transform()
    lidar = _rectangle().copy()
    lidar[2] += np.array([0.07, 0.0, 0.0])
    camera = (R @ lidar.T).T + t
    results = analyze_symmetry_candidates(lidar, camera, include_all_permutations=True)

    assert results[0].permutation == [0, 1, 2, 3]
    assert results[0].fit_stats_mm["rmse"] < 1e-9
    assert results[1].fit_stats_mm["rmse"] > 10.0


def test_symmetric_four_hole_minimum_candidate_set_has_four_candidates():
    perms = candidate_permutations(include_all=False)
    assert [name for name, _ in perms] == ["NORMAL", "REVERSED", "MIRROR", "HALF_TURN"]


def test_physically_implausible_translation_is_not_preferred():
    lidar = _rectangle()
    camera = lidar + np.array([20.0, 0.0, 0.0])
    results = analyze_symmetry_candidates(
        lidar,
        camera,
        include_all_permutations=False,
        thresholds=PlausibilityThresholds(max_translation_norm_m=5.0),
    )
    decision = decide_candidate(results, PlausibilityThresholds(max_translation_norm_m=5.0))
    assert all("TRANSLATION_NORM_IMPLAUSIBLE" in r.plausibility_warnings for r in results)
    assert decision.status == "CALIBRATION_INVALID"


def test_seed_nearest_candidate_can_be_prior_evidence_without_physical_id_claim():
    R, t = _rigid_transform()
    lidar = _rectangle()
    camera = (R @ lidar.T).T + t
    results = analyze_symmetry_candidates(
        lidar,
        camera,
        seed_R=R,
        seed_t=t,
        include_all_permutations=False,
        thresholds=PlausibilityThresholds(seed_gate_enabled=False),
    )
    decision = decide_candidate(results, PlausibilityThresholds(seed_gate_enabled=False))
    assert decision.status == "CALIBRATION_AMBIGUOUS"
    assert decision.selection_source == "AMBIGUOUS"
    assert decision.preferred_candidate_name is None


def test_projection_good_candidate_can_be_prior_evidence_without_match_rate_collapse():
    R, t = _rigid_transform()
    lidar = _rectangle()
    camera = (R @ lidar.T).T + t
    results = analyze_symmetry_candidates(lidar, camera, include_all_permutations=False)
    _mark_cross_scene_good(results)
    for r in results:
        r.projection_summary = {
            "ok": True,
            "matched_median_px": 12.0,
            "matched_p95_px": 20.0,
            "match_rate": 0.8,
            "within_5px": 0.2,
            "within_10px": 0.4,
        }
    normal = next(r for r in results if r.name == "NORMAL")
    normal.projection_summary = {
        "ok": True,
        "matched_median_px": 2.0,
        "matched_p95_px": 5.0,
        "match_rate": 0.75,
        "within_5px": 0.7,
        "within_10px": 0.9,
    }

    decision = decide_candidate(results)
    assert decision.status == "CALIBRATION_ACCEPTABLE"
    assert decision.selection_source == "PRIOR_DISAMBIGUATION"
    assert decision.preferred_candidate_name == "NORMAL"


def test_seed_does_not_override_projection_and_cross_scene_evidence():
    R, t = _rigid_transform()
    lidar = _rectangle()
    camera = (R @ lidar.T).T + t
    results = analyze_symmetry_candidates(
        lidar,
        camera,
        seed_R=R,
        seed_t=t,
        include_all_permutations=False,
        thresholds=PlausibilityThresholds(seed_gate_enabled=False),
    )
    _mark_cross_scene_good(results)
    for r in results:
        r.projection_summary = {
            "ok": True,
            "matched_median_px": 15.0,
            "matched_p95_px": 25.0,
            "match_rate": 0.8,
        }
    half = next(r for r in results if r.name == "HALF_TURN")
    half.projection_summary = {
        "ok": True,
        "matched_median_px": 2.0,
        "matched_p95_px": 5.0,
        "match_rate": 0.8,
    }

    decision = decide_candidate(results, PlausibilityThresholds(seed_gate_enabled=False))
    assert decision.status == "CALIBRATION_ACCEPTABLE"
    assert decision.preferred_candidate_name == "HALF_TURN"


def test_near_tie_projection_stays_ambiguous():
    R, t = _rigid_transform()
    lidar = _rectangle()
    camera = (R @ lidar.T).T + t
    results = analyze_symmetry_candidates(lidar, camera, include_all_permutations=False)
    _mark_cross_scene_good(results)
    for r in results:
        r.projection_summary = {
            "ok": True,
            "matched_median_px": 5.0,
            "matched_p95_px": 8.0,
            "match_rate": 0.8,
        }

    decision = decide_candidate(results)
    assert decision.status == "CALIBRATION_AMBIGUOUS"


def test_physical_ids_from_both_sensors_would_be_separate_selection_source():
    R, t = _rigid_transform()
    lidar = _rectangle().copy()
    lidar[2] += np.array([0.07, 0.0, 0.0])
    camera = (R @ lidar.T).T + t
    results = analyze_symmetry_candidates(lidar, camera, include_all_permutations=True)

    decision = decide_candidate(results, physical_ids_from_both_sensors=True)
    assert decision.status == "CALIBRATION_ACCEPTABLE"
    assert decision.selection_source == "PHYSICAL_ID"


def test_analyze_calibration_symmetry_uses_camera_physical_order_and_exports_no_yaml(tmp_path):
    R, t = _rigid_transform()
    lidar1 = _rectangle()
    lidar2 = _rectangle() + np.array([0.2, 0.1, 0.0])
    camera1 = (R @ lidar1.T).T + t
    camera2 = (R @ lidar2.T).T + t
    scenes = [("a", lidar1, camera1), ("b", lidar2, camera2)]

    candidates, decision = analyze_calibration_symmetry(scenes, include_all_permutations=False)
    out = export_candidate_selection_evidence(candidates, decision, tmp_path)

    assert decision.status == "CALIBRATION_AMBIGUOUS"
    assert (out / "candidate_selection_evidence.json").exists()
    assert not list(tmp_path.glob("*.yaml"))
    normal = next(c for c in candidates if c.name == "NORMAL")
    assert normal.mapping[0] == "lidar_ordered_0->camera_physical_0"


def test_pca_axis_sign_flip_preserves_distances_but_changes_xy_sign():
    pts = _rectangle()
    xy = pca_plane_xy(pts)
    flipped = pts.copy()
    flipped[:, 0] *= -1.0
    xy_flipped = pca_plane_xy(flipped)

    assert pairwise_distance_matrix(pts) == pytest.approx(pairwise_distance_matrix(flipped))
    assert np.abs(xy) == pytest.approx(np.abs(xy_flipped))


def test_candidate_permutations_include_named_first_and_24_total():
    perms = candidate_permutations(include_all=True)
    assert [name for name, _ in perms[:4]] == ["NORMAL", "REVERSED", "MIRROR", "HALF_TURN"]
    assert len(perms) == 24


def test_lightweight_projection_proxy_prefers_points_on_image_edges():
    image = np.zeros((80, 100, 3), np.uint8)
    image[:, 50:] = 255
    u = np.repeat(np.arange(45, 56), 20)
    v = np.tile(np.linspace(0, 79, 20), 11)
    depth = np.where(u < 50, 4.0, 8.0)
    uv = np.column_stack([u, v]).astype(float)

    result = lightweight_projection_proxy(
        image,
        uv,
        depth,
        ProjectionProxyParams(max_projected_points=5000, max_lidar_edge_points=1000, grid_px=2),
    )

    assert result["ok"]
    assert result["matched_median_px"] < 2.0
    assert result["within_5px"] > 0.9
