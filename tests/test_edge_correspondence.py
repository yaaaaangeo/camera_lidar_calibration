"""Unit tests for gui.core.evaluation.edge_correspondence."""

from __future__ import annotations

import cv2
import numpy as np

from gui.core.evaluation.edge_correspondence import (
    CorrespondenceResult,
    EdgeOrientationMap,
    apply_local_consistency_filter,
    compute_edge_orientation_map,
    estimate_lidar_edge_orientations,
    find_correspondences,
)


def test_compute_edge_orientation_map_vertical_edge_is_vertical_tangent():
    img = np.zeros((100, 100), dtype=np.uint8)
    img[:, 50:] = 255
    om = compute_edge_orientation_map(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    # A vertical intensity step has a horizontal gradient, so its tangent
    # (gradient + 90) is vertical: ~90 degrees.
    col = om.orientation_deg[40:60, 49:51]
    strong = om.strength[40:60, 49:51] > 1.0
    assert np.allclose(col[strong], 90.0, atol=5.0)


def test_estimate_lidar_edge_orientations_vertical_line():
    pixels = np.column_stack([np.full(10, 100.0), np.linspace(0, 90, 10)])
    orientations = estimate_lidar_edge_orientations(pixels, k=4)
    assert np.allclose(orientations, 90.0, atol=1.0)


def test_estimate_lidar_edge_orientations_single_point_is_nan():
    orientations = estimate_lidar_edge_orientations(np.array([[10.0, 10.0]]))
    assert np.isnan(orientations).all()


def _manual_orientation_map(h=100, w=100):
    """A vertical edge segment and a perpendicular (horizontal) decoy segment,
    close to a LiDAR point whose own local orientation is vertical -- the
    orientation filter should prefer the farther, correctly-oriented one."""
    edge_mask = np.zeros((h, w), dtype=np.uint8)
    orientation_deg = np.zeros((h, w))
    strength = np.zeros((h, w))

    decoy_rc = (50, 51)   # 1px from the LiDAR point below -> nearest by distance
    real_rc = (50, 55)    # 5px away, but orientation-correct

    edge_mask[decoy_rc] = 255
    orientation_deg[decoy_rc] = 0.0     # horizontal tangent
    strength[decoy_rc] = 100.0

    edge_mask[real_rc] = 255
    orientation_deg[real_rc] = 90.0     # vertical tangent
    strength[real_rc] = 100.0

    return EdgeOrientationMap(edge_mask=edge_mask, orientation_deg=orientation_deg, strength=strength)


def test_find_correspondences_prefers_orientation_over_nearest_distance():
    om = _manual_orientation_map()
    lidar_pixels = np.array([[50.0, 50.0]])
    lidar_orientations = np.array([90.0])   # matches the farther, "real" edge

    result = find_correspondences(
        lidar_pixels, lidar_orientations, om, radii_px=(6.0,), max_orientation_diff_deg=30.0,
    )
    assert result.matched[0]
    assert np.allclose(result.matched_pixels[0], [55.0, 50.0])
    assert np.isclose(result.distance_px[0], 5.0)


def test_find_correspondences_grows_radius_when_first_radius_empty():
    om = _manual_orientation_map()
    lidar_pixels = np.array([[50.0, 50.0]])
    lidar_orientations = np.array([90.0])

    # Radius 2 finds nothing at all (nearest candidate is 5px away); radius 10
    # should then pick it up.
    result = find_correspondences(
        lidar_pixels, lidar_orientations, om, radii_px=(2.0, 10.0), max_orientation_diff_deg=30.0,
    )
    assert result.matched[0]
    assert np.isclose(result.distance_px[0], 5.0)


def test_find_correspondences_unmatched_point_gets_max_radius_penalty():
    om = _manual_orientation_map()
    lidar_pixels = np.array([[50.0, 50.0]])
    lidar_orientations = np.array([0.0])  # matches only the decoy, but decoy is filtered by radius below

    result = find_correspondences(
        lidar_pixels, lidar_orientations, om, radii_px=(0.5,), max_orientation_diff_deg=30.0,
    )
    assert not result.matched[0]
    assert result.distance_px[0] == 0.5
    assert result.rejection_reason[0] == "no_candidate"


def test_find_correspondences_never_drops_a_point_matched_or_not():
    """One matchable point plus two that cannot possibly find a candidate
    (nothing within radius, or wrong orientation) -- all three must still come
    back as a MATCHED-or-UNMATCHED entry, never removed from the arrays."""
    om = _manual_orientation_map()
    lidar_pixels = np.array([[50.0, 50.0], [0.0, 0.0], [90.0, 10.0]])
    lidar_orientations = np.array([90.0, 90.0, np.nan])

    result = find_correspondences(
        lidar_pixels, lidar_orientations, om, radii_px=(6.0,), max_orientation_diff_deg=30.0,
    )
    n = len(lidar_pixels)
    assert len(result.matched) == n
    assert len(result.distance_px) == n
    assert len(result.matched_pixels) == n
    assert len(result.rejection_reason) == n
    assert int(result.matched.sum()) + int((~result.matched).sum()) == n
    # Only the first point is anywhere near a candidate; the isolated point and
    # the NaN-orientation point must both survive as UNMATCHED, not vanish.
    assert result.matched[0]
    assert not result.matched[1]
    assert not result.matched[2]


def test_apply_local_consistency_filter_demotes_lone_bad_match():
    lidar_pixels = np.array([[float(i) * 10, 0.0] for i in range(6)])
    matched_pixels = lidar_pixels + np.array([2.0, 0.0])
    # One point's "match" points in a wildly different direction than its
    # neighbours' shared [2, 0] displacement.
    matched_pixels[3] = lidar_pixels[3] + np.array([2.0, 30.0])

    n = len(lidar_pixels)
    raw = CorrespondenceResult(
        matched=np.ones(n, dtype=bool),
        matched_pixels=matched_pixels,
        distance_px=np.linalg.norm(matched_pixels - lidar_pixels, axis=1),
        orientation_diff_deg=np.zeros(n),
        strength=np.full(n, 50.0),
        rejection_reason=[None] * n,
    )

    filtered = apply_local_consistency_filter(
        lidar_pixels, raw, k=4, max_angle_diff_deg=45.0, penalty_distance_px=15.0,
    )
    assert not filtered.matched[3]
    assert filtered.rejection_reason[3] == "consistency"
    assert filtered.distance_px[3] == 15.0
    # Everyone else was mutually consistent and should be untouched.
    assert filtered.matched[[0, 1, 2, 4, 5]].all()
    # A demotion must never shrink the arrays -- the point stays present,
    # just flipped to unmatched with the penalty distance.
    assert len(filtered.matched) == n
    assert len(filtered.distance_px) == n


def test_apply_local_consistency_filter_leaves_too_few_matches_alone():
    lidar_pixels = np.array([[0.0, 0.0], [10.0, 0.0]])
    matched_pixels = lidar_pixels + np.array([1.0, 0.0])
    raw = CorrespondenceResult(
        matched=np.array([True, True]),
        matched_pixels=matched_pixels,
        distance_px=np.array([1.0, 1.0]),
        orientation_diff_deg=np.zeros(2),
        strength=np.full(2, 50.0),
        rejection_reason=[None, None],
    )
    filtered = apply_local_consistency_filter(lidar_pixels, raw)
    assert filtered.matched.all()
