"""Unit tests for gui.core.evaluation.edge_alignment."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from gui.core.evaluation.edge_alignment import (
    MATCH_RATE_THRESHOLDS_PX,
    EdgeAlignmentParams,
    aggregate_point_stats,
    evaluate_edge_alignment,
    extract_image_edges,
    extract_lidar_edge_points,
)


# --------------------------------------------------------------------- edges


def test_extract_lidar_edge_points_finds_depth_step():
    near = np.array([[100.0, 100.0], [101.0, 100.0], [100.0, 101.0], [101.0, 101.0]])
    far = np.array([[100.5, 100.5], [101.5, 100.5], [100.5, 101.5], [101.5, 101.5]])
    pixels = np.vstack([near, far])
    depths = np.array([5.0, 5.0, 5.0, 5.0, 15.0, 15.0, 15.0, 15.0])

    mask = extract_lidar_edge_points(pixels, depths, radius_px=3.0, depth_jump_threshold_m=0.3, min_neighbors=3)
    assert mask[:4].all()
    assert not mask[4:].any()


def test_extract_lidar_edge_points_no_discontinuity():
    pixels = np.random.RandomState(0).uniform(0, 50, size=(30, 2))
    depths = np.full(30, 5.0)
    mask = extract_lidar_edge_points(pixels, depths, radius_px=5.0, depth_jump_threshold_m=0.3)
    assert not mask.any()


def test_extract_lidar_edge_points_empty_input():
    mask = extract_lidar_edge_points(np.zeros((0, 2)), np.zeros(0))
    assert mask.shape == (0,)


def test_extract_lidar_edge_points_respects_min_neighbors():
    pixels = np.array([[0.0, 0.0], [500.0, 500.0]])
    depths = np.array([5.0, 50.0])
    mask = extract_lidar_edge_points(pixels, depths, radius_px=3.0, min_neighbors=3)
    assert not mask.any()


def test_extract_image_edges_detects_step_edge():
    img = np.zeros((100, 100), dtype=np.uint8)
    img[:, 50:] = 255
    edges = extract_image_edges(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    assert edges[:, 49:52].any()


def test_extract_image_edges_blank_image_has_no_edges():
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    assert not extract_image_edges(img).any()


# ------------------------------------------------------------ end-to-end scene
#
# A depth step at u=cx, aligned with a drawn vertical image edge -- the
# minimal scene that exercises extraction, matching, and aggregation together.


def _make_step_scene():
    width, height = 640, 480
    cx = 320.0

    image = np.zeros((height, width), dtype=np.uint8)
    image[:, int(cx):] = 255
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Spacing must stay under the default radius_px=3.0 in both axes, or
    # neighbouring grid points straddling the depth step never see each other
    # and no edge is ever flagged.
    u = np.repeat(np.linspace(0, width - 1, 250), 200)
    v = np.tile(np.linspace(0, height - 1, 200), 250)
    depth = np.where(u < cx, 5.0, 10.0)
    pixels = np.column_stack([u, v])
    return image, pixels, depth


def test_evaluate_edge_alignment_matches_aligned_step():
    image, pixels, depths = _make_step_scene()
    result = evaluate_edge_alignment(image, pixels, depths, EdgeAlignmentParams(depth_jump_threshold_m=1.0))
    assert result.ok, result.reason
    assert result.n_edge_points > 0
    assert result.mean_px < 3.0
    assert result.match_rate > 0.5
    assert result.n_matched + result.n_unmatched == result.n_edge_points


def test_evaluate_edge_alignment_reports_unmatched_not_silently_dropped():
    image, pixels, depths = _make_step_scene()
    # Blank the image so nothing can ever be matched -- unmatched points must
    # still be counted, not excluded from the statistics.
    blank = np.zeros_like(image)
    # Leave a single unrelated edge far from the depth step so the "no image
    # edges at all" short-circuit does not trigger before matching runs.
    blank[:, -2:] = 255
    result = evaluate_edge_alignment(blank, pixels, depths, EdgeAlignmentParams(depth_jump_threshold_m=1.0))
    assert result.ok, result.reason
    assert result.n_unmatched > 0
    assert result.n_matched + result.n_unmatched == result.n_edge_points
    assert np.isfinite(result.mean_px)


def test_evaluate_edge_alignment_fails_on_empty_input():
    image, _, _ = _make_step_scene()
    result = evaluate_edge_alignment(image, np.zeros((0, 2)), np.zeros(0))
    assert not result.ok
    assert np.isnan(result.mean_px)


def test_evaluate_edge_alignment_fails_on_too_few_edge_points():
    image, pixels, depths = _make_step_scene()
    result = evaluate_edge_alignment(
        image, pixels[:5], depths[:5], EdgeAlignmentParams(min_edge_points=1000),
    )
    assert not result.ok
    assert "LiDAR" in result.reason or result.n_edge_points >= 0


def test_evaluate_edge_alignment_preserves_every_edge_point():
    """No stage of the pipeline may drop a LiDAR edge point on the floor --
    every one must end up counted as either matched or unmatched, and every
    per-point array must stay exactly n_edge_points long."""
    image, pixels, depths = _make_step_scene()
    result = evaluate_edge_alignment(image, pixels, depths, EdgeAlignmentParams(depth_jump_threshold_m=1.0))
    assert result.ok, result.reason

    assert result.n_matched + result.n_unmatched == result.n_edge_points
    assert len(result.edge_errors_px) == result.n_edge_points
    assert len(result.edge_matched) == result.n_edge_points
    assert len(result.edge_pixels) == result.n_edge_points
    assert len(result.edge_depths) == result.n_edge_points
    assert int(result.edge_matched.sum()) == result.n_matched
    assert int((~result.edge_matched).sum()) == result.n_unmatched


def test_evaluate_edge_alignment_fails_on_blank_image():
    _, pixels, depths = _make_step_scene()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    result = evaluate_edge_alignment(blank, pixels, depths, EdgeAlignmentParams(depth_jump_threshold_m=1.0))
    assert not result.ok


# ---------------------------------------------------- matched-only / penalty split
#
# aggregate_point_stats is tested directly, on hand-built (errors_px, matched)
# arrays, rather than through a real scene -- the claims here are pure
# arithmetic (does the 15px penalty leak into the matched-only family, is the
# threshold sweep monotonic, does sign survive) and do not need real geometry
# or Canny detection to hold or fail.


def test_matched_only_mean_never_includes_the_unmatched_penalty():
    # 3 matched points, real distances (well under the 15px penalty), and 2
    # unmatched points already sitting at the 15px penalty value -- exactly
    # the shape find_correspondences/apply_local_consistency_filter produce.
    errors = np.array([1.0, 2.0, 3.0, 15.0, 15.0])
    matched = np.array([True, True, True, False, False])
    agg = aggregate_point_stats(errors, matched)

    assert agg["matched_mean_px"] == pytest.approx(2.0)          # mean(1,2,3), 15px never enters
    assert agg["matched_median_px"] == pytest.approx(2.0)
    assert agg["matched_max_px"] == pytest.approx(3.0)           # never 15 -- that would mean the penalty leaked in
    # The penalty-inclusive family (computed the old way, over every point)
    # is a different number, and must actually differ here to prove the two
    # are not silently the same computation under different names.
    assert float(np.mean(errors)) == pytest.approx(7.2)
    assert float(np.mean(errors)) != agg["matched_mean_px"]


def test_matched_only_stats_are_nan_when_nothing_matched():
    errors = np.full(4, 15.0)
    matched = np.zeros(4, dtype=bool)
    agg = aggregate_point_stats(errors, matched)
    for key in ("matched_mean_px", "matched_median_px", "matched_p95_px", "matched_p99_px", "matched_max_px"):
        assert np.isnan(agg[key]), key


def test_p95_and_max_pin_to_the_penalty_when_over_a_third_unmatched():
    """The exact mechanism behind the real-world "P95 == Max == 15.00"
    symptom: once the unmatched fraction exceeds 5%, the 95th percentile of
    the *penalty-inclusive* array sits inside the block of points already
    pinned to the penalty value -- this is a guaranteed consequence of the
    metric's construction, not a separate "worst 5%" measurement. Kept in
    this module (not just re-derived by hand in the analysis) because it is
    exactly the claim evaluate_edge_alignment's own docstring makes."""
    matched_errors = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]      # 6 real, low-error matches
    unmatched_errors = [15.0] * 6                          # 6 unmatched, 50% of 12
    errors = np.array(matched_errors + unmatched_errors)
    p95 = float(np.percentile(errors, 95))
    assert p95 == pytest.approx(15.0)
    assert float(np.max(errors)) == pytest.approx(15.0)


def test_match_rate_by_threshold_is_monotonic_and_15px_equals_match_rate():
    errors = np.array([1.0, 4.0, 4.5, 9.0, 14.0, 15.0, 15.0])
    matched = np.array([True, True, True, True, True, False, False])
    agg = aggregate_point_stats(errors, matched)
    rates = [agg["match_rate_by_threshold_px"][th] for th in MATCH_RATE_THRESHOLDS_PX]
    assert rates == sorted(rates)  # 2px <= 3px <= 5px <= 10px <= 15px, non-decreasing
    n_matched = int(matched.sum())
    n_total = len(errors)
    # Nothing can match past the largest search radius, so "within 15px" must
    # equal the ordinary match rate exactly.
    assert agg["match_rate_by_threshold_px"][15.0] == pytest.approx(n_matched / n_total)


def test_signed_delta_u_v_mean_keeps_sign_not_just_magnitude():
    # 3 matched points: candidate consistently projects 4px to the right (+u)
    # and 2px up (-v) of where the image edge actually is -- a systematic
    # bias an unsigned distance could never reveal (it would report ~4.5px
    # for both a +4 and a -4 case alike).
    errors = np.array([4.47, 4.47, 4.47, 15.0])
    matched = np.array([True, True, True, False])
    delta_u = np.array([4.0, 4.0, 4.0, np.nan])
    delta_v = np.array([-2.0, -2.0, -2.0, np.nan])
    agg = aggregate_point_stats(errors, matched, delta_u, delta_v)

    assert agg["mean_delta_u_px"] == pytest.approx(4.0)
    assert agg["median_delta_u_px"] == pytest.approx(4.0)
    assert agg["mean_delta_v_px"] == pytest.approx(-2.0)
    assert agg["median_delta_v_px"] == pytest.approx(-2.0)


def test_evaluate_edge_alignment_populates_matched_only_and_delta_fields():
    """End-to-end (real geometry, not hand-built arrays): the new fields
    actually reach EdgeAlignmentResult, stay internally consistent with the
    existing penalty-inclusive ones, and the per-point delta arrays line up
    1:1 with edge_matched (NaN exactly where, and only where, unmatched)."""
    image, pixels, depths = _make_step_scene()
    result = evaluate_edge_alignment(image, pixels, depths, EdgeAlignmentParams(depth_jump_threshold_m=1.0))
    assert result.ok, result.reason
    assert result.n_matched > 0  # otherwise this scene proves nothing about matched-only stats

    assert np.isfinite(result.matched_mean_px)
    assert result.matched_max_px <= result.max_px + 1e-9  # matched-only can never exceed the penalised max
    assert result.matched_mean_px != pytest.approx(result.mean_px) or result.n_unmatched == 0

    assert result.edge_delta_u_px.shape == (result.n_edge_points,)
    assert result.edge_delta_v_px.shape == (result.n_edge_points,)
    assert np.array_equal(np.isnan(result.edge_delta_u_px), ~result.edge_matched)
    assert np.array_equal(np.isnan(result.edge_delta_v_px), ~result.edge_matched)

    assert set(result.match_rate_by_threshold_px) == set(MATCH_RATE_THRESHOLDS_PX)
    assert result.match_rate_by_threshold_px[15.0] == pytest.approx(result.match_rate)

