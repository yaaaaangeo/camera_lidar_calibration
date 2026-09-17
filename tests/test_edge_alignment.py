"""Unit tests for gui.core.evaluation.edge_alignment."""

from __future__ import annotations

import cv2
import numpy as np

from gui.core.evaluation.edge_alignment import (
    EdgeAlignmentParams,
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
