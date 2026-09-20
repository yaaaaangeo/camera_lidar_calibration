"""Unit tests for gui.core.evaluation.spatial_analysis.

Builds synthetic EdgeAlignmentResult-shaped data directly rather than running
the full projection/edge pipeline -- this module only ever regroups arrays
`evaluate_edge_alignment` already produced, so that is what should be tested
here in isolation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gui.core.evaluation.edge_alignment import EdgeAlignmentResult
from gui.core.evaluation.spatial_analysis import analyze_spatial


def _result(pixels, depths, errors, matched=None, delta_u=None, delta_v=None) -> EdgeAlignmentResult:
    pixels = np.asarray(pixels, float)
    return EdgeAlignmentResult(
        ok=True,
        edge_pixels=pixels,
        edge_depths=np.asarray(depths, float),
        edge_errors_px=np.asarray(errors, float),
        edge_matched=np.asarray(matched, bool) if matched is not None else np.ones(len(pixels), bool),
        edge_delta_u_px=np.asarray(delta_u, float) if delta_u is not None else None,
        edge_delta_v_px=np.asarray(delta_v, float) if delta_v is not None else None,
    )


def test_analyze_spatial_none_when_result_failed():
    failed = EdgeAlignmentResult(ok=False, reason="x")
    assert analyze_spatial(failed, 640, 480) is None


def test_analyze_spatial_depth_bins_one_point_each():
    pixels = np.full((5, 2), 150.0)  # all land in the CENTER/CENTER region
    depths = [5.0, 15.0, 25.0, 40.0, 60.0]
    errors = [1.0, 2.0, 3.0, 4.0, 5.0]
    sa = analyze_spatial(_result(pixels, depths, errors), image_width=300, image_height=300)

    for label, expected in zip(
        ("0-10m", "10-20m", "20-30m", "30-50m", "50m+"), (1.0, 2.0, 3.0, 4.0, 5.0)
    ):
        stats = sa.depth_bins[label]
        assert stats.n_points == 1
        assert stats.mean_px == expected
        assert stats.median_px == expected

    # All five points sit at u=v=150 in a 300x300 image -> the CENTER/CENTER cell.
    assert sa.horizontal["CENTER"].n_points == 5
    assert sa.horizontal["LEFT"].n_points == 0
    assert math.isnan(sa.horizontal["LEFT"].mean_px)


def test_analyze_spatial_horizontal_regions():
    pixels = [[10.0, 150.0], [150.0, 150.0], [290.0, 150.0]]
    errors = [1.0, 2.0, 3.0]
    matched = [True, True, False]
    sa = analyze_spatial(_result(pixels, [5.0, 5.0, 5.0], errors, matched), image_width=300, image_height=300)

    assert sa.horizontal["LEFT"].mean_px == 1.0
    assert sa.horizontal["LEFT"].n_matched == 1 and sa.horizontal["LEFT"].n_unmatched == 0
    assert sa.horizontal["CENTER"].mean_px == 2.0
    assert sa.horizontal["RIGHT"].mean_px == 3.0
    assert sa.horizontal["RIGHT"].n_matched == 0 and sa.horizontal["RIGHT"].n_unmatched == 1


def test_analyze_spatial_vertical_regions():
    pixels = [[150.0, 10.0], [150.0, 150.0], [150.0, 290.0]]
    errors = [10.0, 20.0, 30.0]
    sa = analyze_spatial(_result(pixels, [5.0, 5.0, 5.0], errors), image_width=300, image_height=300)

    assert sa.vertical["TOP"].mean_px == 10.0
    assert sa.vertical["CENTER"].mean_px == 20.0
    assert sa.vertical["BOTTOM"].mean_px == 30.0


def test_analyze_spatial_std_and_p95_over_multiple_points():
    pixels = np.full((4, 2), 150.0)
    errors = [1.0, 2.0, 3.0, 4.0]
    sa = analyze_spatial(_result(pixels, [5.0] * 4, errors), image_width=300, image_height=300)
    bin_ = sa.depth_bins["0-10m"]
    assert bin_.n_points == 4
    assert np.isclose(bin_.std_px, np.std(errors))
    assert np.isclose(bin_.p95_px, np.percentile(errors, 95))


# ------------------------------------------------------ matched-only per bin


def test_analyze_spatial_left_center_right_match_rate_computed_independently():
    """Each of LEFT/CENTER/RIGHT gets its own match rate from only its own
    points -- a region with a different mix of matched/unmatched must not
    leak into another region's rate."""
    pixels = [
        [10.0, 150.0], [10.0, 150.0],       # LEFT: 2 points, both matched
        [150.0, 150.0], [150.0, 150.0], [150.0, 150.0], [150.0, 150.0],  # CENTER: 4 points, 1 matched
        [290.0, 150.0],                      # RIGHT: 1 point, unmatched
    ]
    errors = [1.0, 2.0, 3.0, 15.0, 15.0, 15.0, 15.0]
    matched = [True, True, True, False, False, False, False]
    sa = analyze_spatial(_result(pixels, [5.0] * 7, errors, matched), image_width=300, image_height=300)

    assert sa.horizontal["LEFT"].match_rate == pytest.approx(1.0)
    assert sa.horizontal["LEFT"].matched_mean_px == pytest.approx(1.5)  # mean(1, 2) -- 15px never enters
    assert sa.horizontal["CENTER"].match_rate == pytest.approx(0.25)
    assert sa.horizontal["CENTER"].matched_mean_px == pytest.approx(3.0)  # the one matched point only
    assert sa.horizontal["RIGHT"].match_rate == pytest.approx(0.0)
    assert math.isnan(sa.horizontal["RIGHT"].matched_mean_px)  # nothing matched -- must not read as 0 or 15


def test_analyze_spatial_top_center_bottom_match_rate_same_as_horizontal():
    pixels = [[150.0, 10.0], [150.0, 10.0], [150.0, 150.0], [150.0, 290.0]]
    errors = [1.0, 3.0, 15.0, 5.0]
    matched = [True, True, False, True]
    sa = analyze_spatial(_result(pixels, [5.0] * 4, errors, matched), image_width=300, image_height=300)

    assert sa.vertical["TOP"].match_rate == pytest.approx(1.0)
    assert sa.vertical["TOP"].matched_mean_px == pytest.approx(2.0)
    assert sa.vertical["CENTER"].match_rate == pytest.approx(0.0)
    assert sa.vertical["BOTTOM"].match_rate == pytest.approx(1.0)


def test_analyze_spatial_distance_bins_carry_match_rate_and_matched_mean():
    """0-10m vs 10-20m with different unmatched ratios -- the whole point of
    requirement 3: a near-field error increase must be distinguishable from
    a near-field match-rate collapse."""
    pixels = np.full((6, 2), 150.0)
    depths = [5.0, 5.0, 5.0, 15.0, 15.0, 15.0]
    errors = [2.0, 4.0, 15.0, 6.0, 6.0, 6.0]           # 0-10m: 1/3 unmatched; 10-20m: all matched
    matched = [True, True, False, True, True, True]
    sa = analyze_spatial(_result(pixels, depths, errors, matched), image_width=300, image_height=300)

    near, far = sa.depth_bins["0-10m"], sa.depth_bins["10-20m"]
    assert near.match_rate == pytest.approx(2 / 3)
    assert near.matched_mean_px == pytest.approx(3.0)   # mean(2, 4)
    assert far.match_rate == pytest.approx(1.0)
    assert far.matched_mean_px == pytest.approx(6.0)


def test_analyze_spatial_signed_delta_u_v_per_bin():
    pixels = [[10.0, 150.0], [10.0, 150.0], [290.0, 150.0]]
    errors = [4.0, 4.0, 15.0]
    matched = [True, True, False]
    delta_u = [4.0, 4.0, float("nan")]
    delta_v = [-1.0, -1.0, float("nan")]
    sa = analyze_spatial(_result(pixels, [5.0] * 3, errors, matched, delta_u, delta_v),
                          image_width=300, image_height=300)

    assert sa.horizontal["LEFT"].mean_delta_u_px == pytest.approx(4.0)
    assert sa.horizontal["LEFT"].median_delta_u_px == pytest.approx(4.0)
    assert sa.horizontal["LEFT"].mean_delta_v_px == pytest.approx(-1.0)
    assert math.isnan(sa.horizontal["RIGHT"].mean_delta_u_px)  # no matched point in RIGHT -- no signed offset to report


def test_analyze_spatial_gracefully_handles_missing_delta_arrays():
    """A result built without edge_delta_u_px/edge_delta_v_px (e.g. an older
    caller, or a unit test that only cares about distance) must not raise --
    it should just read as "no signed-offset data available"."""
    pixels = np.full((3, 2), 150.0)
    sa = analyze_spatial(_result(pixels, [5.0] * 3, [1.0, 2.0, 3.0]), image_width=300, image_height=300)
    assert math.isnan(sa.depth_bins["0-10m"].mean_delta_u_px)
