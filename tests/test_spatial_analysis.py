"""Unit tests for gui.core.evaluation.spatial_analysis.

Builds synthetic EdgeAlignmentResult-shaped data directly rather than running
the full projection/edge pipeline -- this module only ever regroups arrays
`evaluate_edge_alignment` already produced, so that is what should be tested
here in isolation.
"""

from __future__ import annotations

import math

import numpy as np

from gui.core.evaluation.edge_alignment import EdgeAlignmentResult
from gui.core.evaluation.spatial_analysis import analyze_spatial


def _result(pixels, depths, errors, matched=None) -> EdgeAlignmentResult:
    pixels = np.asarray(pixels, float)
    return EdgeAlignmentResult(
        ok=True,
        edge_pixels=pixels,
        edge_depths=np.asarray(depths, float),
        edge_errors_px=np.asarray(errors, float),
        edge_matched=np.asarray(matched, bool) if matched is not None else np.ones(len(pixels), bool),
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
