from __future__ import annotations

import numpy as np

from gui.core.evaluation.lidar_target_debug import board_likeness, extent_metrics, plane_local_frame
from gui.core.project import Target


class _Det:
    ok = False
    reason = "debug"
    n_circles = 0


def test_plane_local_frame_and_extent():
    pts = np.array([
        [-1.0, -0.5, 0.0],
        [1.0, -0.5, 0.0],
        [1.0, 0.5, 0.0],
        [-1.0, 0.5, 0.0],
    ])
    center, axes, local = plane_local_frame(pts)
    ext = extent_metrics(local)

    assert np.allclose(center, [0.0, 0.0, 0.0])
    assert axes.shape == (3, 3)
    assert np.isclose(ext["width_m"], 2.0)
    assert np.isclose(ext["height_m"], 1.0)


def test_board_likeness_returns_scores():
    target = Target(delta_width_circles=0.5, delta_height_circles=0.4, circle_radius=0.1)
    xy = np.array([
        [-0.35, -0.3],
        [0.35, -0.3],
        [0.35, 0.3],
        [-0.35, 0.3],
    ])
    score = board_likeness(xy, xy[:2], _Det(), target)

    assert score["n"] == 4
    assert 0.0 <= score["size_similarity"] <= 1.0
    assert score["boundary_score"] == 0.5
